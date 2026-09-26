//! Certified rank, kernels, membership, and exact solving over Fq\[x\].
//!
//! Every routine reduces a copy of its input to shifted weak Popov form
//! with an explicit unimodular witness `U` satisfying `U * A = R`. Zero rows
//! of `R` mark the kernel generators: the matching rows of `U` span the
//! left kernel, whose dimension plus the nonzero-row count equals the row
//! count. Right kernels and column solves transpose through the same
//! primitive; rank is over Fq(x) while every witness carries Fq[x]
//! coefficients.

use alloc::vec::Vec;

use fgf::kernel::FieldKernels;
use poly_ring::Polynomial;

use crate::matrix::{MatrixError, PolynomialMatrix};

/// Row-module membership of `target` in the rows of `basis`.
#[derive(Debug, Clone)]
pub struct MembershipWitness<F: FieldKernels> {
    /// Coefficients with `sum_i quotients[i] * basis[i] == target`.
    pub quotients: Vec<Polynomial<F>>,
}

/// A polynomial solution of `matrix * solution == target`.
#[derive(Debug, Clone)]
pub struct ExactSolution<F: FieldKernels> {
    /// One particular polynomial solution.
    pub particular: Vec<Polynomial<F>>,
    /// Basis of the complete right-kernel module; empty when the solution
    /// is unique.
    pub kernel: PolynomialMatrix<F>,
}

impl<F: FieldKernels> PolynomialMatrix<F> {
    /// Returns the rank over Fq(x): the count of nonzero rows after
    /// shifted weak Popov reduction.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`PolynomialMatrix::reduce_weak_popov`].
    pub fn rank_weak_popov(&self, shifts: &[usize]) -> Result<usize, MatrixError> {
        let mut reduced = self.clone();
        reduced.reduce_weak_popov(shifts)?;
        Ok(reduced.nonzero_row_count())
    }

    /// Returns the shifted leading column of every row, or `None` for a
    /// zero row.
    ///
    /// The input must already be reduced; the query reads leading terms
    /// without verifying that precondition.
    #[must_use]
    pub fn reduced_leading_columns(&self, shifts: &[usize]) -> Vec<Option<usize>> {
        (0..self.shape().0)
            .map(|row| self.unchecked_leading_column(row, shifts))
            .collect()
    }

    /// Returns a basis of the left kernel: every row `k` of the output
    /// satisfies `k * self == 0`.
    ///
    /// The output has one row per kernel generator and `m` columns, where
    /// `m` is the input row count; it is empty when the rows are
    /// independent. Completeness is certified by the tracked reduction:
    /// `U * A = R` with `U` unimodular, so the rows of `U` matching zero
    /// rows of `R` generate the entire kernel.
    ///
    /// # Errors
    ///
    /// Returns the same errors as
    /// [`PolynomialMatrix::reduce_weak_popov_tracked`].
    pub fn left_kernel_weak_popov(
        &self,
        shifts: &[usize],
    ) -> Result<PolynomialMatrix<F>, MatrixError> {
        let mut reduced = self.clone();
        let transform = reduced.reduce_weak_popov_tracked(shifts)?;
        let mut generators = Vec::new();
        for row in 0..reduced.shape().0 {
            if reduced.row_is_zero(row) {
                for column in 0..self.shape().0 {
                    generators.push(transform.entry(row, column).cloned().unwrap_or_default());
                }
            }
        }
        let rows_out = if self.shape().0 == 0 {
            0
        } else {
            generators.len() / self.shape().0
        };
        if generators.is_empty() {
            return PolynomialMatrix::zeros(0, self.shape().0);
        }
        PolynomialMatrix::from_entries(rows_out, self.shape().0, generators)
    }

    /// Returns a basis of the right kernel: every column `n` of the output
    /// satisfies `self * n == 0`, returned as rows of the transposed
    /// problem. The transpose carries zero shifts: column operations on the
    /// input are row operations on the transpose with no column weighting.
    ///
    /// The output has one row per kernel generator and `n` columns, where
    /// `n` is the input column count; it is empty when the columns are
    /// independent.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`PolynomialMatrix::left_kernel_weak_popov`].
    pub fn right_kernel_weak_popov(
        &self,
        _shifts: &[usize],
    ) -> Result<PolynomialMatrix<F>, MatrixError> {
        let transposed = self.transposed()?;
        let column_shifts = alloc::vec![0; self.shape().0];
        transposed.left_kernel_weak_popov(&column_shifts)
    }

    /// Reduces `target` against the nonzero rows of an already reduced
    /// basis, returning polynomial quotients with
    /// `sum_i quotients[i] * self[i] == target`.
    ///
    /// The input must already be in shifted weak Popov form under `shifts`;
    /// the routine reads leading terms without verifying that precondition.
    /// Each step cancels the target leading term against the pivot in the
    /// same column when the pivot degree permits a polynomial multiple, and
    /// strictly decreases the target leading monomial. A nonzero
    /// unreducible remainder proves non-membership.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::GeometryOverflow`] when `target` has the wrong
    /// length or no pivot cancels its leading term, plus polynomial errors.
    pub fn membership_weak_popov(
        &self,
        target: &[Polynomial<F>],
        shifts: &[usize],
    ) -> Result<MembershipWitness<F>, MatrixError> {
        let (rows, columns) = self.shape();
        if target.len() != columns {
            return Err(MatrixError::GeometryOverflow {
                context: "membership target length",
            });
        }
        let pivots = self.row_pivots(shifts);
        let mut remainder: Vec<Polynomial<F>> = target.to_vec();
        let mut quotients: Vec<Polynomial<F>> = alloc::vec![Polynomial::zero(); rows];
        while let Some((column, degree, coefficient)) = leading_at(&remainder, shifts) {
            // `leading_at` observes only the first `shifts.len()` entries;
            // a target longer than the shift vector has no reduction rule.
            if column >= shifts.len() {
                return Err(MatrixError::GeometryOverflow {
                    context: "membership irreducible remainder",
                });
            }
            let Some((pivot, pivot_degree, pivot_coefficient)) = pivots.iter().find_map(|pivot| {
                (pivot.column == column && pivot.degree <= degree).then_some((
                    pivot.row,
                    pivot.degree,
                    pivot.coefficient,
                ))
            }) else {
                return Err(MatrixError::GeometryOverflow {
                    context: "membership irreducible remainder",
                });
            };
            let ratio = coefficient.mul(pivot_coefficient.inv());
            let shift = degree - pivot_degree;
            let pivot_row = self.row_polynomial(pivot);
            // The quotient accumulates +ratio * x^shift while the remainder
            // subtracts it; the reducer's cancelling scalar is its negation.
            let scale = ratio.neg();
            let unit =
                Polynomial::from_coefficients(&[F::Elem::ONE]).map_err(MatrixError::Polynomial)?;
            let mut term = Polynomial::zero();
            term.add_scaled_shifted_assign(ratio, &unit, shift)?;
            quotients[pivot].add_assign(&term)?;
            for (column, entry) in remainder.iter_mut().enumerate() {
                entry.add_scaled_shifted_assign(scale, &pivot_row[column], shift)?;
            }
        }
        for entry in &remainder {
            if !entry.is_zero() {
                return Err(MatrixError::GeometryOverflow {
                    context: "membership irreducible remainder",
                });
            }
        }
        Ok(MembershipWitness { quotients })
    }

    /// Solves `self * solution == target` over Fq\[x\].
    ///
    /// Returns a particular polynomial solution plus the complete
    /// right-kernel basis. The equation `x z == 1` has no polynomial
    /// solution and reports non-membership; `x z == x` solves with `1`.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`PolynomialMatrix::membership_weak_popov`]
    /// and [`PolynomialMatrix::right_kernel_weak_popov`].
    ///
    /// # Panics
    ///
    /// Panics when the transpose reduction reports a different row count
    /// than the input column count; the tracked reducer preserves row
    /// counts, so this never fires on a successful reduction.
    pub fn solve_weak_popov(
        &self,
        target: &[Polynomial<F>],
        shifts: &[usize],
    ) -> Result<ExactSolution<F>, MatrixError> {
        let (rows, _) = self.shape();
        if target.len() != rows {
            return Err(MatrixError::GeometryOverflow {
                context: "solve target length",
            });
        }
        // Column solve `A z = b` is row membership in the transpose: with
        // `R = U * A^T` reduced and `sum_i q[i] * R[i] == b`, the original
        // coordinates satisfy `(q * U) * A^T == b`, so
        // `z[j] = sum_i q[i] * U[i,j]`. The transpose has `n` rows and `m`
        // columns, so it reduces under `m` zero shifts regardless of the
        // caller-supplied column shifts.
        let (rows, columns) = self.shape();
        let transposed = self.transposed()?;
        let mut reduced = transposed;
        let row_shifts = alloc::vec![0; rows];
        let transform = reduced.reduce_weak_popov_tracked(&row_shifts)?;
        let witness = reduced.membership_weak_popov(target, &row_shifts)?;
        assert_eq!(witness.quotients.len(), columns);
        let mut particular = alloc::vec![Polynomial::zero(); columns];
        for (index, quotient) in witness.quotients.iter().enumerate() {
            if quotient.is_zero() {
                continue;
            }
            for (column, slot) in particular.iter_mut().enumerate().take(columns) {
                let entry = transform.entry(index, column).cloned().unwrap_or_default();
                if entry.is_zero() {
                    continue;
                }
                let contribution = entry.multiply(quotient)?;
                slot.add_assign(&contribution)?;
            }
        }
        let kernel = self.right_kernel_weak_popov(shifts)?;
        Ok(ExactSolution { particular, kernel })
    }

    fn nonzero_row_count(&self) -> usize {
        (0..self.shape().0)
            .filter(|&row| !self.row_is_zero(row))
            .count()
    }

    fn row_is_zero(&self, row: usize) -> bool {
        let (_, columns) = self.shape();
        (0..columns).all(|column| self.entry(row, column).is_none_or(Polynomial::is_zero))
    }

    fn unchecked_leading_column(&self, row: usize, shifts: &[usize]) -> Option<usize> {
        let (_, columns) = self.shape();
        let mut best: Option<(usize, usize)> = None;
        for (column, &shift) in shifts.iter().enumerate().take(columns) {
            let Some(entry) = self.entry(row, column) else {
                continue;
            };
            let Some(degree) = entry.degree() else {
                continue;
            };
            let Some(shifted) = degree.checked_add(shift) else {
                continue;
            };
            if best.is_none_or(|(best_column, current)| (shifted, column) > (current, best_column))
            {
                best = Some((column, shifted));
            }
        }
        best.map(|(column, _)| column)
    }

    fn row_pivots(&self, shifts: &[usize]) -> Vec<RowPivot<F>> {
        let (rows, columns) = self.shape();
        let mut pivots = Vec::new();
        for row in 0..rows {
            let mut best: Option<(usize, usize, F::Elem)> = None;
            for (column, &shift) in shifts.iter().enumerate().take(columns) {
                let Some(entry) = self.entry(row, column) else {
                    continue;
                };
                let Some(degree) = entry.degree() else {
                    continue;
                };
                let Some(shifted) = degree.checked_add(shift) else {
                    continue;
                };
                let coefficient = entry.coefficient(degree);
                if best.is_none_or(|(best_column, current, _)| {
                    (shifted, column) > (current, best_column)
                }) {
                    best = Some((column, shifted, coefficient));
                }
            }
            if let Some((column, _, coefficient)) = best {
                let degree = self
                    .entry(row, column)
                    .and_then(Polynomial::degree)
                    .unwrap_or(0);
                pivots.push(RowPivot {
                    row,
                    column,
                    degree,
                    coefficient,
                });
            }
        }
        pivots
    }

    fn row_polynomial(&self, row: usize) -> Vec<Polynomial<F>> {
        let (_, columns) = self.shape();
        let mut entries = Vec::with_capacity(columns);
        for column in 0..columns {
            entries.push(self.entry(row, column).cloned().unwrap_or_default());
        }
        entries
    }
}

struct RowPivot<F: FieldKernels> {
    row: usize,
    column: usize,
    degree: usize,
    coefficient: F::Elem,
}

fn leading_at<F: FieldKernels>(
    row: &[Polynomial<F>],
    shifts: &[usize],
) -> Option<(usize, usize, F::Elem)> {
    let mut best: Option<(usize, usize, F::Elem)> = None;
    for (column, entry) in row.iter().enumerate() {
        let shift = *shifts.get(column)?;
        let degree = entry.degree()?;
        let shifted = degree.checked_add(shift)?;
        let coefficient = entry.coefficient(degree);
        if best.is_none_or(|(best_column, current, _)| (shifted, column) > (current, best_column)) {
            best = Some((column, shifted, coefficient));
        }
    }
    best.map(|(column, shifted, coefficient)| {
        let shift = shifts.get(column).copied().unwrap_or(0);
        (column, shifted - shift, coefficient)
    })
}

use fgf::field::Elem;
