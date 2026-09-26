//! Scalar shifted approximant bases.
//!
//! For `F` in Fq\[x\]^(m by n) and orders `d[j]`, the approximant module
//! holds the rows `p` in Fq\[x\]^(1 by m) with `(p * F)[j] = 0` modulo
//! `x^d[j]` for every column `j`. A zero order imposes no constraint; no
//! constraints at all return `I_m`. Shifts act on the `m` coordinates of
//! `p`: the output reduces to the requested shifted form.
//!
//! The scalar constructor starts from `I_m` and imposes one coefficient
//! constraint at a time in a fixed column/order sequence. For column `j`
//! and order `k`, the discrepancies `delta[i] = coeff_k((B * F)[i,j])`
//! come from the unchanged current basis. When all vanish the basis is
//! kept. Otherwise the pivot minimizes its shifted row-leading pair
//! `(shifted degree, leading column)`, then row index, among
//! nonzero-discrepancy rows; every other such row cancels against the old
//! pivot row, and the pivot multiplies by `x` last. Degrees and leading
//! metadata recompute before the next constraint, so all previously
//! imposed constraints stay satisfied and the basis spans their entire
//! solution module.
//!
//! Every independent scalar constraint multiplies the module index by
//! the field size and raises the determinant degree by one; redundant
//! constraints do neither. The constructor reports the
//! independent-constraint count for verification.

use alloc::vec::Vec;

use fgf::field::Elem;
use fgf::kernel::FieldKernels;
use poly_ring::Polynomial;

use crate::matrix::{MatrixError, PolynomialMatrix};

/// A shifted approximant basis with its independent-constraint count.
#[derive(Debug, Clone)]
pub struct ApproximantBasis<F: FieldKernels> {
    /// Canonical m-by-m module basis.
    pub basis: PolynomialMatrix<F>,
    /// Number of independent scalar constraints imposed.
    pub independent_constraints: usize,
}

impl<F: FieldKernels> PolynomialMatrix<F> {
    /// Returns the shifted approximant basis for `F = self` with per-column
    /// orders `d[j]`, reduced to canonical shifted row Popov form.
    ///
    /// `orders` holds exactly one entry per column; a zero order imposes
    /// no constraint. `shifts` holds one entry per solution coordinate
    /// (`m` entries).
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::GeometryOverflow`] when the order count
    /// disagrees with the shape, [`MatrixError::ShiftCount`] when the
    /// shift count does, plus polynomial and reduction errors.
    pub fn approximant_basis(
        &self,
        orders: &[usize],
        shifts: &[usize],
    ) -> Result<ApproximantBasis<F>, MatrixError> {
        let (rows, columns) = self.shape();
        if orders.len() != columns {
            return Err(MatrixError::GeometryOverflow {
                context: "approximant order count",
            });
        }
        if shifts.len() != rows {
            return Err(MatrixError::ShiftCount {
                columns: rows,
                shifts: shifts.len(),
            });
        }
        let mut basis = PolynomialMatrix::identity(rows)?;
        let mut independent = 0usize;
        for (column, &order) in orders.iter().enumerate() {
            for coefficient in 0..order {
                if Self::impose_scalar_constraint(self, &mut basis, column, coefficient, shifts)? {
                    independent += 1;
                }
            }
        }
        basis.reduce_popov(shifts)?;
        Ok(ApproximantBasis {
            basis,
            independent_constraints: independent,
        })
    }

    /// Imposes one scalar constraint `(B * F)[i,column] = 0` at `x^order`.
    ///
    /// Returns whether the constraint was independent (a nonzero
    /// discrepancy existed).
    fn impose_scalar_constraint(
        input: &PolynomialMatrix<F>,
        basis: &mut PolynomialMatrix<F>,
        column: usize,
        order: usize,
        shifts: &[usize],
    ) -> Result<bool, MatrixError> {
        let (rows, _) = basis.shape();
        // Discrepancies from the unchanged current basis.
        let mut discrepancies = Vec::with_capacity(rows);
        for row in 0..rows {
            discrepancies.push(Self::residual_coefficient(
                input, basis, row, column, order,
            )?);
        }
        if discrepancies.iter().all(|delta| delta.is_zero()) {
            return Ok(false);
        }
        // Pivot: minimum shifted row-leading pair among nonzero-discrepancy
        // rows, then row index.
        let mut pivot: Option<usize> = None;
        let mut pivot_key: Option<(usize, usize)> = None;
        for (row, discrepancy) in discrepancies.iter().enumerate().take(rows) {
            if discrepancy.is_zero() {
                continue;
            }
            let key = Self::shifted_row_key(basis, row, shifts)?;
            if pivot_key.is_none_or(|current| key < current) {
                pivot_key = Some(key);
                pivot = Some(row);
            }
        }
        let pivot = pivot.ok_or(MatrixError::GeometryOverflow {
            context: "approximant pivot selection",
        })?;
        let pivot_discrepancy = discrepancies[pivot];
        let pivot_row: Vec<Polynomial<F>> = {
            let (_, basis_columns) = basis.shape();
            let mut entries = Vec::with_capacity(basis_columns);
            for other in 0..basis_columns {
                entries.push(basis.entry(pivot, other).cloned().unwrap_or_default());
            }
            entries
        };
        // Cancel every other nonzero-discrepancy row against the OLD pivot.
        for (row, discrepancy) in discrepancies.iter().enumerate().take(rows) {
            if row == pivot || discrepancy.is_zero() {
                continue;
            }
            let ratio = discrepancy.mul(pivot_discrepancy.inv()).neg();
            for (other, pivot_entry) in pivot_row.iter().enumerate() {
                let mut target = basis.entry(row, other).cloned().unwrap_or_default();
                target.add_scaled_assign(ratio, pivot_entry)?;
                basis.set_entry(row, other, target)?;
            }
        }
        // Multiply the pivot by x last.
        let (_, basis_columns) = basis.shape();
        for other in 0..basis_columns {
            let entry = basis.entry(pivot, other).cloned().unwrap_or_default();
            basis.set_entry(pivot, other, entry.shifted(1)?)?;
        }
        Ok(true)
    }

    /// Returns the `x^order` coefficient of row `row` of `basis * input`
    /// restricted to one column.
    fn residual_coefficient(
        input: &PolynomialMatrix<F>,
        basis: &PolynomialMatrix<F>,
        row: usize,
        column: usize,
        order: usize,
    ) -> Result<F::Elem, MatrixError> {
        let (input_rows, _) = input.shape();
        let (_, basis_columns) = basis.shape();
        let mut accumulator = Polynomial::zero();
        for inner in 0..input_rows.min(basis_columns) {
            let left = basis.entry(row, inner).cloned().unwrap_or_default();
            let right = input.entry(inner, column).cloned().unwrap_or_default();
            if left.is_zero() || right.is_zero() {
                continue;
            }
            let product = left.multiply(&right)?;
            accumulator.add_assign(&product)?;
        }
        Ok(accumulator.coefficient(order))
    }

    /// Returns the shifted row-leading pair `(shifted degree, column)` of
    /// a basis row, or `MAX` when the row is zero.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::Reduction`] on shifted-degree overflow.
    fn shifted_row_key(
        basis: &PolynomialMatrix<F>,
        row: usize,
        shifts: &[usize],
    ) -> Result<(usize, usize), MatrixError> {
        let (_, columns) = basis.shape();
        let mut best: Option<(usize, usize)> = None;
        for (column, &shift) in shifts.iter().enumerate().take(columns) {
            let Some(entry) = basis.entry(row, column) else {
                continue;
            };
            let Some(degree) = entry.degree() else {
                continue;
            };
            let shifted = degree.checked_add(shift).ok_or(MatrixError::Reduction(
                crate::ReduceError::DegreeOverflow { degree, shift },
            ))?;
            if best.is_none_or(|(best_shifted, best_column)| {
                (shifted, column) > (best_shifted, best_column)
            }) {
                best = Some((shifted, column));
            }
        }
        Ok(best.unwrap_or((usize::MAX, usize::MAX)))
    }
}
