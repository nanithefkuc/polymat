//! Confluent interpolation bases and Padé-family module relations.
//!
//! A confluent constraint set holds triples `(point, column, multiplicity)`
//! requiring `(p * F)[column] = 0` modulo `(x - point)^multiplicity`. The
//! solution module is a congruence module whose column moduli are products
//! of the local factors for that column; the scalar recurrence below
//! imposes one Hasse coefficient at a time in a deterministic
//! point/column/multiplicity order.
//!
//! Duplicate constraints for the same point and column merge by maximum
//! multiplicity; distinct columns stay independent even at equal points.
//! An order of zero imposes no condition. All discrepancies use Hasse
//! (Taylor) coefficients from `poly-ring`, never ordinary derivatives
//! divided by factorials, so multiplicities at and above the
//! characteristic are sound.
//!
//! The Padé constructors are thin module relations over the scalar
//! approximant recurrence with explicit signs and numerator/denominator
//! orientation. They own no decoding radius, locator selection, or
//! candidate policy.

use alloc::vec::Vec;

use fgf::field::Elem;
use fgf::kernel::FieldKernels;
use poly_ring::Polynomial;

use crate::matrix::{MatrixError, PolynomialMatrix};

/// One confluent interpolation constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InterpolationConstraint<F: FieldKernels> {
    /// Evaluation point.
    pub point: F::Elem,
    /// Input column the constraint applies to.
    pub column: usize,
    /// Multiplicity: the first `multiplicity` Hasse coefficients vanish.
    pub multiplicity: usize,
}

impl<F: FieldKernels> InterpolationConstraint<F> {
    /// Builds a constraint of `multiplicity` at `point` on `column`.
    #[must_use]
    pub const fn new(point: F::Elem, column: usize, multiplicity: usize) -> Self {
        Self {
            point,
            column,
            multiplicity,
        }
    }
}

/// A confluent interpolation basis with its independent-constraint count.
#[derive(Debug, Clone)]
pub struct InterpolationBasis<F: FieldKernels> {
    /// Canonical m-by-m module basis.
    pub basis: PolynomialMatrix<F>,
    /// Number of independent scalar constraints imposed.
    pub independent_constraints: usize,
}

/// A Padé-type relation `numerator - target * denominator = 0` modulo the
/// given order, with explicit orientation.
#[derive(Debug, Clone)]
pub struct PadeRelation<F: FieldKernels> {
    /// Numerator polynomial.
    pub numerator: Polynomial<F>,
    /// Denominator polynomial.
    pub denominator: Polynomial<F>,
}

impl<F: FieldKernels> PolynomialMatrix<F> {
    /// Returns the confluent interpolation basis for `F = self` under
    /// `constraints`, reduced to canonical shifted row Popov form.
    ///
    /// Constraints merge deterministically: same point and column keep the
    /// maximum multiplicity; ordering is by column, then multiplicity,
    /// with distinct points keeping first-appearance order. `shifts` holds one entry per solution coordinate (`m`
    /// entries). Every point/column pair must name an existing column; a
    /// zero multiplicity imposes nothing.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::GeometryOverflow`] for an out-of-range
    /// column, [`MatrixError::ShiftCount`] for a shift-count mismatch, plus
    /// polynomial and reduction errors.
    pub fn interpolation_basis(
        &self,
        constraints: &[InterpolationConstraint<F>],
        shifts: &[usize],
    ) -> Result<InterpolationBasis<F>, MatrixError> {
        let (rows, columns) = self.shape();
        if shifts.len() != rows {
            return Err(MatrixError::ShiftCount {
                columns: rows,
                shifts: shifts.len(),
            });
        }
        for constraint in constraints {
            if constraint.column >= columns {
                return Err(MatrixError::GeometryOverflow {
                    context: "interpolation constraint column",
                });
            }
        }
        let merged = Self::merge_constraints(constraints, columns);
        let mut basis = PolynomialMatrix::identity(rows)?;
        let mut independent = 0usize;
        for (column, point, order) in merged {
            for hasse in 0..order {
                if Self::impose_hasse_constraint(self, &mut basis, column, point, hasse, shifts)? {
                    independent += 1;
                }
            }
        }
        basis.reduce_popov(shifts)?;
        Ok(InterpolationBasis {
            basis,
            independent_constraints: independent,
        })
    }

    /// Returns the local moduli per column for a constraint set: the
    /// product of `(x - point)^multiplicity` over merged constraints.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::GeometryOverflow`] for an out-of-range
    /// column, plus polynomial errors.
    pub fn interpolation_moduli(
        &self,
        constraints: &[InterpolationConstraint<F>],
    ) -> Result<Vec<Polynomial<F>>, MatrixError> {
        let (_, columns) = self.shape();
        for constraint in constraints {
            if constraint.column >= columns {
                return Err(MatrixError::GeometryOverflow {
                    context: "interpolation constraint column",
                });
            }
        }
        let merged = Self::merge_constraints(constraints, columns);
        let mut moduli = Vec::with_capacity(columns);
        for _ in 0..columns {
            moduli.push(Polynomial::one().map_err(MatrixError::Polynomial)?);
        }
        for (column, point, order) in merged {
            if order == 0 {
                continue;
            }
            let mut factor = Polynomial::one().map_err(MatrixError::Polynomial)?;
            for _ in 0..order {
                factor = factor.multiply_x_plus(point.neg())?;
            }
            let product = moduli[column].multiply(&factor)?;
            moduli[column] = product;
        }
        Ok(moduli)
    }

    /// Merges constraints by maximum multiplicity per point and column in
    /// deterministic order: by column, then multiplicity. Distinct points
    /// in the same column keep first-appearance order through the stable
    /// sort.
    fn merge_constraints(
        constraints: &[InterpolationConstraint<F>],
        columns: usize,
    ) -> Vec<(usize, F::Elem, usize)> {
        let mut merged: Vec<(usize, F::Elem, usize)> = Vec::new();
        for constraint in constraints {
            if constraint.column >= columns || constraint.multiplicity == 0 {
                continue;
            }
            match merged.iter_mut().find(|(column, point, _)| {
                *column == constraint.column && *point == constraint.point
            }) {
                Some(slot) => {
                    slot.2 = slot.2.max(constraint.multiplicity);
                }
                None => merged.push((constraint.column, constraint.point, constraint.multiplicity)),
            }
        }
        merged.sort_by(|left, right| left.0.cmp(&right.0).then(left.2.cmp(&right.2)));
        merged
    }

    /// Imposes one Hasse constraint `H_hasse((B * F)[column], point) = 0`.
    ///
    /// Returns whether the constraint was independent.
    fn impose_hasse_constraint(
        input: &PolynomialMatrix<F>,
        basis: &mut PolynomialMatrix<F>,
        column: usize,
        point: F::Elem,
        hasse: usize,
        shifts: &[usize],
    ) -> Result<bool, MatrixError> {
        let (rows, _) = basis.shape();
        let mut discrepancies = Vec::with_capacity(rows);
        for row in 0..rows {
            discrepancies.push(Self::hasse_residual(
                input, basis, row, column, point, hasse,
            )?);
        }
        if discrepancies.iter().all(|delta| delta.is_zero()) {
            return Ok(false);
        }
        let mut pivot: Option<usize> = None;
        let mut pivot_key: Option<(usize, usize)> = None;
        for (row, discrepancy) in discrepancies.iter().enumerate().take(rows) {
            if discrepancy.is_zero() {
                continue;
            }
            let key = Self::pivot_key(basis, row, shifts)?;
            if pivot_key.is_none_or(|current| key < current) {
                pivot_key = Some(key);
                pivot = Some(row);
            }
        }
        let pivot = pivot.ok_or(MatrixError::GeometryOverflow {
            context: "interpolation pivot selection",
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
        // Multiply the pivot by (x - point) last: the factor raises the
        // Hasse valuation at `point` by exactly one while lower Hasse
        // coefficients stay zero by the ordering of imposed constraints.
        let (_, basis_columns) = basis.shape();
        for other in 0..basis_columns {
            let entry = basis.entry(pivot, other).cloned().unwrap_or_default();
            basis.set_entry(pivot, other, entry.multiply_x_plus(point.neg())?)?;
        }
        Ok(true)
    }

    /// Returns the `hasse`-th Hasse coefficient of row `row` of
    /// `basis * input` at `point`, restricted to one column.
    fn hasse_residual(
        input: &PolynomialMatrix<F>,
        basis: &PolynomialMatrix<F>,
        row: usize,
        column: usize,
        point: F::Elem,
        hasse: usize,
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
        Ok(accumulator.evaluate_hasse(point, hasse))
    }

    /// Returns the shifted row-leading pair of a basis row for pivot
    /// selection: the shared approximant key.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::Reduction`] on shifted-degree overflow.
    fn pivot_key(
        basis: &PolynomialMatrix<F>,
        row: usize,
        shifts: &[usize],
    ) -> Result<(usize, usize), MatrixError> {
        PolynomialMatrix::shifted_row_key(basis, row, shifts)
    }

    /// Returns the scalar Padé relation for `series` of order `order`:
    /// `numerator - series * denominator = 0` modulo `x^order` with
    /// `degree(numerator) < order - degree(denominator)` and denominator
    /// of the requested degree.
    ///
    /// The relation is a length-2 approximant module row selected by the
    /// balanced shifted form: `F = [[series], [-1]]` with order `[order]`
    /// under shifts `[numerator_degree, denominator_degree]`, where
    /// `numerator_degree = order - denominator_degree - 1`. Basis row
    /// `[a, b]` satisfies `a * series - b = 0`, so the denominator is
    /// entry 0 and the numerator is entry 1. The predictable-degree
    /// property guarantees the minimum-shifted-degree row meets the
    /// bounds whenever any solution does; requiring the denominator
    /// degree exactly is a post-check that fails honestly on
    /// degree-dropping data.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::GeometryOverflow`] when no row meets the
    /// degree bounds, plus polynomial and reduction errors.
    pub fn pade_relation(
        series: &Polynomial<F>,
        order: usize,
        denominator_degree: usize,
    ) -> Result<PadeRelation<F>, MatrixError> {
        // F = [[series],[-1]] as 2x1: row [a, b] gives
        // a*series - b, vanishing mod x^order for a Padé relation.
        let input = PolynomialMatrix::from_entries(
            2,
            1,
            alloc::vec![series.clone(), Polynomial::one()?.negated()],
        )?;
        let numerator_bound = order.saturating_sub(denominator_degree);
        let numerator_degree = numerator_bound.saturating_sub(1);
        let result = input.approximant_basis(&[order], &[numerator_degree, denominator_degree])?;
        let row = Self::select_pade_row(&result.basis, denominator_degree, numerator_bound)?;
        Ok(PadeRelation {
            denominator: result.basis.entry(row, 0).cloned().unwrap_or_default(),
            numerator: result.basis.entry(row, 1).cloned().unwrap_or_default(),
        })
    }

    /// Selects the approximant row with entry-0 (denominator) degree
    /// exactly `denominator_degree` and entry-1 (numerator) below
    /// `numerator_bound`.
    fn select_pade_row(
        basis: &PolynomialMatrix<F>,
        denominator_degree: usize,
        numerator_bound: usize,
    ) -> Result<usize, MatrixError> {
        let (rows, _) = basis.shape();
        for row in 0..rows {
            let denominator = basis.entry(row, 0).cloned().unwrap_or_default();
            let numerator = basis.entry(row, 1).cloned().unwrap_or_default();
            let numerator_ok = numerator
                .degree()
                .is_none_or(|degree| degree < numerator_bound);
            let denominator_ok = denominator.degree() == Some(denominator_degree)
                || (denominator_degree == 0 && denominator.is_zero());
            if numerator_ok && denominator_ok {
                return Ok(row);
            }
        }
        Err(MatrixError::GeometryOverflow {
            context: "pade relation selection",
        })
    }

    /// Returns simultaneous Padé relations for `series` with a shared
    /// denominator of `denominator_degree`: entry `i` of the output holds
    /// `(numerator_i, denominator)` with
    /// `numerator_i - series[i] * denominator = 0` modulo `x^order`.
    ///
    /// Built over the stacked module with unknowns
    /// `(numerators.., denominator)`: `F` is `(k+1)`-by-`k` with
    /// `F[i,j] = delta[i,j]` and `F[k,j] = -series[j]`, orders
    /// `[order; k]`, balanced shifts `[denominator_degree; k]` then
    /// `numerator_degree` with `numerator_degree = bound - 1` and
    /// `bound = order - denominator_degree`. The selected basis row has
    /// last-coordinate (denominator) degree exactly `denominator_degree`
    /// and every numerator below `bound`.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::GeometryOverflow`] for an empty series or
    /// no row meeting the degree bounds, plus polynomial and reduction
    /// errors.
    pub fn simultaneous_pade(
        series: &[Polynomial<F>],
        order: usize,
        denominator_degree: usize,
    ) -> Result<Vec<PadeRelation<F>>, MatrixError> {
        if series.is_empty() {
            return Err(MatrixError::GeometryOverflow {
                context: "simultaneous-pade empty series",
            });
        }
        let count = series.len();
        let bound = order.saturating_sub(denominator_degree);
        let numerator_degree = bound.saturating_sub(1);
        let one = Polynomial::one().map_err(MatrixError::Polynomial)?;
        let mut entries = alloc::vec![Polynomial::zero(); (count + 1) * count];
        for (index, target) in series.iter().enumerate() {
            entries[index * count + index].clone_from(&one);
            entries[count * count + index] = target.negated();
        }
        let input = PolynomialMatrix::from_entries(count + 1, count, entries)?;
        let orders = alloc::vec![order; count];
        let mut shifts = alloc::vec![denominator_degree; count];
        shifts.push(numerator_degree);
        let result = input.approximant_basis(&orders, &shifts)?;
        let row = Self::select_simultaneous_row(&result.basis, denominator_degree, bound)?;
        let denominator = result.basis.entry(row, count).cloned().unwrap_or_default();
        series
            .iter()
            .enumerate()
            .map(|(index, _)| {
                Ok(PadeRelation {
                    numerator: result.basis.entry(row, index).cloned().unwrap_or_default(),
                    denominator: denominator.clone(),
                })
            })
            .collect()
    }

    /// Selects the stacked row with shared-denominator degree exactly
    /// `denominator_degree` and every numerator below `bound`.
    fn select_simultaneous_row(
        basis: &PolynomialMatrix<F>,
        denominator_degree: usize,
        bound: usize,
    ) -> Result<usize, MatrixError> {
        let (rows, columns) = basis.shape();
        if columns == 0 {
            return Err(MatrixError::GeometryOverflow {
                context: "simultaneous-pade empty series",
            });
        }
        for row in 0..rows {
            let denominator = basis.entry(row, columns - 1).cloned().unwrap_or_default();
            if denominator.degree() != Some(denominator_degree)
                && !(denominator_degree == 0 && denominator.is_zero())
            {
                continue;
            }
            let mut ok = true;
            for column in 0..columns - 1 {
                let numerator = basis.entry(row, column).cloned().unwrap_or_default();
                if numerator.degree().is_some_and(|degree| degree >= bound) {
                    ok = false;
                    break;
                }
            }
            if ok {
                return Ok(row);
            }
        }
        Err(MatrixError::GeometryOverflow {
            context: "simultaneous-pade relation selection",
        })
    }

    /// Returns the Hermite-Padé relation for `series` with per-row degree
    /// bounds `bounds`: `sum_i series[i] * relation[i] = 0` modulo
    /// `x^order` with `degree(relation[i]) < bounds[i]`.
    ///
    /// Built as the approximant basis of the row `F = [series]` under
    /// shifts `max_bound - bounds[i]`, returning the minimum-degree basis
    /// row that satisfies every bound; the shifts guarantee a compliant
    /// row exists whenever any solution does.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::GeometryOverflow`] for an empty series, a
    /// bound count mismatch, or no compliant basis row, plus polynomial
    /// and reduction errors.
    pub fn hermite_pade(
        series: &[Polynomial<F>],
        order: usize,
        bounds: &[usize],
    ) -> Result<Vec<Polynomial<F>>, MatrixError> {
        if series.is_empty() || bounds.len() != series.len() {
            return Err(MatrixError::GeometryOverflow {
                context: "hermite-pade series bounds",
            });
        }
        let maximum = bounds.iter().copied().max().unwrap_or(0);
        let shifts: Vec<usize> = bounds.iter().map(|bound| maximum - bound).collect();
        let input = PolynomialMatrix::from_entries(1, series.len(), series.to_vec())?;
        let transposed = input.transposed()?;
        let result = transposed.approximant_basis(&alloc::vec![order], &shifts)?;
        // Return the minimum-max-degree row across the basis rows that
        // satisfy every per-coordinate bound; normal data carries one.
        let mut best: Option<Vec<Polynomial<F>>> = None;
        let mut best_degree: Option<usize> = None;
        for candidate in 0..series.len() {
            let row_vec: Vec<Polynomial<F>> = (0..series.len())
                .map(|column| {
                    result
                        .basis
                        .entry(candidate, column)
                        .cloned()
                        .unwrap_or_default()
                })
                .collect();
            let mut compliant = true;
            for (entry, &bound) in row_vec.iter().zip(bounds) {
                if entry.degree().is_some_and(|degree| degree >= bound) {
                    compliant = false;
                    break;
                }
            }
            if !compliant {
                continue;
            }
            let degree = row_vec.iter().filter_map(Polynomial::degree).max();
            if best_degree.is_none_or(|current| degree < Some(current)) {
                best_degree = degree;
                best = Some(row_vec);
            }
        }
        best.ok_or(MatrixError::GeometryOverflow {
            context: "hermite-pade relation selection",
        })
    }
}
