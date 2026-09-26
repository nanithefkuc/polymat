//! Row Hermite form, Smith form, determinant, and polynomial inverse.
//!
//! The row Hermite form uses the leftmost-pivot convention: nonzero rows
//! first with pivots in strictly increasing columns, zeros below each
//! pivot, monic pivots, and entries above each pivot degree-reduced. The
//! Smith form is diagonal with monic invariant factors `d[i]` dividing
//! `d[i+1]` and zeros last. Both come with unimodular witnesses
//! (`U * A = H`, `U * A * V = D`); every elementary row and column
//! operation below is a transvection, swap, or nonzero-constant scaling,
//! hence unimodular with nonzero-constant determinant.
//!
//! The determinant keeps its field-valued unit: it is zero on singular
//! input and one for 0-by-0. The polynomial inverse exists only for a
//! unimodular square matrix (nonzero constant determinant) and is derived
//! from the transformation certificate; a rational inverse is a different
//! operation and outside this surface.

use alloc::vec::Vec;

use fgf::field::Elem;
use fgf::kernel::FieldKernels;
use poly_ring::Polynomial;

use crate::matrix::{MatrixError, PolynomialMatrix};

/// A row Hermite form with its left witness.
#[derive(Debug, Clone)]
pub struct HermiteForm<F: FieldKernels> {
    /// Row Hermite form of the input.
    pub form: PolynomialMatrix<F>,
    /// Unimodular left witness with `witness * input = form`.
    pub witness: PolynomialMatrix<F>,
}

/// A Smith form with both witnesses.
#[derive(Debug, Clone)]
pub struct SmithForm<F: FieldKernels> {
    /// Diagonal Smith form of the input.
    pub form: PolynomialMatrix<F>,
    /// Unimodular left witness.
    pub left: PolynomialMatrix<F>,
    /// Unimodular right witness with `left * input * right = form`.
    pub right: PolynomialMatrix<F>,
}

impl<F: FieldKernels> PolynomialMatrix<F> {
    /// Reduces to row Hermite form with the left witness.
    ///
    /// Processes columns left to right: selects the lowest-degree nonzero
    /// entry at or below the current row as pivot, swaps it into place,
    /// clears entries below via exact division remainders, makes the pivot
    /// monic, and degree-reduces entries above. Rank-deficient and
    /// rectangular inputs keep their shape with zero rows last.
    ///
    /// # Errors
    ///
    /// Returns polynomial errors and [`MatrixError`] reservation errors.
    pub fn hermite_form(&self) -> Result<HermiteForm<F>, MatrixError> {
        let (rows, columns) = self.shape();
        let mut form = self.clone();
        let mut witness = PolynomialMatrix::identity(rows)?;
        let mut pivot_row = 0usize;
        for column in 0..columns {
            if pivot_row >= rows {
                break;
            }
            // Select the lowest-degree nonzero entry at or below pivot_row.
            let mut selected: Option<(usize, usize)> = None;
            for row in pivot_row..rows {
                let entry = form.entry(row, column).cloned().unwrap_or_default();
                let Some(degree) = entry.degree() else {
                    continue;
                };
                if selected.is_none_or(|(_, best)| degree < best) {
                    selected = Some((row, degree));
                }
            }
            let Some((selected_row, _)) = selected else {
                continue;
            };
            form.swap_rows(pivot_row, selected_row)?;
            witness.swap_rows(pivot_row, selected_row)?;
            // Clear entries below the pivot with Euclidean remainders.
            loop {
                let pivot_entry = form.entry(pivot_row, column).cloned().unwrap_or_default();
                if pivot_entry.is_zero() {
                    break;
                }
                let mut progress = false;
                for row in pivot_row + 1..rows {
                    let entry = form.entry(row, column).cloned().unwrap_or_default();
                    if entry.is_zero() {
                        continue;
                    }
                    let (_, remainder) = entry.div_rem(&pivot_entry)?;
                    if remainder.is_zero() {
                        continue;
                    }
                    // Reduce row by the quotient multiple, then swap the
                    // smaller remainder upward; both are unimodular steps.
                    let (quotient, _) = entry.div_rem(&pivot_entry)?;
                    form.add_row_multiple(row, pivot_row, &quotient.negated(), &mut witness)?;
                    form.swap_rows(row, pivot_row)?;
                    witness.swap_rows(row, pivot_row)?;
                    progress = true;
                    break;
                }
                if !progress {
                    break;
                }
            }
            // Eliminate below-pivot entries exactly (now all divisible).
            let pivot_entry = form.entry(pivot_row, column).cloned().unwrap_or_default();
            for row in pivot_row + 1..rows {
                let entry = form.entry(row, column).cloned().unwrap_or_default();
                if entry.is_zero() {
                    continue;
                }
                let (quotient, _) = entry.div_rem(&pivot_entry)?;
                form.add_row_multiple(row, pivot_row, &quotient.negated(), &mut witness)?;
            }
            // Monic normalization with mirrored witness scaling.
            Self::scale_row_monic(&mut form, Some(&mut witness), pivot_row, column)?;
            // Degree-reduce entries above the pivot.
            let divisor = form.entry(pivot_row, column).cloned().unwrap_or_default();
            for row in 0..pivot_row {
                let entry = form.entry(row, column).cloned().unwrap_or_default();
                if entry.is_zero() {
                    continue;
                }
                let (quotient, _) = entry.div_rem(&divisor)?;
                if quotient.is_zero() {
                    continue;
                }
                form.add_row_multiple(row, pivot_row, &quotient.negated(), &mut witness)?;
            }
            pivot_row += 1;
        }
        Ok(HermiteForm { form, witness })
    }

    /// Reduces to Smith form with both witnesses.
    ///
    /// Repeatedly Hermite-reduces rows and (transposed) columns, repairing
    /// divisibility: when the pivot does not divide the remaining
    /// submatrix, a witnessing row or column addition restores progress.
    /// Diagonal entries are monic with `d[i]` dividing `d[i+1]`; zero rows
    /// and columns end last.
    ///
    /// # Errors
    ///
    /// Returns polynomial errors and [`MatrixError`] reservation errors.
    pub fn smith_form(&self) -> Result<SmithForm<F>, MatrixError> {
        let (rows, columns) = self.shape();
        let mut form = self.clone();
        let mut left = PolynomialMatrix::identity(rows)?;
        let mut right = PolynomialMatrix::identity(columns)?;
        let diagonal = rows.min(columns);
        for pivot in 0..diagonal {
            // Every pass either breaks or strictly reduces the pivot
            // degree through a Bezout combination, so the loop terminates.
            loop {
                if !Self::smith_pivot_nonzero(&mut form, &mut left, &mut right, pivot)? {
                    break;
                }
                // Eliminate the pivot row rightward and column downward.
                Self::smith_eliminate(&mut form, &mut left, &mut right, pivot)?;
                // A leftover remainder shares the pivot row or column:
                // fold it into the pivot with a Bezout step, which drops
                // the pivot degree below the remainder degree.
                if let Some((row, column)) = Self::smith_remainder(&form, pivot) {
                    if row == pivot {
                        Self::smith_column_bezout(&mut form, &mut right, pivot, column)?;
                    } else {
                        Self::smith_row_bezout(&mut form, &mut left, pivot, row)?;
                    }
                    continue;
                }
                // Pivot row and column are clear; check the submatrix.
                match Self::smith_witness(&form, pivot)? {
                    None => {
                        Self::scale_row_monic(&mut form, Some(&mut left), pivot, pivot)?;
                        break;
                    }
                    Some((row, column)) => {
                        // Fold the witness into the pivot row, then fold
                        // the resulting pivot-row entry into the pivot.
                        // The witness is not divisible by the pivot, so
                        // the Bezout combination is strictly smaller.
                        let one = Polynomial::one().map_err(MatrixError::Polynomial)?;
                        form.add_row_multiple(pivot, row, &one, &mut left)?;
                        Self::smith_column_bezout(&mut form, &mut right, pivot, column)?;
                    }
                }
            }
        }
        // Order nonzero diagonal entries first (they already sit in pivot
        // order); enforce divisibility d[i] | d[i+1] is structural by the
        // repair loop above.
        Ok(SmithForm { form, left, right })
    }

    /// Returns the determinant of a square matrix with its field unit.
    ///
    /// Zero on singular input; one for 0-by-0. Computed by Bareiss
    /// fraction-free elimination with swap-sign tracking: each row swap
    /// negates the determinant, and every division is exact over Fq\[x\].
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::GeometryOverflow`] for a non-square input,
    /// plus polynomial errors.
    pub fn determinant(&self) -> Result<Polynomial<F>, MatrixError> {
        let (rows, columns) = self.shape();
        if rows != columns {
            return Err(MatrixError::GeometryOverflow {
                context: "polynomial determinant shape",
            });
        }
        if rows == 0 {
            return Polynomial::one().map_err(MatrixError::Polynomial);
        }
        // Bareiss fraction-free elimination: work[i,j] = (work[i,j] *
        // prev - work[i,k] * work[k,j]) / prev_prev, exact at every step
        // over the GCD domain Fq[x]. Row swaps negate the determinant.
        let mut work = self.clone();
        let mut sign = F::Elem::ONE;
        let mut previous = Polynomial::one().map_err(MatrixError::Polynomial)?;
        for pivot in 0..rows {
            let mut selected: Option<usize> = None;
            for row in pivot..rows {
                let entry = work.entry(row, pivot).cloned().unwrap_or_default();
                if !entry.is_zero() && selected.is_none() {
                    selected = Some(row);
                    break;
                }
            }
            let Some(selected_row) = selected else {
                return Ok(Polynomial::zero());
            };
            if selected_row != pivot {
                work.swap_rows(pivot, selected_row)?;
                sign = sign.neg();
            }
            let pivot_entry = work.entry(pivot, pivot).cloned().unwrap_or_default();
            for row in pivot + 1..rows {
                for column in pivot + 1..columns {
                    let entry = work.entry(row, column).cloned().unwrap_or_default();
                    let sub = work.entry(row, pivot).cloned().unwrap_or_default();
                    let factor = work.entry(pivot, column).cloned().unwrap_or_default();
                    let mut next = entry.multiply(&pivot_entry)?;
                    let subtrahend = sub.multiply(&factor)?;
                    next.sub_assign(&subtrahend)?;
                    let exact = next.divide_exact(&previous)?;
                    work.set_entry(row, column, exact)?;
                }
            }
            for row in pivot + 1..rows {
                work.set_entry(row, pivot, Polynomial::zero())?;
            }
            previous = pivot_entry;
        }
        let mut determinant = work
            .entry(rows - 1, columns - 1)
            .cloned()
            .unwrap_or_default();
        determinant.scale_assign(sign);
        Ok(determinant)
    }

    /// Returns the Leibniz determinant of a tiny square matrix: the signed
    /// sum over all permutations.
    ///
    /// An independent oracle for [`PolynomialMatrix::determinant`] on small
    /// inputs; production code paths use Bareiss elimination.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::GeometryOverflow`] for a non-square or large
    /// input (more than 6 rows), plus polynomial errors.
    pub fn determinant_leibniz(&self) -> Result<Polynomial<F>, MatrixError> {
        let (rows, columns) = self.shape();
        if rows != columns || rows > 6 {
            return Err(MatrixError::GeometryOverflow {
                context: "leibniz determinant shape",
            });
        }
        if rows == 0 {
            return Polynomial::one().map_err(MatrixError::Polynomial);
        }
        let mut total = Polynomial::zero();
        let mut permutation: Vec<usize> = (0..rows).collect();
        loop {
            let mut term = Polynomial::one().map_err(MatrixError::Polynomial)?;
            for (row, &column) in permutation.iter().enumerate() {
                let entry = self.entry(row, column).cloned().unwrap_or_default();
                term = term.multiply(&entry)?;
            }
            if permutation_sign(&permutation) < 0 {
                term.scale_assign(F::Elem::ONE.neg());
            }
            total.add_assign(&term)?;
            if !next_permutation(&mut permutation) {
                break;
            }
        }
        Ok(total)
    }

    /// Returns the polynomial inverse with `left * self = I` and
    /// `self * left = I`, or `None` for a non-unimodular matrix.
    ///
    /// Derives the inverse from the Hermite witness: when the form is the
    /// identity, the witness is the inverse. A non-identity form (or a
    /// non-constant determinant) reports non-invertibility without error.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::GeometryOverflow`] for a non-square input,
    /// plus polynomial errors.
    pub fn polynomial_inverse(&self) -> Result<Option<PolynomialMatrix<F>>, MatrixError> {
        let (rows, columns) = self.shape();
        if rows != columns {
            return Err(MatrixError::GeometryOverflow {
                context: "polynomial inverse shape",
            });
        }
        let determinant = self.determinant()?;
        let is_constant = determinant.degree().is_none_or(|degree| degree == 0);
        if determinant.is_zero() || !is_constant {
            return Ok(None);
        }
        let hermite = self.hermite_form()?;
        for row in 0..rows {
            for column in 0..columns {
                let entry = hermite.form.entry(row, column).cloned().unwrap_or_default();
                let expected = row == column;
                if entry.is_one() != expected || (!expected && !entry.is_zero()) {
                    return Ok(None);
                }
            }
        }
        Ok(Some(hermite.witness))
    }

    /// Returns the monic gcd of all k-by-k minors: the k-th determinantal
    /// divisor, an independent check of Smith invariant factors.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::GeometryOverflow`] for an out-of-range `k`,
    /// plus polynomial errors.
    pub fn minor_gcd(&self, size: usize) -> Result<Polynomial<F>, MatrixError> {
        let (rows, columns) = self.shape();
        if size == 0 || size > rows.min(columns) {
            return Err(MatrixError::GeometryOverflow {
                context: "minor gcd size",
            });
        }
        let mut accumulator: Option<Polynomial<F>> = None;
        for row_set in combinations(rows, size) {
            for column_set in combinations(columns, size) {
                let mut entries = Vec::with_capacity(size * size);
                for &row in &row_set {
                    for &column in &column_set {
                        entries.push(self.entry(row, column).cloned().unwrap_or_default());
                    }
                }
                let minor = PolynomialMatrix::from_entries(size, size, entries)?;
                let determinant = minor.determinant_leibniz()?;
                accumulator = Some(match accumulator {
                    None => determinant.monic(),
                    Some(current) => current.gcd(&determinant)?,
                });
            }
        }
        Ok(accumulator.unwrap_or_default().monic())
    }

    /// Swaps two rows in place.
    fn swap_rows(&mut self, left: usize, right: usize) -> Result<(), MatrixError> {
        if left == right {
            return Ok(());
        }
        let (_, columns) = self.shape();
        for column in 0..columns {
            let left_entry = self.entry(left, column).cloned().unwrap_or_default();
            let right_entry = self.entry(right, column).cloned().unwrap_or_default();
            self.set_entry(left, column, right_entry)?;
            self.set_entry(right, column, left_entry)?;
        }
        Ok(())
    }

    /// Swaps two columns in place.
    fn swap_columns(&mut self, left: usize, right: usize) -> Result<(), MatrixError> {
        if left == right {
            return Ok(());
        }
        let (rows, _) = self.shape();
        for row in 0..rows {
            let left_entry = self.entry(row, left).cloned().unwrap_or_default();
            let right_entry = self.entry(row, right).cloned().unwrap_or_default();
            self.set_entry(row, left, right_entry)?;
            self.set_entry(row, right, left_entry)?;
        }
        Ok(())
    }

    /// Adds `scale * pivot` to `target`, mirroring into `witness`.
    fn add_row_multiple(
        &mut self,
        target: usize,
        pivot: usize,
        scale: &Polynomial<F>,
        witness: &mut PolynomialMatrix<F>,
    ) -> Result<(), MatrixError> {
        let (_, columns) = self.shape();
        for column in 0..columns {
            let pivot_entry = self.entry(pivot, column).cloned().unwrap_or_default();
            let multiple = scale.multiply(&pivot_entry)?;
            let mut entry = self.entry(target, column).cloned().unwrap_or_default();
            entry.add_assign(&multiple)?;
            self.set_entry(target, column, entry)?;
        }
        let (witness_rows, _) = witness.shape();
        for column in 0..witness_rows {
            let pivot_entry = witness.entry(pivot, column).cloned().unwrap_or_default();
            let multiple = scale.multiply(&pivot_entry)?;
            let mut entry = witness.entry(target, column).cloned().unwrap_or_default();
            entry.add_assign(&multiple)?;
            witness.set_entry(target, column, entry)?;
        }
        Ok(())
    }

    /// Adds `scale * pivot_column` to `target_column`, mirroring into the
    /// right witness.
    fn add_column_multiple(
        &mut self,
        target: usize,
        pivot: usize,
        scale: &Polynomial<F>,
        witness: &mut PolynomialMatrix<F>,
    ) -> Result<(), MatrixError> {
        let (rows, _) = self.shape();
        for row in 0..rows {
            let pivot_entry = self.entry(row, pivot).cloned().unwrap_or_default();
            let multiple = scale.multiply(&pivot_entry)?;
            let mut entry = self.entry(row, target).cloned().unwrap_or_default();
            entry.add_assign(&multiple)?;
            self.set_entry(row, target, entry)?;
        }
        let (witness_rows, _) = witness.shape();
        for row in 0..witness_rows {
            let pivot_entry = witness.entry(row, pivot).cloned().unwrap_or_default();
            let multiple = scale.multiply(&pivot_entry)?;
            let mut entry = witness.entry(row, target).cloned().unwrap_or_default();
            entry.add_assign(&multiple)?;
            witness.set_entry(row, target, entry)?;
        }
        Ok(())
    }

    /// Scales one row to monic, mirroring into an optional witness.
    fn scale_row_monic(
        form: &mut PolynomialMatrix<F>,
        witness: Option<&mut PolynomialMatrix<F>>,
        row: usize,
        column: usize,
    ) -> Result<(), MatrixError> {
        let entry = form.entry(row, column).cloned().unwrap_or_default();
        let Some(leading) = entry.leading_coefficient() else {
            return Ok(());
        };
        if leading == F::Elem::ONE {
            return Ok(());
        }
        let inverse = leading.inv();
        let (_, columns) = form.shape();
        for other in 0..columns {
            let mut target = form.entry(row, other).cloned().unwrap_or_default();
            target.scale_assign(inverse);
            form.set_entry(row, other, target)?;
        }
        if let Some(witness) = witness {
            let (witness_rows, _) = witness.shape();
            for other in 0..witness_rows {
                let mut target = witness.entry(row, other).cloned().unwrap_or_default();
                target.scale_assign(inverse);
                witness.set_entry(row, other, target)?;
            }
        }
        Ok(())
    }

    /// Moves the minimum-degree nonzero submatrix entry into the pivot
    /// position; returns whether the pivot is now nonzero.
    fn smith_pivot_nonzero(
        form: &mut PolynomialMatrix<F>,
        left: &mut PolynomialMatrix<F>,
        right: &mut PolynomialMatrix<F>,
        pivot: usize,
    ) -> Result<bool, MatrixError> {
        let (rows, columns) = form.shape();
        let mut best: Option<(usize, usize, usize)> = None;
        for row in pivot..rows {
            for column in pivot..columns {
                let entry = form.entry(row, column).cloned().unwrap_or_default();
                let Some(degree) = entry.degree() else {
                    continue;
                };
                if best.is_none_or(|(_, _, current)| degree < current) {
                    best = Some((row, column, degree));
                }
            }
        }
        let Some((row, column, _)) = best else {
            return Ok(false);
        };
        if row != pivot {
            form.swap_rows(row, pivot)?;
            left.swap_rows(row, pivot)?;
        }
        if column != pivot {
            form.swap_columns(column, pivot)?;
            right.swap_columns(column, pivot)?;
        }
        Ok(true)
    }

    /// Eliminates the pivot row rightward and column downward with
    /// quotients. Entries that are not divisible leave remainders for the
    /// Bezout steps below.
    fn smith_eliminate(
        form: &mut PolynomialMatrix<F>,
        left: &mut PolynomialMatrix<F>,
        right: &mut PolynomialMatrix<F>,
        pivot: usize,
    ) -> Result<(), MatrixError> {
        let (rows, columns) = form.shape();
        let divisor = form.entry(pivot, pivot).cloned().unwrap_or_default();
        for column in pivot + 1..columns {
            let entry = form.entry(pivot, column).cloned().unwrap_or_default();
            if entry.is_zero() {
                continue;
            }
            let (quotient, _) = entry.div_rem(&divisor)?;
            if quotient.is_zero() {
                continue;
            }
            form.add_column_multiple(column, pivot, &quotient.negated(), right)?;
        }
        for row in 0..rows {
            if row == pivot {
                continue;
            }
            let entry = form.entry(row, pivot).cloned().unwrap_or_default();
            if entry.is_zero() {
                continue;
            }
            let (quotient, _) = entry.div_rem(&divisor)?;
            if quotient.is_zero() {
                continue;
            }
            form.add_row_multiple(row, pivot, &quotient.negated(), left)?;
        }
        Ok(())
    }

    /// Finds a nonzero remainder in the pivot row or column: evidence the
    /// pivot does not divide its line. Returns the witness coordinates.
    fn smith_remainder(form: &PolynomialMatrix<F>, pivot: usize) -> Option<(usize, usize)> {
        let (rows, columns) = form.shape();
        for column in pivot + 1..columns {
            let entry = form.entry(pivot, column).cloned().unwrap_or_default();
            if !entry.is_zero() {
                return Some((pivot, column));
            }
        }
        for row in 0..rows {
            if row == pivot {
                continue;
            }
            let entry = form.entry(row, pivot).cloned().unwrap_or_default();
            if !entry.is_zero() {
                return Some((row, pivot));
            }
        }
        None
    }

    /// Folds pivot-row entry `column` into the pivot with a Bezout
    /// combination, strictly reducing the pivot degree. With
    /// `g = gcd(p, e) = s*p + t*e`, the pivot column becomes `g` and the
    /// combination matrix has determinant one.
    fn smith_column_bezout(
        form: &mut PolynomialMatrix<F>,
        right: &mut PolynomialMatrix<F>,
        pivot: usize,
        column: usize,
    ) -> Result<(), MatrixError> {
        let (rows, _) = form.shape();
        let pivot_entry = form.entry(pivot, pivot).cloned().unwrap_or_default();
        let other = form.entry(pivot, column).cloned().unwrap_or_default();
        let relation = pivot_entry.extended_gcd(&other)?;
        let (quotient, _) = pivot_entry.div_rem(&relation.gcd)?;
        let (other_quotient, _) = other.div_rem(&relation.gcd)?;
        for row in 0..rows {
            let left_value = form.entry(row, pivot).cloned().unwrap_or_default();
            let right_value = form.entry(row, column).cloned().unwrap_or_default();
            let mut next_pivot = relation.a_cofactor.multiply(&left_value)?;
            let term = relation.b_cofactor.multiply(&right_value)?;
            next_pivot.add_assign(&term)?;
            let mut next_other = quotient.negated().multiply(&left_value)?;
            let term = other_quotient.multiply(&right_value)?;
            next_other.add_assign(&term)?;
            form.set_entry(row, pivot, next_pivot)?;
            form.set_entry(row, column, next_other)?;
        }
        let (witness_rows, _) = right.shape();
        for row in 0..witness_rows {
            let left_value = right.entry(row, pivot).cloned().unwrap_or_default();
            let right_value = right.entry(row, column).cloned().unwrap_or_default();
            let mut next_pivot = relation.a_cofactor.multiply(&left_value)?;
            let term = relation.b_cofactor.multiply(&right_value)?;
            next_pivot.add_assign(&term)?;
            let mut next_other = quotient.negated().multiply(&left_value)?;
            let term = other_quotient.multiply(&right_value)?;
            next_other.add_assign(&term)?;
            right.set_entry(row, pivot, next_pivot)?;
            right.set_entry(row, column, next_other)?;
        }
        Ok(())
    }

    /// Folds pivot-column entry `row` into the pivot with a Bezout
    /// combination, strictly reducing the pivot degree. Mirrors
    /// [`PolynomialMatrix::smith_column_bezout`] on rows with the left
    /// witness.
    fn smith_row_bezout(
        form: &mut PolynomialMatrix<F>,
        left: &mut PolynomialMatrix<F>,
        pivot: usize,
        row: usize,
    ) -> Result<(), MatrixError> {
        let (_, columns) = form.shape();
        let pivot_entry = form.entry(pivot, pivot).cloned().unwrap_or_default();
        let other = form.entry(row, pivot).cloned().unwrap_or_default();
        let relation = pivot_entry.extended_gcd(&other)?;
        let (quotient, _) = pivot_entry.div_rem(&relation.gcd)?;
        let (other_quotient, _) = other.div_rem(&relation.gcd)?;
        for column in 0..columns {
            let upper = form.entry(pivot, column).cloned().unwrap_or_default();
            let lower = form.entry(row, column).cloned().unwrap_or_default();
            let mut next_pivot = relation.a_cofactor.multiply(&upper)?;
            let term = relation.b_cofactor.multiply(&lower)?;
            next_pivot.add_assign(&term)?;
            let mut next_other = quotient.negated().multiply(&upper)?;
            let term = other_quotient.multiply(&lower)?;
            next_other.add_assign(&term)?;
            form.set_entry(pivot, column, next_pivot)?;
            form.set_entry(row, column, next_other)?;
        }
        let (witness_rows, _) = left.shape();
        for column in 0..witness_rows {
            let upper = left.entry(pivot, column).cloned().unwrap_or_default();
            let lower = left.entry(row, column).cloned().unwrap_or_default();
            let mut next_pivot = relation.a_cofactor.multiply(&upper)?;
            let term = relation.b_cofactor.multiply(&lower)?;
            next_pivot.add_assign(&term)?;
            let mut next_other = quotient.negated().multiply(&upper)?;
            let term = other_quotient.multiply(&lower)?;
            next_other.add_assign(&term)?;
            left.set_entry(pivot, column, next_pivot)?;
            left.set_entry(row, column, next_other)?;
        }
        Ok(())
    }

    /// Finds a submatrix entry not divisible by the pivot, with the pivot
    /// row and column already clear. Returns the witness coordinates.
    fn smith_witness(
        form: &PolynomialMatrix<F>,
        pivot: usize,
    ) -> Result<Option<(usize, usize)>, MatrixError> {
        let (rows, columns) = form.shape();
        let divisor = form.entry(pivot, pivot).cloned().unwrap_or_default();
        if divisor.is_zero() {
            return Ok(None);
        }
        for row in pivot + 1..rows {
            for column in pivot + 1..columns {
                let entry = form.entry(row, column).cloned().unwrap_or_default();
                if entry.is_zero() {
                    continue;
                }
                let (_, remainder) = entry.div_rem(&divisor)?;
                if !remainder.is_zero() {
                    return Ok(Some((row, column)));
                }
            }
        }
        Ok(None)
    }
}

/// Returns all k-subsets of `0..count` in lexicographic order.
fn combinations(count: usize, size: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut current: Vec<usize> = (0..size).collect();
    if size > count {
        return result;
    }
    if size == 0 {
        result.push(Vec::new());
        return result;
    }
    loop {
        result.push(current.clone());
        let mut index = size;
        for candidate in (0..size).rev() {
            if current[candidate] + size - candidate < count {
                index = candidate;
                break;
            }
        }
        if index == size {
            break;
        }
        current[index] += 1;
        for next in index + 1..size {
            current[next] = current[next - 1] + 1;
        }
    }
    result
}

/// Returns the sign of a permutation: +1 or -1.
fn permutation_sign(permutation: &[usize]) -> i32 {
    let mut inversions = 0usize;
    for (i, &left) in permutation.iter().enumerate() {
        for &right in &permutation[i + 1..] {
            if left > right {
                inversions += 1;
            }
        }
    }
    if inversions.is_multiple_of(2) { 1 } else { -1 }
}

/// Advances a permutation lexicographically; returns whether one exists.
fn next_permutation(permutation: &mut [usize]) -> bool {
    let count = permutation.len();
    if count < 2 {
        return false;
    }
    let mut pivot: Option<usize> = None;
    for index in (0..count - 1).rev() {
        if permutation[index] < permutation[index + 1] {
            pivot = Some(index);
            break;
        }
    }
    let Some(pivot) = pivot else {
        return false;
    };
    for index in (pivot + 1..count).rev() {
        if permutation[index] > permutation[pivot] {
            permutation.swap(index, pivot);
            break;
        }
    }
    permutation[pivot + 1..].reverse();
    true
}
