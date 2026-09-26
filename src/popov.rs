//! Ordered weak and canonical shifted Popov forms.
//!
//! Weak Popov form gives every nonzero row a distinct shifted leading
//! column. The ordered form sorts nonzero rows by increasing leading column
//! with zero rows last. The canonical row Popov form additionally makes
//! every pivot monic and reduces every other entry in each pivot column
//! below the pivot degree. Canonical output is invariant under unimodular
//! changes of generators: two generating matrices for the same row module
//! reduce to the same nonzero rows, up to zero-row padding.
//!
//! All three transformations are unimodular row operations. Ordering is a
//! row permutation; monic scaling multiplies a row by a nonzero constant;
//! column reduction subtracts polynomial multiples of pivot rows. The
//! optional witness `U` with `U * A = R` tracks each step.

use alloc::vec::Vec;

use fgf::field::Elem;
use fgf::kernel::FieldKernels;
use poly_ring::Polynomial;

use crate::matrix::{MatrixError, PolynomialMatrix};

impl<F: FieldKernels> PolynomialMatrix<F> {
    /// Sorts nonzero rows by increasing shifted leading column, zero rows
    /// last, returning the row permutation applied.
    ///
    /// The input must already be in shifted weak Popov form under `shifts`;
    /// the routine reads leading terms without verifying that precondition.
    /// The permutation maps new positions to old rows: `result[i]` holds
    /// the old row now at position `i`.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::ShiftCount`] when `shifts` does not hold
    /// exactly one entry per column.
    pub fn order_weak_popov(&mut self, shifts: &[usize]) -> Result<Vec<usize>, MatrixError> {
        if shifts.len() != self.shape().1 {
            return Err(MatrixError::ShiftCount {
                columns: self.shape().1,
                shifts: shifts.len(),
            });
        }
        let (rows, columns) = self.shape();
        // Zero rows sort last via `usize::MAX`; nonzero rows sort by
        // increasing leading column, then shifted degree.
        let mut keys: Vec<(usize, usize, usize)> = Vec::with_capacity(rows);
        for row in 0..rows {
            match self.leading_key(row, shifts) {
                None => keys.push((1, usize::MAX, usize::MAX)),
                Some((shifted, column)) => keys.push((0, column, shifted)),
            }
        }
        let mut permutation: Vec<usize> = (0..rows).collect();
        permutation.sort_by_key(|&row| keys[row]);
        let snapshot = self.clone();
        for (new, &old) in permutation.iter().enumerate() {
            for column in 0..columns {
                let entry = snapshot.entry(old, column).cloned().unwrap_or_default();
                self.set_entry(new, column, entry)?;
            }
        }
        Ok(permutation)
    }

    /// Reduces to canonical shifted row Popov form.
    ///
    /// Runs weak reduction, orders rows, makes every pivot monic, and
    /// reduces every other entry in each pivot column below the pivot
    /// degree, repeating until all pivot-column inequalities hold. Zero
    /// rows end last; rectangular and rank-deficient inputs keep their
    /// shape with zero-row padding.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`PolynomialMatrix::reduce_weak_popov`].
    pub fn reduce_popov(&mut self, shifts: &[usize]) -> Result<(), MatrixError> {
        self.reduce_weak_popov(shifts)?;
        self.order_weak_popov(shifts)?;
        self.canonicalize_popov(shifts)
    }

    /// Reduces to canonical form with the unimodular witness `U * self = R`.
    ///
    /// Ordering permutes both matrices identically; monic scaling and
    /// column reduction apply the same elementary operation to each.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`PolynomialMatrix::reduce_weak_popov`].
    pub fn reduce_popov_tracked(
        &mut self,
        shifts: &[usize],
    ) -> Result<PolynomialMatrix<F>, MatrixError> {
        let mut transform = self.reduce_weak_popov_tracked(shifts)?;
        self.apply_permutation_tracked(shifts, &mut transform)?;
        self.canonicalize_popov_tracked(shifts, &mut transform)?;
        Ok(transform)
    }

    /// Reports whether every nonzero row has a distinct shifted leading
    /// column.
    #[must_use]
    pub fn is_weak_popov(&self, shifts: &[usize]) -> bool {
        let (rows, columns) = self.shape();
        if shifts.len() != columns {
            return false;
        }
        let mut seen = alloc::vec![false; columns];
        for row in 0..rows {
            match self.leading_key(row, shifts) {
                None => {}
                Some((_, column)) => {
                    if seen[column] {
                        return false;
                    }
                    seen[column] = true;
                }
            }
        }
        true
    }

    /// Reports whether the matrix is weak Popov with nonzero rows ordered
    /// by increasing leading column and zero rows last.
    #[must_use]
    pub fn is_ordered_weak_popov(&self, shifts: &[usize]) -> bool {
        if !self.is_weak_popov(shifts) {
            return false;
        }
        let (rows, _) = self.shape();
        let mut last_column: Option<usize> = None;
        let mut seen_zero = false;
        for row in 0..rows {
            match self.leading_key(row, shifts) {
                None => seen_zero = true,
                Some((_, column)) => {
                    if seen_zero {
                        return false;
                    }
                    if last_column.is_some_and(|last| column <= last) {
                        return false;
                    }
                    last_column = Some(column);
                }
            }
        }
        true
    }

    /// Reports whether the matrix is canonical row Popov: ordered weak
    /// Popov with monic pivots and every other entry in each pivot column
    /// strictly below the pivot degree.
    #[must_use]
    pub fn is_popov(&self, shifts: &[usize]) -> bool {
        if !self.is_ordered_weak_popov(shifts) {
            return false;
        }
        let (rows, _) = self.shape();
        for pivot in 0..rows {
            let Some((_, pivot_column)) = self.leading_key(pivot, shifts) else {
                continue;
            };
            let pivot_entry = self.entry(pivot, pivot_column).cloned().unwrap_or_default();
            if pivot_entry.leading_coefficient() != Some(F::Elem::ONE) {
                return false;
            }
            let Some(pivot_degree) = pivot_entry.degree() else {
                continue;
            };
            for row in 0..rows {
                if row == pivot {
                    continue;
                }
                let entry = self.entry(row, pivot_column).cloned().unwrap_or_default();
                if entry.degree().is_some_and(|degree| degree >= pivot_degree) {
                    return false;
                }
            }
        }
        true
    }

    /// Returns the predictable degree of `combination`: the maximum over
    /// nonzero rows of `shifted row degree + combination degree`.
    ///
    /// Applies to combinations of nonzero reduced rows; it is not a
    /// guarantee of an arbitrary generating matrix.
    #[must_use]
    pub fn predictable_degree(
        &self,
        shifts: &[usize],
        combination: &[Polynomial<F>],
    ) -> Option<usize> {
        let (rows, _) = self.shape();
        let mut best: Option<usize> = None;
        for (row, factor) in combination.iter().enumerate().take(rows) {
            if factor.is_zero() {
                continue;
            }
            let Some((shifted, _)) = self.leading_key(row, shifts) else {
                continue;
            };
            let candidate = shifted.checked_add(factor.degree()?)?;
            if best.is_none_or(|current| candidate > current) {
                best = Some(candidate);
            }
        }
        best
    }

    fn leading_key(&self, row: usize, shifts: &[usize]) -> Option<(usize, usize)> {
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
            if best.is_none_or(|(best_shifted, best_column)| {
                (shifted, column) > (best_shifted, best_column)
            }) {
                best = Some((shifted, column));
            }
        }
        best
    }

    fn canonicalize_popov(&mut self, shifts: &[usize]) -> Result<(), MatrixError> {
        loop {
            let (rows, columns) = self.shape();
            let mut pivots: Vec<(usize, usize, usize)> = Vec::new();
            for row in 0..rows {
                if let Some((shifted, column)) = self.leading_key(row, shifts) {
                    let degree = self
                        .entry(row, column)
                        .and_then(Polynomial::degree)
                        .unwrap_or(0);
                    pivots.push((row, column, degree));
                    let _ = shifted;
                }
            }
            // Monic normalization first: scaling preserves shifted degrees.
            let mut changed = false;
            for &(row, column, _) in &pivots {
                if self.scale_pivot_monic(row, column, None)? {
                    changed = true;
                }
            }
            if changed {
                continue;
            }
            // Column reduction: eliminate entries at or above pivot degree.
            let mut reduced = false;
            for &(pivot, pivot_column, pivot_degree) in &pivots {
                let pivot_row: Vec<Polynomial<F>> = (0..columns)
                    .map(|column| self.entry(pivot, column).cloned().unwrap_or_default())
                    .collect();
                let Some(divisor) = ({
                    let entry = self.entry(pivot, pivot_column).cloned().unwrap_or_default();
                    if entry.is_zero() { None } else { Some(entry) }
                }) else {
                    continue;
                };
                for row in 0..rows {
                    if row == pivot {
                        continue;
                    }
                    if self.reduce_row_against_pivot(
                        row,
                        pivot,
                        pivot_column,
                        pivot_degree,
                        &pivot_row,
                        &divisor,
                        None,
                    )? {
                        reduced = true;
                    }
                }
            }
            if !reduced {
                return Ok(());
            }
        }
    }

    fn apply_permutation_tracked(
        &mut self,
        shifts: &[usize],
        transform: &mut PolynomialMatrix<F>,
    ) -> Result<Vec<usize>, MatrixError> {
        let permutation = self.order_weak_popov(shifts)?;
        let snapshot = transform.clone();
        let (rows, _) = transform.shape();
        for (new, &old) in permutation.iter().enumerate().take(rows) {
            for column in 0..rows {
                let entry = snapshot.entry(old, column).cloned().unwrap_or_default();
                transform.set_entry(new, column, entry)?;
            }
        }
        Ok(permutation)
    }

    fn canonicalize_popov_tracked(
        &mut self,
        shifts: &[usize],
        transform: &mut PolynomialMatrix<F>,
    ) -> Result<(), MatrixError> {
        loop {
            let (rows, columns) = self.shape();
            let mut pivots: Vec<(usize, usize, usize)> = Vec::new();
            for row in 0..rows {
                if let Some((_, column)) = self.leading_key(row, shifts) {
                    let degree = self
                        .entry(row, column)
                        .and_then(Polynomial::degree)
                        .unwrap_or(0);
                    pivots.push((row, column, degree));
                }
            }
            let mut changed = false;
            for &(row, column, _) in &pivots {
                if self.scale_pivot_monic(row, column, Some(transform))? {
                    changed = true;
                }
            }
            if changed {
                continue;
            }
            let mut reduced = false;
            for &(pivot, pivot_column, pivot_degree) in &pivots {
                let pivot_row: Vec<Polynomial<F>> = (0..columns)
                    .map(|column| self.entry(pivot, column).cloned().unwrap_or_default())
                    .collect();
                let divisor = self.entry(pivot, pivot_column).cloned().unwrap_or_default();
                if divisor.is_zero() {
                    continue;
                }
                for row in 0..rows {
                    if row == pivot {
                        continue;
                    }
                    if self.reduce_row_against_pivot(
                        row,
                        pivot,
                        pivot_column,
                        pivot_degree,
                        &pivot_row,
                        &divisor,
                        Some(transform),
                    )? {
                        reduced = true;
                    }
                }
            }
            if !reduced {
                return Ok(());
            }
        }
    }

    /// Scales one pivot row to monic, mirroring into `transform` when given.
    ///
    /// Returns whether the row changed.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::GeometryOverflow`] for out-of-bounds pivot
    /// coordinates.
    fn scale_pivot_monic(
        &mut self,
        row: usize,
        column: usize,
        transform: Option<&mut PolynomialMatrix<F>>,
    ) -> Result<bool, MatrixError> {
        let entry = self.entry(row, column).cloned().unwrap_or_default();
        let Some(leading) = entry.leading_coefficient() else {
            return Ok(false);
        };
        if leading == F::Elem::ONE {
            return Ok(false);
        }
        let inverse = leading.inv();
        let (_, columns) = self.shape();
        for other in 0..columns {
            let target = self.entry(row, other).cloned().unwrap_or_default();
            let mut scaled = target.clone();
            scaled.scale_assign(inverse);
            self.set_entry(row, other, scaled)?;
        }
        if let Some(transform) = transform {
            let (rows, _) = transform.shape();
            for other in 0..rows {
                let target = transform.entry(row, other).cloned().unwrap_or_default();
                let mut scaled = target.clone();
                scaled.scale_assign(inverse);
                transform.set_entry(row, other, scaled)?;
            }
        }
        Ok(true)
    }

    /// Reduces one row against a pivot row, mirroring into `transform` when
    /// given. Returns whether the row changed.
    ///
    /// # Errors
    ///
    /// Returns polynomial division errors and [`MatrixError`] index errors.
    #[allow(clippy::too_many_arguments)]
    fn reduce_row_against_pivot(
        &mut self,
        row: usize,
        pivot: usize,
        pivot_column: usize,
        pivot_degree: usize,
        pivot_row: &[Polynomial<F>],
        divisor: &Polynomial<F>,
        transform: Option<&mut PolynomialMatrix<F>>,
    ) -> Result<bool, MatrixError> {
        let entry = self.entry(row, pivot_column).cloned().unwrap_or_default();
        let Some(degree) = entry.degree() else {
            return Ok(false);
        };
        if degree < pivot_degree {
            return Ok(false);
        }
        let (quotient, _) = entry.div_rem(divisor)?;
        if quotient.is_zero() {
            return Ok(false);
        }
        for (column, pivot_entry) in pivot_row.iter().enumerate() {
            let mut target = self.entry(row, column).cloned().unwrap_or_default();
            let mut multiple = quotient.multiply(pivot_entry)?;
            multiple.scale_assign(F::Elem::ONE.neg());
            target.add_assign(&multiple)?;
            self.set_entry(row, column, target)?;
        }
        if let Some(transform) = transform {
            let (rows, _) = transform.shape();
            for column in 0..rows {
                let mut target = transform.entry(row, column).cloned().unwrap_or_default();
                let pivot_witness = transform.entry(pivot, column).cloned().unwrap_or_default();
                let mut multiple = quotient.multiply(&pivot_witness)?;
                multiple.scale_assign(F::Elem::ONE.neg());
                target.add_assign(&multiple)?;
                transform.set_entry(row, column, target)?;
            }
        }
        Ok(true)
    }
}
