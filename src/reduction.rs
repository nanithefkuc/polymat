//! Generic shifted weak-Popov row reduction.
//!
//! The reducer owns the Mulders–Storjohann pivot schedule without owning a
//! polynomial representation. Consumers supply row access and one shifted row
//! update through [`WeakPopovRow`] or [`WeakPopovBasis`].

use core::cmp::Ordering;

use alloc::vec::Vec;

use fgf::FieldKernels;
use fgf::field::Elem;

use crate::ReduceError;

/// A leading polynomial term under a caller-supplied column shift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PopovLeadingTerm {
    /// Degree before applying the column shift.
    pub degree: usize,
    /// Polynomial column containing the term.
    pub column: usize,
    /// `degree + shifts[column]`.
    pub shifted_degree: usize,
}

/// Reusable leading-row schedule for shifted weak-Popov reduction.
#[derive(Debug, Default)]
pub struct WeakPopovScratch {
    leading_rows: Vec<Option<usize>>,
}

impl WeakPopovScratch {
    /// Constructs empty reduction scratch.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            leading_rows: Vec::new(),
        }
    }

    /// Returns retained schedule capacity available to a later reduction.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.leading_rows.capacity()
    }

    /// Returns the heap bytes this scratch retains for the schedule.
    ///
    /// A consumer reporting its own retained memory adds this instead of
    /// multiplying [`WeakPopovScratch::capacity`] by an assumed entry width.
    #[must_use]
    pub fn retained_bytes(&self) -> usize {
        self.leading_rows.capacity() * size_of::<Option<usize>>()
    }

    fn prepare(&mut self, columns: usize) -> Result<(), ReduceError> {
        if self.leading_rows.capacity() < columns {
            self.leading_rows
                .try_reserve_exact(columns - self.leading_rows.len())
                .map_err(|_| ReduceError::AllocationFailed { entries: columns })?;
        }
        self.leading_rows.resize(columns, None);
        Ok(())
    }
}

/// One caller-owned polynomial row accepted by [`weak_popov`].
///
/// Implementations expose polynomial degrees and coefficients and apply one
/// shifted row addition. If `column_count()` is less than the shift length,
/// degree queries for trailing columns must return `None` and coefficient
/// queries must return zero.
pub trait WeakPopovRow<F: FieldKernels> {
    /// Consumer error type.
    type Error: From<ReduceError>;

    /// Returns the number of logical polynomial columns in this row.
    fn column_count(&self) -> usize;

    /// Returns the degree of one polynomial column, or `None` when it is zero.
    fn degree(&self, column: usize) -> Option<usize>;

    /// Returns the coefficient of `X^degree` in one polynomial column.
    fn coefficient(&self, column: usize, degree: usize) -> F::Elem;

    /// Returns the leading term under `shifts`.
    ///
    /// An implementation may override this scan with cached metadata. The
    /// reducer checks the result against the degree surface before using it.
    ///
    /// # Errors
    ///
    /// Returns [`ReduceError::DegreeOverflow`] if a shifted degree cannot be
    /// represented.
    fn leading_term(&self, shifts: &[usize]) -> Result<Option<PopovLeadingTerm>, Self::Error>
    where
        Self: Sized,
    {
        scan_row_leading_term::<F, Self>(self, shifts).map_err(Self::Error::from)
    }

    /// Adds `scale * X^shift * pivot` to this row.
    ///
    /// # Errors
    ///
    /// Returns the row type's native error when the shifted update cannot be
    /// represented or applied. A failure may leave the row partially updated.
    fn add_scaled_shifted_assign(
        &mut self,
        scale: F::Elem,
        pivot: &Self,
        shift: usize,
    ) -> Result<(), Self::Error>;
}

/// A polynomial basis whose rows may share one backing allocation.
///
/// Unlike [`WeakPopovRow`], this interface identifies rows by index, allowing
/// a consumer to expose a contiguous slab without manufacturing aliased mutable
/// row views. If `column_count(row)` is less than the shift length, degree
/// queries for trailing columns must return `None` and coefficient queries must
/// return zero.
pub trait WeakPopovBasis<F: FieldKernels> {
    /// Consumer error type.
    type Error: From<ReduceError>;

    /// Returns the number of basis rows.
    fn row_count(&self) -> usize;

    /// Returns the number of logical polynomial columns in one row.
    fn column_count(&self, row: usize) -> usize;

    /// Returns the degree of one polynomial column, or `None` when it is zero.
    fn degree(&self, row: usize, column: usize) -> Option<usize>;

    /// Returns the coefficient of `X^degree` in one polynomial column.
    fn coefficient(&self, row: usize, column: usize, degree: usize) -> F::Elem;

    /// Returns the leading term under `shifts`.
    ///
    /// Implementations with cached metadata may override this scan. The
    /// reducer checks the result against the degree surface before using it.
    ///
    /// # Errors
    ///
    /// Returns [`ReduceError::DegreeOverflow`] if a shifted degree cannot be
    /// represented.
    fn leading_term(
        &self,
        row: usize,
        shifts: &[usize],
    ) -> Result<Option<PopovLeadingTerm>, Self::Error> {
        scan_basis_leading_term::<F, Self>(self, row, shifts).map_err(Self::Error::from)
    }

    /// Adds `scale * X^shift * pivot` to `target`.
    ///
    /// `shifts` lets an implementation refresh cached leading metadata without
    /// retaining a second copy.
    ///
    /// # Errors
    ///
    /// Returns the basis-native error when the shifted update cannot be
    /// represented or applied. A failure may leave the basis partially updated.
    fn add_scaled_shifted_assign(
        &mut self,
        target: usize,
        pivot: usize,
        scale: F::Elem,
        shift: usize,
        shifts: &[usize],
    ) -> Result<(), Self::Error>;
}

struct RowBasis<'a, R>(&'a mut [R]);

impl<F, R> WeakPopovBasis<F> for RowBasis<'_, R>
where
    F: FieldKernels,
    R: WeakPopovRow<F>,
{
    type Error = R::Error;

    fn row_count(&self) -> usize {
        self.0.len()
    }

    fn column_count(&self, row: usize) -> usize {
        self.0[row].column_count()
    }

    fn degree(&self, row: usize, column: usize) -> Option<usize> {
        self.0[row].degree(column)
    }

    fn coefficient(&self, row: usize, column: usize, degree: usize) -> F::Elem {
        self.0[row].coefficient(column, degree)
    }

    fn leading_term(
        &self,
        row: usize,
        shifts: &[usize],
    ) -> Result<Option<PopovLeadingTerm>, Self::Error> {
        self.0[row].leading_term(shifts)
    }

    fn add_scaled_shifted_assign(
        &mut self,
        target: usize,
        pivot: usize,
        scale: F::Elem,
        shift: usize,
        _shifts: &[usize],
    ) -> Result<(), Self::Error> {
        let (target_row, pivot_row) = if target < pivot {
            let (lower, upper) = self.0.split_at_mut(pivot);
            (&mut lower[target], &upper[0])
        } else {
            let (lower, upper) = self.0.split_at_mut(target);
            (&mut upper[0], &lower[pivot])
        };
        target_row.add_scaled_shifted_assign(scale, pivot_row, shift)
    }
}

/// Reduces `basis` to shifted weak Popov form.
///
/// `shifts[column]` is added to every degree in that polynomial column. Ties
/// in shifted degree choose the greater column. Every nonzero result row has a
/// distinct leading column.
///
/// This allocating form creates schedule storage for the call. Use
/// [`weak_popov_scratch`] to retain it. Row-update callbacks may allocate
/// independently.
///
/// # Errors
///
/// Returns the row's native error or a checked [`ReduceError`]. Any failure
/// reported after an applied row update, including a callback failure and a
/// later validation failure, may leave the basis partially reduced.
pub fn weak_popov<F, R>(basis: &mut [R], shifts: &[usize]) -> Result<(), R::Error>
where
    F: FieldKernels,
    R: WeakPopovRow<F>,
{
    weak_popov_scratch::<F, R>(basis, shifts, &mut WeakPopovScratch::new())
}

/// Reduces caller-owned rows using retained schedule storage.
///
/// Once `scratch` has capacity for the shift length, the schedule itself does
/// not allocate. Row-update callbacks retain their own allocation behavior.
///
/// # Errors
///
/// Returns the same errors and has the same partial-mutation behavior as
/// [`weak_popov`].
pub fn weak_popov_scratch<F, R>(
    basis: &mut [R],
    shifts: &[usize],
    scratch: &mut WeakPopovScratch,
) -> Result<(), R::Error>
where
    F: FieldKernels,
    R: WeakPopovRow<F>,
{
    weak_popov_basis_scratch::<F, _>(&mut RowBasis(basis), shifts, scratch)
}

/// Reduces an indexed, potentially slab-backed basis using retained schedule
/// storage.
///
/// # Errors
///
/// Returns [`ReduceError::ShiftCount`] for a row wider than `shifts`,
/// [`ReduceError::DegreeOverflow`] for an unrepresentable shifted degree,
/// [`ReduceError::InvalidLeadingTerm`] when cached metadata disagrees with the
/// adapter's degree and coefficient surface, [`ReduceError::Diverged`] for a
/// non-decreasing update, or the basis-native update error. Any failure
/// reported after an applied row update, including a callback failure and a
/// later validation failure, may leave the basis partially reduced. No
/// rollback is performed.
pub fn weak_popov_basis_scratch<F, B>(
    basis: &mut B,
    shifts: &[usize],
    scratch: &mut WeakPopovScratch,
) -> Result<(), B::Error>
where
    F: FieldKernels,
    B: WeakPopovBasis<F> + ?Sized,
{
    validate_shape::<F, B>(basis, shifts)?;
    let columns = shifts.len();
    scratch.prepare(columns).map_err(B::Error::from)?;
    let leading_rows = &mut scratch.leading_rows;
    let mut ceiling = 0usize;
    let mut iterations = 0usize;

    leading_rows.fill(None);
    let mut collision = None;
    for row in 0..basis.row_count() {
        let Some(leading) = checked_basis_leading_term::<F, B>(basis, row, shifts)? else {
            continue;
        };
        ceiling = ceiling.saturating_add(row_measure(columns, leading));
        if collision.is_some() {
            continue;
        }
        if let Some(previous) = leading_rows[leading.column] {
            collision = Some((previous, row));
        } else {
            leading_rows[leading.column] = Some(row);
        }
    }

    loop {
        let Some((left, right)) = collision else {
            return Ok(());
        };
        if iterations >= ceiling {
            return Err(ReduceError::Diverged {
                iterations,
                ceiling,
            }
            .into());
        }
        let target = reduce_checked_pair::<F, B>(basis, left, right, shifts)?;
        iterations += 1;

        if let Some(old_column) = leading_rows.iter().position(|slot| *slot == Some(target)) {
            leading_rows[old_column] = None;
        }
        let new_leading = checked_basis_leading_term::<F, B>(basis, target, shifts)?;
        collision = match new_leading {
            None => None,
            Some(leading) => {
                if let Some(other) = leading_rows[leading.column] {
                    Some((other, target))
                } else {
                    leading_rows[leading.column] = Some(target);
                    None
                }
            }
        };
        if collision.is_some() {
            continue;
        }

        for row in 0..basis.row_count() {
            let Some(leading) = checked_basis_leading_term::<F, B>(basis, row, shifts)? else {
                continue;
            };
            if leading_rows[leading.column] == Some(row) {
                continue;
            }
            if let Some(previous) = leading_rows[leading.column] {
                collision = Some((previous, row));
                break;
            }
            leading_rows[leading.column] = Some(row);
        }
    }
}

/// Reduces an indexed basis by rescanning every row after each update.
///
/// This deliberately untuned implementation is exposed only through the
/// `internals` feature as an independent schedule control.
///
/// # Errors
///
/// Returns the same validation, callback, and partial-mutation errors as
/// [`weak_popov_basis_scratch`].
#[cfg_attr(not(feature = "internals"), allow(dead_code))]
pub fn weak_popov_basis_reference<F, B>(basis: &mut B, shifts: &[usize]) -> Result<(), B::Error>
where
    F: FieldKernels,
    B: WeakPopovBasis<F> + ?Sized,
{
    validate_shape::<F, B>(basis, shifts)?;
    let columns = shifts.len();
    let mut ceiling = 0usize;
    for row in 0..basis.row_count() {
        if let Some(leading) = scan_basis_leading_term::<F, B>(basis, row, shifts)? {
            ceiling = ceiling.saturating_add(row_measure(columns, leading));
        }
    }

    let mut iterations = 0usize;
    loop {
        let Some((left, right)) = first_scanned_collision::<F, B>(basis, shifts)? else {
            return Ok(());
        };
        if iterations >= ceiling {
            return Err(ReduceError::Diverged {
                iterations,
                ceiling,
            }
            .into());
        }
        reduce_scanned_pair::<F, B>(basis, left, right, shifts)?;
        iterations += 1;
    }
}

fn validate_shape<F, B>(basis: &B, shifts: &[usize]) -> Result<(), B::Error>
where
    F: FieldKernels,
    B: WeakPopovBasis<F> + ?Sized,
{
    for row in 0..basis.row_count() {
        let columns = basis.column_count(row);
        if columns > shifts.len() {
            return Err(ReduceError::ShiftCount {
                columns,
                shifts: shifts.len(),
            }
            .into());
        }
    }
    Ok(())
}

fn row_measure(columns: usize, leading: PopovLeadingTerm) -> usize {
    columns
        .saturating_mul(leading.shifted_degree)
        .saturating_add(leading.column)
        .saturating_add(1)
}

fn checked_basis_leading_term<F, B>(
    basis: &B,
    row: usize,
    shifts: &[usize],
) -> Result<Option<PopovLeadingTerm>, B::Error>
where
    F: FieldKernels,
    B: WeakPopovBasis<F> + ?Sized,
{
    let reported = basis.leading_term(row, shifts)?;
    let scanned = scan_basis_leading_term::<F, B>(basis, row, shifts)?;
    if reported != scanned {
        return Err(ReduceError::InvalidLeadingTerm { row }.into());
    }
    if let Some(term) = reported
        && basis.coefficient(row, term.column, term.degree).is_zero()
    {
        return Err(ReduceError::InvalidLeadingTerm { row }.into());
    }
    Ok(reported)
}

fn scan_basis_leading_term<F, B>(
    basis: &B,
    row: usize,
    shifts: &[usize],
) -> Result<Option<PopovLeadingTerm>, ReduceError>
where
    F: FieldKernels,
    B: WeakPopovBasis<F> + ?Sized,
{
    scan_leading_term(shifts, |column| basis.degree(row, column))
}

fn scan_row_leading_term<F, R>(
    row: &R,
    shifts: &[usize],
) -> Result<Option<PopovLeadingTerm>, ReduceError>
where
    F: FieldKernels,
    R: WeakPopovRow<F> + ?Sized,
{
    scan_leading_term(shifts, |column| row.degree(column))
}

fn scan_leading_term(
    shifts: &[usize],
    mut degree_at: impl FnMut(usize) -> Option<usize>,
) -> Result<Option<PopovLeadingTerm>, ReduceError> {
    let mut leading = None;
    for (column, &shift) in shifts.iter().enumerate() {
        let Some(degree) = degree_at(column) else {
            continue;
        };
        let shifted_degree = degree
            .checked_add(shift)
            .ok_or(ReduceError::DegreeOverflow { degree, shift })?;
        let candidate = PopovLeadingTerm {
            degree,
            column,
            shifted_degree,
        };
        if leading.is_none_or(|current: PopovLeadingTerm| {
            (candidate.shifted_degree, candidate.column) > (current.shifted_degree, current.column)
        }) {
            leading = Some(candidate);
        }
    }
    Ok(leading)
}

fn first_scanned_collision<F, B>(
    basis: &B,
    shifts: &[usize],
) -> Result<Option<(usize, usize)>, B::Error>
where
    F: FieldKernels,
    B: WeakPopovBasis<F> + ?Sized,
{
    for right in 0..basis.row_count() {
        let Some(right_term) = scan_basis_leading_term::<F, B>(basis, right, shifts)? else {
            continue;
        };
        for left in 0..right {
            let Some(left_term) = scan_basis_leading_term::<F, B>(basis, left, shifts)? else {
                continue;
            };
            if left_term.column == right_term.column {
                return Ok(Some((left, right)));
            }
        }
    }
    Ok(None)
}

fn reduce_checked_pair<F, B>(
    basis: &mut B,
    left: usize,
    right: usize,
    shifts: &[usize],
) -> Result<usize, B::Error>
where
    F: FieldKernels,
    B: WeakPopovBasis<F> + ?Sized,
{
    let left_term = checked_basis_leading_term::<F, B>(basis, left, shifts)?
        .expect("a colliding row has a leading term");
    let right_term = checked_basis_leading_term::<F, B>(basis, right, shifts)?
        .expect("a colliding row has a leading term");
    reduce_pair_with_terms::<F, B>(basis, left, right, left_term, right_term, shifts)
}

fn reduce_scanned_pair<F, B>(
    basis: &mut B,
    left: usize,
    right: usize,
    shifts: &[usize],
) -> Result<usize, B::Error>
where
    F: FieldKernels,
    B: WeakPopovBasis<F> + ?Sized,
{
    let left_term = scan_basis_leading_term::<F, B>(basis, left, shifts)?
        .expect("a colliding row has a leading term");
    let right_term = scan_basis_leading_term::<F, B>(basis, right, shifts)?
        .expect("a colliding row has a leading term");
    reduce_pair_with_terms::<F, B>(basis, left, right, left_term, right_term, shifts)
}

fn reduce_pair_with_terms<F, B>(
    basis: &mut B,
    left: usize,
    right: usize,
    left_term: PopovLeadingTerm,
    right_term: PopovLeadingTerm,
    shifts: &[usize],
) -> Result<usize, B::Error>
where
    F: FieldKernels,
    B: WeakPopovBasis<F> + ?Sized,
{
    let target_is_left = match left_term.degree.cmp(&right_term.degree) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => left < right,
    };
    let (target, pivot, target_term, pivot_term) = if target_is_left {
        (left, right, left_term, right_term)
    } else {
        (right, left, right_term, left_term)
    };
    let target_coefficient = basis.coefficient(target, target_term.column, target_term.degree);
    if target_coefficient.is_zero() {
        return Err(ReduceError::InvalidLeadingTerm { row: target }.into());
    }
    let pivot_coefficient = basis.coefficient(pivot, pivot_term.column, pivot_term.degree);
    if pivot_coefficient.is_zero() {
        return Err(ReduceError::InvalidLeadingTerm { row: pivot }.into());
    }
    let scale = target_coefficient.mul(pivot_coefficient.inv()).neg();
    let shift = target_term.degree - pivot_term.degree;
    basis.add_scaled_shifted_assign(target, pivot, scale, shift, shifts)?;
    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn impossible_schedule_reservation_is_reported() {
        let mut scratch = WeakPopovScratch::new();
        assert_eq!(
            scratch.prepare(usize::MAX),
            Err(ReduceError::AllocationFailed {
                entries: usize::MAX
            })
        );
    }
}
