//! Owned rectangular matrices over Fq\[x\].
//!
//! The matrix owns an m-by-n row-major vector of normalized [`Polynomial`]
//! entries, including the empty 0-by-n and m-by-0 shapes. Checked
//! construction, entry access, and destination-first arithmetic compose the
//! public scalar operations of `poly-ring`; no polynomial arithmetic is
//! re-implemented here.

use alloc::vec::Vec;
use core::fmt;

use crate::ReduceError;
use crate::reduction::{WeakPopovBasis, weak_popov_basis_scratch};
use fgf::kernel::FieldKernels;
use poly_ring::{Polynomial, PolynomialError};

/// A checked operation on an owned polynomial matrix failed.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MatrixError {
    /// A checked row, column, or degree product overflowed `usize`, or two
    /// shapes disagreed.
    GeometryOverflow {
        /// Static description of the failing check.
        context: &'static str,
    },
    /// Storage for entries or scratch could not be reserved.
    AllocationFailed {
        /// Number of entries requested.
        entries: usize,
    },
    /// A shift vector did not hold exactly one entry per matrix column.
    ShiftCount {
        /// Matrix columns.
        columns: usize,
        /// Entries in the shift vector.
        shifts: usize,
    },
    /// Normalizing signed shifts overflowed the representable span.
    ShiftSpan {
        /// Minimum signed shift observed.
        minimum: i64,
        /// Maximum signed shift observed.
        maximum: i64,
    },
    /// A supporting polynomial operation failed.
    Polynomial(PolynomialError),
    /// An owned-basis reduction failed validation or termination.
    Reduction(ReduceError),
}

impl fmt::Display for MatrixError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, formatter)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for MatrixError {}

impl From<PolynomialError> for MatrixError {
    fn from(error: PolynomialError) -> Self {
        Self::Polynomial(error)
    }
}

impl From<ReduceError> for MatrixError {
    fn from(error: ReduceError) -> Self {
        Self::Reduction(error)
    }
}

/// A shift vector normalized to nonnegative entries.
///
/// Signed preparation subtracts the common minimum with checked arithmetic;
/// `offset` records the subtracted minimum so a caller can report original
/// shifted degrees. Normalization preserves pivot choice but not the
/// reducer's termination ceiling or its reported shifted degrees.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShiftPreparation {
    /// Nonnegative shifts driving the reducer.
    pub shifts: Vec<usize>,
    /// Subtracted common minimum; equals the minimum input shift.
    pub offset: i64,
}

impl ShiftPreparation {
    /// Normalizes signed `shifts` by subtracting their minimum.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::ShiftSpan`] when the input is empty or its span
    /// overflows `usize`.
    pub fn from_signed(shifts: &[i64]) -> Result<Self, MatrixError> {
        let (&minimum, &maximum) = match (shifts.iter().min(), shifts.iter().max()) {
            (Some(minimum), Some(maximum)) => (minimum, maximum),
            (None, None) => {
                return Err(MatrixError::ShiftSpan {
                    minimum: 0,
                    maximum: 0,
                });
            }
            (None, Some(_)) | (Some(_), None) => unreachable!("min and max agree on emptiness"),
        };
        let span = usize::try_from(i128::from(maximum) - i128::from(minimum))
            .map_err(|_| MatrixError::ShiftSpan { minimum, maximum })?;
        let _ = span;
        let mut normalized = Vec::with_capacity(shifts.len());
        for &shift in shifts {
            let entry = usize::try_from(i128::from(shift) - i128::from(minimum))
                .map_err(|_| MatrixError::ShiftSpan { minimum, maximum })?;
            normalized.push(entry);
        }
        Ok(Self {
            shifts: normalized,
            offset: minimum,
        })
    }
}

/// An owned m-by-n matrix over Fq\[x\] in row-major order.
///
/// Entries are normalized [`Polynomial`] values; the zero polynomial is the
/// empty buffer and [`Polynomial::degree`] of it is `None`. The shape is
/// explicit: a 0-by-n matrix holds no rows, an m-by-0 matrix holds rows with
/// no entries, and logical trailing zero columns are part of the shape, never
/// dropped by entry normalization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolynomialMatrix<F: FieldKernels> {
    rows: usize,
    columns: usize,
    entries: Vec<Polynomial<F>>,
}

impl<F: FieldKernels> PolynomialMatrix<F> {
    /// Builds an m-by-n matrix from row-major `entries`, normalizing each
    /// entry to the canonical representation.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::GeometryOverflow`] when `rows * columns`
    /// overflows or disagrees with the entry count.
    pub fn from_entries(
        rows: usize,
        columns: usize,
        mut entries: Vec<Polynomial<F>>,
    ) -> Result<Self, MatrixError> {
        let count = rows
            .checked_mul(columns)
            .ok_or(MatrixError::GeometryOverflow {
                context: "polynomial matrix entries",
            })?;
        if count != entries.len() {
            return Err(MatrixError::GeometryOverflow {
                context: "polynomial matrix entries",
            });
        }
        for entry in &mut entries {
            entry.normalize();
        }
        Ok(Self {
            rows,
            columns,
            entries,
        })
    }

    /// Builds an m-by-n zero matrix.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::AllocationFailed`] when the entry vector cannot
    /// be reserved.
    pub fn zeros(rows: usize, columns: usize) -> Result<Self, MatrixError> {
        let count = rows
            .checked_mul(columns)
            .ok_or(MatrixError::GeometryOverflow {
                context: "polynomial matrix entries",
            })?;
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(count)
            .map_err(|_| MatrixError::AllocationFailed { entries: count })?;
        entries.resize_with(count, Polynomial::zero);
        Ok(Self {
            rows,
            columns,
            entries,
        })
    }

    /// Builds the n-by-n identity matrix.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`PolynomialMatrix::zeros`], plus the
    /// polynomial error when the unit constant cannot be built.
    pub fn identity(size: usize) -> Result<Self, MatrixError> {
        let mut matrix = Self::zeros(size, size)?;
        let one = Polynomial::one()?;
        for index in 0..size {
            matrix.entries[index * size + index].clone_from(&one);
        }
        Ok(matrix)
    }

    /// Returns the explicit `(rows, columns)` shape.
    #[must_use]
    pub const fn shape(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }

    /// Returns the entry at `(row, column)`, or `None` when out of bounds.
    #[must_use]
    pub fn entry(&self, row: usize, column: usize) -> Option<&Polynomial<F>> {
        if row >= self.rows || column >= self.columns {
            return None;
        }
        self.entries.get(row * self.columns + column)
    }

    /// Replaces the entry at `(row, column)` with a normalized polynomial.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::GeometryOverflow`] when the index product
    /// overflows or the coordinates are out of bounds.
    pub fn set_entry(
        &mut self,
        row: usize,
        column: usize,
        mut value: Polynomial<F>,
    ) -> Result<(), MatrixError> {
        let index = self.checked_index(row, column)?;
        value.normalize();
        self.entries[index] = value;
        Ok(())
    }

    /// Returns the transpose, exchanging rows and columns.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::AllocationFailed`] when the transposed entry
    /// vector cannot be reserved.
    pub fn transposed(&self) -> Result<Self, MatrixError> {
        let count = self
            .rows
            .checked_mul(self.columns)
            .ok_or(MatrixError::GeometryOverflow {
                context: "polynomial matrix entries",
            })?;
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(count)
            .map_err(|_| MatrixError::AllocationFailed { entries: count })?;
        for column in 0..self.columns {
            for row in 0..self.rows {
                entries.push(self.entries[row * self.columns + column].clone());
            }
        }
        Ok(Self {
            rows: self.columns,
            columns: self.rows,
            entries,
        })
    }

    /// Writes `self + other` into `out`, leaving both inputs unchanged.
    ///
    /// The sum is staged in a fresh matrix and moved into `out` only after
    /// every entry succeeds, so any failure leaves `out` unchanged.
    /// Aliasing `out` with an input is rejected by the borrow checker: `out`
    /// is borrowed mutably while the inputs are borrowed immutably.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::GeometryOverflow`] for mismatched shapes or an
    /// overflowing entry count, [`MatrixError::AllocationFailed`] when the
    /// staged matrix cannot be reserved, or the polynomial error.
    pub fn add_into(&self, other: &Self, out: &mut Self) -> Result<(), MatrixError> {
        self.checked_same_shape(other)?;
        let mut staged = Self::zeros(self.rows, self.columns)?;
        for ((left, right), target) in self
            .entries
            .iter()
            .zip(&other.entries)
            .zip(&mut staged.entries)
        {
            *target = left.add(right)?;
        }
        *out = staged;
        Ok(())
    }

    /// Writes `self - other` into `out`.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`PolynomialMatrix::add_into`].
    pub fn sub_into(&self, other: &Self, out: &mut Self) -> Result<(), MatrixError> {
        self.checked_same_shape(other)?;
        let mut staged = Self::zeros(self.rows, self.columns)?;
        for ((left, right), target) in self
            .entries
            .iter()
            .zip(&other.entries)
            .zip(&mut staged.entries)
        {
            *target = left.sub(right)?;
        }
        *out = staged;
        Ok(())
    }

    /// Writes the full product `self * other` into `out`, where
    /// `C[i,j] = sum_k A[i,k] B[k,j]` over Fq\[x\].
    ///
    /// The product is staged in a fresh matrix and moved into `out` only
    /// after every entry succeeds, so any failure leaves `out` unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::GeometryOverflow`] for an inner-dimension
    /// mismatch or an overflowing index product, plus reservation and
    /// polynomial errors.
    pub fn mul_into(&self, other: &Self, out: &mut Self) -> Result<(), MatrixError> {
        self.mul_truncated_into(other, usize::MAX, out)
    }

    /// Writes the product truncated modulo `x^bound` into `out`.
    ///
    /// `usize::MAX` is the full product; `0` is the zero matrix of the
    /// product shape.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`PolynomialMatrix::mul_into`].
    pub fn mul_truncated_into(
        &self,
        other: &Self,
        bound: usize,
        out: &mut Self,
    ) -> Result<(), MatrixError> {
        if self.columns != other.rows {
            return Err(MatrixError::GeometryOverflow {
                context: "polynomial matrix inner dimension",
            });
        }
        let mut staged = Self::zeros(self.rows, other.columns)?;
        for row in 0..self.rows {
            for inner in 0..self.columns {
                let left = &self.entries[row * self.columns + inner];
                if left.is_zero() {
                    continue;
                }
                for column in 0..other.columns {
                    let right = &other.entries[inner * other.columns + column];
                    if right.is_zero() {
                        continue;
                    }
                    let target = &mut staged.entries[row * other.columns + column];
                    if bound == usize::MAX {
                        let product = left.multiply(right)?;
                        target.add_assign(&product)?;
                    } else {
                        let product = left.multiply_truncated(right, bound)?;
                        target.add_assign(&product)?;
                        target.truncate(bound);
                    }
                }
            }
        }
        *out = staged;
        Ok(())
    }

    /// Evaluates every entry at `point`, returning row-major field elements.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::AllocationFailed`] when the output buffer
    /// cannot be reserved.
    pub fn evaluate_to_vec(&self, point: F::Elem) -> Result<Vec<F::Elem>, MatrixError> {
        let count = self
            .rows
            .checked_mul(self.columns)
            .ok_or(MatrixError::GeometryOverflow {
                context: "polynomial matrix entries",
            })?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| MatrixError::AllocationFailed { entries: count })?;
        for entry in &self.entries {
            values.push(entry.evaluate(point));
        }
        Ok(values)
    }

    /// Reduces the rows to shifted weak Popov form with exactly `n` shifts.
    ///
    /// Zero rows are allowed; every nonzero result row has a distinct
    /// leading column. Any failure reported after an applied row update,
    /// including a callback failure and a later [`MatrixError::Reduction`]
    /// validation or termination failure, may leave the matrix partially
    /// reduced; no rollback is performed.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::ShiftCount`] when `shifts` does not hold
    /// exactly one entry per column, or the reduction error.
    pub fn reduce_weak_popov(&mut self, shifts: &[usize]) -> Result<(), MatrixError> {
        if shifts.len() != self.columns {
            return Err(MatrixError::ShiftCount {
                columns: self.columns,
                shifts: shifts.len(),
            });
        }
        let mut adapter = MatrixRows { matrix: self };
        let mut scratch = crate::WeakPopovScratch::new();
        weak_popov_basis_scratch::<F, _>(&mut adapter, shifts, &mut scratch)?;
        Ok(())
    }

    /// Reduces with signed shifts normalized by common-offset subtraction.
    ///
    /// Returns the [`ShiftPreparation`] carrying the normalized shifts and
    /// the subtracted offset.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`PolynomialMatrix::reduce_weak_popov`],
    /// plus [`MatrixError::ShiftSpan`] for an empty or unrepresentable span.
    pub fn reduce_weak_popov_signed(
        &mut self,
        shifts: &[i64],
    ) -> Result<ShiftPreparation, MatrixError> {
        let preparation = ShiftPreparation::from_signed(shifts)?;
        self.reduce_weak_popov(&preparation.shifts)?;
        Ok(preparation)
    }

    fn checked_index(&self, row: usize, column: usize) -> Result<usize, MatrixError> {
        if row >= self.rows || column >= self.columns {
            return Err(MatrixError::GeometryOverflow {
                context: "polynomial matrix index",
            });
        }
        row.checked_mul(self.columns)
            .and_then(|base| base.checked_add(column))
            .ok_or(MatrixError::GeometryOverflow {
                context: "polynomial matrix index",
            })
    }

    fn checked_same_shape(&self, other: &Self) -> Result<(), MatrixError> {
        if self.rows != other.rows || self.columns != other.columns {
            return Err(MatrixError::GeometryOverflow {
                context: "polynomial matrix shape",
            });
        }
        Ok(())
    }
}

/// Row view borrowing target and pivot rows out of one backing vector.
///
/// `split_at_mut` yields disjoint borrows; the cost is one split per row
/// update with no copied coefficients.
struct MatrixRows<'a, F: FieldKernels> {
    matrix: &'a mut PolynomialMatrix<F>,
}

impl<F: FieldKernels> WeakPopovBasis<F> for MatrixRows<'_, F> {
    type Error = MatrixError;

    fn row_count(&self) -> usize {
        self.matrix.rows
    }

    fn column_count(&self, _row: usize) -> usize {
        self.matrix.columns
    }

    fn degree(&self, row: usize, column: usize) -> Option<usize> {
        self.matrix.entries[row * self.matrix.columns + column].degree()
    }

    fn coefficient(&self, row: usize, column: usize, degree: usize) -> F::Elem {
        self.matrix.entries[row * self.matrix.columns + column].coefficient(degree)
    }

    fn add_scaled_shifted_assign(
        &mut self,
        target: usize,
        pivot: usize,
        scale: F::Elem,
        shift: usize,
        _shifts: &[usize],
    ) -> Result<(), Self::Error> {
        let columns = self.matrix.columns;
        let (target_base, pivot_base) = (target * columns, pivot * columns);
        for column in 0..columns {
            let (target_entry, pivot_entry) = if target_base + column < pivot_base + column {
                let (lower, upper) = self.matrix.entries.split_at_mut(pivot_base + column);
                (&mut lower[target_base + column], &upper[0])
            } else {
                let (lower, upper) = self.matrix.entries.split_at_mut(target_base + column);
                (&mut upper[0], &lower[pivot_base + column])
            };
            target_entry.add_scaled_shifted_assign(scale, pivot_entry, shift)?;
        }
        Ok(())
    }
}
