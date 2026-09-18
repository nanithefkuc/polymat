//! Polynomial-module reduction errors.

use core::fmt;

/// A shifted polynomial-row reduction failed validation or termination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReduceError {
    /// The reduction exceeded its checked iteration ceiling.
    Diverged {
        /// Iterations actually performed.
        iterations: usize,
        /// The ceiling that was exceeded.
        ceiling: usize,
    },
    /// A row exposes more polynomial columns than the shift vector.
    ShiftCount {
        /// Polynomial columns in the row.
        columns: usize,
        /// Entries in the shift vector.
        shifts: usize,
    },
    /// Adding a polynomial degree and its shift overflowed `usize`.
    DegreeOverflow {
        /// Unshifted polynomial degree.
        degree: usize,
        /// Shift assigned to that polynomial column.
        shift: usize,
    },
    /// Storage for the leading-row schedule could not be reserved.
    AllocationFailed {
        /// Number of schedule entries requested.
        entries: usize,
    },
    /// A custom leading-term result disagrees with the adapter's row data.
    InvalidLeadingTerm {
        /// Basis row whose metadata was rejected.
        row: usize,
    },
}

impl fmt::Display for ReduceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, formatter)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ReduceError {}
