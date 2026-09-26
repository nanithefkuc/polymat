//! Polynomial matrices and finitely generated row modules over Fq\[x\].
//!
//! The owned [`PolynomialMatrix`] holds explicit m-by-n row-major normalized
//! polynomial entries with checked construction and destination-first
//! arithmetic. The public reducer operates on caller-owned polynomial rows
//! or indexed, slab-backed bases. It preserves the generated row module and
//! produces shifted weak Popov form without imposing a storage
//! representation.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]
#![warn(clippy::pedantic)]

extern crate alloc;

mod error;
mod kernel;
mod matrix;
mod popov;
mod reduction;

#[cfg(feature = "internals")]
pub mod internals;

pub use error::ReduceError;
pub use kernel::{ExactSolution, MembershipWitness};
pub use matrix::{MatrixError, PolynomialMatrix, ShiftPreparation};
pub use reduction::{
    PopovLeadingTerm, WeakPopovBasis, WeakPopovRow, WeakPopovScratch, weak_popov,
    weak_popov_basis_scratch, weak_popov_scratch,
};
