//! Generic polynomial congruence module bases.
//!
//! For `F` in Fq\[x\]^(m by n) and nonzero moduli `M[j]` in Fq\[x\], the
//! congruence module holds the rows `p` in Fq\[x\]^(1 by m) with
//! `(p * F)[j]` divisible by `M[j]` for every column `j`. A unit (nonzero
//! constant) modulus imposes no constraint; a zero modulus is rejected and
//! exact kernels have their own API. Moduli need not split into linear
//! factors and may share non-trivial common divisors.
//!
//! The constructor builds the block matrix `[F; -diag(M)]` in row
//! orientation: the first `m` rows carry `F`, and `n` further rows carry
//! the negated moduli on the diagonal. Its complete left kernel projects
//! onto the first `m` coordinates, which identifies exactly the requested
//! module: a kernel row `(p, q)` satisfies `p * F = q * diag(M)`, so `p`
//! satisfies every congruence, and every solution lifts with the
//! componentwise quotients. The projected generators reduce to the
//! requested shifted form, returned as an m-by-m full-rank basis.

use alloc::vec::Vec;

use fgf::kernel::FieldKernels;
use poly_ring::Polynomial;

use crate::matrix::{MatrixError, PolynomialMatrix};

/// A congruence module basis with its projection-completeness witness.
///
/// `block_kernel` holds the complete left kernel of `[F; -diag(M)]`
/// and `block` the block itself, so a verifier recomputes
/// `block_kernel * block == 0`, checks the kernel row count against
/// the block row count minus its full column rank, and confirms `basis`
/// rows are the kernel's first-`m` projections reduced to canonical form.
#[derive(Debug, Clone)]
pub struct CongruenceCertificate<F: FieldKernels> {
    /// Canonical m-by-m module basis.
    pub basis: PolynomialMatrix<F>,
    /// Complete left kernel of the congruence block.
    pub block_kernel: PolynomialMatrix<F>,
    /// The `[F; -diag(M)]` block the kernel annihilates.
    pub block: PolynomialMatrix<F>,
}

impl<F: FieldKernels> PolynomialMatrix<F> {
    /// Returns an m-by-m basis of the congruence module for `F = self`
    ///
    /// Each modulus normalizes to monic form. When every modulus is a
    /// nonzero constant, no constraint is effective and the output is the
    /// identity. The output reduces to canonical shifted row Popov form
    /// under `shifts`, which carries one entry per solution coordinate
    /// (`m` entries).
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::GeometryOverflow`] when `moduli` does not
    /// hold exactly one entry per column or holds a zero modulus, plus
    /// kernel, reduction, and polynomial errors.
    pub fn congruence_basis(
        &self,
        moduli: &[Polynomial<F>],
        shifts: &[usize],
    ) -> Result<PolynomialMatrix<F>, MatrixError> {
        Ok(self.congruence_basis_certified(moduli, shifts)?.basis)
    }

    /// Returns the congruence basis with its completeness witness.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`PolynomialMatrix::congruence_basis`].
    pub fn congruence_basis_certified(
        &self,
        moduli: &[Polynomial<F>],
        shifts: &[usize],
    ) -> Result<CongruenceCertificate<F>, MatrixError> {
        let (rows, columns) = self.shape();
        if moduli.len() != columns {
            return Err(MatrixError::GeometryOverflow {
                context: "congruence modulus count",
            });
        }
        for modulus in moduli {
            if modulus.is_zero() {
                return Err(MatrixError::GeometryOverflow {
                    context: "congruence zero modulus",
                });
            }
        }
        if shifts.len() != rows {
            return Err(MatrixError::ShiftCount {
                columns: rows,
                shifts: shifts.len(),
            });
        }
        if moduli.iter().all(|modulus| modulus.degree() == Some(0)) {
            let basis = PolynomialMatrix::identity(rows)?;
            let block = self.congruence_block(moduli)?;
            let block_kernel = block.left_kernel_weak_popov(&alloc::vec![0; columns])?;
            return Ok(CongruenceCertificate {
                basis,
                block_kernel,
                block,
            });
        }
        let block = self.congruence_block(moduli)?;
        let kernel = block.left_kernel_weak_popov(&alloc::vec![0; columns])?;
        let mut generators = Vec::new();
        for row in 0..kernel.shape().0 {
            for column in 0..rows {
                generators.push(kernel.entry(row, column).cloned().unwrap_or_default());
            }
        }
        let generator_rows = if rows == 0 {
            0
        } else {
            generators.len() / rows.max(1)
        };
        let mut basis = if generators.is_empty() {
            PolynomialMatrix::zeros(0, rows)?
        } else {
            PolynomialMatrix::from_entries(generator_rows, rows, generators)?
        };
        // The block has full column rank, so the projection yields exactly
        // `rows` generators; canonical reduction preserves the row count.
        // Zero rows pad any rank shortfall to the square m-by-m basis.
        basis.reduce_popov(shifts)?;
        let basis = Self::complete_to_square(&basis, rows)?;
        Ok(CongruenceCertificate {
            basis,
            block_kernel: kernel,
            block,
        })
    }

    /// Builds the `[F; -diag(M)]` block matrix in row orientation.
    ///
    /// The first `m` rows carry `F`; `n` further rows carry the negated
    /// monic moduli on the diagonal. All moduli are nonzero on entry.
    fn congruence_block(
        &self,
        moduli: &[Polynomial<F>],
    ) -> Result<PolynomialMatrix<F>, MatrixError> {
        let (rows, columns) = self.shape();
        let mut block = PolynomialMatrix::zeros(rows + columns, columns)?;
        for row in 0..rows {
            for column in 0..columns {
                block.set_entry(
                    row,
                    column,
                    self.entry(row, column).cloned().unwrap_or_default(),
                )?;
            }
        }
        for (index, modulus) in moduli.iter().enumerate() {
            let monic = modulus.monic();
            let mut negated = Polynomial::zero();
            negated.sub_assign(&monic)?;
            block.set_entry(rows + index, index, negated)?;
        }
        Ok(block)
    }

    /// Truncates or zero-pads a reduced generator set to the square
    /// m-by-m module basis.
    ///
    /// The block always has full column rank (every column carries its
    /// nonzero negated modulus), so the kernel projection yields exactly
    /// `size` generators; canonical reduction keeps at most `size` nonzero
    /// rows and this pads any shortfall with zero rows.
    fn complete_to_square(
        basis: &PolynomialMatrix<F>,
        size: usize,
    ) -> Result<PolynomialMatrix<F>, MatrixError> {
        let (rows, columns) = basis.shape();
        debug_assert_eq!(columns, size);
        let mut entries: Vec<Polynomial<F>> = Vec::with_capacity(size * size);
        for row in 0..rows.min(size) {
            for column in 0..size {
                entries.push(basis.entry(row, column).cloned().unwrap_or_default());
            }
        }
        for _ in rows.min(size)..size {
            for _ in 0..size {
                entries.push(Polynomial::zero());
            }
        }
        PolynomialMatrix::from_entries(size, size, entries)
    }
}
