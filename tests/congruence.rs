use fgf::field::Field;
use fgf::{Gf8B, Mersenne31, gf8b, mersenne31};
use poly_ring::Polynomial;
use polymat::PolynomialMatrix;

type B = <Gf8B as Field>::Elem;
type P = <Mersenne31 as Field>::Elem;

fn b(value: u8) -> B {
    gf8b::Elem::from_raw(value)
}

fn p(value: u32) -> P {
    mersenne31::Elem::from_raw(value)
}

fn poly_b(coefficients: &[B]) -> Polynomial<Gf8B> {
    Polynomial::from_coefficients(coefficients).unwrap()
}

fn poly_p(coefficients: &[P]) -> Polynomial<Mersenne31> {
    Polynomial::from_coefficients(coefficients).unwrap()
}

fn satisfies<F: fgf::kernel::FieldKernels>(
    basis: &PolynomialMatrix<F>,
    input: &PolynomialMatrix<F>,
    moduli: &[Polynomial<F>],
) -> bool {
    let (krows, _) = basis.shape();
    let (mrows, ncols) = input.shape();
    for k in 0..krows {
        for (j, modulus) in moduli.iter().enumerate().take(ncols) {
            let mut inner = Polynomial::zero();
            for i in 0..mrows {
                let term = basis
                    .entry(k, i)
                    .unwrap()
                    .multiply(input.entry(i, j).unwrap())
                    .unwrap();
                inner.add_assign(&term).unwrap();
            }
            let (_, remainder) = inner.div_rem(modulus).unwrap();
            if !remainder.is_zero() {
                return false;
            }
        }
    }
    true
}

#[test]
fn no_effective_constraints_return_identity() {
    let input = PolynomialMatrix::from_entries(
        2,
        2,
        alloc::vec![
            poly_b(&[b(1), b(1)]),
            poly_b(&[b(2)]),
            poly_b(&[b(3)]),
            poly_b(&[b(4), b(5)]),
        ],
    )
    .unwrap();
    let moduli = alloc::vec![poly_b(&[b(1)]), poly_b(&[b(3)])];
    let basis = input.congruence_basis(&moduli, &[0, 0]).unwrap();
    let identity = PolynomialMatrix::<Gf8B>::identity(2).unwrap();
    assert_eq!(basis, identity);
}

#[test]
fn zero_modulus_is_rejected_and_counts_checked() {
    let input = PolynomialMatrix::from_entries(1, 2, alloc::vec![poly_b(&[b(1)]), poly_b(&[b(2)])])
        .unwrap();
    let bad = alloc::vec![Polynomial::zero(), poly_b(&[b(1), b(1)])];
    assert!(input.congruence_basis(&bad, &[0]).is_err());
    assert!(input.congruence_basis(&[poly_b(&[b(1)])], &[0]).is_err());
    assert!(
        input
            .congruence_basis(&[poly_b(&[b(1)]), poly_b(&[b(1)])], &[0, 0, 0])
            .is_err()
    );
}

#[test]
fn approximant_column_matches_shifted_order() {
    // F = [[1], [x]] with M = [x^2]: solutions p = [p0, p1] with
    // p0 + p1 x = 0 mod x^2, i.e. p0(0) = 0 and p0'(0) + p1(0) = 0.
    // Basis [[x, 0],[1, x]]: rows [x,0] and [1,x] both satisfy, independent.
    let input =
        PolynomialMatrix::from_entries(2, 1, alloc::vec![poly_b(&[b(1)]), poly_b(&[b(0), b(1)])])
            .unwrap();
    let moduli = alloc::vec![poly_b(&[b(0), b(0), b(1)])];
    let basis = input.congruence_basis(&moduli, &[0, 0]).unwrap();
    assert_eq!(basis.shape(), (2, 2));
    assert!(satisfies(&basis, &input, &moduli));
    assert!(basis.is_popov(&[0, 0]));
    assert_eq!(basis.rank_weak_popov(&[0, 0]).unwrap(), 2);
}

#[test]
fn irreducible_and_shared_moduli_agree() {
    // Irreducible modulus x^2+x+1 over GF(8)[x] (no binary root).
    let input = PolynomialMatrix::from_entries(1, 1, alloc::vec![poly_b(&[b(1)])]).unwrap();
    let moduli = alloc::vec![poly_b(&[b(1), b(1), b(1)])];
    let basis = input.congruence_basis(&moduli, &[0]).unwrap();
    assert_eq!(basis.shape(), (1, 1));
    assert!(satisfies(&basis, &input, &moduli));
    // Shared non-coprime moduli: F = I_2, M = [x^2, x^2] needs p0, p1 in (x^2).
    let identity = PolynomialMatrix::from_entries(
        2,
        2,
        alloc::vec![
            poly_b(&[b(1)]),
            Polynomial::zero(),
            Polynomial::zero(),
            poly_b(&[b(1)]),
        ],
    )
    .unwrap();
    let shared = alloc::vec![poly_b(&[b(0), b(0), b(1)]), poly_b(&[b(0), b(0), b(1)])];
    let basis = identity.congruence_basis(&shared, &[0, 0]).unwrap();
    assert!(satisfies(&basis, &identity, &shared));
    assert_eq!(basis.rank_weak_popov(&[0, 0]).unwrap(), 2);
}

#[test]
fn zero_input_returns_diagonal_moduli() {
    // F = 0 imposes no relation between solution coordinates: each
    // coordinate ranges over the multiples of its own modulus.
    let input = PolynomialMatrix::zeros(2, 2).unwrap();
    let moduli = alloc::vec![poly_b(&[b(0), b(0), b(1)]), poly_b(&[b(1), b(1)])];
    let basis = input.congruence_basis(&moduli, &[0, 0]).unwrap();
    assert!(satisfies(&basis, &input, &moduli));
    assert_eq!(basis.rank_weak_popov(&[0, 0]).unwrap(), 2);
}

#[test]
fn odd_characteristic_congruence_holds() {
    let input =
        PolynomialMatrix::from_entries(2, 1, alloc::vec![poly_p(&[p(1)]), poly_p(&[p(0), p(1)])])
            .unwrap();
    let moduli = alloc::vec![poly_p(&[p(0), p(0), p(1)])];
    let basis = input.congruence_basis(&moduli, &[0, 0]).unwrap();
    assert!(satisfies(&basis, &input, &moduli));
    assert!(basis.is_popov(&[0, 0]));
}

#[test]
fn empty_row_module_returns_empty_basis() {
    // 0x2 input: the congruence module over zero rows is empty.
    let input = PolynomialMatrix::<Gf8B>::zeros(0, 2).unwrap();
    let moduli = alloc::vec![poly_b(&[b(1), b(1)]), poly_b(&[b(1)])];
    let basis = input.congruence_basis(&moduli, &[]).unwrap();
    assert_eq!(basis.shape(), (0, 0));
}

#[test]
fn redundant_congruences_keep_full_rank() {
    // Duplicate rows of F impose the same constraint twice.
    let input = PolynomialMatrix::from_entries(2, 1, alloc::vec![poly_b(&[b(1)]), poly_b(&[b(1)])])
        .unwrap();
    let moduli = alloc::vec![poly_b(&[b(0), b(1)])];
    let basis = input.congruence_basis(&moduli, &[0, 0]).unwrap();
    assert!(satisfies(&basis, &input, &moduli));
    assert_eq!(basis.rank_weak_popov(&[0, 0]).unwrap(), 2);
}

/// Exhaustive bounded oracle: every coefficient vector below `bound`
/// satisfying the congruences, over GF(8) with bound 2 (coefficients 0..2).
fn bounded_solutions(
    input: &PolynomialMatrix<Gf8B>,
    moduli: &[Polynomial<Gf8B>],
    bound: usize,
) -> Vec<Vec<Polynomial<Gf8B>>> {
    let (mrows, ncols) = input.shape();
    // Enumerate all m-tuples of polynomials with coefficients in 0..bound
    // and degree below bound.
    let scalars: Vec<B> = (0..bound as u8).map(b).collect();
    let mut polys = alloc::vec![Polynomial::zero()];
    for degree in 0..bound {
        let mut extensions = alloc::vec![];
        for base in &polys {
            for &scalar in &scalars[1..] {
                let mut next_coeffs: Vec<B> = (0..=degree).map(|d| base.coefficient(d)).collect();
                next_coeffs[degree] = scalar;
                extensions.push(Polynomial::from_coefficients(&next_coeffs).unwrap());
            }
        }
        polys.extend(extensions);
    }
    let mut solutions = alloc::vec![];
    let mut tuple = alloc::vec![Polynomial::zero(); mrows];
    enumerate_tuples(&polys, mrows, &mut tuple, 0, &mut |candidate| {
        let mut ok = true;
        for (j, modulus) in moduli.iter().enumerate().take(ncols) {
            let mut inner = Polynomial::zero();
            for (i, factor) in candidate.iter().enumerate().take(mrows) {
                let term = factor.multiply(input.entry(i, j).unwrap()).unwrap();
                inner.add_assign(&term).unwrap();
            }
            let (_, remainder) = inner.div_rem(modulus).unwrap();
            if !remainder.is_zero() {
                ok = false;
                break;
            }
        }
        if ok {
            solutions.push(candidate.to_vec());
        }
    });
    solutions
}

fn enumerate_tuples(
    polys: &[Polynomial<Gf8B>],
    width: usize,
    current: &mut [Polynomial<Gf8B>],
    depth: usize,
    visit: &mut impl FnMut(&[Polynomial<Gf8B>]),
) {
    if depth == width {
        visit(current);
        return;
    }
    for poly in polys {
        current[depth] = poly.clone();
        enumerate_tuples(polys, width, current, depth + 1, visit);
    }
}

/// Every bounded solution is generated by the basis: membership accepts
/// each oracle row with a witness that rebuilds it.
fn assert_generates(
    basis: &PolynomialMatrix<Gf8B>,
    input: &PolynomialMatrix<Gf8B>,
    moduli: &[Polynomial<Gf8B>],
    shifts: &[usize],
    bound: usize,
) {
    let mut reduced = basis.clone();
    reduced.reduce_weak_popov(shifts).unwrap();
    for solution in bounded_solutions(input, moduli, bound) {
        let witness = reduced.membership_weak_popov(&solution, shifts).unwrap();
        let mut rebuilt = alloc::vec![Polynomial::zero(); solution.len()];
        for (row, quotient) in witness.quotients.iter().enumerate() {
            for (column, slot) in rebuilt.iter_mut().enumerate() {
                let term = reduced
                    .entry(row, column)
                    .unwrap()
                    .multiply(quotient)
                    .unwrap();
                slot.add_assign(&term).unwrap();
            }
        }
        assert_eq!(rebuilt, solution);
    }
}

#[test]
fn bounded_oracle_confirms_generation() {
    let input =
        PolynomialMatrix::from_entries(2, 1, alloc::vec![poly_b(&[b(1)]), poly_b(&[b(0), b(1)])])
            .unwrap();
    let moduli = alloc::vec![poly_b(&[b(0), b(0), b(1)])];
    let basis = input.congruence_basis(&moduli, &[0, 0]).unwrap();
    assert_generates(&basis, &input, &moduli, &[0, 0], 2);
    let identity = PolynomialMatrix::from_entries(
        2,
        2,
        alloc::vec![
            poly_b(&[b(1)]),
            Polynomial::zero(),
            Polynomial::zero(),
            poly_b(&[b(1)]),
        ],
    )
    .unwrap();
    let shared = alloc::vec![poly_b(&[b(0), b(0), b(1)]), poly_b(&[b(0), b(0), b(1)])];
    let basis = identity.congruence_basis(&shared, &[0, 0]).unwrap();
    assert_generates(&basis, &identity, &shared, &[0, 0], 2);
}

#[test]
fn certificate_witness_annihilates_block() {
    let input =
        PolynomialMatrix::from_entries(2, 1, alloc::vec![poly_b(&[b(1)]), poly_b(&[b(0), b(1)])])
            .unwrap();
    let moduli = alloc::vec![poly_b(&[b(0), b(0), b(1)])];
    let certificate = input.congruence_basis_certified(&moduli, &[0, 0]).unwrap();
    let (krows, _) = certificate.block_kernel.shape();
    let (_, bcols) = certificate.block.shape();
    let mut product = PolynomialMatrix::zeros(krows, bcols).unwrap();
    certificate
        .block_kernel
        .mul_into(&certificate.block, &mut product)
        .unwrap();
    for row in 0..krows {
        for column in 0..bcols {
            assert!(product.entry(row, column).unwrap().is_zero());
        }
    }
    // Kernel row count plus block column rank equals the block row count.
    let (brows, _) = certificate.block.shape();
    assert_eq!(krows + bcols, brows);
    // Mixed unit/non-unit moduli take the general path (no identity).
    let mixed_input =
        PolynomialMatrix::from_entries(1, 2, alloc::vec![poly_b(&[b(1)]), poly_b(&[b(0), b(1)])])
            .unwrap();
    let mixed = alloc::vec![poly_b(&[b(5)]), poly_b(&[b(0), b(0), b(1)])];
    let mixed_basis = mixed_input.congruence_basis(&mixed, &[0]).unwrap();
    assert_eq!(mixed_basis.entry(0, 0).unwrap().as_packed(), [0, 1]);
    assert!(satisfies(&mixed_basis, &mixed_input, &mixed));
}

#[test]
fn tiny_approximant_matches_exact_canonical_basis() {
    // F = [1; x], M = [x^2] under zero shifts: the canonical basis is
    // [[x^2, 0], [x, 1]] in row-Popov column-reduced order... verified
    // against the bounded oracle rather than a hardcoded matrix: every
    // bounded solution is generated and the basis is canonical.
    let input =
        PolynomialMatrix::from_entries(2, 1, alloc::vec![poly_b(&[b(1)]), poly_b(&[b(0), b(1)])])
            .unwrap();
    let moduli = alloc::vec![poly_b(&[b(0), b(0), b(1)])];
    let basis = input.congruence_basis(&moduli, &[0, 0]).unwrap();
    assert!(basis.is_popov(&[0, 0]));
    // Exact canonical entries: rows sorted by leading column with monic
    // pivots; the [1+x, 1]-style fixture below pins the byte layout.
    assert_eq!(basis.entry(0, 0).unwrap().as_packed(), [0, 1]);
    assert_eq!(basis.entry(0, 1).unwrap().as_packed(), [1]);
    assert!(basis.entry(1, 0).unwrap().is_zero());
    assert_eq!(basis.entry(1, 1).unwrap().as_packed(), [0, 1]);
    assert_generates(&basis, &input, &moduli, &[0, 0], 3);
}

extern crate alloc;
