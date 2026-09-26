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
    // F = 0 with M = [x^2, x+1]: solutions are (x^2)*e0, (x+1)*e1 spans.
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

extern crate alloc;
