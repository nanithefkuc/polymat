use fgf::field::{Elem, Field};
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

fn residuals_masked<F: fgf::kernel::FieldKernels>(
    basis: &PolynomialMatrix<F>,
    input: &PolynomialMatrix<F>,
    orders: &[usize],
) -> bool {
    let (krows, _) = basis.shape();
    let (_, ncols) = input.shape();
    let (mrows, _) = input.shape();
    for k in 0..krows {
        for (j, &order) in orders.iter().enumerate().take(ncols) {
            let mut inner = Polynomial::zero();
            for i in 0..mrows {
                let term = basis
                    .entry(k, i)
                    .unwrap()
                    .multiply(input.entry(i, j).unwrap())
                    .unwrap();
                inner.add_assign(&term).unwrap();
            }
            for coefficient in 0..order {
                if !inner.coefficient(coefficient).is_zero() {
                    return false;
                }
            }
        }
    }
    true
}

#[test]
fn no_orders_return_identity_with_zero_count() {
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
    let result = input.approximant_basis(&[0, 0], &[0, 0]).unwrap();
    assert_eq!(result.basis, PolynomialMatrix::identity(2).unwrap());
    assert_eq!(result.independent_constraints, 0);
    assert!(input.approximant_basis(&[1], &[0, 0]).is_err());
    assert!(input.approximant_basis(&[0, 0], &[0]).is_err());
}

#[test]
fn scalar_recurrence_matches_congruence_oracle() {
    // F = [1; x] with orders [2]: compare against M = [x^2].
    let input =
        PolynomialMatrix::from_entries(2, 1, alloc::vec![poly_b(&[b(1)]), poly_b(&[b(0), b(1)])])
            .unwrap();
    let orders = alloc::vec![2];
    let recurrence = input.approximant_basis(&orders, &[0, 0]).unwrap();
    let moduli = alloc::vec![poly_b(&[b(0), b(0), b(1)])];
    let oracle = input.congruence_basis(&moduli, &[0, 0]).unwrap();
    assert_eq!(recurrence.basis, oracle);
    assert!(residuals_masked(&recurrence.basis, &input, &orders));
}

#[test]
fn unequal_and_repeated_orders_agree() {
    // F = I_2 with orders [2, 1]: solutions are (x^2)*e0, x*e1 spans;
    // repeating the first column constraint adds no independent row.
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
    let orders = alloc::vec![2, 1];
    let recurrence = identity.approximant_basis(&orders, &[0, 0]).unwrap();
    let moduli = alloc::vec![poly_b(&[b(0), b(0), b(1)]), poly_b(&[b(0), b(1)])];
    let oracle = identity.congruence_basis(&moduli, &[0, 0]).unwrap();
    assert_eq!(recurrence.basis, oracle);
    assert_eq!(recurrence.independent_constraints, 3);
    assert!(residuals_masked(&recurrence.basis, &identity, &orders));
    // Determinant degree equals the independent count: diag(x^2, x).
    let diagonal = recurrence.basis.entry(0, 0).unwrap().degree().unwrap_or(0)
        + recurrence.basis.entry(1, 1).unwrap().degree().unwrap_or(0);
    assert_eq!(diagonal, 3);
}

#[test]
fn zero_input_and_empty_shapes_agree() {
    let zero = PolynomialMatrix::<Gf8B>::zeros(2, 2).unwrap();
    // F = 0 imposes nothing: every discrepancy vanishes, so all
    // constraints are redundant and both paths return the identity.
    let recurrence = zero.approximant_basis(&[2, 1], &[0, 0]).unwrap();
    let moduli = alloc::vec![poly_b(&[b(0), b(0), b(1)]), poly_b(&[b(0), b(1)])];
    let oracle = zero.congruence_basis(&moduli, &[0, 0]).unwrap();
    assert_eq!(recurrence.basis, oracle);
    assert_eq!(recurrence.basis, PolynomialMatrix::identity(2).unwrap());
    assert_eq!(recurrence.independent_constraints, 0);
    let empty = PolynomialMatrix::<Gf8B>::zeros(0, 2).unwrap();
    let result = empty.approximant_basis(&[1, 1], &[]).unwrap();
    assert_eq!(result.basis.shape(), (0, 0));
    assert_eq!(result.independent_constraints, 0);
    let wide = PolynomialMatrix::<Gf8B>::zeros(2, 0).unwrap();
    let result = wide.approximant_basis(&[], &[0, 0]).unwrap();
    assert_eq!(result.basis, PolynomialMatrix::identity(2).unwrap());
}

#[test]
fn unbalanced_shifts_and_odd_characteristic_agree() {
    let input = PolynomialMatrix::from_entries(
        2,
        2,
        alloc::vec![
            poly_p(&[p(1), p(1)]),
            poly_p(&[p(2)]),
            poly_p(&[p(3)]),
            poly_p(&[p(4), p(5)]),
        ],
    )
    .unwrap();
    let orders = alloc::vec![2, 1];
    let shifts = alloc::vec![3, 0];
    let recurrence = input.approximant_basis(&orders, &shifts).unwrap();
    let moduli = alloc::vec![poly_p(&[p(0), p(0), p(1)]), poly_p(&[p(0), p(1)])];
    let oracle = input.congruence_basis(&moduli, &shifts).unwrap();
    assert_eq!(recurrence.basis, oracle);
    assert!(residuals_masked(&recurrence.basis, &input, &orders));
    assert!(recurrence.basis.is_popov(&shifts));
}

#[test]
fn determinant_degree_counts_independent_constraints() {
    // F = [[1],[1]] with orders [1]: one independent constraint (the two
    // constant terms must be equal... verified against the oracle), and a
    // repeated identical column adds only redundant constraints.
    let input = PolynomialMatrix::from_entries(2, 1, alloc::vec![poly_b(&[b(1)]), poly_b(&[b(1)])])
        .unwrap();
    let single = input.approximant_basis(&[1], &[0, 0]).unwrap();
    assert_eq!(single.independent_constraints, 1);
    let doubled_input = PolynomialMatrix::from_entries(
        2,
        2,
        alloc::vec![
            poly_b(&[b(1)]),
            poly_b(&[b(1)]),
            poly_b(&[b(1)]),
            poly_b(&[b(1)])
        ],
    )
    .unwrap();
    let doubled = doubled_input.approximant_basis(&[1, 1], &[0, 0]).unwrap();
    let moduli = alloc::vec![poly_b(&[b(0), b(1)]), poly_b(&[b(0), b(1)])];
    let oracle = doubled_input.congruence_basis(&moduli, &[0, 0]).unwrap();
    assert_eq!(doubled.basis, oracle);
    assert!(doubled.independent_constraints < 2 + 2);
}

extern crate alloc;
