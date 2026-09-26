use fgf::field::Field;
use fgf::{Gf8B, Mersenne31, gf8b, mersenne31};
use poly_ring::Polynomial;
use polymat::PolynomialMatrix;
use polymat::internals::leibniz_determinant;

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

#[test]
fn determinant_matches_leibniz_with_signs() {
    // Swap matrix [[0,1],[1,0]] has determinant -1 = 1 in char 2, -1 odd.
    let swap8 = PolynomialMatrix::from_entries(
        2,
        2,
        alloc::vec![
            Polynomial::zero(),
            poly_b(&[b(1)]),
            poly_b(&[b(1)]),
            Polynomial::zero(),
        ],
    )
    .unwrap();
    assert_eq!(
        swap8.determinant().unwrap(),
        leibniz_determinant(&swap8).unwrap()
    );
    let swap31 = PolynomialMatrix::from_entries(
        2,
        2,
        alloc::vec![
            Polynomial::zero(),
            poly_p(&[p(1)]),
            poly_p(&[p(1)]),
            Polynomial::zero(),
        ],
    )
    .unwrap();
    assert_eq!(
        swap31.determinant().unwrap(),
        leibniz_determinant(&swap31).unwrap()
    );
    let mut neg_one = poly_p(&[p(1)]);
    neg_one.scale_assign(p(1).neg());
    assert_eq!(swap31.determinant().unwrap(), neg_one);
    // Singular input gives zero; 0-by-0 gives one.
    let singular = PolynomialMatrix::from_entries(
        2,
        2,
        alloc::vec![
            poly_b(&[b(1), b(1)]),
            poly_b(&[b(1)]),
            poly_b(&[b(1), b(1)]),
            poly_b(&[b(1)]),
        ],
    )
    .unwrap();
    assert!(singular.determinant().unwrap().is_zero());
    let empty = PolynomialMatrix::<Gf8B>::zeros(0, 0).unwrap();
    assert!(empty.determinant().unwrap().is_one());
    // 3x3 agreement with Leibniz, both fields.
    let three = PolynomialMatrix::from_entries(
        3,
        3,
        alloc::vec![
            poly_b(&[b(1), b(1)]),
            poly_b(&[b(2)]),
            poly_b(&[b(0), b(1)]),
            poly_b(&[b(1)]),
            poly_b(&[b(1), b(0), b(1)]),
            poly_b(&[b(3)]),
            poly_b(&[b(0), b(1)]),
            poly_b(&[b(4)]),
            poly_b(&[b(1), b(1), b(1)]),
        ],
    )
    .unwrap();
    assert_eq!(
        three.determinant().unwrap(),
        leibniz_determinant(&three).unwrap()
    );
    assert!(leibniz_determinant(&three).is_ok());
    assert!(
        PolynomialMatrix::<Gf8B>::zeros(2, 3)
            .unwrap()
            .determinant()
            .is_err()
    );
    assert!(
        leibniz_determinant(&PolynomialMatrix::<Gf8B>::zeros(0, 0).unwrap())
            .unwrap()
            .is_one()
    );
}

#[test]
fn leibniz_oracle_rejects_oversize_input() {
    assert!(leibniz_determinant(&PolynomialMatrix::<Gf8B>::zeros(7, 7).unwrap()).is_err());
}

extern crate alloc;
