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

fn mul<F: fgf::kernel::FieldKernels>(
    left: &PolynomialMatrix<F>,
    right: &PolynomialMatrix<F>,
) -> PolynomialMatrix<F> {
    let (rows, _) = left.shape();
    let (_, columns) = right.shape();
    let mut out = PolynomialMatrix::zeros(rows, columns).unwrap();
    left.mul_into(right, &mut out).unwrap();
    out
}

#[test]
fn hermite_form_holds_witness_and_shape() {
    let input = PolynomialMatrix::from_entries(
        2,
        3,
        alloc::vec![
            poly_b(&[b(1), b(1)]),
            poly_b(&[b(1)]),
            poly_b(&[b(0), b(1)]),
            poly_b(&[b(1), b(0), b(1)]),
            poly_b(&[b(2)]),
            poly_b(&[b(1), b(1)]),
        ],
    )
    .unwrap();
    let hermite = input.hermite_form().unwrap();
    let product = mul(&hermite.witness, &input);
    assert_eq!(product, hermite.form);
    // Leftmost pivots increase; entries above pivots are degree-reduced.
    let (rows, columns) = hermite.form.shape();
    assert_eq!((rows, columns), (2, 3));
    let mut previous: Option<usize> = None;
    for row in 0..rows {
        let pivot =
            (0..columns).find(|&column| !hermite.form.entry(row, column).unwrap().is_zero());
        if let Some(pivot) = pivot {
            assert!(
                hermite
                    .form
                    .entry(row, pivot)
                    .unwrap()
                    .leading_coefficient()
                    == Some(b(1))
            );
            if let Some(last) = previous {
                assert!(pivot > last);
            }
            previous = Some(pivot);
            for upper in 0..row {
                let entry = hermite.form.entry(upper, pivot).unwrap();
                let pivot_entry = hermite.form.entry(row, pivot).unwrap();
                let below = entry.degree().unwrap_or(0) < pivot_entry.degree().unwrap_or(0)
                    || entry.is_zero();
                assert!(below);
            }
        }
    }
    // Rank-deficient input pads zero rows last.
    let dependent = PolynomialMatrix::from_entries(
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
    let hermite = dependent.hermite_form().unwrap();
    assert_eq!(mul(&hermite.witness, &dependent), hermite.form);
    assert!(
        hermite.form.entry(1, 0).unwrap().is_zero() && hermite.form.entry(1, 1).unwrap().is_zero()
    );
}

#[test]
fn smith_form_holds_divisibility_and_witnesses() {
    let input = PolynomialMatrix::from_entries(
        2,
        2,
        alloc::vec![
            poly_b(&[b(1), b(1)]),
            poly_b(&[b(0), b(1)]),
            poly_b(&[b(1)]),
            poly_b(&[b(1), b(0), b(1)]),
        ],
    )
    .unwrap();
    let smith = input.smith_form().unwrap();
    let product = mul(&mul(&smith.left, &input), &smith.right);
    assert_eq!(product, smith.form);
    // Diagonal, monic, divisibility d0 | d1.
    let off =
        smith.form.entry(0, 1).unwrap().is_zero() && smith.form.entry(1, 0).unwrap().is_zero();
    assert!(off);
    let first = smith.form.entry(0, 0).unwrap().clone();
    let second = smith.form.entry(1, 1).unwrap().clone();
    assert_eq!(first.leading_coefficient(), Some(b(1)));
    if !second.is_zero() {
        let (_, remainder) = second.div_rem(&first).unwrap();
        assert!(remainder.is_zero());
    }
    // Rectangular input keeps its shape.
    let wide = PolynomialMatrix::from_entries(
        2,
        3,
        alloc::vec![
            poly_b(&[b(1), b(1)]),
            poly_b(&[b(1)]),
            poly_b(&[b(0), b(1)]),
            poly_b(&[b(1)]),
            poly_b(&[b(2)]),
            poly_b(&[b(3), b(1)]),
        ],
    )
    .unwrap();
    let smith = wide.smith_form().unwrap();
    assert_eq!(smith.form.shape(), (2, 3));
    assert_eq!(mul(&mul(&smith.left, &wide), &smith.right), smith.form);
}

#[test]
fn smith_minors_match_determinantal_divisors() {
    // Diagonal diag(x, x(1+x)): invariant factors x | x(1+x); 1-minors gcd
    // is x, 2-minors gcd is x^2(1+x).
    let input = PolynomialMatrix::from_entries(
        2,
        2,
        alloc::vec![
            poly_b(&[b(0), b(1)]),
            Polynomial::zero(),
            Polynomial::zero(),
            poly_b(&[b(0), b(1), b(1)]),
        ],
    )
    .unwrap();
    let smith = input.smith_form().unwrap();
    let first = smith.form.entry(0, 0).unwrap().clone();
    let second = smith.form.entry(1, 1).unwrap().clone();
    assert_eq!(input.minor_gcd(1).unwrap(), first.monic());
    let product = first.multiply(&second).unwrap().monic();
    assert_eq!(input.minor_gcd(2).unwrap(), product);
}

#[test]
fn hermite_euclidean_remainder_path_holds() {
    // Column entries sharing a non-trivial gcd force the Euclidean
    // remainder loop (not just exact elimination): gcd(x^2, 1+x) = 1.
    let input = PolynomialMatrix::from_entries(
        2,
        2,
        alloc::vec![
            poly_b(&[b(0), b(0), b(1)]),
            poly_b(&[b(1)]),
            poly_b(&[b(1), b(1)]),
            poly_b(&[b(0), b(1)]),
        ],
    )
    .unwrap();
    let hermite = input.hermite_form().unwrap();
    assert_eq!(mul(&hermite.witness, &input), hermite.form);
    assert!(hermite.form.entry(1, 0).unwrap().is_zero());
    // Odd-characteristic Hermite with a remainder step.
    let odd = PolynomialMatrix::from_entries(
        2,
        2,
        alloc::vec![
            poly_p(&[p(0), p(0), p(1)]),
            poly_p(&[p(1)]),
            poly_p(&[p(1), p(1)]),
            poly_p(&[p(0), p(1)]),
        ],
    )
    .unwrap();
    let hermite = odd.hermite_form().unwrap();
    assert_eq!(mul(&hermite.witness, &odd), hermite.form);
    // Empty-column selection skips fully-zero columns.
    let sparse = PolynomialMatrix::from_entries(
        2,
        3,
        alloc::vec![
            Polynomial::zero(),
            poly_b(&[b(1), b(1)]),
            poly_b(&[b(1)]),
            Polynomial::zero(),
            poly_b(&[b(1)]),
            poly_b(&[b(0), b(1)]),
        ],
    )
    .unwrap();
    let hermite = sparse.hermite_form().unwrap();
    assert_eq!(mul(&hermite.witness, &sparse), hermite.form);
}

#[test]
fn error_arms_reject_bad_geometry() {
    // Non-square and out-of-range queries report geometry errors.
    assert!(
        PolynomialMatrix::<Gf8B>::zeros(2, 3)
            .unwrap()
            .determinant()
            .is_err()
    );
    assert!(
        PolynomialMatrix::<Gf8B>::zeros(2, 3)
            .unwrap()
            .polynomial_inverse()
            .is_err()
    );
    let tiny = PolynomialMatrix::from_entries(
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
    assert!(tiny.minor_gcd(0).is_err());
    assert!(tiny.minor_gcd(3).is_err());
    // Smith on a zero matrix keeps zero witnesses consistent.
    let zero = PolynomialMatrix::<Gf8B>::zeros(2, 2).unwrap();
    let smith = zero.smith_form().unwrap();
    assert_eq!(smith.form, zero);
    // Smith repair path: pivot needs submatrix divisibility work.
    let repair = PolynomialMatrix::from_entries(
        2,
        2,
        alloc::vec![
            poly_b(&[b(0), b(1)]),
            poly_b(&[b(1)]),
            poly_b(&[b(1)]),
            poly_b(&[b(0), b(1)]),
        ],
    )
    .unwrap();
    let smith = repair.smith_form().unwrap();
    let product = mul(&mul(&smith.left, &repair), &smith.right);
    assert_eq!(product, smith.form);
    let first = smith.form.entry(0, 0).unwrap().clone();
    let second = smith.form.entry(1, 1).unwrap().clone();
    if !first.is_zero() && !second.is_zero() {
        let (_, remainder) = second.div_rem(&first).unwrap();
        assert!(remainder.is_zero());
    }
}

#[test]
fn smith_bezout_paths_reduce_pivot_degree() {
    // Pivot-row remainder: (0,1) = x+1 leaves remainder 1 under x^2,
    // forcing a column Bezout step that drops the pivot to degree 0.
    let row_case = PolynomialMatrix::from_entries(
        2,
        2,
        alloc::vec![
            poly_b(&[b(0), b(0), b(1)]),
            poly_b(&[b(1), b(1)]),
            poly_b(&[b(0), b(1)]),
            Polynomial::zero(),
        ],
    )
    .unwrap();
    let smith = row_case.smith_form().unwrap();
    assert_eq!(mul(&mul(&smith.left, &row_case), &smith.right), smith.form);
    // Pivot-column remainder below: (1,0) = x^2+1 leaves remainder 1
    // under x, forcing a row Bezout step.
    let column_case = PolynomialMatrix::from_entries(
        2,
        2,
        alloc::vec![
            poly_b(&[b(0), b(1)]),
            Polynomial::zero(),
            poly_b(&[b(1), b(0), b(1)]),
            poly_b(&[b(0), b(1)]),
        ],
    )
    .unwrap();
    let smith = column_case.smith_form().unwrap();
    assert_eq!(
        mul(&mul(&smith.left, &column_case), &smith.right),
        smith.form
    );
    // Submatrix witness: pivot row and column clear, but (1,2) is not
    // divisible by the pivot, forcing the fold-in repair path.
    let witness_case = PolynomialMatrix::from_entries(
        3,
        3,
        alloc::vec![
            poly_b(&[b(0), b(1)]),
            Polynomial::zero(),
            Polynomial::zero(),
            Polynomial::zero(),
            poly_b(&[b(0), b(0), b(1)]),
            poly_b(&[b(1), b(0), b(1)]),
            Polynomial::zero(),
            Polynomial::zero(),
            poly_b(&[b(0), b(1)]),
        ],
    )
    .unwrap();
    let smith = witness_case.smith_form().unwrap();
    assert_eq!(
        mul(&mul(&smith.left, &witness_case), &smith.right),
        smith.form
    );
    let first = smith.form.entry(0, 0).unwrap().clone();
    let second = smith.form.entry(1, 1).unwrap().clone();
    if !first.is_zero() && !second.is_zero() {
        let (_, remainder) = second.div_rem(&first).unwrap();
        assert!(remainder.is_zero());
    }
}

#[test]
fn smith_bezout_witnesses_stay_unimodular() {
    // Regression: the Bezout second row once used -(p/g) and (e/g),
    // giving determinant (s*e + t*p)/g instead of one. On [[x, x+1],
    // [0, x]] the transformation changed the determinant; the diagonal
    // below is the true Smith form with unit witnesses.
    let input = PolynomialMatrix::from_entries(
        2,
        2,
        alloc::vec![
            poly_b(&[b(0), b(1)]),
            poly_b(&[b(1), b(1)]),
            Polynomial::zero(),
            poly_b(&[b(0), b(1)]),
        ],
    )
    .unwrap();
    let smith = input.smith_form().unwrap();
    assert_eq!(mul(&mul(&smith.left, &input), &smith.right), smith.form);
    let left_det = smith.left.determinant().unwrap();
    assert!(left_det.degree().is_none_or(|degree| degree == 0) && !left_det.is_zero());
    let right_det = smith.right.determinant().unwrap();
    assert!(right_det.degree().is_none_or(|degree| degree == 0) && !right_det.is_zero());
    // Determinant preserved up to units: det(input) = det(form)/det(U V).
    let input_det = input.determinant().unwrap();
    let form_det = smith.form.determinant().unwrap();
    let witness_det = left_det.multiply(&right_det).unwrap();
    let scaled = form_det.div_rem(&witness_det).unwrap();
    assert!(scaled.1.is_zero());
    assert_eq!(scaled.0.monic(), input_det.monic());
}

#[test]
fn polynomial_inverse_holds_both_orders() {
    // Unimodular [[1+x, x],[x, 1+x+x^2]]... use elementary [[1, x],[0, 1]] with inverse [[1, x],[0, 1]] in char 2.
    let unitriangular = PolynomialMatrix::from_entries(
        2,
        2,
        alloc::vec![
            poly_b(&[b(1)]),
            poly_b(&[b(0), b(1)]),
            Polynomial::zero(),
            poly_b(&[b(1)]),
        ],
    )
    .unwrap();
    let inverse = unitriangular
        .polynomial_inverse()
        .unwrap()
        .expect("unimodular");
    assert_eq!(
        mul(&unitriangular, &inverse),
        PolynomialMatrix::identity(2).unwrap()
    );
    assert_eq!(
        mul(&inverse, &unitriangular),
        PolynomialMatrix::identity(2).unwrap()
    );
    // Nonconstant determinant is not invertible.
    let nonunit = PolynomialMatrix::from_entries(
        2,
        2,
        alloc::vec![
            poly_b(&[b(0), b(1)]),
            Polynomial::zero(),
            Polynomial::zero(),
            poly_b(&[b(1)]),
        ],
    )
    .unwrap();
    assert_eq!(nonunit.polynomial_inverse().unwrap(), None);
    // Non-square rejected.
    assert!(
        PolynomialMatrix::<Gf8B>::zeros(2, 3)
            .unwrap()
            .polynomial_inverse()
            .is_err()
    );
}

extern crate alloc;
