use fgf::field::Field;
use fgf::{Gf8B, Mersenne31, gf8b, mersenne31};
use poly_ring::Polynomial;
use polymat::{MatrixError, PolynomialMatrix, ShiftPreparation};

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

fn matrix<F: fgf::kernel::FieldKernels>(
    rows: usize,
    columns: usize,
    flat: Vec<Polynomial<F>>,
) -> PolynomialMatrix<F> {
    PolynomialMatrix::from_entries(rows, columns, flat).unwrap()
}

#[test]
fn construction_preserves_empty_shapes() {
    let empty_rows = PolynomialMatrix::<Gf8B>::zeros(0, 3).unwrap();
    assert_eq!(empty_rows.shape(), (0, 3));
    let empty_columns = PolynomialMatrix::<Gf8B>::zeros(2, 0).unwrap();
    assert_eq!(empty_columns.shape(), (2, 0));
    let mut reduced = empty_rows;
    reduced.reduce_weak_popov(&[0, 0, 0]).unwrap();
    assert_eq!(reduced.shape(), (0, 3));
}

#[test]
fn construction_rejects_entry_count_mismatch() {
    let entries = alloc::vec![Polynomial::<Gf8B>::zero()];
    assert!(PolynomialMatrix::from_entries(2, 2, entries).is_err());
    assert!(PolynomialMatrix::<Gf8B>::from_entries(usize::MAX, 2, alloc::vec![]).is_err());
}

#[test]
fn transpose_exchanges_rows_and_columns() {
    let left = matrix(
        2,
        3,
        alloc::vec![
            poly_b(&[b(1)]),
            poly_b(&[b(2), b(3)]),
            poly_b(&[b(4)]),
            poly_b(&[b(5), b(0), b(6)]),
            Polynomial::zero(),
            poly_b(&[b(7)]),
        ],
    );
    let transposed = left.transposed().unwrap();
    assert_eq!(transposed.shape(), (3, 2));
    assert_eq!(transposed.entry(0, 0).unwrap(), left.entry(0, 0).unwrap());
    assert_eq!(transposed.entry(2, 0).unwrap(), left.entry(0, 2).unwrap());
    assert_eq!(transposed.entry(0, 1).unwrap(), left.entry(1, 0).unwrap());
    assert_eq!(transposed.entry(2, 1).unwrap(), left.entry(1, 2).unwrap());
}

#[test]
fn add_and_subtract_are_inverse() {
    let left = matrix(
        2,
        2,
        alloc::vec![
            poly_b(&[b(1), b(2)]),
            poly_b(&[b(3)]),
            Polynomial::zero(),
            poly_b(&[b(4), b(5), b(6)]),
        ],
    );
    let right = matrix(
        2,
        2,
        alloc::vec![
            poly_b(&[b(7)]),
            poly_b(&[b(1), b(1)]),
            poly_b(&[b(2), b(2)]),
            poly_b(&[b(3)]),
        ],
    );
    let mut sum = PolynomialMatrix::zeros(9, 9).unwrap();
    left.add_into(&right, &mut sum).unwrap();
    assert_eq!(sum.shape(), (2, 2));
    let mut back = PolynomialMatrix::zeros(1, 1).unwrap();
    sum.sub_into(&right, &mut back).unwrap();
    assert_eq!(back, left);
}

#[test]
fn add_rejects_mismatched_shapes_without_mutation() {
    let left = PolynomialMatrix::<Gf8B>::zeros(2, 2).unwrap();
    let right = PolynomialMatrix::<Gf8B>::zeros(2, 3).unwrap();
    let mut out = matrix(1, 1, alloc::vec![poly_b(&[b(9), b(9)])]);
    let before = out.clone();
    assert!(left.add_into(&right, &mut out).is_err());
    assert_eq!(out, before);
    assert!(left.mul_into(&right, &mut out).is_ok());
}

#[test]
fn non_square_product_matches_independent_oracle() {
    // A = [[1+x, x^2],[2, 1]], B = [[1, x, 0],[3x, 1+x^2, 5]] over GF(8)[x].
    let left = matrix(
        2,
        2,
        alloc::vec![
            poly_b(&[b(1), b(1)]),
            poly_b(&[b(0), b(0), b(1)]),
            poly_b(&[b(2)]),
            poly_b(&[b(1)]),
        ],
    );
    let right = matrix(
        2,
        3,
        alloc::vec![
            poly_b(&[b(1)]),
            poly_b(&[b(0), b(1)]),
            Polynomial::zero(),
            poly_b(&[b(0), b(3)]),
            poly_b(&[b(1), b(0), b(1)]),
            poly_b(&[b(5)]),
        ],
    );
    let mut product = PolynomialMatrix::zeros(0, 0).unwrap();
    left.mul_into(&right, &mut product).unwrap();
    assert_eq!(product.shape(), (2, 3));
    // Independent oracle: expand each output as a coefficient sum.
    for row in 0..2 {
        for column in 0..3 {
            let mut expected = Polynomial::<Gf8B>::zero();
            for inner in 0..2 {
                let term = left
                    .entry(row, inner)
                    .unwrap()
                    .multiply(right.entry(inner, column).unwrap())
                    .unwrap();
                expected.add_assign(&term).unwrap();
            }
            assert_eq!(product.entry(row, column).unwrap(), &expected);
        }
    }
}

#[test]
fn truncated_product_and_order_zero_agree() {
    let left = matrix(
        1,
        2,
        alloc::vec![poly_b(&[b(1), b(2), b(3)]), poly_b(&[b(4), b(5)])],
    );
    let right = matrix(2, 1, alloc::vec![poly_b(&[b(6), b(7)]), poly_b(&[b(1)])]);
    let mut full = PolynomialMatrix::zeros(0, 0).unwrap();
    left.mul_truncated_into(&right, usize::MAX, &mut full)
        .unwrap();
    let mut truncated = PolynomialMatrix::zeros(0, 0).unwrap();
    left.mul_truncated_into(&right, 2, &mut truncated).unwrap();
    let mut expected = full.entry(0, 0).unwrap().clone();
    expected.truncate(2);
    assert_eq!(truncated.entry(0, 0).unwrap(), &expected);
    let mut order_zero = PolynomialMatrix::zeros(0, 0).unwrap();
    left.mul_truncated_into(&right, 0, &mut order_zero).unwrap();
    assert!(order_zero.entry(0, 0).unwrap().is_zero());
}

#[test]
fn inner_dimension_zero_yields_zero_product() {
    let left = PolynomialMatrix::<Gf8B>::zeros(2, 0).unwrap();
    let right = PolynomialMatrix::<Gf8B>::zeros(0, 3).unwrap();
    let mut product = PolynomialMatrix::zeros(5, 5).unwrap();
    left.mul_into(&right, &mut product).unwrap();
    assert_eq!(product.shape(), (2, 3));
    for row in 0..2 {
        for column in 0..3 {
            assert!(product.entry(row, column).unwrap().is_zero());
        }
    }
}

#[test]
fn degree_zero_matrices_agree_with_field_arithmetic() {
    // Constants over Mersenne31: [[2,3],[5,7]] * [[11],[13]] = [[2*11+3*13],[5*11+7*13]].
    let left = matrix(
        2,
        2,
        alloc::vec![
            poly_p(&[p(2)]),
            poly_p(&[p(3)]),
            poly_p(&[p(5)]),
            poly_p(&[p(7)])
        ],
    );
    let right = matrix(2, 1, alloc::vec![poly_p(&[p(11)]), poly_p(&[p(13)])]);
    let mut product = PolynomialMatrix::zeros(0, 0).unwrap();
    left.mul_into(&right, &mut product).unwrap();
    let top = p(2).mul(p(11)).add(p(3).mul(p(13)));
    let bottom = p(5).mul(p(11)).add(p(7).mul(p(13)));
    assert_eq!(product.entry(0, 0).unwrap().coefficient(0), top);
    assert_eq!(product.entry(1, 0).unwrap().coefficient(0), bottom);
}

#[test]
fn evaluation_returns_row_major_field_elements() {
    let left = matrix(
        2,
        2,
        alloc::vec![
            poly_p(&[p(1), p(2)]),
            poly_p(&[p(3)]),
            Polynomial::zero(),
            poly_p(&[p(4), p(0), p(5)]),
        ],
    );
    let values = left.evaluate_to_vec(p(6)).unwrap();
    assert_eq!(values.len(), 4);
    assert_eq!(values[0], left.entry(0, 0).unwrap().evaluate(p(6)));
    assert_eq!(values[3], left.entry(1, 1).unwrap().evaluate(p(6)));
}

#[test]
fn owned_reduction_gives_distinct_leading_columns() {
    let basis = matrix(
        2,
        2,
        alloc::vec![
            poly_b(&[b(1), b(1)]),
            poly_b(&[b(1)]),
            poly_b(&[b(1), b(0), b(1)]),
            poly_b(&[b(1), b(1)]),
        ],
    );
    let mut reduced = basis;
    reduced.reduce_weak_popov(&[0, 0]).unwrap();
    let mut columns = [false; 2];
    for row in 0..2 {
        let mut best: Option<(usize, usize)> = None;
        for column in 0..2 {
            if let Some(degree) = reduced.entry(row, column).unwrap().degree()
                && best.is_none_or(|(_, d)| degree > d)
            {
                best = Some((column, degree));
            }
        }
        if let Some((column, _)) = best {
            assert!(!columns[column]);
            columns[column] = true;
        }
    }
}

#[test]
fn signed_shifts_match_equivalent_nonnegative_shifts() {
    let entries = || {
        alloc::vec![
            poly_b(&[b(1), b(1)]),
            poly_b(&[b(1)]),
            poly_b(&[b(1), b(0), b(1)]),
            poly_b(&[b(1), b(1)]),
        ]
    };
    let mut signed = matrix(2, 2, entries());
    let preparation = signed.reduce_weak_popov_signed(&[-1, 1]).unwrap();
    assert_eq!(preparation.shifts, alloc::vec![0, 2]);
    assert_eq!(preparation.offset, -1);
    let mut plain = matrix(2, 2, entries());
    plain.reduce_weak_popov(&[0, 2]).unwrap();
    assert_eq!(signed, plain);
}

#[test]
fn extreme_signed_spans_are_checked() {
    let extreme = ShiftPreparation::from_signed(&[i64::MIN, 0]).unwrap();
    assert_eq!(extreme.offset, i64::MIN);
    assert_eq!(
        extreme.shifts,
        alloc::vec![0, usize::try_from(i64::MAX).unwrap() + 1]
    );
    let full = ShiftPreparation::from_signed(&[i64::MIN, i64::MAX]).unwrap();
    assert_eq!(full.offset, i64::MIN);
    assert_eq!(full.shifts, alloc::vec![0, usize::MAX]);
    let preparation = ShiftPreparation::from_signed(&[3, 5]).unwrap();
    assert_eq!(preparation.shifts, alloc::vec![0, 2]);
    assert_eq!(preparation.offset, 3);
}

#[test]
fn construction_normalizes_denormalized_entries() {
    let mut denormalized = poly_b(&[b(1), b(2)]);
    denormalized.resize_coefficients(4).unwrap();
    assert_eq!(denormalized.degree(), Some(3));
    let owned = matrix(1, 1, alloc::vec![denormalized]);
    assert_eq!(owned.entry(0, 0).unwrap().degree(), Some(1));
}

#[test]
fn owned_reduction_rejects_wide_shifts() {
    let mut basis = PolynomialMatrix::<Gf8B>::zeros(1, 2).unwrap();
    let error = basis.reduce_weak_popov(&[0]).unwrap_err();
    assert_eq!(
        error,
        MatrixError::ShiftCount {
            columns: 2,
            shifts: 1
        }
    );
    assert!(ShiftPreparation::from_signed(&[]).is_err());
}

#[test]
fn identity_entry_accessors_and_error_display_hold() {
    let mut identity = PolynomialMatrix::<Gf8B>::identity(2).unwrap();
    assert_eq!(identity.shape(), (2, 2));
    assert!(identity.entry(0, 0).unwrap().is_one());
    assert!(identity.entry(0, 1).unwrap().is_zero());
    assert!(identity.entry(2, 0).is_none());
    identity.set_entry(0, 1, poly_b(&[b(3), b(4)])).unwrap();
    assert_eq!(identity.entry(0, 1).unwrap().degree(), Some(1));
    assert!(identity.set_entry(2, 0, poly_b(&[b(1)])).is_err());
    let display = alloc::format!("{identity:?}");
    assert!(display.contains("PolynomialMatrix"));
    let errors = [
        MatrixError::GeometryOverflow { context: "shape" },
        MatrixError::AllocationFailed { entries: 3 },
        MatrixError::ShiftCount {
            columns: 2,
            shifts: 1,
        },
        MatrixError::ShiftSpan {
            minimum: -1,
            maximum: 2,
        },
    ];
    for error in errors {
        let text = alloc::format!("{error}");
        assert!(!text.is_empty());
        let debug = alloc::format!("{error:?}");
        assert!(!debug.is_empty());
    }
    let from_reduce: MatrixError = polymat::ReduceError::AllocationFailed { entries: 4 }.into();
    assert_eq!(
        from_reduce,
        MatrixError::Reduction(polymat::ReduceError::AllocationFailed { entries: 4 })
    );
    let from_degree: MatrixError = polymat::ReduceError::DegreeOverflow {
        degree: 1,
        shift: usize::MAX,
    }
    .into();
    assert_eq!(
        from_degree,
        MatrixError::Reduction(polymat::ReduceError::DegreeOverflow {
            degree: 1,
            shift: usize::MAX,
        })
    );
}

extern crate alloc;
