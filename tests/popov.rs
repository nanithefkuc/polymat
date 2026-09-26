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

fn mul_matrix(
    left: &PolynomialMatrix<Gf8B>,
    right: &PolynomialMatrix<Gf8B>,
) -> PolynomialMatrix<Gf8B> {
    let (rows, _) = left.shape();
    let (_, columns) = right.shape();
    let mut out = PolynomialMatrix::zeros(rows, columns).unwrap();
    left.mul_into(right, &mut out).unwrap();
    out
}

#[test]
fn validators_distinguish_three_forms() {
    // Weak but unordered: leading columns [1, 0].
    let mut unordered = PolynomialMatrix::from_entries(
        2,
        2,
        alloc::vec![
            Polynomial::zero(),
            poly_b(&[b(1)]),
            poly_b(&[b(1)]),
            Polynomial::zero()
        ],
    )
    .unwrap();
    assert!(unordered.is_weak_popov(&[0, 0]));
    assert!(!unordered.is_ordered_weak_popov(&[0, 0]));
    assert!(!unordered.is_popov(&[0, 0]));
    let permutation = unordered.order_weak_popov(&[0, 0]).unwrap();
    assert_eq!(permutation, alloc::vec![1, 0]);
    assert!(unordered.is_ordered_weak_popov(&[0, 0]));
    // Ordered but not canonical: pivot column 0 has an entry of equal degree.
    let mut unreduced = PolynomialMatrix::from_entries(
        2,
        2,
        alloc::vec![
            poly_b(&[b(1), b(1)]),
            poly_b(&[b(1)]),
            poly_b(&[b(2), b(2)]),
            poly_b(&[b(0), b(1)]),
        ],
    )
    .unwrap();
    unreduced.reduce_weak_popov(&[0, 0]).unwrap();
    unreduced.order_weak_popov(&[0, 0]).unwrap();
    assert!(unreduced.is_ordered_weak_popov(&[0, 0]));
    // Duplicate leading columns are not weak.
    let duplicate = PolynomialMatrix::from_entries(
        2,
        2,
        alloc::vec![
            poly_b(&[b(1)]),
            Polynomial::zero(),
            poly_b(&[b(2)]),
            Polynomial::zero()
        ],
    )
    .unwrap();
    assert!(!duplicate.is_weak_popov(&[0, 0]));
    assert!(!duplicate.is_popov(&[1]));
}

#[test]
fn canonical_form_is_monic_and_column_reduced() {
    let mut basis = PolynomialMatrix::from_entries(
        2,
        2,
        alloc::vec![
            poly_b(&[b(2), b(2)]),
            poly_b(&[b(1)]),
            poly_b(&[b(3), b(0), b(3)]),
            poly_b(&[b(1), b(1)]),
        ],
    )
    .unwrap();
    basis.reduce_popov(&[0, 0]).unwrap();
    assert!(basis.is_popov(&[0, 0]));
    // Every pivot is monic.
    for row in 0..2 {
        let columns = basis.reduced_leading_columns(&[0, 0]);
        let column = columns[row].unwrap();
        assert_eq!(
            basis.entry(row, column).unwrap().leading_coefficient(),
            Some(b(1))
        );
    }
}

#[test]
fn canonical_form_is_invariant_under_unimodular_change() {
    // Two generating sets for the same module: the second adds (1+x) times
    // row 0 to row 1 and swaps the rows.
    let first = PolynomialMatrix::from_entries(
        2,
        2,
        alloc::vec![
            poly_b(&[b(1), b(1)]),
            poly_b(&[b(1)]),
            poly_b(&[b(1), b(0), b(1)]),
            poly_b(&[b(1), b(1)]),
        ],
    )
    .unwrap();
    let scale = poly_b(&[b(1), b(1)]);
    let mut second_entries = alloc::vec![];
    for row in 0..2 {
        for column in 0..2 {
            second_entries.push(first.entry(row, column).unwrap().clone());
        }
    }
    // row1 += (1+x) * row0, then swap.
    for column in 0..2 {
        let mut updated = second_entries[2 + column].clone();
        let multiple = second_entries[column].multiply(&scale).unwrap();
        updated.add_assign(&multiple).unwrap();
        second_entries[2 + column] = updated;
    }
    second_entries.swap(0, 2);
    second_entries.swap(1, 3);
    let second = PolynomialMatrix::from_entries(2, 2, second_entries).unwrap();
    let mut canonical_first = first.clone();
    canonical_first.reduce_popov(&[0, 0]).unwrap();
    let mut canonical_second = second.clone();
    canonical_second.reduce_popov(&[0, 0]).unwrap();
    assert!(canonical_first.is_popov(&[0, 0]));
    assert!(canonical_second.is_popov(&[0, 0]));
    // Same nonzero rows after normalization.
    let first_rows: Vec<Vec<Polynomial<Gf8B>>> = (0..2)
        .map(|row| {
            (0..2)
                .map(|column| canonical_first.entry(row, column).unwrap().clone())
                .collect()
        })
        .collect();
    let second_rows: Vec<Vec<Polynomial<Gf8B>>> = (0..2)
        .map(|row| {
            (0..2)
                .map(|column| canonical_second.entry(row, column).unwrap().clone())
                .collect()
        })
        .collect();
    assert_eq!(first_rows, second_rows);
}

#[test]
fn tracked_canonical_preserves_transformation_identity() {
    let original = PolynomialMatrix::from_entries(
        2,
        2,
        alloc::vec![
            poly_b(&[b(1), b(1)]),
            poly_b(&[b(1)]),
            poly_b(&[b(1), b(0), b(1)]),
            poly_b(&[b(1), b(1)]),
        ],
    )
    .unwrap();
    let mut reduced = original.clone();
    let transform = reduced.reduce_popov_tracked(&[0, 0]).unwrap();
    assert!(reduced.is_popov(&[0, 0]));
    let product = mul_matrix(&transform, &original);
    assert_eq!(product, reduced);
    // Rank-deficient input keeps its shape with zero-row padding.
    let mut dependent = original.clone();
    dependent.reduce_popov(&[0, 0]).unwrap();
    assert_eq!(dependent.shape(), (2, 2));
}

#[test]
fn predictable_degree_bounds_combinations() {
    let mut basis = PolynomialMatrix::from_entries(
        2,
        2,
        alloc::vec![
            poly_b(&[b(1)]),
            Polynomial::zero(),
            Polynomial::zero(),
            poly_b(&[b(1), b(1)])
        ],
    )
    .unwrap();
    basis.reduce_popov(&[0, 1]).unwrap();
    // Row degrees: row0 shifted 0, row1 shifted 1+1=2. Combination [x^2, x]
    // predicts max(0+2, 2+1) = 3.
    let combination = alloc::vec![poly_b(&[b(0), b(0), b(1)]), poly_b(&[b(0), b(1)])];
    assert_eq!(basis.predictable_degree(&[0, 1], &combination), Some(3));
    let zero_combination = alloc::vec![Polynomial::zero(), Polynomial::zero()];
    assert_eq!(basis.predictable_degree(&[0, 1], &zero_combination), None);
}

#[test]
fn odd_characteristic_canonical_is_monic() {
    let mut basis = PolynomialMatrix::from_entries(
        2,
        2,
        alloc::vec![
            poly_p(&[p(2), p(2)]),
            poly_p(&[p(1)]),
            poly_p(&[p(3), p(0), p(3)]),
            poly_p(&[p(1), p(1)]),
        ],
    )
    .unwrap();
    basis.reduce_popov(&[0, 0]).unwrap();
    assert!(basis.is_popov(&[0, 0]));
    for row in 0..2 {
        let column = basis.reduced_leading_columns(&[0, 0])[row].unwrap();
        assert_eq!(
            basis.entry(row, column).unwrap().leading_coefficient(),
            Some(p(1))
        );
    }
}

#[test]
fn ordering_rejects_bad_shifts_and_zero_rows_sort_last() {
    let mut basis = PolynomialMatrix::from_entries(
        3,
        2,
        alloc::vec![
            Polynomial::zero(),
            Polynomial::zero(),
            poly_b(&[b(1)]),
            Polynomial::zero(),
            Polynomial::zero(),
            poly_b(&[b(1), b(1)]),
        ],
    )
    .unwrap();
    assert!(basis.order_weak_popov(&[0]).is_err());
    let permutation = basis.order_weak_popov(&[0, 0]).unwrap();
    assert_eq!(permutation, alloc::vec![1, 2, 0]);
    assert!(basis.is_ordered_weak_popov(&[0, 0]));
    // Nonzero row after a zero row is not ordered.
    let mut swapped = basis.clone();
    let first: Vec<_> = (0..2)
        .map(|c| swapped.entry(0, c).unwrap().clone())
        .collect();
    let last: Vec<_> = (0..2)
        .map(|c| swapped.entry(2, c).unwrap().clone())
        .collect();
    for (c, e) in first.iter().enumerate() {
        swapped.set_entry(2, c, e.clone()).unwrap();
    }
    for (c, e) in last.iter().enumerate() {
        swapped.set_entry(0, c, e.clone()).unwrap();
    }
    assert!(!swapped.is_ordered_weak_popov(&[0, 0]));
    assert!(!swapped.is_popov(&[0, 0]));
}

#[test]
fn validators_reject_nonmonic_and_unreduced_pivots() {
    // Non-monic pivot: ordered but not canonical.
    let mut basis = PolynomialMatrix::from_entries(
        2,
        2,
        alloc::vec![
            poly_b(&[b(2), b(2)]),
            Polynomial::zero(),
            Polynomial::zero(),
            poly_b(&[b(1)]),
        ],
    )
    .unwrap();
    basis.reduce_weak_popov(&[0, 0]).unwrap();
    basis.order_weak_popov(&[0, 0]).unwrap();
    assert!(basis.is_ordered_weak_popov(&[0, 0]));
    assert!(!basis.is_popov(&[0, 0]));
    // Pivot column with an entry above pivot degree is not canonical, yet
    // still ordered weak: under shifts [0, 10] row 1 leads in column 1
    // (shifted 0+10) while its column-0 entry (degree 5) exceeds the
    // column-0 pivot degree 1.
    let raw = PolynomialMatrix::from_entries(
        2,
        2,
        alloc::vec![
            poly_b(&[b(1), b(1)]),
            Polynomial::zero(),
            poly_b(&[b(0), b(0), b(0), b(0), b(0), b(1)]),
            poly_b(&[b(1)]),
        ],
    )
    .unwrap();
    assert!(raw.is_weak_popov(&[0, 10]));
    assert!(raw.is_ordered_weak_popov(&[0, 10]));
    assert!(!raw.is_popov(&[0, 10]));
}

#[test]
fn tracked_odd_characteristic_canonical_holds_identity() {
    // Exercises the tracked monic-scaling path (pivots 2, 3 need inverses)
    // and the tracked column-reduction witness update.
    let original = PolynomialMatrix::from_entries(
        2,
        2,
        alloc::vec![
            poly_p(&[p(2), p(2)]),
            poly_p(&[p(1), p(0), p(1)]),
            poly_p(&[p(3), p(0), p(3)]),
            poly_p(&[p(1), p(1)]),
        ],
    )
    .unwrap();
    let mut reduced = original.clone();
    let transform = reduced.reduce_popov_tracked(&[0, 0]).unwrap();
    assert!(reduced.is_popov(&[0, 0]));
    let mut product = PolynomialMatrix::zeros(2, 2).unwrap();
    transform.mul_into(&original, &mut product).unwrap();
    assert_eq!(product, reduced);
}

#[test]
fn empty_and_zero_matrices_have_stable_forms() {
    let mut empty = PolynomialMatrix::<Gf8B>::zeros(0, 2).unwrap();
    empty.reduce_popov(&[0, 0]).unwrap();
    assert!(empty.is_popov(&[0, 0]));
    let mut zero = PolynomialMatrix::<Gf8B>::zeros(2, 2).unwrap();
    zero.reduce_popov(&[0, 0]).unwrap();
    assert!(zero.is_popov(&[0, 0]));
    assert_eq!(
        zero.reduced_leading_columns(&[0, 0]),
        alloc::vec![None, None]
    );
}

extern crate alloc;
