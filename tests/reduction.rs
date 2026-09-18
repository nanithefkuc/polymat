mod common;

use fgf::field::Field;
use fgf::{Gf8B, Mersenne31, gf8b, mersenne31};
use polymat::{
    PopovLeadingTerm, ReduceError, WeakPopovRow, WeakPopovScratch, weak_popov,
    weak_popov_basis_scratch,
};

use common::{IndexedBasis, Row, multiply_transform};

type B = <Gf8B as Field>::Elem;
type P = <Mersenne31 as Field>::Elem;

fn b(value: u8) -> B {
    gf8b::Elem::from_raw(value)
}

fn p(value: u32) -> P {
    mersenne31::Elem::from_raw(value)
}

#[test]
fn binary_fixture_preserves_schedule_and_module() {
    let original = vec![
        vec![vec![b(0), b(1)], vec![b(1)]],
        vec![vec![b(1)], Vec::new()],
    ];
    let mut basis = vec![
        Row::<Gf8B>::with_transform(original[0].clone(), vec![vec![b(1)], Vec::new()]),
        Row::<Gf8B>::with_transform(original[1].clone(), vec![Vec::new(), vec![b(1)]]),
    ];

    weak_popov::<Gf8B, _>(&mut basis, &[0, 0]).unwrap();

    let leading: Vec<_> = basis
        .iter()
        .filter_map(|row| row.leading_term(&[0, 0]).unwrap())
        .map(|term| term.column)
        .collect();
    assert_eq!(leading, [1, 0]);
    assert_eq!(
        multiply_transform(&basis, &original),
        basis
            .iter()
            .map(|row| row.columns.clone())
            .collect::<Vec<_>>()
    );
}

#[test]
fn odd_characteristic_update_uses_negative_ratio() {
    let mut basis = [
        Row::<Mersenne31>::new(vec![vec![p(2)]]),
        Row::<Mersenne31>::new(vec![vec![p(1)]]),
    ];

    weak_popov::<Mersenne31, _>(&mut basis, &[0]).unwrap();

    assert_eq!(basis[0].columns, [Vec::<P>::new()]);
    assert_eq!(basis[1].columns, [vec![p(1)]]);
    assert_ne!(p(2).add(p(2)), P::ZERO);
}

#[test]
fn row_and_indexed_adapters_agree() {
    let rows = vec![
        Row::<Gf8B>::new(vec![vec![b(0), b(1)], vec![b(1)]]),
        Row::<Gf8B>::new(vec![vec![b(1)], Vec::new()]),
        Row::<Gf8B>::new(vec![Vec::new(), Vec::new()]),
    ];
    let mut row_basis = rows.clone();
    let mut indexed = IndexedBasis(rows);
    let mut scratch = WeakPopovScratch::new();

    weak_popov::<Gf8B, _>(&mut row_basis, &[1, 0]).unwrap();
    weak_popov_basis_scratch::<Gf8B, _>(&mut indexed, &[1, 0], &mut scratch).unwrap();

    assert_eq!(indexed.0, row_basis);

    let rows = vec![
        Row::<Mersenne31>::new(vec![vec![p(2), p(3)], vec![p(1)]]),
        Row::<Mersenne31>::new(vec![vec![p(1)], vec![p(4)]]),
        Row::<Mersenne31>::new(vec![Vec::new(), Vec::new()]),
    ];
    let mut row_basis = rows.clone();
    let mut indexed = IndexedBasis(rows);

    weak_popov::<Mersenne31, _>(&mut row_basis, &[1, 0]).unwrap();
    weak_popov_basis_scratch::<Mersenne31, _>(&mut indexed, &[1, 0], &mut scratch).unwrap();

    assert_eq!(indexed.0, row_basis);
}

#[test]
fn greater_column_breaks_shifted_degree_ties() {
    let row = Row::<Gf8B>::new(vec![vec![b(1), b(1)], vec![b(1)]]);
    assert_eq!(
        row.leading_term(&[0, 1]).unwrap(),
        Some(PopovLeadingTerm {
            degree: 0,
            column: 1,
            shifted_degree: 1,
        })
    );
}

#[test]
fn empty_zero_and_trailing_columns_terminate() {
    let mut empty: [Row<Gf8B>; 0] = [];
    weak_popov::<Gf8B, _>(&mut empty, &[]).unwrap();

    let mut zero = [Row::<Gf8B>::new(vec![Vec::new()])];
    weak_popov::<Gf8B, _>(&mut zero, &[0, 9]).unwrap();
    assert_eq!(zero[0].columns, [Vec::<B>::new()]);
}

#[test]
fn shape_and_degree_overflow_are_checked() {
    let mut wide = [Row::<Gf8B>::new(vec![vec![b(1)], vec![b(1)]])];
    assert_eq!(
        weak_popov::<Gf8B, _>(&mut wide, &[0]),
        Err(ReduceError::ShiftCount {
            columns: 2,
            shifts: 1,
        })
    );

    let mut overflow = [Row::<Gf8B>::new(vec![vec![b(1), b(1)]])];
    assert_eq!(
        weak_popov::<Gf8B, _>(&mut overflow, &[usize::MAX]),
        Err(ReduceError::DegreeOverflow {
            degree: 1,
            shift: usize::MAX,
        })
    );
}

#[test]
fn non_decreasing_callback_hits_termination_ceiling() {
    let mut left = Row::<Gf8B>::new(vec![vec![b(0), b(1)]]);
    left.apply_updates = false;
    let mut right = Row::<Gf8B>::new(vec![vec![b(1)]]);
    right.apply_updates = false;
    let mut basis = [left, right];

    assert!(matches!(
        weak_popov::<Gf8B, _>(&mut basis, &[0]),
        Err(ReduceError::Diverged { .. })
    ));
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdapterError {
    Reduce(ReduceError),
    UpdateRejected,
}

impl From<ReduceError> for AdapterError {
    fn from(error: ReduceError) -> Self {
        Self::Reduce(error)
    }
}

#[derive(Debug)]
struct RejectingRow(Row<Gf8B>);

impl WeakPopovRow<Gf8B> for RejectingRow {
    type Error = AdapterError;

    fn column_count(&self) -> usize {
        self.0.column_count()
    }

    fn degree(&self, column: usize) -> Option<usize> {
        self.0.degree(column)
    }

    fn coefficient(&self, column: usize, degree: usize) -> B {
        self.0.coefficient(column, degree)
    }

    fn add_scaled_shifted_assign(
        &mut self,
        _scale: B,
        _pivot: &Self,
        _shift: usize,
    ) -> Result<(), Self::Error> {
        Err(AdapterError::UpdateRejected)
    }
}

#[test]
fn native_callback_error_is_preserved() {
    let mut basis = [
        RejectingRow(Row::<Gf8B>::new(vec![vec![b(0), b(1)]])),
        RejectingRow(Row::<Gf8B>::new(vec![vec![b(1)]])),
    ];
    assert_eq!(
        weak_popov::<Gf8B, _>(&mut basis, &[0]),
        Err(AdapterError::UpdateRejected),
    );
}

#[derive(Debug)]
struct InvalidMetadata(Row<Gf8B>);

impl WeakPopovRow<Gf8B> for InvalidMetadata {
    type Error = ReduceError;

    fn column_count(&self) -> usize {
        self.0.column_count()
    }

    fn degree(&self, column: usize) -> Option<usize> {
        self.0.degree(column)
    }

    fn coefficient(&self, column: usize, degree: usize) -> B {
        self.0.coefficient(column, degree)
    }

    fn leading_term(&self, _shifts: &[usize]) -> Result<Option<PopovLeadingTerm>, Self::Error> {
        Ok(None)
    }

    fn add_scaled_shifted_assign(
        &mut self,
        scale: B,
        pivot: &Self,
        shift: usize,
    ) -> Result<(), Self::Error> {
        self.0.add_scaled_shifted_assign(scale, &pivot.0, shift)
    }
}

#[test]
fn invalid_cached_leading_metadata_is_rejected() {
    let mut basis = [InvalidMetadata(Row::<Gf8B>::new(vec![vec![b(1)]]))];
    assert_eq!(
        weak_popov::<Gf8B, _>(&mut basis, &[0]),
        Err(ReduceError::InvalidLeadingTerm { row: 0 })
    );
}

#[derive(Debug)]
struct ZeroLeadingCoefficient(Row<Gf8B>);

impl WeakPopovRow<Gf8B> for ZeroLeadingCoefficient {
    type Error = ReduceError;

    fn column_count(&self) -> usize {
        self.0.column_count()
    }

    fn degree(&self, column: usize) -> Option<usize> {
        self.0.degree(column)
    }

    fn coefficient(&self, _column: usize, _degree: usize) -> B {
        B::ZERO
    }

    fn add_scaled_shifted_assign(
        &mut self,
        scale: B,
        pivot: &Self,
        shift: usize,
    ) -> Result<(), Self::Error> {
        self.0.add_scaled_shifted_assign(scale, &pivot.0, shift)
    }
}

#[test]
fn agreeing_metadata_with_zero_leading_coefficient_is_rejected() {
    let mut basis = [ZeroLeadingCoefficient(Row::<Gf8B>::new(vec![vec![b(1)]]))];
    assert_eq!(
        weak_popov::<Gf8B, _>(&mut basis, &[0]),
        Err(ReduceError::InvalidLeadingTerm { row: 0 })
    );
}

#[test]
fn odd_characteristic_reduction_preserves_module() {
    let original = vec![
        vec![vec![p(3), p(2)], vec![p(5)]],
        vec![vec![p(1), p(4)], vec![p(6), p(1)]],
        vec![vec![p(2)], vec![p(2)]],
    ];
    let mut basis: Vec<_> = original
        .iter()
        .enumerate()
        .map(|(row, columns)| {
            let transform = (0..original.len())
                .map(|index| if index == row { vec![p(1)] } else { Vec::new() })
                .collect();
            Row::<Mersenne31>::with_transform(columns.clone(), transform)
        })
        .collect();

    weak_popov::<Mersenne31, _>(&mut basis, &[0, 1]).unwrap();

    assert_eq!(
        multiply_transform(&basis, &original),
        basis
            .iter()
            .map(|row| row.columns.clone())
            .collect::<Vec<_>>()
    );

    let mut columns: Vec<_> = basis
        .iter()
        .filter_map(|row| row.leading_term(&[0, 1]).unwrap())
        .map(|term| term.column)
        .collect();
    let distinct = columns.len();
    columns.sort_unstable();
    columns.dedup();
    assert_eq!(columns.len(), distinct);
}

#[test]
fn scratch_reuse_survives_changing_column_counts() {
    let wide = vec![
        Row::<Gf8B>::new(vec![vec![b(0), b(1)], vec![b(1)], vec![b(1), b(1)]]),
        Row::<Gf8B>::new(vec![vec![b(1)], Vec::new(), vec![b(1)]]),
    ];
    let narrow = vec![
        Row::<Gf8B>::new(vec![vec![b(0), b(1)], vec![b(1)]]),
        Row::<Gf8B>::new(vec![vec![b(1)], Vec::new()]),
    ];

    let mut reused = WeakPopovScratch::new();
    let mut first = IndexedBasis(wide.clone());
    weak_popov_basis_scratch::<Gf8B, _>(&mut first, &[0, 0, 0], &mut reused).unwrap();
    let mut second = IndexedBasis(narrow.clone());
    weak_popov_basis_scratch::<Gf8B, _>(&mut second, &[0, 0], &mut reused).unwrap();
    let mut third = IndexedBasis(wide.clone());
    weak_popov_basis_scratch::<Gf8B, _>(&mut third, &[0, 0, 0], &mut reused).unwrap();

    let mut fresh_narrow = IndexedBasis(narrow);
    weak_popov_basis_scratch::<Gf8B, _>(&mut fresh_narrow, &[0, 0], &mut WeakPopovScratch::new())
        .unwrap();
    let mut fresh_wide = IndexedBasis(wide);
    weak_popov_basis_scratch::<Gf8B, _>(&mut fresh_wide, &[0, 0, 0], &mut WeakPopovScratch::new())
        .unwrap();

    assert_eq!(second, fresh_narrow);
    assert_eq!(third, fresh_wide);
}
