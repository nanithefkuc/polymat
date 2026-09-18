#![cfg(feature = "internals")]

mod common;

use fgf::field::Field;
use fgf::{Gf8B, Mersenne31, gf8b, mersenne31};
use polymat::internals::weak_popov_basis_reference;
use polymat::{WeakPopovRow, WeakPopovScratch, weak_popov_basis_scratch};

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
fn cached_schedule_matches_full_rescan_in_binary_field() {
    let rows = vec![
        Row::<Gf8B>::new(vec![vec![b(1), b(1), b(1)], vec![b(1)]]),
        Row::<Gf8B>::new(vec![vec![b(0), b(1)], vec![b(1), b(1)]]),
        Row::<Gf8B>::new(vec![vec![b(1)], Vec::new()]),
    ];
    let mut cached = IndexedBasis(rows.clone());
    let mut reference = IndexedBasis(rows);

    weak_popov_basis_scratch::<Gf8B, _>(&mut cached, &[0, 1], &mut WeakPopovScratch::new())
        .unwrap();
    weak_popov_basis_reference::<Gf8B, _>(&mut reference, &[0, 1]).unwrap();

    assert_eq!(cached, reference);
}

#[test]
fn cached_schedule_matches_full_rescan_in_prime_field() {
    let original = vec![
        vec![vec![p(2), p(3)], vec![p(1)]],
        vec![vec![p(1)], vec![p(4)]],
        vec![vec![p(5)], Vec::new()],
    ];
    let rows = vec![
        Row::<Mersenne31>::with_transform(
            original[0].clone(),
            vec![vec![p(1)], Vec::new(), Vec::new()],
        ),
        Row::<Mersenne31>::with_transform(
            original[1].clone(),
            vec![Vec::new(), vec![p(1)], Vec::new()],
        ),
        Row::<Mersenne31>::with_transform(
            original[2].clone(),
            vec![Vec::new(), Vec::new(), vec![p(1)]],
        ),
    ];
    let mut cached = IndexedBasis(rows.clone());
    let mut reference = IndexedBasis(rows);

    weak_popov_basis_scratch::<Mersenne31, _>(&mut cached, &[1, 0], &mut WeakPopovScratch::new())
        .unwrap();
    weak_popov_basis_reference::<Mersenne31, _>(&mut reference, &[1, 0]).unwrap();

    for reduced in [&cached, &reference] {
        let columns = reduced
            .0
            .iter()
            .map(|row| row.columns.clone())
            .collect::<Vec<_>>();
        assert_eq!(multiply_transform(&reduced.0, &original), columns);

        let mut leading_columns = reduced
            .0
            .iter()
            .filter_map(|row| row.leading_term(&[1, 0]).unwrap())
            .map(|term| term.column)
            .collect::<Vec<_>>();
        let rank = leading_columns.len();
        leading_columns.sort_unstable();
        leading_columns.dedup();
        assert_eq!(leading_columns.len(), rank);
    }
    assert_eq!(
        cached
            .0
            .iter()
            .filter(|row| row.leading_term(&[1, 0]).unwrap().is_some())
            .count(),
        reference
            .0
            .iter()
            .filter(|row| row.leading_term(&[1, 0]).unwrap().is_some())
            .count(),
    );
}
