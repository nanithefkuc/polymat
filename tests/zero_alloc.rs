use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

use fgf::field::{Elem, Field};
use fgf::{Gf8B, gf8b};
use polymat::{ReduceError, WeakPopovRow, WeakPopovScratch, weak_popov_scratch};

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

type E = <Gf8B as Field>::Elem;

struct FixedRow {
    columns: Vec<Vec<E>>,
    capacity: usize,
}

impl FixedRow {
    fn new(columns: &[&[u8]], capacity: usize) -> Self {
        let columns = columns
            .iter()
            .map(|column| {
                let mut buffer = Vec::with_capacity(capacity);
                buffer.extend(column.iter().map(|&value| gf8b::Elem::from_raw(value)));
                buffer
            })
            .collect();
        Self { columns, capacity }
    }
}

impl WeakPopovRow<Gf8B> for FixedRow {
    type Error = ReduceError;

    fn column_count(&self) -> usize {
        self.columns.len()
    }

    fn degree(&self, column: usize) -> Option<usize> {
        self.columns.get(column).and_then(|polynomial| {
            polynomial
                .iter()
                .rposition(|coefficient| !coefficient.is_zero())
        })
    }

    fn coefficient(&self, column: usize, degree: usize) -> E {
        self.columns
            .get(column)
            .and_then(|polynomial| polynomial.get(degree))
            .copied()
            .unwrap_or(E::ZERO)
    }

    fn add_scaled_shifted_assign(
        &mut self,
        scale: E,
        pivot: &Self,
        shift: usize,
    ) -> Result<(), Self::Error> {
        for (target, source) in self.columns.iter_mut().zip(&pivot.columns) {
            let required = source.len() + shift;
            assert!(required <= self.capacity);
            if required > target.len() {
                target.resize(required, E::ZERO);
            }
            for (degree, &coefficient) in source.iter().enumerate() {
                let position = degree + shift;
                target[position] = target[position].add(scale.mul(coefficient));
            }
            while target
                .last()
                .is_some_and(|coefficient| coefficient.is_zero())
            {
                target.pop();
            }
        }
        Ok(())
    }
}

fn basis(first: &[&[u8]], second: &[&[u8]], capacity: usize) -> [FixedRow; 2] {
    [
        FixedRow::new(first, capacity),
        FixedRow::new(second, capacity),
    ]
}

#[test]
fn warmed_schedule_reduces_without_allocating() {
    let capacity = 4;
    let shifts = [0usize, 0usize];
    let mut scratch = WeakPopovScratch::new();

    let mut warm = basis(&[&[0, 1], &[1]], &[&[1], &[]], capacity);
    weak_popov_scratch::<Gf8B, _>(&mut warm, &shifts, &mut scratch).unwrap();
    let retained_capacity = scratch.capacity();
    let retained_bytes = scratch.retained_bytes();
    assert!(retained_capacity >= shifts.len());
    assert!(retained_bytes >= retained_capacity);

    let mut first = basis(&[&[0, 1], &[1]], &[&[1], &[]], capacity);
    let mut second = basis(&[&[1, 1], &[1]], &[&[0, 1], &[1]], capacity);

    let before = ALLOCATIONS.load(Ordering::SeqCst);
    weak_popov_scratch::<Gf8B, _>(black_box(&mut first), &shifts, &mut scratch).unwrap();
    weak_popov_scratch::<Gf8B, _>(black_box(&mut second), &shifts, &mut scratch).unwrap();
    let allocations = ALLOCATIONS.load(Ordering::SeqCst) - before;
    assert_eq!(scratch.capacity(), retained_capacity);
    assert_eq!(scratch.retained_bytes(), retained_bytes);

    assert_eq!(allocations, 0);
}
