#![allow(dead_code)]

use core::marker::PhantomData;

use fgf::FieldKernels;
use fgf::field::{Elem, Field};
use polymat::{ReduceError, WeakPopovBasis, WeakPopovRow};

pub type E<F> = <F as Field>::Elem;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row<F: FieldKernels> {
    pub columns: Vec<Vec<E<F>>>,
    pub transform: Vec<Vec<E<F>>>,
    pub apply_updates: bool,
    field: PhantomData<F>,
}

impl<F: FieldKernels> Row<F> {
    pub fn new(columns: Vec<Vec<E<F>>>) -> Self {
        Self {
            columns,
            transform: Vec::new(),
            apply_updates: true,
            field: PhantomData,
        }
    }

    pub fn with_transform(columns: Vec<Vec<E<F>>>, transform: Vec<Vec<E<F>>>) -> Self {
        Self {
            columns,
            transform,
            apply_updates: true,
            field: PhantomData,
        }
    }

    pub fn normalize(&mut self) {
        for polynomial in self.columns.iter_mut().chain(&mut self.transform) {
            while polynomial
                .last()
                .is_some_and(|coefficient| coefficient.is_zero())
            {
                polynomial.pop();
            }
        }
    }
}

impl<F: FieldKernels> WeakPopovRow<F> for Row<F> {
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

    fn coefficient(&self, column: usize, degree: usize) -> E<F> {
        self.columns
            .get(column)
            .and_then(|polynomial| polynomial.get(degree))
            .copied()
            .unwrap_or(E::<F>::ZERO)
    }

    fn add_scaled_shifted_assign(
        &mut self,
        scale: E<F>,
        pivot: &Self,
        shift: usize,
    ) -> Result<(), Self::Error> {
        if !self.apply_updates {
            return Ok(());
        }
        add_scaled_rows(&mut self.columns, &pivot.columns, scale, shift);
        add_scaled_rows(&mut self.transform, &pivot.transform, scale, shift);
        self.normalize();
        Ok(())
    }
}

fn add_scaled_rows<E: Elem>(target: &mut [Vec<E>], pivot: &[Vec<E>], scale: E, shift: usize) {
    for (column, target_polynomial) in target.iter_mut().enumerate() {
        let Some(pivot_polynomial) = pivot.get(column) else {
            continue;
        };
        let required = pivot_polynomial.len().saturating_add(shift);
        if required > target_polynomial.len() {
            target_polynomial.resize(required, E::ZERO);
        }
        for (degree, &coefficient) in pivot_polynomial.iter().enumerate() {
            let position = degree + shift;
            target_polynomial[position] = target_polynomial[position].add(scale.mul(coefficient));
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexedBasis<F: FieldKernels>(pub Vec<Row<F>>);

impl<F: FieldKernels> WeakPopovBasis<F> for IndexedBasis<F> {
    type Error = ReduceError;

    fn row_count(&self) -> usize {
        self.0.len()
    }

    fn column_count(&self, row: usize) -> usize {
        self.0[row].column_count()
    }

    fn degree(&self, row: usize, column: usize) -> Option<usize> {
        self.0[row].degree(column)
    }

    fn coefficient(&self, row: usize, column: usize, degree: usize) -> E<F> {
        self.0[row].coefficient(column, degree)
    }

    fn add_scaled_shifted_assign(
        &mut self,
        target: usize,
        pivot: usize,
        scale: E<F>,
        shift: usize,
        _shifts: &[usize],
    ) -> Result<(), Self::Error> {
        let (target_row, pivot_row) = if target < pivot {
            let (lower, upper) = self.0.split_at_mut(pivot);
            (&mut lower[target], &upper[0])
        } else {
            let (lower, upper) = self.0.split_at_mut(target);
            (&mut upper[0], &lower[pivot])
        };
        target_row.add_scaled_shifted_assign(scale, pivot_row, shift)
    }
}

pub fn polynomial_add<E: Elem>(target: &mut Vec<E>, source: &[E]) {
    if source.len() > target.len() {
        target.resize(source.len(), E::ZERO);
    }
    for (destination, &coefficient) in target.iter_mut().zip(source) {
        *destination = destination.add(coefficient);
    }
    while target
        .last()
        .is_some_and(|coefficient| coefficient.is_zero())
    {
        target.pop();
    }
}

pub fn polynomial_multiply<E: Elem>(left: &[E], right: &[E]) -> Vec<E> {
    if left.is_empty() || right.is_empty() {
        return Vec::new();
    }
    let mut product = vec![E::ZERO; left.len() + right.len() - 1];
    for (left_degree, &left_coefficient) in left.iter().enumerate() {
        for (right_degree, &right_coefficient) in right.iter().enumerate() {
            let degree = left_degree + right_degree;
            product[degree] = product[degree].add(left_coefficient.mul(right_coefficient));
        }
    }
    while product
        .last()
        .is_some_and(|coefficient| coefficient.is_zero())
    {
        product.pop();
    }
    product
}

pub fn multiply_transform<F: FieldKernels>(
    transform_rows: &[Row<F>],
    original: &[Vec<Vec<E<F>>>],
) -> Vec<Vec<Vec<E<F>>>> {
    transform_rows
        .iter()
        .map(|row| {
            (0..original.first().map_or(0, Vec::len))
                .map(|column| {
                    let mut value = Vec::new();
                    for (coefficient, original_row) in row.transform.iter().zip(original) {
                        let term = polynomial_multiply(coefficient, &original_row[column]);
                        polynomial_add(&mut value, &term);
                    }
                    value
                })
                .collect()
        })
        .collect()
}
