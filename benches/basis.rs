//! Public-API basis workloads on representative small shapes.
//!
//! Small dense matrices with nonuniform shifts and per-column orders stand
//! in for interpolation-module geometry; they are approximations, not
//! wired engine plans. The workloads time the module operations a consumer
//! calls without claiming decoder throughput.

use core::time::Duration;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fgf::field::Field;
use fgf::{Gf8B, Gf16};
use poly_ring::Polynomial;
use polymat::PolynomialMatrix;

fn dense<F: fgf::kernel::FieldKernels>(
    rows: usize,
    columns: usize,
    seed: &mut u64,
) -> PolynomialMatrix<F>
where
    <F as Field>::Elem: Clone,
{
    let mut entries = Vec::with_capacity(rows * columns);
    for _ in 0..rows * columns {
        let degree = (*seed % 8) as usize;
        *seed = seed.wrapping_add(1);
        let mut coefficients = Vec::with_capacity(degree + 1);
        for _ in 0..=degree {
            *seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bytes = seed.to_le_bytes();
            coefficients.push(F::decode(&bytes[..F::BYTES]));
        }
        entries.push(Polynomial::from_coefficients(&coefficients).unwrap());
    }
    PolynomialMatrix::from_entries(rows, columns, entries).unwrap()
}

fn shifts(columns: usize, stride: usize) -> Vec<usize> {
    (0..columns).map(|column| column * stride).collect()
}

fn run_field<F>(criterion: &mut Criterion, name: &str)
where
    F: fgf::kernel::FieldKernels,
    <F as Field>::Elem: Clone,
{
    let mut group = criterion.benchmark_group(format!("basis/{name}"));
    group.measurement_time(Duration::from_secs(2));
    // Consumer interpolation geometry: small multiplicity module.
    let mut seed = 0x1234_5678_9abc_def0;
    let basis = dense::<F>(3, 4, &mut seed);
    let shifts = shifts(4, 3);
    group.throughput(Throughput::Elements(12));
    group.bench_function(BenchmarkId::new("weak_popov", "3x4"), |bencher| {
        bencher.iter(|| {
            let mut work = basis.clone();
            work.reduce_weak_popov(&shifts).unwrap();
            black_box(work);
        });
    });
    // Coupled-congruence shape: taller module with per-column orders.
    let coupled = dense::<F>(4, 3, &mut seed);
    let orders = [4, 3, 5];
    group.bench_function(BenchmarkId::new("approximant", "4x3"), |bencher| {
        bencher.iter(|| {
            let result = coupled.approximant_basis(&orders, &[0, 0, 0, 0]).unwrap();
            black_box(result);
        });
    });
    // Canonical form on the same interpolation geometry.
    group.bench_function(BenchmarkId::new("popov", "3x4"), |bencher| {
        bencher.iter(|| {
            let mut work = basis.clone();
            work.reduce_popov(&shifts).unwrap();
            black_box(work);
        });
    });
    group.finish();
}

fn basis(criterion: &mut Criterion) {
    run_field::<Gf8B>(criterion, "gf8");
    run_field::<Gf16>(criterion, "gf16");
}

criterion_group!(benches, basis);
criterion_main!(benches);
