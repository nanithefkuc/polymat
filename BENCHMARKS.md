# Benchmarks

Public-API basis timings on representative small shapes. The crate ships one
strategy per operation — scalar weak-Popov reduction, the scalar
discrepancy recurrence for approximants, and canonical Popov reduction —
so there is no crossover or dispatch decision to record.

| Workload | Field | Time (ns) |
| --- | --- | --- |
| `weak_popov` 3x4 shifts 0,3,6,9 | GF(2^8) | 320 |
| `approximant` 4x3 orders 4,3,5 | GF(2^8) | 10672 |
| `popov` 3x4 shifts 0,3,6,9 | GF(2^8) | 1118 |
| `weak_popov` 3x4 shifts 0,3,6,9 | GF(2^16) | 2845 |
| `approximant` 4x3 orders 4,3,5 | GF(2^16) | 18886 |
| `popov` 3x4 shifts 0,3,6,9 | GF(2^16) | 5190 |

Caveats and measurement details:

- Host: Intel Core Ultra 7 258V (`x86_64`), Rust 1.98.0, pinned to CPU 3
  via `FEC_GOLDEN_CORE=3`.
- Criterion reports the mean of 100 samples per workload; values above
  are rounded to whole nanoseconds.
- Workloads live in `benches/basis.rs`: cloned inputs per iteration
  (allocation outside the reducer is included), shifts and orders as
  tabulated. Shapes approximate interpolation-module geometry; they are
  not wired engine plans.

## Competitors

No direct non-copyleft Rust competitor was found for generic shifted polynomial
matrix reduction over binary and prime fields. FLINT, NTL, and SageMath cover
related polynomial-matrix or module operations under copyleft licenses and do
not enter this crate's build.
