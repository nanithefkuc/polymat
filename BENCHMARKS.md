# Benchmarks

Public-API basis timings at consumer geometry. The crate ships one
strategy per operation — scalar weak-Popov reduction, the scalar
discrepancy recurrence for approximants, Hasse interpolation, and the
block-kernel congruence oracle — so there is no crossover or dispatch
decision to record. The scalar paths are the retained correctness
controls; a faster basis strategy promotes only with a same-session
pinned baseline/control pair per the measurement policy.

| Workload | Field | Time (ns) |
| --- | --- | --- |
| `weak_popov` 3x4 shifts 0,3,6,9 | GF(8) binary | 320 |
| `approximant` 4x3 orders 4,3,5 | GF(8) binary | 10672 |
| `popov` 3x4 shifts 0,3,6,9 | GF(8) binary | 1118 |
| `weak_popov` 3x4 shifts 0,3,6,9 | GF(16) binary | 2845 |
| `approximant` 4x3 orders 4,3,5 | GF(16) binary | 18886 |
| `popov` 3x4 shifts 0,3,6,9 | GF(16) binary | 5190 |

Caveats and measurement details:

- Host: Intel Core Ultra 7 258V (`x86_64`), Rust 1.98.0, pinned to CPU 3
  via `FEC_GOLDEN_CORE=3`.
- Criterion reports the mean of 100 samples per workload; values above
  are rounded to whole nanoseconds.
- Workloads live in `benches/basis.rs`: cloned inputs per iteration
  (allocation outside the reducer is included), shifts and orders as
  tabulated.
- The approximant workload constructs the full discrepancy recurrence
  with per-constraint residual products; the profile is dominated by
  `poly-ring` residual multiplication, not schedule overhead, so no
  divide-and-conquer candidate is proposed from this round.
- No faster basis strategy is promoted: one scalar strategy per
  operation stands, with the scalar paths retained as correctness
  oracles for later optimization work.

## Competitors

No direct non-copyleft Rust competitor was found for generic shifted polynomial
matrix reduction over binary and prime fields. FLINT, NTL, and SageMath cover
related polynomial-matrix or module operations under copyleft licenses and do
not enter this crate's build.
