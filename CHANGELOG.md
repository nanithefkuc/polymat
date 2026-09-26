# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Hermite/Smith normal forms, determinant, and inverse: `hermite_form`
  with the leftmost-pivot convention and `U A = H`, `smith_form` with
  monic divisibility-ordered invariants and `U A V = D` via Euclidean
  row/column Bezout steps, Bareiss `determinant` with swap signs
  cross-checked against the Leibniz oracle, `minor_gcd` determinantal
  divisors, and `polynomial_inverse` for unimodular inputs only.
- Confluent interpolation and Padé relations: `interpolation_basis`
  imposes merged `(point, column, multiplicity)` Hasse constraints with
  the old-pivot/`(x - point)` recurrence and cross-checks against
  `interpolation_moduli` congruence products and zero-point
  approximants; thin `pade_relation`/`simultaneous_pade`/`hermite_pade`
  constructors carry explicit numerator/denominator orientation, with
  selection failure on singular-block data rather than decoder verdicts.
- Scalar shifted approximant bases: `approximant_basis` runs the
  discrepancy recurrence from `I_m` over per-column orders (zero orders
  impose nothing), picks the minimum shifted-leading pivot, cancels
  against the old pivot, multiplies it by `x` last, and reports the
  independent-constraint count with canonical Popov output agreeing
  against the congruence oracle.
- Generic congruence module bases: `congruence_basis` spans `p F = 0 mod
  M[j]` via the `[F; -diag(M)]` left-kernel projection with monic modulus
  normalization, unit-modulus identity fast path, zero-modulus rejection,
  and canonical shifted Popov output as an m-by-m full-rank basis.
- Ordered weak and canonical shifted Popov forms: `order_weak_popov` with
  the applied row permutation, `reduce_popov` and `reduce_popov_tracked`
  with `U A = R` through ordering, monic scaling, and pivot-column
  reduction, `is_weak_popov`/`is_ordered_weak_popov`/`is_popov`
  validators, and `predictable_degree` for reduced-row combinations.
  Canonical nonzero rows are invariant under unimodular generator changes.
- Certified kernels and exact solving: `rank_weak_popov`,
  `reduced_leading_columns` with verified-reduced contracts,
  `left_kernel_weak_popov`/`right_kernel_weak_popov` with `K A = 0` /
  `A N = 0` and rank-nullity, `membership_weak_popov` with polynomial
  quotient witnesses, and `solve_weak_popov` returning a particular
  solution plus the complete right-kernel module through the transpose
  membership identity with the `U` back-mapping. `x z = 1` reports
  non-membership.
- Owned `PolynomialMatrix` over `poly-ring` polynomials: explicit m-by-n
  row-major storage with 0-by-n and m-by-0 shapes, checked construction and
  normalization-preserving setters, transpose, destination-first
  `add_into`/`sub_into`/`mul_into`/`mul_truncated_into` staged so any
  failure leaves the destination unchanged, `evaluate_to_vec` to row-major
  field elements, an owned weak-Popov reduction adapter using the
  reducer's default leading-term scan, and common-offset signed-shift
  preparation. `MatrixError` carries geometry, allocation, shift-count,
  shift-span, polynomial, and reduction (`Reduction`) failures; the
  reducer's termination and metadata errors are preserved, not flattened.
- Shifted weak-Popov reduction for caller-owned rows and indexed, slab-backed
  bases: `PopovLeadingTerm`, `WeakPopovRow`, `WeakPopovBasis`,
  `WeakPopovScratch`, `weak_popov`, `weak_popov_scratch`, and
  `weak_popov_basis_scratch`.
- `ReduceError` preserves checked geometry, degree, allocation, termination,
  and adapter-metadata failures without erasing consumer-native errors.
- A full-rescan scalar reducer behind `internals` provides an untuned control
  for the cached collision schedule.
- `WeakPopovScratch::retained_bytes` reports the heap bytes the schedule
  holds, so a consumer summing its own retained memory does not multiply
  `capacity` by an assumed entry width.

### Changed

- **Breaking, relative to `gfm`:** polynomial-module reduction moves to
  `polymat`. Add a `polymat` dependency and import the reducer types and
  functions from its crate root. Reduction failures are now
  `polymat::ReduceError`; Fq matrix storage and elimination remain in `gfm`.
- Cancelling row updates use the negative leading-coefficient ratio. Binary
  field results are unchanged, while odd-characteristic fields now cancel the
  advertised leading term correctly.
- **Breaking, relative to `gfm`:** a collision between two rows of equal
  leading degree reduces the lower-indexed row, so the result no longer
  depends on the order in which the cached schedule discovered the pair. Rows
  of distinct leading degree, and therefore every basis without a tie, reduce
  exactly as before; a consumer that recorded output for a tied basis records
  it again.
- `ReduceError::InvalidLeadingTerm` names the row whose advertised leading
  coefficient is zero rather than always naming the reduction target.
