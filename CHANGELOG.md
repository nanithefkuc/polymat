# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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
