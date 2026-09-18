# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Shifted weak-Popov reduction for caller-owned rows and indexed, slab-backed
  bases: `PopovLeadingTerm`, `WeakPopovRow`, `WeakPopovBasis`,
  `WeakPopovScratch`, `weak_popov`, `weak_popov_scratch`, and
  `weak_popov_basis_scratch`.
- `ReduceError` preserves checked geometry, degree, allocation, termination,
  and adapter-metadata failures without erasing consumer-native errors.
- A full-rescan scalar reducer behind `internals` provides an untuned control
  for the cached collision schedule.

### Changed

- **Breaking, relative to `gfm`:** polynomial-module reduction moves to
  `polymat`. Add a `polymat` dependency and import the reducer types and
  functions from its crate root. Reduction failures are now
  `polymat::ReduceError`; Fq matrix storage and elimination remain in `gfm`.
- Cancelling row updates use the negative leading-coefficient ratio. Binary
  field results are unchanged, while odd-characteristic fields now cancel the
  advertised leading term correctly.
