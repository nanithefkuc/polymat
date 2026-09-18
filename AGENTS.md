# polymat

> `polymat` owns matrices and finitely generated row modules over Fq[x]. Field
> arithmetic comes from `fgf`; scalar polynomial arithmetic belongs to
> `poly-ring`. Ordinary Fq matrices and Gaussian elimination remain in `gfm`.
> Decoder policy and consumer-specific coefficient slabs remain in engines.

## Non-negotiables

1. **Polynomial modules only.** A polynomial pivot is not a field unit. Never
   replace module reduction with Gaussian elimination over Fq or Fq(x).
2. **No `unsafe`.** Forbidden at the crate root.
3. **No private field or polynomial ring.** Scalar field operations use `fgf`.
   Production polynomial arithmetic composes `poly-ring` when that surface is
   needed.
4. **Weak is not canonical.** Weak Popov form guarantees distinct leading
   positions among nonzero rows. It does not imply ordered or canonical Popov
   form.
5. **Cancellation uses subtraction.** An adding row-update callback receives
   the negative leading-coefficient ratio. Addition and subtraction coincide
   only in characteristic two.
6. **Adapter metadata is checked.** A custom leading-term result must agree
   with the adapter's degree scan before it drives indexing or termination.
7. **No decoder policy.** Interpolation geometry, shifts chosen from code
   parameters, candidate selection, and decoding decisions stay in engines.
8. **Oracles stay independent.** The full-rescan reducer remains an untuned
   control for the cached schedule. Tests use coefficient loops, not another
   production reducer, for algebraic certificates.

## Tooling

`just validate` is the pull-request gate; the shared recipe surface is
specified in the umbrella `AGENTS.md`.

- `TIERS` is empty because the reducer uses scalar elements and owns no
  runtime dispatch surface.
- `MIRI` is empty because the crate forbids unsafe code.
- `COV_IGNORE` is empty; every source line counts toward coverage.
- `justfile` is a byte-identical vendored copy. Edit the umbrella canonical
  copy and run `just sync`; crate-specific values belong in `crate.just`.

## Working here

- Edition 2024, MSRV 1.93, matching the current `fgf` dependency floor.
- Features: `default = ["std", "simd"]`; `simd` forwards to `fgf`; `internals`
  changes reachability only and exposes the scalar control.
- New subtrees use module-named files beside their directories. `lib.rs` holds
  crate documentation, declarations, and root re-exports.
- Errors are hand-written in `src/error.rs`, with manual `Display` and a
  `std::error::Error` implementation behind `std`.
- Public behavior belongs in integration tests. Deterministic coefficient
  fixtures cover characteristic two and odd characteristic.
- Commit subjects are at most about ten words, shaped `polymat: short verb
  phrase`. Public files never cite planning material.
