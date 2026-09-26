> [!WARNING]
> This library was made with the help of AI. Audit the code yourself, or with
> your own agent before using.

# polymat — polynomial matrices and modules

`polymat` owns matrices and finitely generated row modules over Fq[x]. Its
public surface holds an owned row-major `PolynomialMatrix` with checked
construction, destination-first arithmetic, certified rank/kernel/
membership/solve queries, ordered and canonical Popov forms with
validators, and unimodular reduction witnesses alongside the shifted
weak Popov reducer with the Mulders–Storjohann schedule. Row and indexed-basis
adapters let a consumer retain a compact coefficient slab instead of
allocating a second polynomial-matrix representation.

Weak Popov form gives each nonzero row a distinct shifted leading column. Ties
in shifted degree choose the greatest column. The operation preserves the
row-generated Fq[x] module; it does not claim ordered or canonical Popov form.
The adding row-update callback receives a negative coefficient ratio, so the
same contract is correct in binary and odd-characteristic fields.

Field arithmetic comes from [`fgf`](https://github.com/nanithefkuc/fgf).
Ordinary Fq matrix storage and elimination remain in
[`gfm`](https://github.com/nanithefkuc/gfm), while decoder-specific basis
construction and policy remain in the consuming engine. The crate contains no
unsafe code.

**Early development.** Nothing here is stable.

## Usage

The MSRV is Rust 1.93.

`polymat` is distributed through git only, and its `fgf` dependency is pinned
to an exact version.

```toml
[dependencies]
polymat = { git = "https://github.com/nanithefkuc/polymat" }
```

Portable `no_std` builds with `alloc` are available:

```toml
[dependencies]
polymat = { git = "https://github.com/nanithefkuc/polymat", default-features = false }
```

### Features

| Feature | Result |
| --- | --- |
| default (`std`, `simd`) | standard-library error integration and `fgf` SIMD forwarding |
| `std` without `simd` | portable field backend |
| `internals` | exposes the full-rescan scalar reducer; not a compatibility promise |
| `--no-default-features` | `no_std` with `alloc` |

## Building

The repository uses the shared `just` command surface:

```sh
just validate
```

`just test` runs one host test pass. `just features` exercises the minimal,
default, and all-feature closures.

## License

MIT — see [LICENSE](LICENSE).
