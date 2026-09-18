# Crate-specific values for the shared `justfile`, which is vendored verbatim
# from the ecosystem's canonical copy and must not be edited here. Recipes
# unique to polymat belong in this file and are documented in `AGENTS.md`.

CRATE := 'polymat'
MSRV := '1.93'

# The reducer uses scalar field elements and owns no runtime dispatch surface.
TIERS := ''

# `cargo miri test` argument sets. Empty because the crate forbids unsafe code.
MIRI := ''

# Every source line executes on ordinary CI hosts.
COV_IGNORE := ''
