# agi-core-codex-minimal

A small 4-pillar scaffold aimed at discovering structure from low-level grid operations.

## Idea
- feedback: score every program on the task examples
- approximability: keep a small frontier of the best imperfect programs
- abstraction/composition: promote repeated useful subtrees into a learned library
- exploration: enumerate a small typed language over primitive values, coordinate remaps, and learned abstractions

## Files
- [minimal.py](/Users/vibhorjain/github/agi-core-codex-minimal/minimal.py): tiny CLI
- [language.py](/Users/vibhorjain/github/agi-core-codex-minimal/language.py): primitive grid and mask operations, coordinate remaps, `pipe(...)`, and `focus(...)`
- [learner.py](/Users/vibhorjain/github/agi-core-codex-minimal/learner.py): deterministic frontier search and abstraction promotion
- [domains.py](/Users/vibhorjain/github/agi-core-codex-minimal/domains.py): synthetic curriculum and ARC dataset loading
- [OBSERVATIONS.md](/Users/vibhorjain/github/agi-core-codex-minimal/OBSERVATIONS.md): manual ARC notes distilled into low-level substrate ideas

## Primitive Substrate
- primitive values: integer cell values, grid coordinates, boolean masks
- primitive operations:
  - `identity : grid -> grid`
  - `nonzero : grid -> mask`
  - `invert : mask -> mask`
- generated coordinate remaps:
  - programs like `remap(row, width-1-col)` or `remap(col, row)`
  - this is where mirror, transpose, and rotation-like behavior are meant to be discovered
- generated coordinate predicates:
  - programs like `coord_eq(row, col)` or `coord_eq(row, zero)`
  - this is where diagonals, borders, corners, and simple anchors can emerge
- mask and cellwise composition:
  - `mask_and(left, right)` and `mask_or(left, right)`
  - `select(mask, when_true, when_false)` for cellwise writeback
- program constructors:
  - `pipe(left, right)` for composition
  - `focus(selector, transform)` for local apply-and-writeback

## Run
- `python minimal.py synthetic`
- `python minimal.py arc`
- `python minimal.py both`

ARC expects JSON task files in:
- `ARC_AGI_1_TRAIN_DIR`
- `data/ARC-AGI/data/training`
- `../agi-core/data/ARC-AGI/data/training`

## Current Readout
- The current design choice is deliberate:
  - mirror, transpose, and rotate should not be named primitives
  - they should emerge as coordinate remap programs
  - local behavior should come from selectors, cellwise selection, and writeback, not ARC-specific tactics
- Synthetic still acts mainly as a scaffold check; it now exercises remaps, selectors, and cellwise selection on the same tiny search loop.
- ARC is still only a proof-of-concept transfer probe, but the lower-level substrate now does slightly more:
  - round 1: `10/400` exact on train
  - round 3: `10/400` exact on train
  - public eval: `0/400` exact, mean test about `0.535`
  - newly surfaced programs include `coord_eq(...)` and `select(...)`, not just remaps
