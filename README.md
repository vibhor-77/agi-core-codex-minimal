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
  - local behavior should come from `focus(selector, transform)`, not ARC-specific tactics
- Synthetic currently demonstrates staged compounding around one learned local remap abstraction.
- ARC is still only a proof-of-concept transfer probe, not the optimization target.
