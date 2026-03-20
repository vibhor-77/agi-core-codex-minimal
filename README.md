# agi-core-codex-minimal

A readability-first reset of the 4-pillar scaffold.

## Idea
- feedback: score every program on the task examples
- approximability: keep a small frontier of the best imperfect programs
- abstraction/composition: promote repeated useful subtrees into a learned library
- exploration: enumerate a small typed language over primitive programs and learned abstractions

## Files
- [minimal.py](/Users/vibhorjain/github/agi-core-codex-minimal/minimal.py): tiny CLI
- [language.py](/Users/vibhorjain/github/agi-core-codex-minimal/language.py): typed primitives plus `pipe(...)` and `focus(...)`
- [learner.py](/Users/vibhorjain/github/agi-core-codex-minimal/learner.py): deterministic frontier search and abstraction promotion
- [domains.py](/Users/vibhorjain/github/agi-core-codex-minimal/domains.py): synthetic curriculum and ARC dataset loading

## Primitive Substrate
- grid transforms: `identity`, `reverse_rows`, `reverse_cols`, `swap_axes`
- perception: `nonzero`
- mask transform: `invert`
- program constructors: `pipe(left, right)` and `focus(selector, transform)`

## Run
- `python minimal.py synthetic`
- `python minimal.py arc`
- `python minimal.py both`

ARC expects JSON task files in:
- `ARC_AGI_1_TRAIN_DIR`
- `data/ARC-AGI/data/training`
- `../agi-core/data/ARC-AGI/data/training`

## Current Readout
- This reset is cleaner and simpler than the previous version, but weaker.
- Synthetic currently learns the first useful abstraction, `focus(nonzero, reverse_cols)`, but does not yet compound through the later stages.
- ARC currently reaches `7/400` exact on train and `0/400` exact on public eval, with mean public-eval score about `0.519`.
- The point of the current code is clarity of the typed substrate and learning loop, not benchmark performance yet.
