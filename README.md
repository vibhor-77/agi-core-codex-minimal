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
- The code is now centered on a small typed substrate and a deterministic local-extension search.
- Synthetic compounds through the full staged curriculum:
  - stage 1 learns `focus(nonzero, reverse_cols)`
  - stage 2 learns branch abstractions by extending it with one primitive
  - stage 3 and stage 4 continue by extending the deepest learned abstractions first
  - the frozen choice probe solves `4/4`
- ARC is still only a transfer probe:
  - round 1: `5/400` exact on train
  - round 2: `7/400` exact on train
  - round 3: `7/400` exact on train
  - public eval: `0/400` exact, mean test about `0.487`
- The current value is not benchmark strength; it is that the same small mechanism now demonstrates clean staged compounding on synthetic tasks and nontrivial reuse on ARC without any ARC-specific tactics.
