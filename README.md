# agi-core-codex-minimal

A readable 4-pillar scaffold built around a tiny coordinate/mask calculus.

## Idea
- feedback: score candidates on train examples and keep a small diverse frontier
- approximability: promote subtrees that improve tasks even before exact solves
- abstraction/composition: reuse learned `GridExpr` and `MaskExpr` subtrees in later rounds
- exploration: enumerate a small typed search space over integer expressions, mask expressions, and grid expressions

## Core Types
- `IntExpr`: `0`, `1`, `row`, `col`, `height`, `width`, `add`, `sub`
- `MaskExpr`: `nonzero`, `eq`, `and`, `or`, `not`
- `GridExpr`: `identity`, `remap`, `compose`, `select`, `focus`

Mirror, transpose, and rotation are not named primitives. They emerge as `remap(...)` programs built from `IntExpr`.

## Files
- [minimal.py](/Users/vibhorjain/github/agi-core-codex-minimal/minimal.py): CLI
- [language.py](/Users/vibhorjain/github/agi-core-codex-minimal/language.py): typed calculus and evaluation
- [learner.py](/Users/vibhorjain/github/agi-core-codex-minimal/learner.py): `explore`, `score`, and `promote`
- [domains.py](/Users/vibhorjain/github/agi-core-codex-minimal/domains.py): synthetic curriculum, ARC loading, and inspection
- [OBSERVATIONS.md](/Users/vibhorjain/github/agi-core-codex-minimal/OBSERVATIONS.md): manual ARC notes

## Run
- `python minimal.py synthetic`
- `python minimal.py arc`
- `python minimal.py inspect TASK_ID`

ARC expects JSON task files in:
- `ARC_AGI_1_TRAIN_DIR`
- `data/ARC-AGI/data/training`
- `../agi-core/data/ARC-AGI/data/training`

## Current Readout
- Synthetic is the primary mechanism gate:
  - stage 1 promotes a reusable local mirror abstraction
  - stage 2 promotes two branch abstractions
  - stages 3 and 4 solve by reusing those learned branches
  - ablation breaks later stages when the learned library is removed
- ARC remains a transfer probe and should be read mainly as a sanity check on the mechanism, not as the optimization target.
  - round 1: `6/400` exact on train, mean train `0.525`
  - round 3: `6/400` exact on train, mean train `0.530`
  - public eval: `0/400` exact, mean test `0.543`
