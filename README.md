# agi-core-codex-minimal

A small, readable scaffold for the 4 pillars:
- feedback: score every attempt on the task examples
- approximability: keep the best frontier, even when it is imperfect
- abstraction/composition: promote repeated useful subtrees into a learned population
- exploration: search new programs by composing primitives with learned abstractions

## Files
- [minimal.py](/Users/vibhorjain/github/agi-core-codex-minimal/minimal.py): tiny CLI entrypoint
- [language.py](/Users/vibhorjain/github/agi-core-codex-minimal/language.py): typed grid language, execution, scoring, and tree utilities
- [learner.py](/Users/vibhorjain/github/agi-core-codex-minimal/learner.py): frontier search, causal credit, abstraction survival, and round reporting
- [domains.py](/Users/vibhorjain/github/agi-core-codex-minimal/domains.py): synthetic curriculum and ARC dataset loading

## Run
- `python minimal.py synthetic`
- `python minimal.py arc`
- `python minimal.py both`

ARC expects JSON task files in one of:
- `ARC_AGI_1_TRAIN_DIR`
- `data/ARC-AGI/data/training`
- `../agi-core/data/ARC-AGI/data/training`

## Current Readout
- Synthetic runs a 4-stage branching curriculum plus a frozen choice probe.
- The shared stage-1 abstraction `local(nonzero_mask, flip_h)` owns stage 2, and later branch abstractions own the later stage-3/stage-4/choice tasks.
- Learned abstractions now count as compressed units during later search, so reuse buys actual search headroom.
- ARC currently reaches `8/400` exact on train and `0/400` exact on public eval, with mean public-eval score about `0.507`.
- The two surviving ARC abstractions are:
  - `chain(local(nonzero_mask, transpose), local(nonzero_mask, flip_v))`
  - `chain(local(nonzero_mask, flip_h), local(nonzero_mask, flip_v))`
