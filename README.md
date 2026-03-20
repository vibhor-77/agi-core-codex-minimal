# agi-core-codex-minimal
Tiny 4-pillar scaffolding: feedback scores every attempt, approximability keeps a small best-so-far frontier per task, abstraction promotes recurring leader subtrees, and exploration searches a tiny deterministic tree language over seeds plus learned programs.
ARC expects training JSONs at `ARC_AGI_1_TRAIN_DIR`, `data/ARC-AGI/data/training`, or `../agi-core/data/ARC-AGI/data/training`.
Run:
- `python minimal.py synthetic`
- `python minimal.py arc`
- `python minimal.py both`
Expected:
- synthetic should go `4/6 -> 6/6`, promoting `overlay(flip_h, nonzero_mask)` in round 1
- ARC is scaffolding only; even `0/10` is acceptable while the loop is still minimal
