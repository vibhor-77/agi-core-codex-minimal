# agi-core-codex-minimal
Tiny 4-pillar scaffolding: feedback scores every attempt, approximability keeps the best near-miss per task, abstraction promotes recurring length-2 subprograms, and exploration searches compositions of the current primitive set.
ARC expects training JSONs at `ARC_AGI_1_TRAIN_DIR`, `data/ARC-AGI/data/training`, or `../agi-core/data/ARC-AGI/data/training`.
Run:
- `python minimal.py synthetic`
- `python minimal.py arc`
- `python minimal.py both`
Expected:
- synthetic should promote a shared pair in round 1 and solve more in round 2
- ARC is scaffolding only; even `0/10` is acceptable while the loop is still minimal
