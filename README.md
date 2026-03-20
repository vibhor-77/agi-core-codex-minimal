# agi-core-codex-minimal
Tiny 4-pillar scaffolding: feedback scores every attempt, approximability keeps a small frontier per task, abstraction promotes subtrees with repeated marginal gain, and exploration searches a tiny deterministic tree language over seeds plus learned programs.
ARC expects JSONs at `ARC_AGI_1_TRAIN_DIR`, `data/ARC-AGI/data/training`, or `../agi-core/data/ARC-AGI/data/training`, and scores the matching public eval split at fixed milestones.
Run:
- `python minimal.py synthetic`
- `python minimal.py arc`
- `python minimal.py both`
Expected:
- synthetic should go `4/6 -> 6/6`, promoting `overlay(flip_h, nonzero_mask)` in round 1
- ARC currently goes `10/400 -> 11/400` on train exact and `1/400` on public eval while staying intentionally minimal
