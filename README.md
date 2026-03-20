# agi-core-codex-minimal
Tiny 4-pillar proof: try tiny seeds, remember exact solves, then reuse them in round 2.
ARC expects training JSONs at `ARC_AGI_1_TRAIN_DIR`, `data/ARC-AGI/data/training`, or `../agi-core/data/ARC-AGI/data/training`.
Run:
- `python minimal.py synthetic`
- `python minimal.py arc`
- `python minimal.py both`
Expected:
- synthetic: `3/6` then `6/6`
- arc smoke: `5/10` then `10/10`
