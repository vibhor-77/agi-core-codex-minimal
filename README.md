# agi-core-codex-minimal

Tiny 4-pillar proof: start from a few generic grid ops, solve easy tasks in round 1, store exact reusable programs, then solve harder tasks in round 2 by composing what was learned.

Assumes ARC-AGI-1 training data is already cloned and discoverable at `ARC_AGI_1_TRAIN_DIR`, `data/ARC-AGI/data/training`, or `../agi-core/data/ARC-AGI/data/training`.

Run:
- `python minimal.py synthetic`
- `python minimal.py arc`
- `python minimal.py both`

Expected:
- synthetic: round 1 `3/6`, round 2 `6/6`
- arc smoke: round 1 `5/10`, round 2 `10/10`
