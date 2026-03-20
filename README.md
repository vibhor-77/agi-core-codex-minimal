# agi-core-codex-minimal
Tiny 4-pillar scaffolding: feedback scores attempts, approximability keeps better partials, abstraction stores improving programs, and exploration grows the library by one primitive per round.
ARC expects training JSONs at `ARC_AGI_1_TRAIN_DIR`, `data/ARC-AGI/data/training`, or `../agi-core/data/ARC-AGI/data/training`.
Run:
- `python minimal.py synthetic`
- `python minimal.py arc`
- `python minimal.py both`
Expected:
- synthetic should improve from round 1 to round 2
- ARC is scaffolding only and may solve `0/10`
