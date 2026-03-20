# agi-core-codex-minimal
Tiny 4-pillar scaffolding: feedback scores every attempt, approximability keeps a small frontier per task, abstraction keeps an evolving population of repeated useful subtrees, and exploration breeds new programs from a tiny typed language with whole-grid transforms, one perception primitive, and one local writeback operator.
ARC expects JSONs at `ARC_AGI_1_TRAIN_DIR`, `data/ARC-AGI/data/training`, or `../agi-core/data/ARC-AGI/data/training`, and scores the matching public eval split at fixed milestones without learning from it.
Run:
- `python minimal.py synthetic`
- `python minimal.py arc`
- `python minimal.py both`
Expected:
- synthetic now runs a 4-stage curriculum:
  - stage 1 learns `local(nonzero_mask, flip_h)`
  - stage 2 reuses it to learn `chain(local(nonzero_mask, flip_h), transpose)`
  - stage 3 reuses that to learn `chain(chain(local(nonzero_mask, flip_h), transpose), flip_h)`
  - stage 4 reuses the depth-3 abstraction to learn `local(nonzero_mask, chain(chain(local(nonzero_mask, flip_h), transpose), flip_h))`
- ARC currently goes `8/400` on train exact and `0/400` on public eval exact, with mean public-eval score about `0.507`
- each round now prints compounding metrics including lineage depth, new population count, primitive-equivalent rejections, library solves, average library-attributed gain, survivor count, average reuse, and pool-per-solve
