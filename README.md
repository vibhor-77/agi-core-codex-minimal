# agi-core-codex-minimal
Tiny 4-pillar scaffolding: feedback scores every attempt, approximability keeps a small frontier per task, abstraction keeps an evolving population of repeated useful subtrees, and exploration breeds new programs from basic primitives plus the current population.
ARC expects JSONs at `ARC_AGI_1_TRAIN_DIR`, `data/ARC-AGI/data/training`, or `../agi-core/data/ARC-AGI/data/training`, and scores the matching public eval split at fixed milestones without learning from it.
Run:
- `python minimal.py synthetic`
- `python minimal.py arc`
- `python minimal.py both`
Expected:
- synthetic now runs a 4-stage curriculum:
  - stage 1 learns `chain(flip_h, nonzero_mask)`
  - stage 2 reuses it to learn `chain(chain(flip_h, nonzero_mask), transpose)`
  - stage 3 reuses the learned population to grow a deeper chain
  - stage 4 still reuses prior abstractions, but currently plateaus at lineage depth 3, which is useful evidence that this frozen chain-only substrate is hitting an equivalence ceiling
- ARC currently goes `8/400` on train exact and `0/400` on public eval exact, while holding mean public-eval score at about `0.527`
- each round now prints compounding metrics including lineage depth, new population count, primitive-equivalent rejections, library solves, average library-attributed gain, survivor count, average reuse, and pool-per-solve
