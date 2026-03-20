# agi-core-codex-minimal
Tiny 4-pillar scaffolding: feedback scores every attempt, approximability keeps a small diverse frontier per task, abstraction keeps an evolving population of repeated useful subtrees, and exploration breeds new programs from a tiny typed language with whole-grid transforms, one perception primitive, and one local writeback operator.
ARC expects JSONs at `ARC_AGI_1_TRAIN_DIR`, `data/ARC-AGI/data/training`, or `../agi-core/data/ARC-AGI/data/training`, and scores the matching public eval split at fixed milestones without learning from it.
Run:
- `python minimal.py synthetic`
- `python minimal.py arc`
- `python minimal.py both`
Expected:
- synthetic now runs a 4-stage branching curriculum plus a frozen choice probe:
  - stage 1 learns `local(nonzero_mask, flip_h)`
  - stage 2 learns two sibling abstractions, `chain(local(nonzero_mask, flip_h), transpose)` and `chain(flip_v, local(nonzero_mask, flip_h))`
  - stage 3 grows one descendant from each branch
  - stage 4 grows one deeper descendant from each branch again
  - the frozen `synthetic choice` probe mixes branch-specific tasks and checks that the learned population picks the right branch without changing the library
- synthetic stages 3 and 4 both show `critical_library_solves=4`; stage 4 shows `ablation_breaks=4`, and the frozen choice probe shows `library_solves=4` and `critical_library_solves=4`
- causal ownership is now assigned to the outermost reused abstraction in a winning program, so the stage-2 pair owns only stage 2, while later branch abstractions own their downstream stage-3/stage-4/choice tasks
- the library is ranked by unique causal coverage over solved tasks, which now makes the branch descendants visibly survive for different task sets instead of inheriting credit through the shared root
- ARC currently goes `8/400` on train exact and `0/400` on public eval exact, with mean public-eval score about `0.507`; the current surviving abstractions have distinct exact-task and near-miss ownership, and the overlap readout is empty after the causal accounting fix
- the current ARC population collapses to two causally useful abstractions:
  - `chain(local(nonzero_mask, transpose), local(nonzero_mask, flip_v))`
  - `chain(local(nonzero_mask, flip_h), local(nonzero_mask, flip_v))`
  Together they cover the two exact train solves `ed36ccf7` and `3c9b0459`, and they split the remaining near-miss help across `4` unsolved tasks instead of one abstraction free-riding on the other
- each round now prints fresh solved programs, critical solved programs, and compounding metrics including lineage depth, new population count, primitive-equivalent rejections, library solves, critical library solves, average library-attributed gain, counterfactual drop, average solved-task coverage, average unsolved-task help coverage, union coverage, union help, average criticality, average impact, survivor count, average reuse, pool-per-solve, no-library ablation, a `top population` summary ranked by causal necessity and unique coverage, and the top overlap pairs
