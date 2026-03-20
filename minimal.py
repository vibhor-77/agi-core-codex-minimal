import json, os, sys
from pathlib import Path

"""Minimal 4-pillar loop: score attempts, keep frontiers, promote useful subtrees, repeat."""

def flip_h(g): return [r[::-1] for r in g]
def flip_v(g): return g[::-1]
def transpose(g): return [list(r) for r in zip(*g)]
def nonzero_mask(g): return [[1 if v else 0 for v in r] for r in g]
def overlay(a, b): return [] if not a or len(a) != len(b) or len(a[0]) != len(b[0]) else [[y if y else x for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]

def box(g):
    pts = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v]
    if not pts: return None
    rs, cs = zip(*pts)
    return min(rs), max(rs) + 1, min(cs), max(cs) + 1

def crop(g, b):
    r0, r1, c0, c1 = b
    return [row[c0:c1] for row in g[r0:r1]]

def paste(g, patch, b):
    r0, r1, c0, c1 = b
    if not patch or len(patch) != r1 - r0 or len(patch[0]) != c1 - c0: return []
    out = [row[:] for row in g]
    for r, row in enumerate(patch, r0): out[r][c0:c1] = row
    return out

SEEDS = {"identity": lambda g: g, "flip_h": flip_h, "flip_v": flip_v, "transpose": transpose, "nonzero_mask": nonzero_mask}

def run(p, g, atoms):
    if isinstance(p, str): return atoms[p](g)
    op, a, b = p
    if op == "chain": return run(b, run(a, g, atoms), atoms)
    if op == "overlay": return overlay(run(a, g, atoms), run(b, g, atoms))
    mask = run(a, g, atoms)
    if not mask or len(mask) != len(g) or len(mask[0]) != len(g[0]): return []
    b0 = box(mask)
    return g if not b0 else paste(g, run(b, crop(g, b0), atoms), b0)

def show(p): return p if isinstance(p, str) else f"{p[0]}({show(p[1])}, {show(p[2])})"
def size(p): return 1 if isinstance(p, str) else 1 + size(p[1]) + size(p[2])
def leaves(p): return [p] if isinstance(p, str) else leaves(p[1]) + leaves(p[2])
def subtrees(p): return [] if isinstance(p, str) else [p, *subtrees(p[1]), *subtrees(p[2])]

def score(p, examples, atoms):
    total = hits = 0
    for inp, out in examples:
        got = run(p, inp, atoms)
        if not got or len(got) != len(out) or len(got[0]) != len(out[0]): continue
        total += sum(len(r) for r in out)
        hits += sum(a == b for ra, rb in zip(got, out) for a, b in zip(ra, rb))
    return hits / total if total else 0.0

def signature(p, inputs, atoms): return tuple(tuple(tuple(row) for row in run(p, g, atoms)) for g in inputs)
def utility(p, useful): return sum(useful.get(n, 1.0) for n in leaves(p))

def candidates(atoms, useful, width=8):
    names = sorted(atoms, key=lambda n: (-useful.get(n, 1.0), n))[:width]
    for n in names: yield n
    for a in names:
        for b in names:
            yield ("chain", a, b)
            yield ("overlay", a, b)
            yield ("local", a, b)

def frontier(task, atoms, useful, keep=3, old=()):
    pool = list(old) + list(candidates(atoms, useful))
    ranked = sorted(((score(p, task["train"], atoms), p) for p in pool), key=lambda item: (-item[0], size(item[1]), -utility(item[1], useful), show(item[1])))
    seen, chosen = set(), []
    for quality, program in ranked:
        name = show(program)
        if name not in seen:
            chosen.append((quality, program))
            seen.add(name)
        if len(chosen) == keep: break
    return chosen

def promote(improved, atoms, useful, inputs, threshold=1.5, limit=4):
    seen = {signature(name, inputs, atoms) for name in atoms}
    weights = {}
    for delta, leader in improved.values():
        for subtree in subtrees(leader):
            if size(subtree) > 1: weights[subtree] = weights.get(subtree, 0.0) + delta
    ranked = sorted(weights.items(), key=lambda item: (-item[1], size(item[0]), show(item[0])))
    new = []
    for program, gain in ranked:
        name, sig = show(program), signature(program, inputs, atoms)
        if gain >= threshold and sig not in seen:
            atoms[name] = lambda g, p=program: run(p, g, atoms)
            useful[name] = useful.get(name, 1.0) + gain
            seen.add(sig)
            new.append((name, gain))
        if len(new) == limit: break
    return new, ranked

def sample_inputs(tasks, limit=8):
    grids = []
    for task in tasks.values():
        for inp, _ in task["train"]:
            grids.append(inp)
            if len(grids) == limit: return grids
    return grids

def evaluate(label, tasks, atoms, useful, keep=3):
    solved = 0
    mean = 0.0
    for task in tasks.values():
        best = frontier(task, atoms, useful, keep)[0][1]
        test_score = score(best, task["test"], atoms)
        solved += test_score == 1.0
        mean += test_score
    print(f"{label}: {solved}/{len(tasks)} exact, mean test {mean / len(tasks):.3f}")

def learn(label, tasks, eval_tasks=None, rounds=3, keep=3):
    atoms, useful, fronts, inputs = SEEDS.copy(), {name: 1.0 for name in SEEDS}, {}, sample_inputs(tasks)
    solved_before = set()
    for round_id in range(1, rounds + 1):
        improved = {}
        for task_id, task in tasks.items():
            before = fronts.get(task_id, [(0.0, "identity")])[0][0]
            fronts[task_id] = frontier(task, atoms, useful, keep, [p for _, p in fronts.get(task_id, [])])
            after, leader = fronts[task_id][0]
            if after > before: improved[task_id] = (after - before, leader)
        for delta, leader in improved.values():
            for name in leaves(leader): useful[name] = useful.get(name, 1.0) + delta
        new, ranked = promote(improved, atoms, useful, inputs)
        solved = sorted(task_id for task_id, items in fronts.items() if items[0][0] == 1.0)
        fresh = sorted(set(solved) - solved_before)
        solved_before = set(solved)
        mean = sum(items[0][0] for items in fronts.values()) / len(tasks)
        reused = sorted({name for task_id in solved for name in leaves(fronts[task_id][0][1]) if name not in SEEDS})
        print(f"{label} round {round_id}: {len(solved)}/{len(tasks)} solved, mean train {mean:.3f}")
        print("newly solved:", fresh[:12], "..." if len(fresh) > 12 else "")
        print("new primitives:", [f"{name} ({gain:.2f})" for name, gain in new[:8]])
        print("reused library:", reused[:8])
        print("top subtree gains:", [f"{show(program)} ({gain:.2f})" for program, gain in ranked[:5]])
        print("library size:", len(atoms) - len(SEEDS))
        if eval_tasks and round_id in {1, rounds}: evaluate(f"public eval after round {round_id}", eval_tasks, atoms, useful, keep)
    return atoms, useful

def task(grid, program): return {"train": [(grid, run(program, grid, SEEDS))], "test": [(grid, run(program, grid, SEEDS))]}

def synthetic():
    pair = ("overlay", "flip_h", "nonzero_mask")
    return {
        "flip_h": task([[1, 0], [0, 0]], "flip_h"),
        "mask": task([[1, 0, 2], [0, 3, 0], [4, 0, 0]], "nonzero_mask"),
        "pair_1": task([[1, 0, 0], [0, 2, 3]], pair),
        "pair_2": task([[0, 4, 5], [6, 0, 0]], pair),
        "hard_1": task([[1, 2, 0], [0, 0, 3]], ("chain", pair, "flip_h")),
        "hard_2": task([[1, 0, 2], [3, 0, 0]], ("chain", pair, "transpose")),
    }

def arc_root():
    here = Path(__file__).resolve().parent
    checked = [os.getenv("ARC_AGI_1_TRAIN_DIR"), here / "data/ARC-AGI/data/training", here / "../agi-core/data/ARC-AGI/data/training"]
    root = next((Path(path) for path in checked if path and Path(path).is_dir()), None)
    if root: return root
    print("missing ARC dataset; checked:")
    for path in checked: print(f"- {path or '(unset) ARC_AGI_1_TRAIN_DIR'}")
    raise SystemExit(1)

def load(path):
    data = json.loads(path.read_text())
    return {"train": [(e["input"], e["output"]) for e in data["train"]], "test": [(e["input"], e["output"]) for e in data["test"]]}

def arc(split):
    root = arc_root()
    base = root if split == "training" else root.parent / split
    return {path.stem: load(path) for path in sorted(base.glob("*.json"))}

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode in {"synthetic", "both"}: learn("synthetic", synthetic(), rounds=2)
    if mode in {"arc", "both"}: learn("arc train", arc("training"), arc("evaluation"))
