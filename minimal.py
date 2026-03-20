import json, os, sys
from pathlib import Path

"""Minimal 4-pillar loop over a tiny program language and an evolving abstraction population."""

def flip_h(g): return [r[::-1] for r in g]
def flip_v(g): return g[::-1]
def transpose(g): return [list(r) for r in zip(*g)]
def nonzero_mask(g): return [[1 if v else 0 for v in r] for r in g]
PRIMS = {"identity": lambda g: g, "flip_h": flip_h, "flip_v": flip_v, "transpose": transpose, "nonzero_mask": nonzero_mask}

def run(p, g):
    return PRIMS[p](g) if isinstance(p, str) else run(p[2], run(p[1], g))

def show(p): return p if isinstance(p, str) else f"chain({show(p[1])}, {show(p[2])})"
def size(p): return 1 if isinstance(p, str) else 1 + size(p[1]) + size(p[2])
def subtrees(p): return [] if isinstance(p, str) else [p, *subtrees(p[1]), *subtrees(p[2])]

def score(p, examples):
    total = hits = 0
    for inp, out in examples:
        got = run(p, inp)
        if len(got) != len(out) or len(got[0]) != len(out[0]): continue
        total += sum(len(r) for r in out)
        hits += sum(a == b for ra, rb in zip(got, out) for a, b in zip(ra, rb))
    return hits / total if total else 0.0

def signature(p, inputs):
    return tuple(tuple(tuple(row) for row in run(p, grid)) for grid in inputs)

def unique(programs):
    seen, out = set(), []
    for p in programs:
        name = show(p)
        if name not in seen:
            out.append(p)
            seen.add(name)
    return out

def spawn(library, leaders, width=6, max_size=7):
    ranked = list(PRIMS) + [e["program"] for e in sorted(library.values(), key=lambda e: (-e["reuse"], -e["gain"], show(e["program"])))] + leaders
    basis = unique(ranked)[:width]
    pool = basis[:]
    for p in basis:
        for q in PRIMS:
            pool += [("chain", q, p), ("chain", p, q)]
    for a in basis[:4]:
        for b in basis[:4]:
            if show(a) != show(b): pool.append(("chain", a, b))
    return [p for p in unique(pool) if size(p) <= max_size]

def frontier(task, pool, keep=3, old=()):
    ranked = sorted(((score(p, task["train"]), p) for p in unique(list(old) + pool)), key=lambda item: (-item[0], size(item[1]), show(item[1])))
    return ranked[:keep]

def evolve(library, improved, inputs, cap=8):
    for entry in library.values():
        entry["age"] += 1
        entry["used"] = False
    gains = {}
    for delta, leader in improved.values():
        trees = {show(t): t for t in subtrees(leader)}
        for name, tree in trees.items():
            item = gains.setdefault(name, {"program": tree, "gain": 0.0, "support": 0})
            item["gain"] += delta
            item["support"] += 1
        used = set(trees)
        for name, entry in library.items():
            if name in used:
                entry["reuse"] += 1
                entry["used"] = True
                entry["age"] = 0
    seen = {signature(entry["program"], inputs) for entry in library.values()}
    ranked = sorted(gains.values(), key=lambda item: (-item["support"], -item["gain"], size(item["program"]), show(item["program"])))
    new = []
    for item in ranked:
        name, sig = show(item["program"]), signature(item["program"], inputs)
        if item["support"] > 1 and sig not in seen and name not in library:
            library[name] = {"program": item["program"], "gain": item["gain"], "support": item["support"], "reuse": 0, "age": 0, "used": False}
            seen.add(sig)
            new.append(item)
    keep = sorted(library.values(), key=lambda e: (-e["reuse"], -e["support"], -e["gain"], e["age"], size(e["program"]), show(e["program"])))[:cap]
    library = {show(e["program"]): e for e in keep}
    return library, new[:cap], ranked[:5]

def sample_inputs(tasks, limit=8):
    grids = []
    for task in tasks.values():
        for inp, _ in task["train"]:
            grids.append(inp)
            if len(grids) == limit: return grids
    return grids

def evaluate(label, tasks, library):
    pool = spawn(library, [])
    exact = mean = 0.0
    for task in tasks.values():
        best = frontier(task, pool, keep=1)[0][1]
        s = score(best, task["test"])
        exact += s == 1.0
        mean += s
    print(f"{label}: {int(exact)}/{len(tasks)} exact, mean test {mean / len(tasks):.3f}")

def learn(label, tasks, eval_tasks=None, rounds=3, keep=3):
    library, fronts, inputs, solved_before = {}, {}, sample_inputs(tasks), set()
    for r in range(1, rounds + 1):
        leaders = [items[0][1] for items in fronts.values()] if fronts else []
        pool = spawn(library, leaders)
        improved = {}
        for task_id, task in tasks.items():
            before = fronts.get(task_id, [(0.0, "identity")])[0][0]
            fronts[task_id] = frontier(task, pool, keep, [p for _, p in fronts.get(task_id, [])])
            after, leader = fronts[task_id][0]
            if after > before: improved[task_id] = (after - before, leader)
        library, new, top = evolve(library, improved, inputs)
        solved = sorted(task_id for task_id, items in fronts.items() if items[0][0] == 1.0)
        fresh = sorted(set(solved) - solved_before)
        solved_before = set(solved)
        mean = sum(items[0][0] for items in fronts.values()) / len(tasks)
        reused = sorted(name for name, entry in library.items() if entry["used"])
        print(f"{label} round {r}: {len(solved)}/{len(tasks)} solved, mean train {mean:.3f}, pool {len(pool)}")
        print("newly solved:", fresh[:12], "..." if len(fresh) > 12 else "")
        print("new abstractions:", [f"{show(item['program'])} s{item['support']} g{item['gain']:.2f}" for item in new])
        print("reused abstractions:", reused[:8])
        print("top candidates:", [f"{show(item['program'])} s{item['support']} g{item['gain']:.2f}" for item in top])
        print("population:", [name for name in library])
        if eval_tasks and r in {1, rounds}: evaluate(f"public eval after round {r}", eval_tasks, library)

def task(grid, program):
    out = run(program, grid)
    return {"train": [(grid, out)], "test": [(grid, out)]}

def synthetic():
    pair = ("chain", "flip_h", "nonzero_mask")
    return {
        "flip_h": task([[1, 0], [0, 0]], "flip_h"),
        "mask": task([[1, 2, 0], [0, 0, 3]], "nonzero_mask"),
        "pair_1": task([[1, 0, 0], [2, 3, 0]], pair),
        "pair_2": task([[0, 4, 5], [6, 0, 0]], pair),
        "hard_1": task([[1, 0, 0], [0, 2, 3]], ("chain", pair, "transpose")),
        "hard_2": task([[1, 0, 2], [3, 0, 0]], ("chain", pair, "flip_v")),
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
