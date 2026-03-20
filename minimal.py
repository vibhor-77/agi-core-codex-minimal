import json, os, sys
from pathlib import Path

ARC_IDS = "1cf80156 67a3c6ac 68b16354 74dd1130 9dfd6313 28bf18c6 4c4377d9 6d0aefbc 6fa7a44f 7468f01a".split()

def flip_h(g): return [r[::-1] for r in g]
def flip_v(g): return g[::-1]
def transpose(g): return [list(r) for r in zip(*g)]
def crop_support(g):
    pts = [(i, j) for i, row in enumerate(g) for j, v in enumerate(row) if v]
    if not pts: return [[0]]
    rows, cols = zip(*pts)
    return [row[min(cols):max(cols)+1] for row in g[min(rows):max(rows)+1]]

SEEDS = {"identity": lambda g: g, "flip_h": flip_h, "flip_v": flip_v, "transpose": transpose, "crop_support": crop_support}
COMPOSE = {
    "overlay": lambda a, b: [[y or x for x, y in zip(r, s)] for r, s in zip(a, b)],
    "hcat": lambda a, b: [r + s for r, s in zip(a, b)],
    "vcat": lambda a, b: a + b,
}

def run(p, g):
    if isinstance(p, str): return SEEDS[p](g)
    op, a, b = p
    return run(b, run(a, g)) if op == "chain" else COMPOSE[op](run(a, g), run(b, g))

def show(p): return p if isinstance(p, str) else f"({p[0]} {show(p[1])} {show(p[2])})"
def subs(p): return [] if isinstance(p, str) else [p, *subs(p[1]), *subs(p[2])]
def size(p): return 1 if isinstance(p, str) else 1 + size(p[1]) + size(p[2])
def acc(a, b):
    if len(a) != len(b) or len(a[0]) != len(b[0]): return 0.0
    return sum(x == y for r, s in zip(a, b) for x, y in zip(r, s)) / sum(len(r) for r in b)
def score(p, task):
    pairs = [(run(p, ex["input"]), ex["output"]) for ex in task["train"] + task["test"]]
    return all(a == b for a, b in pairs), sum(acc(a, b) for a, b in pairs) / len(pairs), size(p)

def make_task(inp, prog):
    test = flip_h(inp)
    return {"train": [{"input": inp, "output": run(prog, inp)}], "test": [{"input": test, "output": run(prog, test)}]}

def synthetic():
    specs = {
        "s1": ([[1, 0], [0, 0]], "flip_h"),
        "s2": ([[0, 2, 0], [0, 0, 0]], "flip_v"),
        "s3": ([[0, 0, 3], [0, 0, 0]], "transpose"),
        "s4": ([[0, 1, 0], [0, 2, 3], [0, 0, 0]], ("chain", "crop_support", "flip_h")),
        "s5": ([[0, 4, 0], [0, 0, 0]], ("vcat", "flip_v", "crop_support")),
        "s6": ([[0, 0, 5], [0, 0, 0]], ("hcat", "crop_support", "flip_h")),
    }
    return {k: make_task(inp, prog) for k, (inp, prog) in specs.items()}

def arc_dir():
    here = Path(__file__).resolve().parent
    checked = [os.getenv("ARC_AGI_1_TRAIN_DIR"), here / "data/ARC-AGI/data/training", here / "../agi-core/data/ARC-AGI/data/training"]
    found = next((Path(p) for p in checked if p and Path(p).is_dir()), None)
    if found: return found
    print("missing ARC dataset; checked:"); [print(f"- {p or '(unset) ARC_AGI_1_TRAIN_DIR'}") for p in checked]; raise SystemExit(1)

def arc(): 
    root = arc_dir()
    return {task_id: json.loads((root / f"{task_id}.json").read_text()) for task_id in ARC_IDS}

def round2(lib):
    pool = list(SEEDS) + lib
    return list(SEEDS) + [(op, a, b) for op in ("chain", "overlay", "hcat", "vcat") for a in pool for b in pool if a in lib or b in lib]

def promote(lib, programs):
    seen = {show(p) for p in lib}
    for p in programs:
        for q in [p, *subs(p)]:
            if isinstance(q, str) and q != p: continue
            if show(q) not in seen: lib.append(q); seen.add(show(q))

def solve(name, tasks):
    lib, solved = [], set()
    for r in (1, 2):
        winners, cand = {}, list(SEEDS) if r == 1 else round2(lib)
        for task_id, task in tasks.items():
            ranked = sorted(((score(p, task), show(p), p) for p in cand), key=lambda x: (-x[0][0], -x[0][1], x[0][2], x[1]))
            if ranked[0][0][0]: winners[task_id] = ranked[0][2]
        new = sorted(task_id for task_id in winners if task_id not in solved)
        solved |= set(new); promote(lib, [winners[k] for k in new])
        print(f"{name} round {r}: {len(solved)}/{len(tasks)} new={new} lib={[show(p) for p in lib]}")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode in ("synthetic", "both"): solve("synthetic", synthetic())
    if mode in ("arc", "both"): solve("arc", arc())
