import json, os, sys
from pathlib import Path

"""Minimal 4-pillar loop: score, keep a small frontier, promote useful subtrees, repeat."""

def flip_h(grid): return [row[::-1] for row in grid]
def flip_v(grid): return grid[::-1]
def transpose(grid): return [list(row) for row in zip(*grid)]
def nonzero_mask(grid): return [[1 if value else 0 for value in row] for row in grid]
def overlay(left, right):
    if len(left) != len(right) or len(left[0]) != len(right[0]): return []
    return [[b if b else a for a, b in zip(row_a, row_b)] for row_a, row_b in zip(left, right)]

SEEDS = {"identity": lambda g: g, "flip_h": flip_h, "flip_v": flip_v, "transpose": transpose, "nonzero_mask": nonzero_mask}
PROBES = [[[1, 0], [0, 2]], [[0, 1, 0], [2, 0, 0]]]

def run(program, grid, atoms):
    if isinstance(program, str): return atoms[program](grid)
    op, left, right = program
    return run(right, run(left, grid, atoms), atoms) if op == "chain" else overlay(run(left, grid, atoms), run(right, grid, atoms))

def show(program):
    if isinstance(program, str): return program
    op, left, right = program
    return f"{op}({show(left)}, {show(right)})"

def size(program):
    if isinstance(program, str): return 1
    _, left, right = program
    return 1 + size(left) + size(right)

def leaves(program):
    if isinstance(program, str): return [program]
    _, left, right = program
    return leaves(left) + leaves(right)

def subtrees(program):
    if isinstance(program, str): return []
    _, left, right = program
    return [program, *subtrees(left), *subtrees(right)]

def quality(program, examples, atoms):
    def cell_score(inp, out):
        got = run(program, inp, atoms)
        if len(got) != len(out) or len(got[0]) != len(out[0]): return 0.0
        total = sum(len(row) for row in out)
        return sum(a == b for row_a, row_b in zip(got, out) for a, b in zip(row_a, row_b)) / total
    return sum(cell_score(inp, out) for inp, out in examples) / len(examples)

def utility(program, usefulness):
    return sum(usefulness.get(name, 1.0) for name in leaves(program))

def candidates(atoms, usefulness):
    names = sorted(atoms, key=lambda name: (-usefulness.get(name, 1.0), name))
    for name in names: yield name
    for left in names:
        for right in names:
            yield ("chain", left, right)
            yield ("overlay", left, right)

def frontier_for(examples, atoms, usefulness, keep, old=()):
    pool = list(old) + list(candidates(atoms, usefulness))
    ranked = sorted(((quality(program, examples, atoms), program) for program in pool), key=lambda item: (-item[0], size(item[1]), -utility(item[1], usefulness), show(item[1])))
    chosen, seen = [], set()
    for score, program in ranked:
        if show(program) not in seen:
            chosen.append((score, program)); seen.add(show(program))
        if len(chosen) == keep: return chosen
    return chosen

def novel(program, atoms):
    outputs = [run(program, probe, atoms) for probe in PROBES]
    return all(outputs != [run(name, probe, atoms) for probe in PROBES] for name in atoms)

def promote(frontier, atoms, threshold=1.5):
    weights = {}
    for programs in frontier.values():
        score, program = programs[0]
        for subtree in subtrees(program):
            if size(subtree) > 1:
                weights[subtree] = weights.get(subtree, 0.0) + score
    ranked = sorted(weights.items(), key=lambda item: (-item[1], show(item[0])))
    new = []
    for program, weight in ranked:
        name = show(program)
        if weight >= threshold and name not in atoms and novel(program, atoms):
            atoms[name] = lambda grid, p=program: run(p, grid, atoms)
            new.append((name, weight))
    return new, ranked

def learn(label, tasks, rounds=2, keep=3):
    atoms, usefulness, frontier = SEEDS.copy(), {name: 1.0 for name in SEEDS}, {}
    for round_number in range(1, rounds + 1):
        improved = {}
        for task_id, examples in tasks.items():
            best_before = frontier.get(task_id, [(0.0, "identity")])[0][0]
            frontier[task_id] = frontier_for(examples, atoms, usefulness, keep, [program for _, program in frontier.get(task_id, [])])
            if frontier[task_id][0][0] > best_before: improved[task_id] = frontier[task_id][0]
        for score, program in improved.values():
            for name in leaves(program): usefulness[name] = usefulness.get(name, 1.0) + score
        new, ranked = promote(frontier, atoms)
        for name, weight in new: usefulness[name] = weight
        solved = sorted(task_id for task_id, programs in frontier.items() if programs[0][0] == 1.0)
        near = sum(0 < programs[0][0] < 1 for programs in frontier.values())
        print(f"{label} round {round_number}: {len(solved)}/{len(tasks)} solved, {near} near-misses")
        print("improved:", [f"{task_id}: {score:.2f} via {show(program)}" for task_id, (score, program) in improved.items()])
        print("solved programs:", [f"{task_id}: {show(frontier[task_id][0][1])}" for task_id in solved])
        print("new primitives:", [f"{name} ({weight:.2f})" for name, weight in new])
        print("top subtree candidates:", [f"{show(program)} ({weight:.2f})" for program, weight in ranked[:3]])
        print("library:", [name for name in atoms if name not in SEEDS])

def task(grid, program): return [(grid, run(program, grid, SEEDS))]

def synthetic_tasks():
    pair = ("overlay", "flip_h", "nonzero_mask")
    return {
        "flip_h": task([[1, 0], [0, 0]], "flip_h"),
        "mask": task([[1, 0, 2], [0, 3, 0], [4, 0, 0]], "nonzero_mask"),
        "pair_1": task([[1, 0, 0], [0, 2, 3]], pair),
        "pair_2": task([[0, 4, 5], [6, 0, 0]], pair),
        "hard_1": task([[1, 2, 0], [0, 0, 3]], ("chain", pair, "flip_h")),
        "hard_2": task([[1, 0, 2], [3, 0, 0]], ("chain", pair, "transpose")),
    }

def arc_tasks(limit=10):
    here = Path(__file__).resolve().parent
    checked = [os.getenv("ARC_AGI_1_TRAIN_DIR"), here / "data/ARC-AGI/data/training", here / "../agi-core/data/ARC-AGI/data/training"]
    root = next((Path(path) for path in checked if path and Path(path).is_dir()), None)
    if not root:
        print("missing ARC dataset; checked:")
        for path in checked: print(f"- {path or '(unset) ARC_AGI_1_TRAIN_DIR'}")
        raise SystemExit(1)
    files = sorted(root.glob("*.json"))[:limit]
    return {file.stem: [(e["input"], e["output"]) for e in json.loads(file.read_text())["train"]] for file in files}

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode in ("synthetic", "both"): learn("synthetic", synthetic_tasks())
    if mode in ("arc", "both"): learn("arc", arc_tasks())
