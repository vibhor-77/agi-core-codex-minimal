import itertools, json, os, sys
from pathlib import Path

"""Minimal 4-pillar loop: explore candidates, score near-misses, promote recurring pairs, repeat."""

def crop_support(grid):
    cells = [(i, j) for i, row in enumerate(grid) for j, v in enumerate(row) if v]
    if not cells: return [[0]]
    rows, cols = zip(*cells)
    return [row[min(cols):max(cols) + 1] for row in grid[min(rows):max(rows) + 1]]

SEEDS = {
    "identity": lambda g: g,
    "flip_h": lambda g: [row[::-1] for row in g],
    "flip_v": lambda g: g[::-1],
    "transpose": lambda g: [list(row) for row in zip(*g)],
    "crop_support": crop_support,
}
PROBES = [[[1, 0], [0, 2]], [[0, 1, 0], [2, 0, 0]]]

def run(program, grid, primitives):
    for name in program: grid = primitives[name](grid)
    return grid

def show(program):
    return " -> ".join(program)

def quality(program, examples, primitives):
    def cell_score(inp, out):
        pred = run(program, inp, primitives)
        if len(pred) != len(out) or len(pred[0]) != len(out[0]): return 0.0
        total = sum(len(row) for row in out)
        return sum(a == b for ra, rb in zip(pred, out) for a, b in zip(ra, rb)) / total
    return sum(cell_score(inp, out) for inp, out in examples) / len(examples)

def candidates(primitives, usefulness, depth=2):
    names = sorted(primitives, key=lambda name: (-usefulness.get(name, 1.0), name))
    for d in range(1, depth + 1): yield from itertools.product(names, repeat=d)

def novel(pair, primitives):
    outputs = [run(pair, probe, primitives) for probe in PROBES]
    return all(outputs != [run((name,), probe, primitives) for probe in PROBES] for name in primitives)

def pair_weights(frontier):
    weights = {}
    for score, program in frontier.values():
        for i in range(len(program) - 1):
            pair = program[i:i + 2]
            weights[pair] = weights.get(pair, 0.0) + score
    return sorted(weights.items(), key=lambda item: (-item[1], item[0]))

def abstract(frontier, primitives, threshold=1.5):
    new = []
    ranked = pair_weights(frontier)
    for pair, weight in ranked:
        name = "+".join(pair)
        if weight >= threshold and name not in primitives and novel(pair, primitives):
            primitives[name] = lambda g, p=pair: run(p, g, primitives)
            new.append((name, weight))
    return new, ranked

def learn(label, tasks, rounds=2):
    primitives, usefulness, frontier = SEEDS.copy(), {name: 1.0 for name in SEEDS}, {}
    for round_number in range(1, rounds + 1):
        improved = {}
        for task_id, examples in tasks.items():
            program = min(
                candidates(primitives, usefulness),
                key=lambda p: (-quality(p, examples, primitives), len(p), -sum(usefulness.get(op, 1.0) for op in p), p),
            )
            score = quality(program, examples, primitives)
            if task_id not in frontier or score > frontier[task_id][0]:
                frontier[task_id] = (score, program)
                improved[task_id] = (score, program)
        for task_id in improved:
            score, program = frontier[task_id]
            for op in program: usefulness[op] = usefulness.get(op, 1.0) + score
        new, ranked = abstract(frontier, primitives)
        for primitive_name, weight in new: usefulness[primitive_name] = weight
        solved = sorted(task_id for task_id, (score, _) in frontier.items() if score == 1.0)
        near = sum(0 < score < 1 for score, _ in frontier.values())
        top_pairs = [f"{show(pair)} ({weight:.2f})" for pair, weight in ranked[:3]]
        print(f"{label} round {round_number}: {len(solved)}/{len(tasks)} solved, {near} near-misses")
        print("improved:", [f"{task_id}: {score:.2f} via {show(program)}" for task_id, (score, program) in improved.items()])
        print("solved programs:", [f"{task_id}: {show(frontier[task_id][1])}" for task_id in solved])
        print("new primitives:", [f"{primitive_name} ({weight:.2f})" for primitive_name, weight in new])
        print("top pair candidates:", top_pairs)
        print("library:", [primitive_name for primitive_name in primitives if primitive_name not in SEEDS])

def make_task(grid, program):
    return [(grid, run(program, grid, SEEDS))]

def synthetic_tasks():
    shared = ("transpose", "flip_v", "flip_h")
    return {
        "flip_h": make_task([[1, 0], [0, 0]], ("flip_h",)),
        "transpose": make_task([[0, 0, 2], [0, 0, 0]], ("transpose",)),
        "hard_1": make_task([[2, 2, 0], [0, 0, 1], [2, 2, 1]], shared),
        "hard_2": make_task([[1, 1, 0], [0, 0, 1], [1, 1, 2]], shared),
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
