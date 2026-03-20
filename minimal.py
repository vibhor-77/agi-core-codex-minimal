import json, os, sys
from pathlib import Path

# 4 pillars: feedback=evaluate, approximability=accuracy, abstraction=library, exploration=expand.

def crop_support(grid):
    cells = [(i, j) for i, row in enumerate(grid) for j, value in enumerate(row) if value]
    if not cells: return [[0]]
    rows, cols = zip(*cells)
    return [row[min(cols):max(cols) + 1] for row in grid[min(rows):max(rows) + 1]]

PRIMITIVES = {
    "identity": lambda g: g,
    "flip_h": lambda g: [row[::-1] for row in g],
    "flip_v": lambda g: g[::-1],
    "transpose": lambda g: [list(row) for row in zip(*g)],
    "crop_support": crop_support,
}

def run(program, grid):
    for name in program: grid = PRIMITIVES[name](grid)
    return grid

def accuracy(predicted, expected):
    if len(predicted) != len(expected) or len(predicted[0]) != len(expected[0]): return 0.0
    total = sum(len(row) for row in expected)
    return sum(a == b for row_a, row_b in zip(predicted, expected) for a, b in zip(row_a, row_b)) / total

def evaluate(program, task):
    scores = [accuracy(run(program, ex["input"]), ex["output"]) for ex in task["train"]]
    return all(score == 1 for score in scores), sum(scores) / len(scores), -len(program)

def explore(library, round_number):
    base = [(name,) for name in PRIMITIVES]
    if round_number == 1: return base
    seen = set(base)
    for learned in library:
        for seed in base:
            seen.add(seed + learned)
            seen.add(learned + seed)
    return sorted(seen, key=lambda program: (len(program), program))

def best_program(task, candidates):
    return min(candidates, key=lambda program: (-evaluate(program, task)[0], -evaluate(program, task)[1], len(program), program))

def learn(name, tasks, rounds=2):
    library, seen = [], set()
    for round_number in range(1, rounds + 1):
        solved = {task_id: best_program(task, explore(library, round_number)) for task_id, task in tasks.items()}
        winners = {task_id: program for task_id, program in solved.items() if evaluate(program, tasks[task_id])[0]}
        new = sorted(task_id for task_id in winners if task_id not in seen)
        seen.update(winners)
        for task_id in new:
            program = winners[task_id]
            if program not in library: library.append(program)
        print(f"{name} round {round_number}: {len(winners)}/{len(tasks)} solved")
        print("new:", new)
        print("library:", [" -> ".join(program) for program in library])

def make_task(grid, program):
    return {"train": [{"input": grid, "output": run(program, grid)}]}

def synthetic_tasks():
    return {
        "flip_h": make_task([[1, 0], [0, 0]], ("flip_h",)),
        "transpose": make_task([[0, 0, 2], [0, 0, 0]], ("transpose",)),
        "crop_then_flip": make_task([[0, 1, 0], [0, 2, 3], [0, 0, 0]], ("crop_support", "flip_h")),
        "flip_then_transpose": make_task([[1, 0], [2, 0], [0, 0]], ("flip_h", "transpose")),
    }

def arc_tasks(limit=10):
    here = Path(__file__).resolve().parent
    checked = [os.getenv("ARC_AGI_1_TRAIN_DIR"), here / "data/ARC-AGI/data/training", here / "../agi-core/data/ARC-AGI/data/training"]
    root = next((Path(path) for path in checked if path and Path(path).is_dir()), None)
    if not root:
        print("missing ARC dataset; checked:"); [print(f"- {path or '(unset) ARC_AGI_1_TRAIN_DIR'}") for path in checked]; raise SystemExit(1)
    files = sorted(root.glob("*.json"))[:limit]
    return {file.stem: json.loads(file.read_text()) for file in files}

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode in ("synthetic", "both"): learn("synthetic", synthetic_tasks())
    if mode in ("arc", "both"): learn("arc", arc_tasks())
