import json
import os
import sys
from pathlib import Path


ARC_TASK_IDS = [
    "1cf80156",
    "67a3c6ac",
    "68b16354",
    "74dd1130",
    "9dfd6313",
    "28bf18c6",
    "4c4377d9",
    "6d0aefbc",
    "6fa7a44f",
    "7468f01a",
]


def flip_h(grid):
    return [row[::-1] for row in grid]


def flip_v(grid):
    return grid[::-1]


def transpose(grid):
    return [list(row) for row in zip(*grid)]


def crop_support(grid):
    points = [(i, j) for i, row in enumerate(grid) for j, value in enumerate(row) if value]
    if not points:
        return [[0]]
    rows = [i for i, _ in points]
    cols = [j for _, j in points]
    top, bottom = min(rows), max(rows) + 1
    left, right = min(cols), max(cols) + 1
    return [row[left:right] for row in grid[top:bottom]]


SEEDS = {
    "identity": lambda g: g,
    "flip_h": flip_h,
    "flip_v": flip_v,
    "transpose": transpose,
    "crop_support": crop_support,
}


def overlay(left, right):
    return [[b or a for a, b in zip(row_a, row_b)] for row_a, row_b in zip(left, right)]


COMPOSITORS = {
    "overlay": overlay,
    "hcat": lambda a, b: [row_a + row_b for row_a, row_b in zip(a, b)],
    "vcat": lambda a, b: a + b,
}


def run(program, grid):
    if isinstance(program, str):
        return SEEDS[program](grid)
    op, left, right = program
    if op == "chain":
        return run(right, run(left, grid))
    return COMPOSITORS[op](run(left, grid), run(right, grid))


def show(program):
    if isinstance(program, str):
        return program
    op, left, right = program
    return f"({op} {show(left)} {show(right)})"


def complexity(program):
    if isinstance(program, str):
        return 1
    _, left, right = program
    return 1 + complexity(left) + complexity(right)


def subprograms(program):
    if isinstance(program, str):
        return []
    _, left, right = program
    return [program, *subprograms(left), *subprograms(right)]


def cell_accuracy(predicted, expected):
    if len(predicted) != len(expected) or len(predicted[0]) != len(expected[0]):
        return 0.0
    cells = sum(len(row) for row in expected)
    correct = sum(a == b for row_a, row_b in zip(predicted, expected) for a, b in zip(row_a, row_b))
    return correct / cells


def task_score(program, task):
    examples = task["train"] + task["test"]
    pairs = [(run(program, ex["input"]), ex["output"]) for ex in examples]
    exact = all(predicted == expected for predicted, expected in pairs)
    mean_accuracy = sum(cell_accuracy(predicted, expected) for predicted, expected in pairs) / len(pairs)
    return exact, mean_accuracy, complexity(program)


def synthetic_tasks():
    programs = {
        "s1": "flip_h",
        "s2": "flip_v",
        "s3": "transpose",
        "s4": ("chain", "crop_support", "flip_h"),
        "s5": ("vcat", "flip_v", "crop_support"),
        "s6": ("hcat", "crop_support", "flip_h"),
    }
    inputs = {
        "s1": [[1, 0], [0, 0]],
        "s2": [[0, 2, 0], [0, 0, 0]],
        "s3": [[0, 0, 3], [0, 0, 0]],
        "s4": [[0, 1, 0], [0, 2, 3], [0, 0, 0]],
        "s5": [[0, 4, 0], [0, 0, 0]],
        "s6": [[0, 0, 5], [0, 0, 0]],
    }
    tasks = {}
    for task_id, program in programs.items():
        train_input = inputs[task_id]
        test_input = flip_h(train_input)
        tasks[task_id] = {
            "train": [{"input": train_input, "output": run(program, train_input)}],
            "test": [{"input": test_input, "output": run(program, test_input)}],
        }
    return tasks


def find_arc_dir():
    here = Path(__file__).resolve().parent
    checked = [
        os.getenv("ARC_AGI_1_TRAIN_DIR"),
        str(here / "data/ARC-AGI/data/training"),
        str(here / "../agi-core/data/ARC-AGI/data/training"),
    ]
    found = next((path for path in checked if path and Path(path).is_dir()), None)
    if found:
        return Path(found)
    print("missing ARC dataset; checked:")
    for path in checked:
        print(f"- {path or '(unset) ARC_AGI_1_TRAIN_DIR'}")
    raise SystemExit(1)


def arc_tasks():
    root = find_arc_dir()
    return {task_id: json.loads((root / f"{task_id}.json").read_text()) for task_id in ARC_TASK_IDS}


def round_two_candidates(library):
    candidates = list(SEEDS)
    pool = list(SEEDS) + library
    for left in pool:
        for right in pool:
            if left not in library and right not in library:
                continue
            candidates.append(("chain", left, right))
            candidates.append(("overlay", left, right))
            candidates.append(("hcat", left, right))
            candidates.append(("vcat", left, right))
    return candidates


def promote(library, solved_programs):
    for program in solved_programs:
        for candidate in [program, *subprograms(program)]:
            name = show(candidate)
            if any(show(existing) == name for existing in library):
                continue
            if isinstance(candidate, str) and candidate not in solved_programs:
                continue
            library.append(candidate)


def solve(name, tasks):
    library = []
    solved = set()
    for round_number in (1, 2):
        candidates = list(SEEDS) if round_number == 1 else round_two_candidates(library)
        winners = {}
        for task_id, task in tasks.items():
            ranked = sorted(
                ((task_score(program, task), program) for program in candidates),
                key=lambda item: (-item[0][0], -item[0][1], item[0][2], show(item[1])),
            )
            if ranked[0][0][0]:
                winners[task_id] = ranked[0][1]
        new_task_ids = sorted(task_id for task_id in winners if task_id not in solved)
        solved.update(new_task_ids)
        promote(library, [winners[task_id] for task_id in new_task_ids])
        print(
            f"{name} round {round_number}: {len(solved)}/{len(tasks)} "
            f"new={new_task_ids} lib={[show(program) for program in library]}"
        )


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode in ("synthetic", "both"):
        solve("synthetic", synthetic_tasks())
    if mode in ("arc", "both"):
        solve("arc", arc_tasks())


if __name__ == "__main__":
    main()
