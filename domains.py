from __future__ import annotations

"""Synthetic curriculum, ARC loading, and task inspection."""

import json
import os
from pathlib import Path

from language import (
    Col,
    Compose,
    Eq,
    Focus,
    GridExpr,
    Height,
    IDENTITY,
    ONE,
    Remap,
    Row,
    Select,
    Sub,
    Width,
    ZERO,
    NonZero,
    evaluate,
)
from learner import ARC_CONFIG, SYNTHETIC_CONFIG, Library, Task, evaluate_tasks, inspect_task, learn


def task(grid, program: GridExpr) -> Task:
    out = evaluate(program, grid)
    return Task(train=[(grid, out)], test=[(grid, out)])


def mirror_h() -> GridExpr:
    return Remap(Row(), Sub(Sub(Width(), ONE), Col()))


def transpose() -> GridExpr:
    return Remap(Col(), Row())


def last_row() -> Eq:
    return Eq(Row(), Sub(Height(), ONE))


def diagonal() -> Eq:
    return Eq(Row(), Col())


def support() -> NonZero:
    return NonZero(IDENTITY)


def synthetic_stages() -> list[tuple[str, dict[str, Task]]]:
    local_mirror = Focus(support(), mirror_h())
    branch_a = Select(diagonal(), local_mirror, IDENTITY)
    branch_b = Select(Eq(Row(), ZERO), transpose(), IDENTITY)
    mix_left = Select(Eq(Col(), ZERO), branch_b, branch_a)
    mix_last = Select(last_row(), branch_b, branch_a)
    mix_top = Select(Eq(Row(), ZERO), branch_b, branch_a)
    return [
        ("stage_1", {
            "mirror_h": task([[1, 2, 0], [3, 0, 4]], mirror_h()),
            "transpose": task([[1, 0, 2], [3, 4, 0]], transpose()),
            "local_mirror_1": task([[1, 2, 0], [3, 0, 0]], local_mirror),
            "local_mirror_2": task([[0, 4, 5], [0, 6, 0]], local_mirror),
        }),
        ("stage_2", {
            "branch_a_1": task([[1, 0, 2], [3, 4, 0], [5, 6, 7]], branch_a),
            "branch_a_2": task([[0, 1, 2, 0], [3, 4, 0, 0], [5, 6, 7, 8], [0, 0, 0, 9]], branch_a),
            "branch_b_1": task([[1, 2, 3], [4, 5, 6], [7, 8, 9]], branch_b),
            "branch_b_2": task([[1, 0, 2], [3, 4, 0], [5, 6, 7]], branch_b),
        }),
        ("stage_3", {
            "mix_left_1": task([[1, 0, 2], [3, 4, 0], [5, 6, 7]], mix_left),
            "mix_left_2": task([[1, 2, 3], [4, 5, 6], [7, 8, 9]], mix_left),
            "mix_last_1": task([[1, 2, 0], [3, 0, 0], [4, 5, 6]], mix_last),
            "mix_last_2": task([[0, 1, 2, 0], [3, 4, 0, 0], [5, 6, 7, 8], [0, 0, 0, 9]], mix_last),
        }),
        ("stage_4", {
            "mix_top_1": task([[1, 2, 3], [4, 5, 6], [7, 8, 9]], mix_top),
            "mix_top_2": task([[0, 1, 2, 0], [3, 4, 0, 0], [5, 6, 7, 8], [0, 0, 0, 9]], mix_top),
            "mix_left_3": task([[0, 1, 2], [3, 4, 0], [5, 6, 7]], mix_left),
            "mix_last_3": task([[1, 0, 2], [3, 4, 0], [5, 6, 7]], mix_last),
        }),
    ]


def synthetic_choice() -> dict[str, Task]:
    local_mirror = Focus(support(), mirror_h())
    branch_a = Select(diagonal(), local_mirror, IDENTITY)
    branch_b = Select(Eq(Row(), ZERO), transpose(), IDENTITY)
    mix_left = Select(Eq(Col(), ZERO), branch_b, branch_a)
    mix_last = Select(last_row(), branch_b, branch_a)
    return {
        "choice_left_1": task([[1, 0, 2], [3, 4, 0], [5, 6, 7]], mix_left),
        "choice_left_2": task([[1, 2, 3], [4, 5, 6], [7, 8, 9]], mix_left),
        "choice_last_1": task([[1, 2, 0], [3, 0, 0], [4, 5, 6]], mix_last),
        "choice_last_2": task([[0, 1, 2, 0], [3, 4, 0, 0], [5, 6, 7, 8], [0, 0, 0, 9]], mix_last),
    }


def run_synthetic() -> None:
    library: Library = {}
    for stage, tasks in synthetic_stages():
        library = learn(f"synthetic {stage}", tasks, rounds=1, library=library, config=SYNTHETIC_CONFIG)
        solved, mean = evaluate_tasks(tasks, library, SYNTHETIC_CONFIG)
        ablated, _ = evaluate_tasks(tasks, {}, SYNTHETIC_CONFIG)
        print(f"{stage} summary: {solved}/{len(tasks)} solved, mean {mean:.3f}, ablation_breaks {solved - ablated}")
    learn("synthetic choice", synthetic_choice(), rounds=1, library=library, freeze=True, config=SYNTHETIC_CONFIG)


def arc_root() -> Path:
    here = Path(__file__).resolve().parent
    checked = [
        os.getenv("ARC_AGI_1_TRAIN_DIR"),
        here / "data/ARC-AGI/data/training",
        here / "../agi-core/data/ARC-AGI/data/training",
    ]
    root = next((Path(path) for path in checked if path and Path(path).is_dir()), None)
    if root:
        return root
    print("missing ARC dataset; checked:")
    for path in checked:
        print(f"- {path or '(unset) ARC_AGI_1_TRAIN_DIR'}")
    raise SystemExit(1)


def load_arc_task(path: Path) -> Task:
    data = json.loads(path.read_text())
    return Task(
        train=[(example["input"], example["output"]) for example in data["train"]],
        test=[(example["input"], example["output"]) for example in data["test"]],
    )


def load_arc_split(split: str) -> dict[str, Task]:
    root = arc_root()
    base = root if split == "training" else root.parent / split
    return {path.stem: load_arc_task(path) for path in sorted(base.glob("*.json"))}


def run_arc() -> None:
    learn("arc train", load_arc_split("training"), load_arc_split("evaluation"), config=ARC_CONFIG)


def observation_note(task_id: str) -> str:
    path = Path(__file__).resolve().parent / "OBSERVATIONS.md"
    lines = [line.rstrip() for line in path.read_text().splitlines() if task_id in line]
    return "\n".join(lines) if lines else "(no observation note yet)"


def run_inspect(task_id: str) -> None:
    training = load_arc_split("training")
    evaluation = load_arc_split("evaluation")
    task = training.get(task_id) or evaluation.get(task_id)
    if task is None:
        raise SystemExit(f"unknown ARC task: {task_id}")
    subset = dict(list(training.items())[:64])
    library = learn("arc inspect build", subset, rounds=1, quiet=True, config=ARC_CONFIG)
    print(inspect_task(task_id, task, library, ARC_CONFIG))
    print("observation note:")
    print(observation_note(task_id))
