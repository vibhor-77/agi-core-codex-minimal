from __future__ import annotations

"""Synthetic curriculum and ARC dataset loading for the minimal scaffold."""

import json
import os
from pathlib import Path

from language import Grid, Program, run
from learner import Library, Task, learn


def make_task(grid: Grid, program: Program) -> Task:
    output = run(program, grid)
    example = (grid, output)
    return Task(train=[example], test=[example])


def synthetic_stages() -> list[tuple[str, dict[str, Task]]]:
    pair = ("local", "nonzero_mask", "flip_h")
    triple_t = ("chain", pair, "transpose")
    triple_v = ("chain", "flip_v", pair)
    quad_t = ("local", "nonzero_mask", triple_t)
    quad_v = ("chain", triple_v, "transpose")
    quint_t = ("chain", quad_t, "transpose")
    quint_v = ("chain", quad_v, "flip_v")
    return [
        ("stage_1", {
            "flip_h": make_task([[1, 0], [0, 0]], "flip_h"),
            "transpose": make_task([[1, 0], [2, 3]], "transpose"),
            "pair_1": make_task([[1, 2, 0], [3, 0, 0]], pair),
            "pair_2": make_task([[0, 4, 5], [0, 6, 0]], pair),
        }),
        ("stage_2", {
            "triple_t_1": make_task([[1, 2, 0], [3, 0, 0]], triple_t),
            "triple_t_2": make_task([[0, 4, 5], [0, 6, 0]], triple_t),
            "triple_v_1": make_task([[0, 2, 3], [0, 2, 2], [0, 2, 1]], triple_v),
            "triple_v_2": make_task([[1, 1, 0], [1, 0, 0], [2, 2, 0]], triple_v),
        }),
        ("stage_3", {
            "quad_t_1": make_task([[3, 0, 2, 3], [3, 2, 3, 2]], quad_t),
            "quad_t_2": make_task([[2, 3, 0], [0, 0, 0], [1, 1, 0]], quad_t),
            "quad_v_1": make_task([[3, 0, 0], [2, 3, 0], [2, 2, 0]], quad_v),
            "quad_v_2": make_task([[0, 3, 0], [0, 3, 1], [0, 3, 3]], quad_v),
        }),
        ("stage_4", {
            "quint_t_1": make_task([[3, 0, 2, 3], [3, 2, 3, 2]], quint_t),
            "quint_t_2": make_task([[1, 2, 0], [3, 4, 0], [5, 0, 0]], quint_t),
            "quint_v_1": make_task([[2, 3, 0], [0, 0, 0], [1, 1, 0]], quint_v),
            "quint_v_2": make_task([[0, 2, 0], [3, 1, 0], [3, 3, 0]], quint_v),
        }),
    ]


def synthetic_choice() -> dict[str, Task]:
    pair = ("local", "nonzero_mask", "flip_h")
    triple_t = ("chain", pair, "transpose")
    triple_v = ("chain", "flip_v", pair)
    quad_t = ("local", "nonzero_mask", triple_t)
    quad_v = ("chain", triple_v, "transpose")
    quint_t = ("chain", quad_t, "transpose")
    quint_v = ("chain", quad_v, "flip_v")
    return {
        "choice_quad_t": make_task([[1, 2, 0], [3, 4, 0], [5, 0, 0]], quad_t),
        "choice_quad_v": make_task([[0, 3, 0], [0, 3, 1], [0, 3, 3]], quad_v),
        "choice_quint_t": make_task([[3, 0, 2, 3], [3, 2, 3, 2]], quint_t),
        "choice_quint_v": make_task([[2, 3, 0], [0, 0, 0], [1, 1, 0]], quint_v),
    }


def run_synthetic() -> None:
    library: Library = {}
    for stage_name, tasks in synthetic_stages():
        library = learn(f"synthetic {stage_name}", tasks, rounds=1, library=library)
    learn("synthetic choice", synthetic_choice(), rounds=1, library=library, freeze=True)


def find_arc_root(script_path: Path | None = None) -> Path:
    here = (script_path or Path(__file__)).resolve().parent
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
    root = find_arc_root()
    base = root if split == "training" else root.parent / split
    return {path.stem: load_arc_task(path) for path in sorted(base.glob("*.json"))}


def run_arc() -> None:
    learn("arc train", load_arc_split("training"), load_arc_split("evaluation"))
