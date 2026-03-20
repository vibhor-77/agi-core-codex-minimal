from __future__ import annotations

"""Tiny typed grid language used by the 4-pillar scaffold."""

from typing import Any, TypeAlias

Grid: TypeAlias = list[list[int]]
Example: TypeAlias = tuple[Grid, Grid]
Program: TypeAlias = str | tuple[str, Any, Any]


def identity(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def flip_h(grid: Grid) -> Grid:
    return [row[::-1] for row in grid]


def flip_v(grid: Grid) -> Grid:
    return grid[::-1]


def transpose(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid)]


def nonzero_mask(grid: Grid) -> Grid:
    return [[1 if cell else 0 for cell in row] for row in grid]


GRID_PRIMITIVES = {
    "identity": identity,
    "flip_h": flip_h,
    "flip_v": flip_v,
    "transpose": transpose,
}

MASK_PRIMITIVES = {
    "nonzero_mask": nonzero_mask,
}

ALL_PRIMITIVES = GRID_PRIMITIVES | MASK_PRIMITIVES


def program_kind(program: Program) -> str | None:
    if isinstance(program, str):
        return "mask" if program in MASK_PRIMITIVES else "grid"
    return "grid" if program[0] in {"chain", "local"} else None


def render(program: Program) -> str:
    if isinstance(program, str):
        return program
    return f"{program[0]}({render(program[1])}, {render(program[2])})"


def size(program: Program) -> int:
    if isinstance(program, str):
        return 1
    return 1 + size(program[1]) + size(program[2])


def cost(program: Program, learned_names: tuple[str, ...] = ()) -> int:
    if size(program) > 1 and render(program) in set(learned_names):
        return 1
    if isinstance(program, str):
        return 1
    return 1 + cost(program[1], learned_names) + cost(program[2], learned_names)


def subtrees(program: Program) -> list[Program]:
    if isinstance(program, str):
        return []
    return [program, *subtrees(program[1]), *subtrees(program[2])]


def pieces(program: Program) -> list[Program]:
    return [program, *subtrees(program)]


def unique(programs: list[Program]) -> list[Program]:
    seen: set[str] = set()
    out: list[Program] = []
    for program in programs:
        name = render(program)
        if name not in seen:
            seen.add(name)
            out.append(program)
    return out


def uses_library(program: Program, learned_names: set[str]) -> bool:
    return any(size(piece) > 1 and render(piece) in learned_names for piece in pieces(program))


def bounding_box(mask: Grid) -> tuple[int, int, int, int] | None:
    cells = [(r, c) for r, row in enumerate(mask) for c, value in enumerate(row) if value]
    if not cells:
        return None
    rows, cols = zip(*cells)
    return min(rows), max(rows) + 1, min(cols), max(cols) + 1


def crop(grid: Grid, box: tuple[int, int, int, int]) -> Grid:
    r0, r1, c0, c1 = box
    return [row[c0:c1] for row in grid[r0:r1]]


def fit(grid: Grid, height: int, width: int) -> Grid:
    return [
        [grid[r][c] if r < len(grid) and c < len(grid[0]) else 0 for c in range(width)]
        for r in range(height)
    ]


def paste(base: Grid, patch: Grid, box: tuple[int, int, int, int]) -> Grid:
    r0, r1, c0, c1 = box
    out = [row[:] for row in base]
    for r in range(r1 - r0):
        for c in range(c1 - c0):
            out[r0 + r][c0 + c] = patch[r][c]
    return out


def run(program: Program, grid: Grid) -> Grid:
    if isinstance(program, str):
        return ALL_PRIMITIVES[program](grid)
    op, left, right = program
    if op == "chain":
        return run(right, run(left, grid))
    mask = run(left, grid)
    box = bounding_box(mask)
    if not box:
        return grid
    patch = run(right, crop(grid, box))
    height, width = box[1] - box[0], box[3] - box[2]
    return paste(grid, fit(patch, height, width), box)


def score(program: Program, examples: list[Example]) -> float:
    if program_kind(program) != "grid":
        return 0.0
    total = hits = 0
    for inp, out in examples:
        got = run(program, inp)
        if len(got) != len(out) or len(got[0]) != len(out[0]):
            continue
        total += sum(len(row) for row in out)
        hits += sum(a == b for row_a, row_b in zip(got, out) for a, b in zip(row_a, row_b))
    return hits / total if total else 0.0


def signature(program: Program, inputs: list[Grid]) -> tuple[tuple[tuple[int, ...], ...], ...]:
    return tuple(tuple(tuple(row) for row in run(program, grid)) for grid in inputs)


def nodes(program: Program, path: tuple[int, ...] = ()) -> list[tuple[tuple[int, ...], Program]]:
    out = [(path, program)]
    if isinstance(program, str):
        return out
    return out + nodes(program[1], path + (1,)) + nodes(program[2], path + (2,))


def replace(program: Program, path: tuple[int, ...], new: Program) -> Program:
    if not path:
        return new
    mutable = list(program)
    mutable[path[0]] = replace(program[path[0]], path[1:], new)
    return tuple(mutable)
