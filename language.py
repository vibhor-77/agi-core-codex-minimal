from __future__ import annotations

"""Typed program language for a tiny grid learner."""

from dataclasses import dataclass
from typing import Callable, Literal

Grid = list[list[int]]
Kind = Literal["grid", "mask"]


@dataclass(frozen=True)
class Primitive:
    name: str
    input_kind: Kind
    output_kind: Kind
    fn: Callable[[Grid], Grid]


@dataclass(frozen=True)
class PrimRef:
    name: str


@dataclass(frozen=True)
class Pipe:
    left: "Program"
    right: "Program"


@dataclass(frozen=True)
class Focus:
    selector: "Program"
    transform: "Program"


@dataclass(frozen=True)
class Remap:
    """Map each output cell back to a source coordinate."""

    row_expr: str
    col_expr: str


Program = PrimRef | Pipe | Focus | Remap


def identity(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def nonzero(grid: Grid) -> Grid:
    return [[1 if cell else 0 for cell in row] for row in grid]


def invert(mask: Grid) -> Grid:
    return [[0 if cell else 1 for cell in row] for row in mask]


PRIMITIVES = {
    primitive.name: primitive
    for primitive in [
        Primitive("identity", "grid", "grid", identity),
        Primitive("nonzero", "grid", "mask", nonzero),
        Primitive("invert", "mask", "mask", invert),
    ]
}

# Candidate source-coordinate formulas. Higher-level moves such as mirror and
# transpose are meant to emerge from these, not appear as named primitives.
ROW_EXPR = ("row", "height-1-row", "col", "height-1-col")
COL_EXPR = ("col", "width-1-col", "row", "width-1-row")


def input_kind(program: Program) -> Kind:
    if isinstance(program, PrimRef):
        return PRIMITIVES[program.name].input_kind
    if isinstance(program, Remap):
        return "grid"
    if isinstance(program, Pipe):
        return input_kind(program.left)
    return "grid"


def output_kind(program: Program) -> Kind:
    if isinstance(program, PrimRef):
        return PRIMITIVES[program.name].output_kind
    if isinstance(program, Remap):
        return "grid"
    if isinstance(program, Pipe):
        return output_kind(program.right)
    return "grid"


def render(program: Program) -> str:
    if isinstance(program, PrimRef):
        return program.name
    if isinstance(program, Remap):
        return f"remap({program.row_expr}, {program.col_expr})"
    if isinstance(program, Pipe):
        return f"pipe({render(program.left)}, {render(program.right)})"
    return f"focus({render(program.selector)}, {render(program.transform)})"


def cost(program: Program, learned: set[str] | None = None) -> int:
    learned = set() if learned is None else learned
    if render(program) in learned:
        return 1
    if isinstance(program, PrimRef):
        return 1
    if isinstance(program, Remap):
        return 1
    if isinstance(program, Pipe):
        return 1 + cost(program.left, learned) + cost(program.right, learned)
    return 1 + cost(program.selector, learned) + cost(program.transform, learned)


def walk(program: Program) -> list[Program]:
    if isinstance(program, (PrimRef, Remap)):
        return [program]
    if isinstance(program, Pipe):
        return [program, *walk(program.left), *walk(program.right)]
    return [program, *walk(program.selector), *walk(program.transform)]


def remap_family() -> list[Remap]:
    return [
        Remap(row_expr, col_expr)
        for row_expr in ROW_EXPR
        for col_expr in COL_EXPR
        if output_axis(row_expr) != output_axis(col_expr)
    ]


def output_axis(expr: str) -> str:
    return "row" if expr.endswith("row") else "col"


def expr_value(expr: str, row: int, col: int, height: int, width: int) -> int:
    return {
        "row": row,
        "height-1-row": height - 1 - row,
        "col": col,
        "height-1-col": height - 1 - col,
        "width-1-col": width - 1 - col,
        "width-1-row": width - 1 - row,
    }[expr]


def remap_shape(program: Remap, height: int, width: int) -> tuple[int, int]:
    out_height = height if output_axis(program.row_expr) == "row" else width
    out_width = width if output_axis(program.col_expr) == "col" else height
    return out_height, out_width


def bbox(mask: Grid) -> tuple[int, int, int, int] | None:
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


def evaluate(program: Program, grid: Grid) -> Grid:
    if isinstance(program, PrimRef):
        return PRIMITIVES[program.name].fn(grid)
    if isinstance(program, Remap):
        height, width = len(grid), len(grid[0])
        out_height, out_width = remap_shape(program, height, width)
        return [
            [
                grid[
                    expr_value(program.row_expr, row, col, height, width)
                ][
                    expr_value(program.col_expr, row, col, height, width)
                ]
                for col in range(out_width)
            ]
            for row in range(out_height)
        ]
    if isinstance(program, Pipe):
        return evaluate(program.right, evaluate(program.left, grid))
    mask = evaluate(program.selector, grid)
    if len(mask) != len(grid) or len(mask[0]) != len(grid[0]):
        return grid
    box = bbox(mask)
    if not box:
        return grid
    patch = evaluate(program.transform, crop(grid, box))
    height, width = box[1] - box[0], box[3] - box[2]
    return paste(grid, fit(patch, height, width), box)


def score(program: Program, examples: list[tuple[Grid, Grid]]) -> float:
    if output_kind(program) != "grid":
        return 0.0
    total = hits = 0
    for inp, out in examples:
        got = evaluate(program, inp)
        if len(got) != len(out) or len(got[0]) != len(out[0]):
            continue
        total += sum(len(row) for row in out)
        hits += sum(a == b for row_a, row_b in zip(got, out) for a, b in zip(row_a, row_b))
    return hits / total if total else 0.0


def signature(program: Program, inputs: list[Grid]) -> tuple[tuple[tuple[int, ...], ...], ...]:
    return tuple(tuple(tuple(row) for row in evaluate(program, grid)) for grid in inputs)


def unique(programs: list[Program]) -> list[Program]:
    seen: set[str] = set()
    out: list[Program] = []
    for program in programs:
        name = render(program)
        if name not in seen:
            seen.add(name)
            out.append(program)
    return out
