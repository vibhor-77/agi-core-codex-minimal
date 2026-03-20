from __future__ import annotations

"""Typed coordinate/mask calculus for a tiny grid learner."""

from dataclasses import dataclass
from typing import Literal

Grid = list[list[int]]
Kind = Literal["int", "value", "mask", "grid"]


@dataclass(frozen=True)
class IntConst:
    value: int


@dataclass(frozen=True)
class Row:
    pass


@dataclass(frozen=True)
class Col:
    pass


@dataclass(frozen=True)
class Height:
    pass


@dataclass(frozen=True)
class Width:
    pass


@dataclass(frozen=True)
class Add:
    left: "IntExpr"
    right: "IntExpr"


@dataclass(frozen=True)
class Sub:
    left: "IntExpr"
    right: "IntExpr"


IntExpr = IntConst | Row | Col | Height | Width | Add | Sub


@dataclass(frozen=True)
class ValueConst:
    value: int


@dataclass(frozen=True)
class Read:
    grid: "GridExpr"
    row: IntExpr
    col: IntExpr


ValueExpr = ValueConst | Read


@dataclass(frozen=True)
class Identity:
    pass


@dataclass(frozen=True)
class Compose:
    left: "GridExpr"
    right: "GridExpr"


@dataclass(frozen=True)
class Remap:
    row: IntExpr
    col: IntExpr


@dataclass(frozen=True)
class Select:
    mask: "MaskExpr"
    when_true: "GridExpr"
    when_false: "GridExpr"


@dataclass(frozen=True)
class Focus:
    mask: "MaskExpr"
    inner: "GridExpr"


@dataclass(frozen=True)
class Paint:
    value: ValueExpr


GridExpr = Identity | Compose | Remap | Select | Focus | Paint


@dataclass(frozen=True)
class NonZero:
    grid: GridExpr


@dataclass(frozen=True)
class Eq:
    left: IntExpr
    right: IntExpr


@dataclass(frozen=True)
class ValueEq:
    left: ValueExpr
    right: ValueExpr


@dataclass(frozen=True)
class And:
    left: "MaskExpr"
    right: "MaskExpr"


@dataclass(frozen=True)
class Or:
    left: "MaskExpr"
    right: "MaskExpr"


@dataclass(frozen=True)
class Not:
    inner: "MaskExpr"


MaskExpr = NonZero | Eq | ValueEq | And | Or | Not
Expr = IntExpr | ValueExpr | MaskExpr | GridExpr


ZERO = IntConst(0)
ONE = IntConst(1)
IDENTITY = Identity()


def copy_grid(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def fit(grid: Grid, height: int, width: int) -> Grid:
    return [
        [grid[r][c] if r < len(grid) and c < len(grid[0]) else 0 for c in range(width)]
        for r in range(height)
    ]


def bbox(mask: Grid) -> tuple[int, int, int, int] | None:
    cells = [(r, c) for r, row in enumerate(mask) for c, value in enumerate(row) if value]
    if not cells:
        return None
    rows, cols = zip(*cells)
    return min(rows), max(rows) + 1, min(cols), max(cols) + 1


def crop(grid: Grid, box: tuple[int, int, int, int]) -> Grid:
    r0, r1, c0, c1 = box
    return [row[c0:c1] for row in grid[r0:r1]]


def paste(base: Grid, patch: Grid, box: tuple[int, int, int, int]) -> Grid:
    r0, r1, c0, c1 = box
    out = copy_grid(base)
    for r in range(r1 - r0):
        for c in range(c1 - c0):
            out[r0 + r][c0 + c] = patch[r][c]
    return out


def kind(expr: Expr) -> Kind:
    if isinstance(expr, (IntConst, Row, Col, Height, Width, Add, Sub)):
        return "int"
    if isinstance(expr, (ValueConst, Read)):
        return "value"
    if isinstance(expr, (NonZero, Eq, ValueEq, And, Or, Not)):
        return "mask"
    return "grid"


def render_int(expr: IntExpr) -> str:
    if isinstance(expr, IntConst):
        return str(expr.value)
    if isinstance(expr, Row):
        return "row"
    if isinstance(expr, Col):
        return "col"
    if isinstance(expr, Height):
        return "height"
    if isinstance(expr, Width):
        return "width"
    if isinstance(expr, Add):
        if expr.right == ONE:
            return f"{render_int(expr.left)}+1"
        return f"add({render_int(expr.left)}, {render_int(expr.right)})"
    if expr.right == ONE:
        return f"{render_int(expr.left)}-1"
    if expr.left in {Height(), Width()} and isinstance(expr.right, Add) and expr.right.right == ONE:
        return f"{render_int(expr.left)}-1-{render_int(expr.right.left)}"
    return f"sub({render_int(expr.left)}, {render_int(expr.right)})"


def render_mask(expr: MaskExpr) -> str:
    if isinstance(expr, NonZero):
        return f"nonzero({render_grid(expr.grid)})"
    if isinstance(expr, Eq):
        return f"eq({render_int(expr.left)}, {render_int(expr.right)})"
    if isinstance(expr, ValueEq):
        return f"value_eq({render_value(expr.left)}, {render_value(expr.right)})"
    if isinstance(expr, And):
        return f"and({render_mask(expr.left)}, {render_mask(expr.right)})"
    if isinstance(expr, Or):
        return f"or({render_mask(expr.left)}, {render_mask(expr.right)})"
    return f"not({render_mask(expr.inner)})"


def render_value(expr: ValueExpr) -> str:
    if isinstance(expr, ValueConst):
        return str(expr.value)
    return f"read({render_grid(expr.grid)}, {render_int(expr.row)}, {render_int(expr.col)})"


def render_grid(expr: GridExpr) -> str:
    if isinstance(expr, Identity):
        return "identity"
    if isinstance(expr, Compose):
        return f"compose({render_grid(expr.left)}, {render_grid(expr.right)})"
    if isinstance(expr, Remap):
        return f"remap({render_int(expr.row)}, {render_int(expr.col)})"
    if isinstance(expr, Select):
        return f"select({render_mask(expr.mask)}, {render_grid(expr.when_true)}, {render_grid(expr.when_false)})"
    if isinstance(expr, Focus):
        return f"focus({render_mask(expr.mask)}, {render_grid(expr.inner)})"
    return f"paint({render_value(expr.value)})"


def render(expr: Expr) -> str:
    if kind(expr) == "int":
        return render_int(expr)  # type: ignore[arg-type]
    if kind(expr) == "value":
        return render_value(expr)  # type: ignore[arg-type]
    if kind(expr) == "mask":
        return render_mask(expr)  # type: ignore[arg-type]
    return render_grid(expr)  # type: ignore[arg-type]


def cost(expr: Expr, learned: set[str] | None = None) -> int:
    learned = set() if learned is None else learned
    if render(expr) in learned:
        return 1
    if isinstance(expr, (IntConst, Row, Col, Height, Width, ValueConst, Identity)):
        return 1
    if isinstance(expr, (Add, Sub)):
        return 1 + cost(expr.left, learned) + cost(expr.right, learned)
    if isinstance(expr, Read):
        return 1 + cost(expr.grid, learned) + cost(expr.row, learned) + cost(expr.col, learned)
    if isinstance(expr, NonZero):
        return 1 + cost(expr.grid, learned)
    if isinstance(expr, Eq):
        return 1 + cost(expr.left, learned) + cost(expr.right, learned)
    if isinstance(expr, ValueEq):
        return 1 + cost(expr.left, learned) + cost(expr.right, learned)
    if isinstance(expr, (And, Or)):
        return 1 + cost(expr.left, learned) + cost(expr.right, learned)
    if isinstance(expr, Not):
        return 1 + cost(expr.inner, learned)
    if isinstance(expr, Compose):
        return 1 + cost(expr.left, learned) + cost(expr.right, learned)
    if isinstance(expr, Remap):
        return 1 + cost(expr.row, learned) + cost(expr.col, learned)
    if isinstance(expr, Select):
        return 1 + cost(expr.mask, learned) + cost(expr.when_true, learned) + cost(expr.when_false, learned)
    if isinstance(expr, Focus):
        return 1 + cost(expr.mask, learned) + cost(expr.inner, learned)
    return 1 + cost(expr.value, learned)


def walk(expr: Expr) -> list[Expr]:
    if isinstance(expr, (IntConst, Row, Col, Height, Width, ValueConst, Identity)):
        return [expr]
    if isinstance(expr, (Add, Sub)):
        return [expr, *walk(expr.left), *walk(expr.right)]
    if isinstance(expr, Read):
        return [expr, *walk(expr.grid), *walk(expr.row), *walk(expr.col)]
    if isinstance(expr, NonZero):
        return [expr, *walk(expr.grid)]
    if isinstance(expr, Eq):
        return [expr, *walk(expr.left), *walk(expr.right)]
    if isinstance(expr, ValueEq):
        return [expr, *walk(expr.left), *walk(expr.right)]
    if isinstance(expr, (And, Or)):
        return [expr, *walk(expr.left), *walk(expr.right)]
    if isinstance(expr, Not):
        return [expr, *walk(expr.inner)]
    if isinstance(expr, Compose):
        return [expr, *walk(expr.left), *walk(expr.right)]
    if isinstance(expr, Remap):
        return [expr, *walk(expr.row), *walk(expr.col)]
    if isinstance(expr, Select):
        return [expr, *walk(expr.mask), *walk(expr.when_true), *walk(expr.when_false)]
    if isinstance(expr, Focus):
        return [expr, *walk(expr.mask), *walk(expr.inner)]
    return [expr, *walk(expr.value)]


def expr_vars(expr: IntExpr) -> set[str]:
    if isinstance(expr, Row):
        return {"row"}
    if isinstance(expr, Col):
        return {"col"}
    if isinstance(expr, Height):
        return {"height"}
    if isinstance(expr, Width):
        return {"width"}
    if isinstance(expr, IntConst):
        return set()
    return expr_vars(expr.left) | expr_vars(expr.right)


def axis(expr: IntExpr) -> Literal["row", "col"] | None:
    vars_used = expr_vars(expr)
    if vars_used <= {"row", "height"} and vars_used:
        return "row"
    if vars_used <= {"col", "width"} and vars_used:
        return "col"
    return None


def eval_int(expr: IntExpr, row: int, col: int, height: int, width: int) -> int:
    if isinstance(expr, IntConst):
        return expr.value
    if isinstance(expr, Row):
        return row
    if isinstance(expr, Col):
        return col
    if isinstance(expr, Height):
        return height
    if isinstance(expr, Width):
        return width
    if isinstance(expr, Add):
        return eval_int(expr.left, row, col, height, width) + eval_int(expr.right, row, col, height, width)
    return eval_int(expr.left, row, col, height, width) - eval_int(expr.right, row, col, height, width)


def eval_value(expr: ValueExpr, grid: Grid, row: int, col: int) -> int:
    height, width = len(grid), len(grid[0])
    if isinstance(expr, ValueConst):
        return expr.value
    source = eval_grid(expr.grid, grid)
    src_r = eval_int(expr.row, row, col, height, width)
    src_c = eval_int(expr.col, row, col, height, width)
    if 0 <= src_r < len(source) and 0 <= src_c < len(source[0]):
        return source[src_r][src_c]
    return 0


def remap_shape(expr: Remap, height: int, width: int) -> tuple[int, int]:
    row_axis = axis(expr.row)
    col_axis = axis(expr.col)
    out_height = height if row_axis == "row" else width
    out_width = width if col_axis == "col" else height
    return out_height, out_width


def eval_mask(expr: MaskExpr, grid: Grid) -> Grid:
    height, width = len(grid), len(grid[0])
    if isinstance(expr, NonZero):
        return [[1 if cell else 0 for cell in row] for row in eval_grid(expr.grid, grid)]
    if isinstance(expr, Eq):
        return [
            [
                1 if eval_int(expr.left, r, c, height, width) == eval_int(expr.right, r, c, height, width) else 0
                for c in range(width)
            ]
            for r in range(height)
        ]
    if isinstance(expr, ValueEq):
        return [
            [1 if eval_value(expr.left, grid, r, c) == eval_value(expr.right, grid, r, c) else 0 for c in range(width)]
            for r in range(height)
        ]
    if isinstance(expr, And):
        left = eval_mask(expr.left, grid)
        right = eval_mask(expr.right, grid)
        return [[1 if a and b else 0 for a, b in zip(row_a, row_b)] for row_a, row_b in zip(left, right)]
    if isinstance(expr, Or):
        left = eval_mask(expr.left, grid)
        right = eval_mask(expr.right, grid)
        return [[1 if a or b else 0 for a, b in zip(row_a, row_b)] for row_a, row_b in zip(left, right)]
    inner = eval_mask(expr.inner, grid)
    return [[0 if cell else 1 for cell in row] for row in inner]


def eval_grid(expr: GridExpr, grid: Grid) -> Grid:
    if isinstance(expr, Identity):
        return copy_grid(grid)
    if isinstance(expr, Compose):
        return eval_grid(expr.right, eval_grid(expr.left, grid))
    if isinstance(expr, Remap):
        height, width = len(grid), len(grid[0])
        row_axis, col_axis = axis(expr.row), axis(expr.col)
        if row_axis is None or col_axis is None or row_axis == col_axis:
            return copy_grid(grid)
        out_height, out_width = remap_shape(expr, height, width)
        out: Grid = []
        for r in range(out_height):
            row_out: list[int] = []
            for c in range(out_width):
                src_r = eval_int(expr.row, r, c, height, width)
                src_c = eval_int(expr.col, r, c, height, width)
                row_out.append(grid[src_r][src_c] if 0 <= src_r < height and 0 <= src_c < width else 0)
            out.append(row_out)
        return out
    if isinstance(expr, Select):
        mask = eval_mask(expr.mask, grid)
        when_true = fit(eval_grid(expr.when_true, grid), len(grid), len(grid[0]))
        when_false = fit(eval_grid(expr.when_false, grid), len(grid), len(grid[0]))
        return [
            [a if use else b for use, a, b in zip(mask_row, true_row, false_row)]
            for mask_row, true_row, false_row in zip(mask, when_true, when_false)
        ]
    if isinstance(expr, Focus):
        mask = eval_mask(expr.mask, grid)
        box = bbox(mask)
        if not box:
            return copy_grid(grid)
        patch = crop(grid, box)
        inner = fit(eval_grid(expr.inner, patch), box[1] - box[0], box[3] - box[2])
        return paste(grid, inner, box)
    height, width = len(grid), len(grid[0])
    return [[eval_value(expr.value, grid, r, c) for c in range(width)] for r in range(height)]


def evaluate(expr: GridExpr, grid: Grid) -> Grid:
    return eval_grid(expr, grid)


def score(expr: GridExpr, examples: list[tuple[Grid, Grid]]) -> float:
    total = hits = 0
    for inp, out in examples:
        got = evaluate(expr, inp)
        if len(got) != len(out) or len(got[0]) != len(out[0]):
            continue
        total += sum(len(row) for row in out)
        hits += sum(a == b for row_a, row_b in zip(got, out) for a, b in zip(row_a, row_b))
    return hits / total if total else 0.0


def signature(expr: Expr, inputs: list[Grid]) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if kind(expr) == "value":
        return tuple(
            tuple(tuple(eval_value(expr, grid, r, c) for c in range(len(grid[0]))) for r in range(len(grid)))
            for grid in inputs
        )  # type: ignore[arg-type]
    if kind(expr) == "mask":
        return tuple(tuple(tuple(row) for row in eval_mask(expr, grid)) for grid in inputs)  # type: ignore[arg-type]
    if kind(expr) == "grid":
        return tuple(tuple(tuple(row) for row in eval_grid(expr, grid)) for grid in inputs)  # type: ignore[arg-type]
    return tuple((((eval_int(expr, 0, 0, len(grid), len(grid[0])),),),) for grid in inputs)  # type: ignore[arg-type]


def unique(exprs: list[Expr]) -> list[Expr]:
    seen: set[str] = set()
    out: list[Expr] = []
    for expr in exprs:
        name = render(expr)
        if name not in seen:
            seen.add(name)
            out.append(expr)
    return out


def row_like_exprs() -> list[IntExpr]:
    return [
        Row(),
        Add(Row(), ONE),
        Sub(Row(), ONE),
        Sub(Sub(Height(), ONE), Row()),
    ]


def col_like_exprs() -> list[IntExpr]:
    return [
        Col(),
        Add(Col(), ONE),
        Sub(Col(), ONE),
        Sub(Sub(Width(), ONE), Col()),
    ]


def grid_seeds() -> list[GridExpr]:
    coords = row_like_exprs() + col_like_exprs()
    remaps = [Remap(row, col) for row in coords for col in coords if axis(row) and axis(col) and axis(row) != axis(col)]
    return unique([IDENTITY] + remaps)


def value_seeds() -> list[ValueExpr]:
    coords = row_like_exprs() + col_like_exprs()
    reads = [Read(IDENTITY, row, col) for row in coords for col in coords if axis(row) and axis(col) and axis(row) != axis(col)]
    return unique(reads)


def mask_seeds() -> list[MaskExpr]:
    last_row = Sub(Height(), ONE)
    last_col = Sub(Width(), ONE)
    return unique([
        NonZero(IDENTITY),
        Eq(Row(), Col()),
        Eq(Row(), ZERO),
        Eq(Col(), ZERO),
        Eq(Row(), last_row),
        Eq(Col(), last_col),
    ])
