from __future__ import annotations

"""Deterministic wake/sleep loop for the coordinate/mask calculus."""

from dataclasses import dataclass, field

from language import (
    And,
    Compose,
    Eq,
    Expr,
    Focus,
    Grid,
    GridExpr,
    IDENTITY,
    Kind,
    MaskExpr,
    Not,
    Or,
    Select,
    cost,
    evaluate,
    grid_seeds,
    kind,
    mask_seeds,
    render,
    score,
    signature,
    unique,
    walk,
)


@dataclass(frozen=True)
class Task:
    train: list[tuple[Grid, Grid]]
    test: list[tuple[Grid, Grid]]


@dataclass
class Abstraction:
    expr: Expr
    support: int = 0
    gain: float = 0.0
    reuse: int = 0
    exact: int = 0
    age: int = 0


Library = dict[str, Abstraction]
Frontier = dict[str, list[tuple[float, GridExpr]]]


@dataclass(frozen=True)
class SearchConfig:
    max_cost: int = 10
    grid_limit: int = 24
    step_limit: int = 12
    mask_limit: int = 18
    compose_left_limit: int = 12
    frontier_keep: int = 4


SYNTHETIC_CONFIG = SearchConfig()
ARC_CONFIG = SearchConfig(max_cost=9, grid_limit=18, step_limit=8, mask_limit=12, compose_left_limit=8, frontier_keep=3)


def sample_inputs(tasks: dict[str, Task], limit: int = 8) -> list[Grid]:
    grids: list[Grid] = []
    for task in tasks.values():
        for inp, _ in task.train:
            grids.append(inp)
            if len(grids) == limit:
                return grids
    return grids


def top_frontier(fronts: Frontier, target: Kind) -> list[Expr]:
    return unique([
        program
        for items in fronts.values()
        for _, program in items
        for piece in walk(program)
        if kind(piece) == target
    ])


def seed_exprs(target: Kind) -> list[Expr]:
    if target == "grid":
        return grid_seeds()
    if target == "mask":
        return mask_seeds()
    return []


def explore_masks(library: Library, fronts: Frontier, config: SearchConfig) -> list[MaskExpr]:
    seeds = [expr for expr in seed_exprs("mask") if isinstance(expr, (Eq, And, Or, Not)) or kind(expr) == "mask"]
    learned = [item.expr for item in library.values() if kind(item.expr) == "mask"]
    frontier = top_frontier(fronts, "mask")
    base = [expr for expr in unique(learned + frontier + seeds) if kind(expr) == "mask"]
    masks = unique(base + [Not(mask) for mask in base])
    masks += [op(left, right) for op in (And, Or) for left in base[:6] for right in base[:6] if render(left) < render(right)]
    return [expr for expr in unique(masks) if kind(expr) == "mask"][:config.mask_limit]  # type: ignore[return-value]


def explore_grids(library: Library, fronts: Frontier, masks: list[MaskExpr], config: SearchConfig) -> list[GridExpr]:
    learned_names = set(library)
    seeds = [expr for expr in seed_exprs("grid") if kind(expr) == "grid"]
    learned_exprs = [item.expr for item in library.values() if kind(item.expr) == "grid"]
    frontier = top_frontier(fronts, "grid")
    base = unique(learned_exprs + frontier + seeds)
    transforms = base[:config.grid_limit]
    steps = unique(learned_exprs + seeds)[:config.step_limit]

    composed = [Compose(left, right) for left in transforms[:config.compose_left_limit] for right in steps if render(right) != "identity"]
    focused = [Focus(mask, grid) for mask in masks[:config.mask_limit] for grid in transforms if render(grid) != "identity"]
    selected = [
        Select(mask, grid, IDENTITY)
        for mask in masks[:config.mask_limit]
        for grid in transforms
        if render(grid) != "identity"
    ] + [
        Select(mask, IDENTITY, grid)
        for mask in masks[:config.mask_limit]
        for grid in transforms
        if render(grid) != "identity"
    ]
    pool = unique(transforms + composed + focused + selected)
    return [expr for expr in pool if cost(expr, learned_names) <= config.max_cost]  # type: ignore[return-value]


def explore(library: Library, fronts: Frontier, config: SearchConfig) -> list[GridExpr]:
    masks = explore_masks(library, fronts, config)
    return explore_grids(library, fronts, masks, config)


def choose_frontier(task: Task, pool: list[GridExpr], prior: list[GridExpr], learned: set[str], keep: int) -> list[tuple[float, GridExpr]]:
    inputs = [inp for inp, _ in task.train]
    ranked = sorted(
        ((score(expr, task.train), expr) for expr in unique(prior + pool)),
        key=lambda item: (-item[0], cost(item[1], learned), render(item[1])),
    )
    chosen: list[tuple[float, GridExpr]] = []
    seen: set[tuple[tuple[tuple[int, ...], ...], ...]] = set()
    for quality, expr in ranked:
        sig = signature(expr, inputs)
        if sig in seen:
            continue
        chosen.append((quality, expr))
        seen.add(sig)
        if len(chosen) == keep:
            break
    return chosen


def prune_library(library: Library, sample_grids: list[Grid], cap: int = 12) -> Library:
    seen = {signature(expr, sample_grids) for expr in seed_exprs("grid") + seed_exprs("mask")}
    kept: list[Abstraction] = []
    ranked = sorted(
        library.values(),
        key=lambda item: (-item.reuse, -item.exact, -item.support, -item.gain, item.age, cost(item.expr), render(item.expr)),
    )
    for item in ranked:
        sig = signature(item.expr, sample_grids)
        if sig in seen:
            continue
        kept.append(item)
        seen.add(sig)
        if len(kept) == cap:
            break
    return {render(item.expr): item for item in kept}


def promote(library: Library, improved: dict[str, tuple[float, GridExpr]], sample_grids: list[Grid]) -> tuple[Library, list[str]]:
    candidates: dict[str, Abstraction] = {}
    for task_id, (delta, winner) in improved.items():
        if delta <= 0:
            continue
        for piece in walk(winner):
            piece_kind = kind(piece)
            if piece_kind == "int" or render(piece) == "identity":
                continue
            item = candidates.setdefault(render(piece), Abstraction(expr=piece))
            item.support += 1
            item.gain += delta
            item.exact += delta == 1.0

    for item in library.values():
        item.age += 1
        item.reuse = 0

    for _, winner in improved.values():
        used = {render(piece) for piece in walk(winner) if render(piece) in library}
        for name in used:
            library[name].reuse += 1
            library[name].age = 0

    added: list[str] = []
    for name, item in sorted(candidates.items(), key=lambda kv: (-kv[1].support, -kv[1].gain, cost(kv[1].expr), kv[0])):
        if name in library:
            library[name].support += item.support
            library[name].gain += item.gain
            library[name].exact += item.exact
            continue
        if not (item.support >= 2 or item.gain >= 1.5):
            continue
        library[name] = item
        added.append(name)

    return prune_library(library, sample_grids), added


def uses_library(expr: GridExpr, learned: set[str]) -> bool:
    return any(render(piece) in learned for piece in walk(expr) if kind(piece) != "int")


def evaluate_tasks(tasks: dict[str, Task], library: Library | None = None, config: SearchConfig = ARC_CONFIG) -> tuple[int, float]:
    library = {} if library is None else library
    pool = explore(library, {}, config)
    exact = mean = 0.0
    for task in tasks.values():
        best = choose_frontier(task, pool, [], set(library), keep=1)[0][1]
        value = score(best, task.test)
        exact += value == 1.0
        mean += value
    return int(exact), mean / len(tasks)


def best_program(task: Task, library: Library | None = None, config: SearchConfig = ARC_CONFIG) -> tuple[float, GridExpr]:
    library = {} if library is None else library
    pool = explore(library, {}, config)
    return choose_frontier(task, pool, [], set(library), keep=1)[0]


def inspect_task(task_id: str, task: Task, library: Library | None = None, config: SearchConfig = ARC_CONFIG) -> str:
    base_score, base_expr = best_program(task, {}, config)
    learned_score, learned_expr = best_program(task, library or {}, config)
    used = sorted({render(piece) for piece in walk(learned_expr) if library and render(piece) in library})
    lines = [
        f"task: {task_id}",
        f"train pairs: {len(task.train)}, test pairs: {len(task.test)}",
        f"best without library: {base_score:.3f} -> {render(base_expr)}",
        f"best with library: {learned_score:.3f} -> {render(learned_expr)}",
        f"reused abstractions: {used}",
    ]
    for index, (inp, out) in enumerate(task.train, start=1):
        lines.append(f"train {index} input: {inp}")
        lines.append(f"train {index} output: {out}")
    for index, (inp, out) in enumerate(task.test, start=1):
        lines.append(f"test {index} input: {inp}")
        lines.append(f"test {index} output: {out}")
    return "\n".join(lines)


def learn(
    label: str,
    tasks: dict[str, Task],
    eval_tasks: dict[str, Task] | None = None,
    rounds: int = 3,
    library: Library | None = None,
    freeze: bool = False,
    quiet: bool = False,
    config: SearchConfig = SYNTHETIC_CONFIG,
) -> Library:
    library = {} if library is None else dict(library)
    fronts: Frontier = {}
    sample_grids = sample_inputs(tasks)
    solved_before: set[str] = set()

    for round_index in range(1, rounds + 1):
        prior = set(library)
        pool = explore(library, fronts, config)
        improved: dict[str, tuple[float, GridExpr]] = {}

        for task_id, task in tasks.items():
            previous = fronts.get(task_id, [(0.0, IDENTITY)])[0][0]
            old_programs = [expr for _, expr in fronts.get(task_id, [])]
            fronts[task_id] = choose_frontier(task, pool, old_programs, prior, keep=config.frontier_keep)
            best_quality, best_expr = fronts[task_id][0]
            if best_quality > previous:
                improved[task_id] = (best_quality - previous, best_expr)

        added: list[str] = []
        if not freeze:
            library, added = promote(library, improved, sample_grids)

        solved = sorted(task_id for task_id, items in fronts.items() if items[0][0] == 1.0)
        fresh = sorted(set(solved) - solved_before)
        solved_before = set(solved)
        fresh_programs = {task_id: render(fronts[task_id][0][1]) for task_id in fresh[:8]}
        reused = sorted(name for name in prior if any(render(piece) == name for items in fronts.values() for piece in walk(items[0][1])))
        library_solves = sum(uses_library(fronts[task_id][0][1], prior) for task_id in solved)
        mean_train = sum(items[0][0] for items in fronts.values()) / len(tasks)

        if not quiet:
            print(f"{label} round {round_index}: {len(solved)}/{len(tasks)} solved, mean train {mean_train:.3f}, pool {len(pool)}")
            print("newly solved:", fresh[:12], "..." if len(fresh) > 12 else "")
            print("fresh programs:", fresh_programs)
            print("new abstractions:", added[:12])
            print("reused abstractions:", reused[:8])
            print("library solves:", library_solves)
            print("population:", list(library))

        if eval_tasks and round_index in {1, rounds}:
            exact, mean = evaluate_tasks(eval_tasks, library, ARC_CONFIG)
            if not quiet:
                print(f"public eval after round {round_index}: {exact}/{len(eval_tasks)} exact, mean test {mean:.3f}")

    return library
