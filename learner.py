from __future__ import annotations

"""Deterministic frontier search and abstraction growth."""

from dataclasses import dataclass, field

from language import (
    Focus,
    Grid,
    Kind,
    Pipe,
    PrimRef,
    Program,
    PRIMITIVES,
    cost,
    input_kind,
    output_kind,
    remap_family,
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
    program: Program
    kind: Kind
    support: int = 0
    gain: float = 0.0
    reuse: int = 0
    age: int = 0
    solved: set[str] = field(default_factory=set)


Library = dict[str, Abstraction]
Frontier = dict[str, list[tuple[float, Program]]]


def seed_programs(kind: Kind) -> list[Program]:
    """True primitives plus a tiny family of discoverable coordinate remaps."""

    base = [PrimRef(name) for name, prim in PRIMITIVES.items() if prim.output_kind == kind]
    if kind == "grid":
        return base + remap_family()
    return base


def sample_inputs(tasks: dict[str, Task], limit: int = 8) -> list[Grid]:
    grids: list[Grid] = []
    for task in tasks.values():
        for inp, _ in task.train:
            grids.append(inp)
            if len(grids) == limit:
                return grids
    return grids


def enumerate_programs(library: Library, fronts: Frontier, max_cost: int = 7) -> list[Program]:
    learned = set(library)
    grid_prims = seed_programs("grid")
    mask_prims = seed_programs("mask")
    frontier_programs = unique([items[0][1] for items in fronts.values() if items])

    learned_transforms = unique(
        [entry.program for entry in library.values() if entry.kind == "grid"]
        + [program for program in frontier_programs if output_kind(program) == "grid"]
    )
    learned_transforms = sorted(learned_transforms, key=lambda program: (-cost(program), render(program)))
    transforms = unique(learned_transforms + grid_prims)[:10]
    selectors = unique(
        [entry.program for entry in library.values() if entry.kind == "mask"]
        + [program for program in frontier_programs if output_kind(program) == "mask"]
        + mask_prims
    )[:6]

    transform_extensions = [
        Pipe(base, primitive)
        for base in learned_transforms
        for primitive in grid_prims
        if output_kind(base) == input_kind(primitive)
    ] + [
        Pipe(primitive, base)
        for primitive in grid_prims
        for base in learned_transforms
        if output_kind(primitive) == input_kind(base)
    ]
    selector_extensions = [
        Pipe(base, primitive)
        for base in learned_transforms + grid_prims
        for primitive in mask_prims
        if output_kind(base) == input_kind(primitive)
    ]

    transforms = unique(transforms + transform_extensions)[:18]
    selectors = unique(selectors + selector_extensions)[:8]
    focuses = [Focus(selector, transform) for selector in selectors for transform in transforms if render(transform) != "identity"]
    pool = unique(transforms + focuses)
    return [program for program in pool if cost(program, learned) <= max_cost]


def choose_frontier(task: Task, pool: list[Program], prior: list[Program], learned: set[str], keep: int = 3) -> list[tuple[float, Program]]:
    inputs = [inp for inp, _ in task.train]
    ranked = sorted(
        ((score(program, task.train), program) for program in unique(prior + pool)),
        key=lambda item: (-item[0], cost(item[1], learned), render(item[1])),
    )
    chosen: list[tuple[float, Program]] = []
    seen: set[tuple[tuple[tuple[int, ...], ...], ...]] = set()
    for quality, program in ranked:
        sig = signature(program, inputs)
        if sig in seen:
            continue
        chosen.append((quality, program))
        seen.add(sig)
        if len(chosen) == keep:
            break
    return chosen


def promote(library: Library, improved: dict[str, list[tuple[float, Program]]], sample_grids: list[Grid], cap: int = 12) -> tuple[Library, list[str]]:
    seen = {signature(program, sample_grids) for program in seed_programs("grid") + seed_programs("mask")}
    seen |= {signature(entry.program, sample_grids) for entry in library.values()}
    candidates: dict[str, Abstraction] = {}

    for task_id, updates in improved.items():
        delta, winner = updates[0]
        if delta <= 0:
            continue
        for subtree in walk(winner):
            if isinstance(subtree, PrimRef):
                continue
            name = render(subtree)
            item = candidates.setdefault(name, Abstraction(program=subtree, kind=output_kind(subtree)))
            item.support += 1
            item.gain += delta
            if delta == 1.0:
                item.solved.add(task_id)

    added: list[str] = []
    for name, item in sorted(candidates.items(), key=lambda kv: (-kv[1].support, -kv[1].gain, kv[0])):
        if item.support < 2 or item.gain / item.support < 0.75:
            continue
        sig = signature(item.program, sample_grids)
        if sig in seen or name in library:
            continue
        seen.add(sig)
        library[name] = item
        added.append(name)

    for entry in library.values():
        entry.age += 1
        entry.reuse = 0
    for updates in improved.values():
        for _, winner in updates:
            names = {render(piece) for piece in walk(winner) if render(piece) in library}
            for name in names:
                library[name].reuse += 1
                library[name].age = 0

    keep = sorted(
        library.values(),
        key=lambda entry: (-entry.reuse, -len(entry.solved), -entry.support, -entry.gain, entry.age, render(entry.program)),
    )[:cap]
    next_library = {render(entry.program): entry for entry in keep}
    return next_library, added[:cap]


def uses_library(program: Program, learned: set[str]) -> bool:
    return any(render(piece) in learned for piece in walk(program) if not isinstance(piece, PrimRef))


def evaluate_split(tasks: dict[str, Task], library: Library) -> tuple[int, float]:
    pool = enumerate_programs(library, {})
    exact = mean = 0.0
    for task in tasks.values():
        best = choose_frontier(task, pool, [], set(library), keep=1)[0][1]
        value = score(best, task.test)
        exact += value == 1.0
        mean += value
    return int(exact), mean / len(tasks)


def learn(label: str, tasks: dict[str, Task], eval_tasks: dict[str, Task] | None = None, rounds: int = 3, library: Library | None = None, freeze: bool = False) -> Library:
    library = {} if library is None else dict(library)
    fronts: Frontier = {}
    sample_grids = sample_inputs(tasks)
    solved_before: set[str] = set()

    for round_index in range(1, rounds + 1):
        prior = set(library)
        pool = enumerate_programs(library, fronts)
        improved: dict[str, list[tuple[float, Program]]] = {}

        for task_id, task in tasks.items():
            previous = fronts.get(task_id, [(0.0, PrimRef("identity"))])[0][0]
            old_programs = [program for _, program in fronts.get(task_id, [])]
            fronts[task_id] = choose_frontier(task, pool, old_programs, prior)
            better = [(quality - previous, program) for quality, program in fronts[task_id] if quality > previous]
            if better:
                improved[task_id] = better

        new_names: list[str] = []
        if not freeze:
            library, new_names = promote(library, improved, sample_grids)

        solved = sorted(task_id for task_id, items in fronts.items() if items[0][0] == 1.0)
        fresh = sorted(set(solved) - solved_before)
        solved_before = set(solved)
        fresh_programs = {task_id: render(fronts[task_id][0][1]) for task_id in fresh[:8]}
        reused = sorted(name for name in prior if any(name == render(piece) for items in fronts.values() for piece in walk(items[0][1])))
        library_solves = sum(uses_library(fronts[task_id][0][1], prior) for task_id in solved)
        mean_train = sum(items[0][0] for items in fronts.values()) / len(tasks)

        print(f"{label} round {round_index}: {len(solved)}/{len(tasks)} solved, mean train {mean_train:.3f}, pool {len(pool)}")
        print("newly solved:", fresh[:12], "..." if len(fresh) > 12 else "")
        print("fresh programs:", fresh_programs)
        print("new abstractions:", new_names)
        print("reused abstractions:", reused[:8])
        print("library solves:", library_solves)
        print("population:", list(library))

        if eval_tasks and round_index in {1, rounds}:
            exact, mean = evaluate_split(eval_tasks, library)
            print(f"public eval after round {round_index}: {exact}/{len(eval_tasks)} exact, mean test {mean:.3f}")

    return library
