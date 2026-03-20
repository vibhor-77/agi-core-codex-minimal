from __future__ import annotations

"""Search, abstraction growth, and reporting for the minimal 4-pillar loop."""

from dataclasses import dataclass, field

from language import (
    GRID_PRIMITIVES,
    MASK_PRIMITIVES,
    Example,
    Grid,
    Program,
    cost,
    nodes,
    pieces,
    program_kind,
    render,
    replace,
    score,
    signature,
    size,
    unique,
    uses_library,
)


@dataclass(frozen=True)
class Task:
    train: list[Example]
    test: list[Example]


@dataclass
class Abstraction:
    program: Program
    gain: float
    support: int
    covers: set[str] = field(default_factory=set)
    helps: set[str] = field(default_factory=set)
    critical: int = 0
    impact: float = 0.0
    reuse: int = 0
    age: int = 0
    used: bool = False
    parents: list[str] = field(default_factory=list)
    depth: int = 0


Library = dict[str, Abstraction]
Frontier = dict[str, list[tuple[float, Program]]]


def parents_of(program: Program) -> list[str]:
    if isinstance(program, str):
        return []
    return [render(program[1]), render(program[2])]


def depth_of(program: Program, library: Library) -> int:
    if isinstance(program, str):
        return 0
    child_depths = []
    for child in program[1:]:
        name = render(child)
        child_depths.append(library[name].depth if name in library else depth_of(child, library))
    return 1 + max(child_depths, default=0)


def sample_inputs(tasks: dict[str, Task], limit: int = 8) -> list[Grid]:
    grids: list[Grid] = []
    for task in tasks.values():
        for inp, _ in task.train:
            grids.append(inp)
            if len(grids) == limit:
                return grids
    return grids


def mutate(program: Program, parts: list[Program], selectors: list[str]) -> list[Program]:
    if isinstance(program, str):
        return []
    out: list[Program] = []
    op, left, right = program
    if op == "chain":
        for part in parts:
            if render(part) != render(left):
                out.append(("chain", part, right))
            if render(part) != render(right):
                out.append(("chain", left, part))
    if op == "local":
        for selector in selectors:
            if selector != left:
                out.append(("local", selector, right))
        for part in parts:
            if render(part) != render(right):
                out.append(("local", left, part))
    for child in program[1:]:
        for candidate in mutate(child, parts, selectors):
            if op == "chain" and child is left:
                out.append(("chain", candidate, right))
            elif op == "chain":
                out.append(("chain", left, candidate))
            else:
                out.append(("local", left, candidate))
    return out


def spawn_candidates(library: Library, fronts: Frontier, width: int = 8, max_cost: int = 9) -> list[Program]:
    learned_names = tuple(library)
    frontier_programs = unique([
        program
        for items in fronts.values()
        for _, program in items
        if program_kind(program) == "grid"
    ])
    ranked_library = sorted(
        [entry.program for entry in library.values() if program_kind(entry.program) == "grid"],
        key=lambda program: library_rank(library[render(program)]),
    )
    evolvers = unique(ranked_library[:width] + frontier_programs[:width])
    selectors = unique(
        list(MASK_PRIMITIVES)
        + [piece for program in evolvers for piece in pieces(program) if program_kind(piece) == "mask"]
    )[:2]
    locals_ = [
        ("local", selector, transform)
        for selector in selectors
        for transform in unique(list(GRID_PRIMITIVES) + evolvers)[:width]
        if render(transform) != "identity"
    ]
    parts = unique([
        piece
        for program in list(GRID_PRIMITIVES) + evolvers + locals_
        for piece in pieces(program)
        if program_kind(piece) == "grid"
    ])[: width * 3]

    pool: list[Program] = list(GRID_PRIMITIVES) + evolvers + locals_
    for program in evolvers:
        for primitive in GRID_PRIMITIVES:
            pool.extend([("chain", primitive, program), ("chain", program, primitive)])
        for selector in selectors:
            if render(program) != "identity":
                pool.append(("local", selector, program))
        pool.extend(mutate(program, parts[:width], selectors))
    for left in parts[:width]:
        for right in parts[:width]:
            if render(left) != render(right):
                pool.append(("chain", left, right))

    return [
        program
        for program in unique(pool)
        if program_kind(program) == "grid" and cost(program, learned_names) <= max_cost
    ]


def select_frontier(
    task: Task,
    pool: list[Program],
    keep: int = 3,
    old: list[Program] | None = None,
    library: Library | None = None,
) -> list[tuple[float, Program]]:
    old = [] if old is None else old
    library = {} if library is None else library
    ranked = sorted(
        ((score(program, task.train), program) for program in unique(old + pool)),
        key=lambda item: (-item[0], cost(item[1], tuple(library)), size(item[1]), render(item[1])),
    )
    chosen: list[tuple[float, Program]] = []
    seen: set[tuple[tuple[tuple[int, ...], ...], ...]] = set()
    inputs = [inp for inp, _ in task.train]
    for quality, program in ranked:
        sig = signature(program, inputs)
        if sig in seen:
            continue
        chosen.append((quality, program))
        seen.add(sig)
        if len(chosen) == keep:
            break
    return chosen


def causal_drop(task: Task, program: Program, learned_names: set[str]) -> float:
    if not uses_library(program, learned_names):
        return 0.0
    base = score(program, task.train)
    alternatives = unique([
        replace(program, path, "identity")
        for path, node in nodes(program)
        if size(node) > 1 and render(node) in learned_names
    ])
    return max(0.0, base - max([score(candidate, task.train) for candidate in alternatives] or [0.0]))


def critical_names(task: Task, program: Program, learned_names: set[str]) -> dict[str, float]:
    if not uses_library(program, learned_names):
        return {}

    base = score(program, task.train)
    hits: list[tuple[tuple[int, ...], str, float]] = []
    for path, node in nodes(program):
        name = render(node)
        if size(node) <= 1 or name not in learned_names:
            continue
        drop = max(0.0, base - score(replace(program, path, "identity"), task.train))
        if drop > 0:
            hits.append((path, name, drop))

    outermost = [
        (path, name, drop)
        for path, name, drop in hits
        if not any(path[: len(other)] == other and other != path for other, _, _ in hits)
    ]
    out: dict[str, float] = {}
    for _, name, drop in outermost:
        out[name] = max(out.get(name, 0.0), drop)
    return out


def keep_alive(entries: list[Abstraction], cap: int) -> list[Abstraction]:
    chosen: list[Abstraction] = []
    covered: set[str] = set()
    while entries and len(chosen) < cap:
        entries.sort(key=lambda entry: survivor_rank(entry, covered))
        best = entries.pop(0)
        chosen.append(best)
        covered |= best.covers | best.helps
    return chosen


def evolve_population(
    library: Library,
    improved: dict[str, list[tuple[float, Program]]],
    sample_grids: list[Grid],
    cap: int = 12,
) -> tuple[Library, list[dict[str, Program | float | int]], list[dict[str, Program | float | int]], int]:
    for entry in library.values():
        entry.age += 1
        entry.used = False

    gains: dict[str, dict[str, Program | float | int]] = {}
    for candidates in improved.values():
        used_names: set[str] = set()
        task_best: dict[str, tuple[Program, float]] = {}
        for delta, leader in candidates:
            for tree in [piece for piece in pieces(leader) if size(piece) > 1]:
                name = render(tree)
                used_names.add(name)
                if name not in task_best or delta > task_best[name][1]:
                    task_best[name] = (tree, delta)
        for name, (tree, delta) in task_best.items():
            item = gains.setdefault(name, {"program": tree, "gain": 0.0, "support": 0})
            item["gain"] += delta
            item["support"] += 1
        for name in used_names:
            if name in library:
                library[name].reuse += 1
                library[name].used = True
                library[name].age = 0

    base_signatures = {
        signature(name, sample_grids) for name in list(GRID_PRIMITIVES) + list(MASK_PRIMITIVES)
    }
    seen = base_signatures | {signature(entry.program, sample_grids) for entry in library.values()}
    ranked = sorted(
        gains.values(),
        key=lambda item: (-item["support"], -item["gain"], size(item["program"]), render(item["program"])),
    )

    new_items: list[dict[str, Program | float | int]] = []
    primitive_equivalent_rejections = 0
    for item in ranked:
        program = item["program"]
        name = render(program)
        sig = signature(program, sample_grids)
        if sig in base_signatures:
            primitive_equivalent_rejections += 1
            continue
        if item["support"] > 1 and item["gain"] / item["support"] >= 0.8 and sig not in seen and name not in library:
            library[name] = Abstraction(
                program=program,
                gain=item["gain"],
                support=item["support"],
                parents=parents_of(program),
                depth=depth_of(program, library),
            )
            seen.add(sig)
            new_items.append(item)
        elif name in library:
            library[name].gain += item["gain"]
            library[name].support += item["support"]

    alive = [
        entry
        for entry in library.values()
        if entry.age == 0 or entry.critical > 0 or entry.helps or entry.impact >= 0.1
    ]
    keep = keep_alive(alive, cap)
    next_library = {render(entry.program): entry for entry in keep}
    return next_library, new_items[:cap], ranked[:5], primitive_equivalent_rejections


def evaluate_split(label: str, tasks: dict[str, Task], library: Library) -> None:
    pool = spawn_candidates(library, {})
    exact = mean = 0.0
    for task in tasks.values():
        best = select_frontier(task, pool, keep=1, library=library)[0][1]
        value = score(best, task.test)
        exact += value == 1.0
        mean += value
    print(f"{label}: {int(exact)}/{len(tasks)} exact, mean test {mean / len(tasks):.3f}")


def ablate_without_library(tasks: dict[str, Task], fronts: Frontier) -> tuple[int, float]:
    pool = spawn_candidates({}, {})
    breaks = gap = 0.0
    for task_id, task in tasks.items():
        baseline = select_frontier(task, pool, keep=1)[0][0]
        current = fronts[task_id][0][0]
        breaks += current == 1.0 and baseline < 1.0
        gap += current - baseline
    return int(breaks), gap / len(tasks)


def learn(
    label: str,
    tasks: dict[str, Task],
    eval_tasks: dict[str, Task] | None = None,
    rounds: int = 3,
    keep: int = 4,
    library: Library | None = None,
    freeze: bool = False,
) -> Library:
    library = {} if library is None else dict(library)
    fronts: Frontier = {}
    inputs = sample_inputs(tasks)
    solved_before: set[str] = set()

    for round_index in range(1, rounds + 1):
        prior_names = set(library)
        pool = spawn_candidates(library, fronts)
        improved: dict[str, list[tuple[float, Program]]] = {}

        for task_id, task in tasks.items():
            previous_best = fronts.get(task_id, [(0.0, "identity")])[0][0]
            prior_programs = [program for _, program in fronts.get(task_id, [])]
            fronts[task_id] = select_frontier(task, pool, keep, prior_programs, library)
            better = [
                (quality - previous_best, program)
                for quality, program in fronts[task_id]
                if quality > previous_best
            ]
            if better:
                improved[task_id] = better

        if prior_names:
            assign_causal_credit(tasks, fronts, library, prior_names)

        if freeze:
            new_items, top_candidates, primitive_equivalent_rejections = [], [], 0
        else:
            library, new_items, top_candidates, primitive_equivalent_rejections = evolve_population(
                library, improved, inputs
            )

        summary = summarize_round(
            tasks,
            fronts,
            library,
            prior_names,
            improved,
            pool,
            solved_before,
            primitive_equivalent_rejections,
            new_items,
            top_candidates,
        )
        solved_before = set(summary["solved"])
        print_round(label, round_index, summary)

        if eval_tasks and round_index in {1, rounds}:
            evaluate_split(f"public eval after round {round_index}", eval_tasks, library)

    return library


def assign_causal_credit(tasks: dict[str, Task], fronts: Frontier, library: Library, prior_names: set[str]) -> None:
    for task_id, task in tasks.items():
        winner = fronts[task_id][0][1]
        winner_score = fronts[task_id][0][0]
        for name, drop in critical_names(task, winner, prior_names).items():
            entry = library[name]
            entry.impact += drop
            if winner_score == 1.0:
                entry.critical += 1
                entry.covers.add(task_id)
            elif drop >= 0.05:
                entry.helps.add(task_id)


def summarize_round(
    tasks: dict[str, Task],
    fronts: Frontier,
    library: Library,
    prior_names: set[str],
    improved: dict[str, list[tuple[float, Program]]],
    pool: list[Program],
    solved_before: set[str],
    primitive_equivalent_rejections: int,
    new_items: list[dict[str, Program | float | int]],
    top_candidates: list[dict[str, Program | float | int]],
) -> dict[str, object]:
    solved = sorted(task_id for task_id, items in fronts.items() if items[0][0] == 1.0)
    fresh = sorted(set(solved) - solved_before)
    reused = sorted(
        name
        for name in prior_names
        if any(
            any(size(piece) > 1 and render(piece) == name for piece in pieces(items[0][1]))
            for items in fronts.values()
        )
    )
    winner_drops = {
        task_id: causal_drop(tasks[task_id], fronts[task_id][0][1], prior_names)
        for task_id in tasks
        if prior_names and uses_library(fronts[task_id][0][1], prior_names)
    }

    fresh_programs = {task_id: render(fronts[task_id][0][1]) for task_id in fresh[:6]}
    critical_tasks = [task_id for task_id in solved if winner_drops.get(task_id, 0) > 0]
    critical_programs = {task_id: render(fronts[task_id][0][1]) for task_id in critical_tasks[:6]}

    solve_reuse = sum(uses_library(fronts[task_id][0][1], prior_names) for task_id in solved)
    gain_reuse = [
        delta
        for items in improved.values()
        for delta, program in items
        if uses_library(program, prior_names)
    ]
    depths = [entry.depth for entry in library.values()]
    ablation_breaks, ablation_gap = ablate_without_library(tasks, fronts)

    return {
        "task_count": len(tasks),
        "solved": solved,
        "fresh": fresh,
        "fresh_programs": fresh_programs,
        "critical_tasks": critical_tasks,
        "critical_programs": critical_programs,
        "mean_train": sum(items[0][0] for items in fronts.values()) / len(tasks),
        "pool_size": len(pool),
        "reused": reused,
        "new_items": [format_candidate(item) for item in new_items],
        "top_candidates": [format_candidate(item) for item in top_candidates],
        "top_population": top_population(library),
        "top_overlaps": overlaps(library),
        "top_niches": niches(library),
        "overlap_tasks": overlap_tasks(library),
        "population": list(library),
        "metrics": {
            "library_solves": solve_reuse,
            "critical_library_solves": sum(task_id in solved and drop > 0 for task_id, drop in winner_drops.items()),
            "avg_library_delta": sum(gain_reuse) / len(gain_reuse) if gain_reuse else 0.0,
            "avg_counterfactual_drop": sum(winner_drops.values()) / len(winner_drops) if winner_drops else 0.0,
            "survivors": len(prior_names & set(library)),
            "avg_reuse": average([entry.reuse for entry in library.values()]),
            "avg_coverage": average([len(entry.covers) for entry in library.values()]),
            "avg_help": average([len(entry.helps) for entry in library.values()]),
            "union_coverage": len(set().union(*[entry.covers for entry in library.values()])) if library else 0,
            "union_help": len(set().union(*[entry.helps for entry in library.values()])) if library else 0,
            "avg_critical": average([entry.critical for entry in library.values()]),
            "avg_impact": average([entry.impact for entry in library.values()]),
            "pool_per_solve": len(pool) / max(1, len(solved)),
            "lineage_depth_max": max(depths, default=0),
            "lineage_depth_avg": average(depths),
            "new_population_count": len(new_items),
            "primitive_equivalent_rejections": primitive_equivalent_rejections,
            "ablation_breaks": ablation_breaks,
            "ablation_gap": ablation_gap,
        },
    }


def average(values: list[float | int]) -> float:
    return sum(values) / len(values) if values else 0.0


def format_candidate(item: dict[str, Program | float | int]) -> str:
    return f"{render(item['program'])} s{item['support']} g{item['gain']:.2f}"


def top_population(library: Library) -> list[str]:
    ranked = sorted(
        library.items(),
        key=lambda item: (
            -(len(item[1].covers) + len(item[1].helps)),
            -len(item[1].covers),
            -item[1].critical,
            -item[1].impact,
            -item[1].reuse,
            render(item[1].program),
        ),
    )
    return [
        f"{name} u{len(entry.covers)} h{len(entry.helps)} c{entry.critical} i{entry.impact:.2f}"
        for name, entry in ranked[:5]
    ]


def overlaps(library: Library) -> list[str]:
    names = list(library)
    pairs: list[tuple[float, int, str, str]] = []
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            left_tasks = library[left].covers | library[left].helps
            right_tasks = library[right].covers | library[right].helps
            inter = len(left_tasks & right_tasks)
            union = len(left_tasks | right_tasks)
            if inter:
                pairs.append((inter / union, inter, left, right))
    return [
        f"{left} ~ {right} j{jaccard:.2f} n{shared}"
        for jaccard, shared, left, right in sorted(pairs, reverse=True)[:3]
    ]


def niches(library: Library) -> list[str]:
    ranked = sorted(
        library.items(),
        key=lambda item: (
            -(len(item[1].covers) + len(item[1].helps)),
            -len(item[1].covers),
            -item[1].critical,
            -item[1].impact,
            render(item[1].program),
        ),
    )
    return [
        f"{name}: covers={sorted(entry.covers)[:6]} helps={sorted(entry.helps)[:6]}"
        for name, entry in ranked[:3]
    ]


def overlap_tasks(library: Library) -> list[str]:
    names = list(library)
    rows: list[tuple[int, str, str, list[str], list[str], list[str]]] = []
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            left_tasks = library[left].covers | library[left].helps
            right_tasks = library[right].covers | library[right].helps
            shared = sorted(left_tasks & right_tasks)
            if shared:
                rows.append((
                    len(shared),
                    left,
                    right,
                    shared[:6],
                    sorted(left_tasks - right_tasks)[:4],
                    sorted(right_tasks - left_tasks)[:4],
                ))
    return [
        f"{left} & {right}: shared={shared} only_a={only_left} only_b={only_right}"
        for _, left, right, shared, only_left, only_right in sorted(rows, reverse=True)[:2]
    ]


def print_round(label: str, round_index: int, summary: dict[str, object]) -> None:
    metrics = summary["metrics"]
    solved = summary["solved"]
    print(
        f"{label} round {round_index}: {len(solved)}/{summary['task_count']} solved, "
        f"mean train {summary['mean_train']:.3f}, pool {summary['pool_size']}"
    )
    print("newly solved:", summary["fresh"][:12], "..." if len(summary["fresh"]) > 12 else "")
    print("fresh programs:", summary["fresh_programs"])
    print("new abstractions:", summary["new_items"])
    print("reused abstractions:", summary["reused"][:8])
    print("critical solves:", summary["critical_tasks"][:12], "..." if len(summary["critical_tasks"]) > 12 else "")
    print("critical programs:", summary["critical_programs"])
    print("top candidates:", summary["top_candidates"])
    print(
        "metrics: "
        f"library_solves={metrics['library_solves']} "
        f"critical_library_solves={metrics['critical_library_solves']} "
        f"avg_library_delta={metrics['avg_library_delta']:.3f} "
        f"avg_counterfactual_drop={metrics['avg_counterfactual_drop']:.3f} "
        f"survivors={metrics['survivors']} "
        f"avg_reuse={metrics['avg_reuse']:.2f} "
        f"avg_coverage={metrics['avg_coverage']:.2f} "
        f"avg_help={metrics['avg_help']:.2f} "
        f"union_coverage={metrics['union_coverage']} "
        f"union_help={metrics['union_help']} "
        f"avg_critical={metrics['avg_critical']:.2f} "
        f"avg_impact={metrics['avg_impact']:.3f} "
        f"pool_per_solve={metrics['pool_per_solve']:.1f} "
        f"lineage_depth_max={metrics['lineage_depth_max']} "
        f"lineage_depth_avg={metrics['lineage_depth_avg']:.2f} "
        f"new_population_count={metrics['new_population_count']} "
        f"primitive_equivalent_rejections={metrics['primitive_equivalent_rejections']} "
        f"ablation_breaks={metrics['ablation_breaks']} "
        f"ablation_gap={metrics['ablation_gap']:.3f}"
    )
    print("top population:", summary["top_population"])
    print("top overlaps:", summary["top_overlaps"])
    print("top niches:", summary["top_niches"])
    print("overlap tasks:", summary["overlap_tasks"])
    print("population:", summary["population"])


def survivor_rank(entry: Abstraction, already_covered: set[str]) -> tuple:
    new_reach = (entry.covers | entry.helps) - already_covered
    new_exact = entry.covers - already_covered
    return (
        -len(new_reach),
        -len(new_exact),
        -entry.critical,
        -entry.impact,
        -entry.reuse,
        -entry.support,
        -entry.gain,
        entry.age,
        size(entry.program),
        render(entry.program),
    )


def library_rank(entry: Abstraction) -> tuple:
    return (
        -(len(entry.covers) + len(entry.helps)),
        -len(entry.covers),
        -entry.critical,
        -entry.impact,
        -entry.reuse,
        -entry.support,
        -entry.gain,
        entry.age,
        size(entry.program),
        render(entry.program),
    )
