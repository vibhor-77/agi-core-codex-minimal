import json, os, sys
from pathlib import Path

"""Minimal 4-pillar loop over a tiny program language and an evolving abstraction population."""

def flip_h(g): return [r[::-1] for r in g]
def flip_v(g): return g[::-1]
def transpose(g): return [list(r) for r in zip(*g)]
def nonzero_mask(g): return [[1 if v else 0 for v in r] for r in g]
GRID = {"identity": lambda g: g, "flip_h": flip_h, "flip_v": flip_v, "transpose": transpose}
MASK = {"nonzero_mask": nonzero_mask}
PRIMS = GRID | MASK

def kind(p):
    if isinstance(p, str): return "mask" if p in MASK else "grid"
    return "grid" if p[0] in {"chain", "local"} else None

def box(mask):
    cells = [(i, j) for i, row in enumerate(mask) for j, v in enumerate(row) if v]
    if not cells: return None
    rows, cols = zip(*cells)
    return min(rows), max(rows) + 1, min(cols), max(cols) + 1

def crop(g, b):
    r0, r1, c0, c1 = b
    return [row[c0:c1] for row in g[r0:r1]]

def fit(g, h, w):
    return [[g[i][j] if i < len(g) and j < len(g[0]) else 0 for j in range(w)] for i in range(h)]

def paste(base, patch, b):
    r0, r1, c0, c1 = b
    out = [row[:] for row in base]
    for i in range(r1 - r0):
        for j in range(c1 - c0):
            out[r0 + i][c0 + j] = patch[i][j]
    return out

def run(p, g):
    if isinstance(p, str): return PRIMS[p](g)
    if p[0] == "chain": return run(p[2], run(p[1], g))
    mask = run(p[1], g)
    b = box(mask)
    if not b: return g
    patch = run(p[2], crop(g, b))
    return paste(g, fit(patch, b[1] - b[0], b[3] - b[2]), b)

def show(p): return p if isinstance(p, str) else f"{p[0]}({show(p[1])}, {show(p[2])})"
def size(p): return 1 if isinstance(p, str) else 1 + size(p[1]) + size(p[2])
def cost(p, names=()):
    if size(p) > 1 and show(p) in set(names): return 1
    return 1 if isinstance(p, str) else 1 + cost(p[1], names) + cost(p[2], names)
def subtrees(p): return [] if isinstance(p, str) else [p, *subtrees(p[1]), *subtrees(p[2])]
def pieces(p): return [p, *subtrees(p)]

def score(p, examples):
    if kind(p) != "grid": return 0.0
    total = hits = 0
    for inp, out in examples:
        got = run(p, inp)
        if len(got) != len(out) or len(got[0]) != len(out[0]): continue
        total += sum(len(r) for r in out)
        hits += sum(a == b for ra, rb in zip(got, out) for a, b in zip(ra, rb))
    return hits / total if total else 0.0

def signature(p, inputs):
    return tuple(tuple(tuple(row) for row in run(p, grid)) for grid in inputs)

def unique(programs):
    seen, out = set(), []
    for p in programs:
        name = show(p)
        if name not in seen:
            out.append(p)
            seen.add(name)
    return out

def uses_library(p, names):
    known = set(names)
    return any(size(t) > 1 and show(t) in known for t in pieces(p))

def mutate(p, parts, selectors):
    out = []
    if isinstance(p, str): return out
    if p[0] == "chain":
        for q in parts:
            if show(q) != show(p[1]): out.append(("chain", q, p[2]))
            if show(q) != show(p[2]): out.append(("chain", p[1], q))
    if p[0] == "local":
        for s in selectors:
            if s != p[1]: out.append(("local", s, p[2]))
        for q in parts:
            if show(q) != show(p[2]): out.append(("local", p[1], q))
    for child in p[1:]:
        for q in mutate(child, parts, selectors):
            out.append(("chain", q, p[2]) if p[0] == "chain" and child is p[1] else ("chain", p[1], q) if p[0] == "chain" else ("local", p[1], q))
    return out

def spawn(library, fronts, width=8, max_size=9):
    frontier_programs = unique([p for items in fronts.values() for _, p in items if kind(p) == "grid"])
    library_programs = [e["program"] for e in sorted(library.values(), key=lambda e: (-(len(e["covers"]) + len(e["helps"])), -len(e["covers"]), -e["critical"], -e["impact"], -e["reuse"], -e["support"], -e["gain"], e["age"], size(e["program"]), show(e["program"]))) if kind(e["program"]) == "grid"]
    evolvers = unique(library_programs[:width] + frontier_programs[:width])
    selectors = unique(list(MASK) + [t for p in evolvers for t in pieces(p) if kind(t) == "mask"])[:2]
    locals_ = [("local", s, t) for s in selectors for t in unique(list(GRID) + evolvers)[:width] if show(t) != "identity"]
    parts = unique([t for p in list(GRID) + evolvers + locals_ for t in pieces(p) if kind(t) == "grid"])[: width * 3]
    pool = list(GRID) + evolvers + locals_
    for p in evolvers:
        for q in GRID:
            pool += [("chain", q, p), ("chain", p, q)]
        for s in selectors:
            if show(p) != "identity": pool.append(("local", s, p))
        pool += mutate(p, parts[:width], selectors)
    for a in parts[:width]:
        for b in parts[:width]:
            if show(a) != show(b): pool.append(("chain", a, b))
    return [p for p in unique(pool) if kind(p) == "grid" and size(p) <= max_size]

def frontier(task, pool, keep=3, old=(), library=()):
    ranked = sorted(((score(p, task["train"]), p) for p in unique(list(old) + pool)), key=lambda item: (-item[0], cost(item[1], library), size(item[1]), show(item[1])))
    chosen, seen = [], set()
    inputs = [inp for inp, _ in task["train"]]
    for quality, program in ranked:
        sig = signature(program, inputs)
        if sig in seen: continue
        chosen.append((quality, program))
        seen.add(sig)
        if len(chosen) == keep: break
    return chosen

def parents_of(program):
    return [] if isinstance(program, str) else [show(program[1]), show(program[2])]

def nodes(p, path=()):
    out = [(path, p)]
    if isinstance(p, str): return out
    return out + nodes(p[1], path + (1,)) + nodes(p[2], path + (2,))

def replace(p, path, new):
    if not path: return new
    q = list(p)
    q[path[0]] = replace(p[path[0]], path[1:], new)
    return tuple(q)

def causal_drop(task, program, names):
    if not uses_library(program, names): return 0.0
    base = score(program, task["train"])
    alts = unique([replace(program, path, "identity") for path, node in nodes(program) if size(node) > 1 and show(node) in names])
    return max(0.0, base - max([score(p, task["train"]) for p in alts] or [0.0]))

def critical_names(task, program, names):
    if not uses_library(program, names): return {}
    base, out = score(program, task["train"]), {}
    for name in names:
        alts = unique([replace(program, path, "identity") for path, node in nodes(program) if size(node) > 1 and show(node) == name])
        if not alts: continue
        drop = max(0.0, base - max([score(p, task["train"]) for p in alts] or [0.0]))
        if drop > 0: out[name] = drop
    return out

def depth_of(program, library):
    if isinstance(program, str): return 0
    child_depths = []
    for child in program[1:]:
        name = show(child)
        child_depths.append(library[name]["depth"] if name in library else depth_of(child, library))
    return 1 + max(child_depths, default=0)

def keep_alive(alive, cap):
    chosen, covered = [], set()
    while alive and len(chosen) < cap:
        alive.sort(key=lambda e: (-len((e["covers"] | e["helps"]) - covered), -len(e["covers"] - covered), -e["critical"], -e["impact"], -e["reuse"], -e["support"], -e["gain"], e["age"], size(e["program"]), show(e["program"])))
        best = alive.pop(0)
        chosen.append(best)
        covered |= best["covers"] | best["helps"]
    return chosen

def overlaps(library):
    names, pairs = list(library), []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sa = library[a]["covers"] | library[a]["helps"]
            sb = library[b]["covers"] | library[b]["helps"]
            inter, union = len(sa & sb), len(sa | sb)
            if inter: pairs.append((inter / union, inter, a, b))
    return [f"{a} ~ {b} j{j:.2f} n{n}" for j, n, a, b in sorted(pairs, reverse=True)[:3]]

def evolve(library, improved, samples, cap=12):
    for entry in library.values():
        entry["age"] += 1
        entry["used"] = False
    gains = {}
    for candidates in improved.values():
        used, task_best = set(), {}
        for delta, leader in candidates:
            for tree in [t for t in pieces(leader) if size(t) > 1]:
                name = show(tree)
                used.add(name)
                if name not in task_best or delta > task_best[name][1]:
                    task_best[name] = (tree, delta)
        for name, (tree, delta) in task_best.items():
            item = gains.setdefault(name, {"program": tree, "gain": 0.0, "support": 0})
            item["gain"] += delta
            item["support"] += 1
        for name in used:
            if name in library:
                library[name]["reuse"] += 1
                library[name]["used"] = True
                library[name]["age"] = 0
    base_sigs = {signature(name, samples) for name in PRIMS}
    seen = base_sigs | {signature(entry["program"], samples) for entry in library.values()}
    ranked = sorted(gains.values(), key=lambda item: (-item["support"], -item["gain"], size(item["program"]), show(item["program"])))
    new, primitive_equivalent_rejections = [], 0
    for item in ranked:
        name, sig = show(item["program"]), signature(item["program"], samples)
        if sig in base_sigs:
            primitive_equivalent_rejections += 1
            continue
        if item["support"] > 1 and item["gain"] / item["support"] >= 0.8 and sig not in seen and name not in library:
            library[name] = {
                "program": item["program"],
                "gain": item["gain"],
                "support": item["support"],
                "covers": set(),
                "helps": set(),
                "critical": 0,
                "impact": 0.0,
                "reuse": 0,
                "age": 0,
                "used": False,
                "parents": parents_of(item["program"]),
                "depth": depth_of(item["program"], library),
            }
            seen.add(sig)
            new.append(item)
        elif name in library:
            library[name]["gain"] += item["gain"]
            library[name]["support"] += item["support"]
    alive = [e for e in library.values() if e["age"] == 0 or e["critical"] > 0 or e["helps"] or e["impact"] >= 0.1]
    keep = keep_alive(alive, cap)
    library = {show(e["program"]): e for e in keep}
    return library, new[:cap], ranked[:5], primitive_equivalent_rejections

def sample_inputs(tasks, limit=8):
    grids = []
    for task in tasks.values():
        for inp, _ in task["train"]:
            grids.append(inp)
            if len(grids) == limit: return grids
    return grids

def evaluate(label, tasks, library):
    pool = spawn(library, {})
    exact = mean = 0.0
    for task in tasks.values():
        best = frontier(task, pool, keep=1, library=library)[0][1]
        s = score(best, task["test"])
        exact += s == 1.0
        mean += s
    print(f"{label}: {int(exact)}/{len(tasks)} exact, mean test {mean / len(tasks):.3f}")

def ablate(tasks, fronts):
    pool = spawn({}, {})
    breaks = gap = 0.0
    for task_id, task in tasks.items():
        base = frontier(task, pool, keep=1)[0][0]
        best = fronts[task_id][0][0]
        breaks += best == 1.0 and base < 1.0
        gap += best - base
    return int(breaks), gap / len(tasks)

def learn(label, tasks, eval_tasks=None, rounds=3, keep=4, library=None, freeze=False):
    library, fronts, inputs, solved_before = ({} if library is None else dict(library)), {}, sample_inputs(tasks), set()
    for r in range(1, rounds + 1):
        prior = set(library)
        pool = spawn(library, fronts)
        improved = {}
        for task_id, task in tasks.items():
            before = fronts.get(task_id, [(0.0, "identity")])[0][0]
            fronts[task_id] = frontier(task, pool, keep, [p for _, p in fronts.get(task_id, [])], library)
            better = [(quality - before, program) for quality, program in fronts[task_id] if quality > before]
            if better: improved[task_id] = better
        if prior:
            for task_id in tasks:
                for name, drop in critical_names(tasks[task_id], fronts[task_id][0][1], prior).items():
                    library[name]["impact"] += drop
                    if fronts[task_id][0][0] == 1.0:
                        library[name]["critical"] += 1
                        library[name]["covers"].add(task_id)
                    elif drop >= 0.05:
                        library[name]["helps"].add(task_id)
        if freeze:
            new, top, primitive_equivalent_rejections = [], [], 0
        else:
            library, new, top, primitive_equivalent_rejections = evolve(library, improved, inputs)
        solved = sorted(task_id for task_id, items in fronts.items() if items[0][0] == 1.0)
        fresh = sorted(set(solved) - solved_before)
        solved_before = set(solved)
        mean = sum(items[0][0] for items in fronts.values()) / len(tasks)
        reused = sorted(name for name in prior if any(any(size(t) > 1 and show(t) == name for t in pieces(items[0][1])) for items in fronts.values()))
        solve_reuse = sum(uses_library(fronts[task_id][0][1], prior) for task_id in solved)
        gain_reuse = [delta for items in improved.values() for delta, program in items if uses_library(program, prior)]
        winner_drops = {task_id: causal_drop(tasks[task_id], fronts[task_id][0][1], prior) for task_id in tasks if prior and uses_library(fronts[task_id][0][1], prior)}
        critical_solves = sum(task_id in solved and drop > 0 for task_id, drop in winner_drops.items())
        survivors = len(prior & set(library))
        avg_reuse = sum(entry["reuse"] for entry in library.values()) / len(library) if library else 0.0
        avg_help = sum(len(entry["helps"]) for entry in library.values()) / len(library) if library else 0.0
        avg_coverage = sum(len(entry["covers"]) for entry in library.values()) / len(library) if library else 0.0
        avg_critical = sum(entry["critical"] for entry in library.values()) / len(library) if library else 0.0
        avg_impact = sum(entry["impact"] for entry in library.values()) / len(library) if library else 0.0
        union_cover = len(set().union(*[entry["covers"] for entry in library.values()])) if library else 0
        union_help = len(set().union(*[entry["helps"] for entry in library.values()])) if library else 0
        depths = [entry["depth"] for entry in library.values()]
        ablation_breaks, ablation_gap = ablate(tasks, fronts)
        fresh_programs = {task_id: show(fronts[task_id][0][1]) for task_id in fresh[:6]}
        critical_tasks = [task_id for task_id in solved if winner_drops.get(task_id, 0) > 0]
        critical_programs = {task_id: show(fronts[task_id][0][1]) for task_id in critical_tasks[:6]}
        top_population = [f"{name} u{len(entry['covers'])} h{len(entry['helps'])} c{entry['critical']} i{entry['impact']:.2f}" for name, entry in sorted(library.items(), key=lambda item: (-(len(item[1]["covers"]) + len(item[1]["helps"])), -len(item[1]["covers"]), -item[1]["critical"], -item[1]["impact"], -item[1]["reuse"], show(item[1]["program"])))[:5]]
        top_overlaps = overlaps(library)
        print(f"{label} round {r}: {len(solved)}/{len(tasks)} solved, mean train {mean:.3f}, pool {len(pool)}")
        print("newly solved:", fresh[:12], "..." if len(fresh) > 12 else "")
        print("fresh programs:", fresh_programs)
        print("new abstractions:", [f"{show(item['program'])} s{item['support']} g{item['gain']:.2f}" for item in new])
        print("reused abstractions:", reused[:8])
        print("critical solves:", critical_tasks[:12], "..." if len(critical_tasks) > 12 else "")
        print("critical programs:", critical_programs)
        print("top candidates:", [f"{show(item['program'])} s{item['support']} g{item['gain']:.2f}" for item in top])
        print(f"metrics: library_solves={solve_reuse} critical_library_solves={critical_solves} avg_library_delta={sum(gain_reuse) / len(gain_reuse) if gain_reuse else 0:.3f} avg_counterfactual_drop={sum(winner_drops.values()) / len(winner_drops) if winner_drops else 0:.3f} survivors={survivors} avg_reuse={avg_reuse:.2f} avg_coverage={avg_coverage:.2f} avg_help={avg_help:.2f} union_coverage={union_cover} union_help={union_help} avg_critical={avg_critical:.2f} avg_impact={avg_impact:.3f} pool_per_solve={len(pool) / max(1, len(solved)):.1f} lineage_depth_max={max(depths, default=0)} lineage_depth_avg={(sum(depths) / len(depths)) if depths else 0:.2f} new_population_count={len(new)} primitive_equivalent_rejections={primitive_equivalent_rejections} ablation_breaks={ablation_breaks} ablation_gap={ablation_gap:.3f}")
        print("top population:", top_population)
        print("top overlaps:", top_overlaps)
        print("population:", [name for name in library])
        if eval_tasks and r in {1, rounds}: evaluate(f"public eval after round {r}", eval_tasks, library)
    return library

def task(grid, program):
    out = run(program, grid)
    return {"train": [(grid, out)], "test": [(grid, out)]}

def synthetic_stages():
    pair = ("local", "nonzero_mask", "flip_h")
    triple_t = ("chain", pair, "transpose")
    triple_v = ("chain", "flip_v", pair)
    quad_t = ("local", "nonzero_mask", triple_t)
    quad_v = ("chain", triple_v, "transpose")
    quint_t = ("chain", quad_t, "transpose")
    quint_v = ("chain", quad_v, "flip_v")
    return [
        ("stage_1", {
            "flip_h": task([[1, 0], [0, 0]], "flip_h"),
            "transpose": task([[1, 0], [2, 3]], "transpose"),
            "pair_1": task([[1, 2, 0], [3, 0, 0]], pair),
            "pair_2": task([[0, 4, 5], [0, 6, 0]], pair),
        }),
        ("stage_2", {
            "triple_t_1": task([[1, 2, 0], [3, 0, 0]], triple_t),
            "triple_t_2": task([[0, 4, 5], [0, 6, 0]], triple_t),
            "triple_v_1": task([[0, 2, 3], [0, 2, 2], [0, 2, 1]], triple_v),
            "triple_v_2": task([[1, 1, 0], [1, 0, 0], [2, 2, 0]], triple_v),
        }),
        ("stage_3", {
            "quad_t_1": task([[3, 0, 2, 3], [3, 2, 3, 2]], quad_t),
            "quad_t_2": task([[2, 3, 0], [0, 0, 0], [1, 1, 0]], quad_t),
            "quad_v_1": task([[3, 0, 0], [2, 3, 0], [2, 2, 0]], quad_v),
            "quad_v_2": task([[0, 3, 0], [0, 3, 1], [0, 3, 3]], quad_v),
        }),
        ("stage_4", {
            "quint_t_1": task([[3, 0, 2, 3], [3, 2, 3, 2]], quint_t),
            "quint_t_2": task([[1, 2, 0], [3, 4, 0], [5, 0, 0]], quint_t),
            "quint_v_1": task([[2, 3, 0], [0, 0, 0], [1, 1, 0]], quint_v),
            "quint_v_2": task([[0, 2, 0], [3, 1, 0], [3, 3, 0]], quint_v),
        }),
    ]

def synthetic_choice():
    pair = ("local", "nonzero_mask", "flip_h")
    triple_t = ("chain", pair, "transpose")
    triple_v = ("chain", "flip_v", pair)
    quad_t = ("local", "nonzero_mask", triple_t)
    quad_v = ("chain", triple_v, "transpose")
    quint_t = ("chain", quad_t, "transpose")
    quint_v = ("chain", quad_v, "flip_v")
    return {
        "choice_quad_t": task([[1, 2, 0], [3, 4, 0], [5, 0, 0]], quad_t),
        "choice_quad_v": task([[0, 3, 0], [0, 3, 1], [0, 3, 3]], quad_v),
        "choice_quint_t": task([[3, 0, 2, 3], [3, 2, 3, 2]], quint_t),
        "choice_quint_v": task([[2, 3, 0], [0, 0, 0], [1, 1, 0]], quint_v),
    }

def synthetic():
    library = {}
    for stage, tasks in synthetic_stages():
        library = learn(f"synthetic {stage}", tasks, rounds=1, library=library)
    learn("synthetic choice", synthetic_choice(), rounds=1, library=library, freeze=True)

def arc_root():
    here = Path(__file__).resolve().parent
    checked = [os.getenv("ARC_AGI_1_TRAIN_DIR"), here / "data/ARC-AGI/data/training", here / "../agi-core/data/ARC-AGI/data/training"]
    root = next((Path(path) for path in checked if path and Path(path).is_dir()), None)
    if root: return root
    print("missing ARC dataset; checked:")
    for path in checked: print(f"- {path or '(unset) ARC_AGI_1_TRAIN_DIR'}")
    raise SystemExit(1)

def load(path):
    data = json.loads(path.read_text())
    return {"train": [(e["input"], e["output"]) for e in data["train"]], "test": [(e["input"], e["output"]) for e in data["test"]]}

def arc(split):
    root = arc_root()
    base = root if split == "training" else root.parent / split
    return {path.stem: load(path) for path in sorted(base.glob("*.json"))}

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode in {"synthetic", "both"}: synthetic()
    if mode in {"arc", "both"}: learn("arc train", arc("training"), arc("evaluation"))
