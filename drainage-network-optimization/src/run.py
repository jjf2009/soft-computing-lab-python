"""Phase 5 (part 2) — Run one algorithm (GA | NSGA-II | MOEA/D) for one seed.

All three share the same encoding, evaluation, operators and evaluation budget
(pop_size * n_gen), so result differences reflect the algorithm, not the setup.

    python -m src.run --algo nsga2 --seed 0
    python -m src.run --algo moead --seed 0
    python -m src.run --algo ga    --seed 0
    python -m src.run --algo nsga2 --seed 0 --serial    # no multiprocessing (debug)

Results are written to outputs/runs/<algo>_seed<seed>.npz (+ .json meta).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from .common import load_config, path
from . import build_graph, encoding, objectives, problem


def _ensure_tables():
    tables = path("tables_dir")
    if not (tables / "candidate_conduits.parquet").exists():
        print("Graph tables missing — building from the SWMM model ...")
        build_graph.build()


def _baseline(norm_path) -> tuple[float, float, float]:
    base = objectives.baseline_objectives()
    with open(norm_path, "w", encoding="utf-8") as fh:
        json.dump({"f1": base[0], "f2": base[1], "f3": base[2]}, fh, indent=2)
    print(f"Baseline (as-built) objectives: f1={base[0]:.1f}  f2={base[1]:.1f}  f3={base[2]:.3g}")
    return base


def _ref_dirs_for(pop_size: int):
    """Pick a das-dennis partition count giving ~pop_size reference directions (3 obj)."""
    best_p, best = 1, None
    for p in range(1, 40):
        n = (p + 1) * (p + 2) // 2  # C(p+2, 2) points for 3 objectives
        if best is None or abs(n - pop_size) < abs(best - pop_size):
            best, best_p = n, p
        if n >= pop_size:
            break
    return get_reference_directions("das-dennis", 3, n_partitions=best_p)


def _make_pool(n_workers: int, serial: bool):
    if serial or n_workers <= 1:
        return None, None
    import multiprocessing as mp
    from pymoo.parallelization import StarmapParallelization

    pool = mp.Pool(n_workers)
    return pool, StarmapParallelization(pool.starmap)


def run_moo(algo: str, seed: int, pop: int, n_gen: int, ops: dict, runner) -> tuple[np.ndarray, np.ndarray, list]:
    prob = problem.DrainageProblem(elementwise_runner=runner) if runner else problem.DrainageProblem()
    if algo == "nsga2":
        algorithm = NSGA2(pop_size=pop, **ops)
    elif algo == "moead":
        ref_dirs = _ref_dirs_for(pop)
        algorithm = MOEAD(ref_dirs, n_neighbors=min(20, len(ref_dirs)),
                          prob_neighbor_mating=0.9, **ops)
    else:
        raise ValueError(algo)

    res = minimize(prob, algorithm, ("n_gen", n_gen), seed=seed,
                   save_history=True, verbose=True)
    X = np.atleast_2d(res.X).astype(np.int8)
    F = np.atleast_2d(res.F).astype(float)
    history = _extract_history(res)
    return X, F, history


def run_ga(seed: int, pop: int, n_gen: int, ops: dict, norm, runner) -> tuple[np.ndarray, np.ndarray, list]:
    """Weighted-sum GA swept over several weight sets -> an approximate trade-off set."""
    cfg = load_config()["optimization"]
    X_rows, F_rows = [], []
    for w in cfg["ga_weight_sets"]:
        prob = (problem.WeightedDrainageProblem(w, norm, elementwise_runner=runner)
                if runner else problem.WeightedDrainageProblem(w, norm))
        algorithm = GA(pop_size=pop, **ops)
        res = minimize(prob, algorithm, ("n_gen", n_gen), seed=seed, verbose=False)
        x = np.asarray(res.X, dtype=np.int8).ravel()
        X_rows.append(x)
        F_rows.append(objectives.evaluate(x))  # true 3-objective vector of the best
        print(f"  GA weights {w} -> f=({F_rows[-1][0]:.1f}, {F_rows[-1][1]:.1f}, {F_rows[-1][2]:.3g})")
    return np.array(X_rows, dtype=np.int8), np.array(F_rows, dtype=float), []


def _extract_history(res) -> list:
    """Per-generation (n_eval, objective matrix) for convergence / hypervolume plots."""
    hist = []
    for entry in (res.history or []):
        F = entry.opt.get("F")
        if F is not None:
            hist.append({"n_eval": int(entry.evaluator.n_eval), "F": np.atleast_2d(F).tolist()})
    return hist


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algo", required=True, choices=["ga", "nsga2", "moead"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pop", type=int, default=None)
    ap.add_argument("--gen", type=int, default=None)
    ap.add_argument("--serial", action="store_true", help="disable multiprocessing")
    args = ap.parse_args()

    cfg = load_config()
    opt = cfg["optimization"]
    pop = args.pop or opt["pop_size"]
    n_gen = args.gen or opt["n_gen"]

    _ensure_tables()
    g = encoding.graph()
    print(f"Problem: {g.n_genes} genes (conduits), scope={cfg['scope']}, "
          f"pop={pop}, gen={n_gen}, budget≈{pop*n_gen} evals")

    norm_path = path("tables_dir") / "baseline.json"
    norm = _baseline(norm_path)

    ops = problem.make_operators()
    pool, runner = _make_pool(opt["n_workers"], args.serial)
    t0 = time.time()
    try:
        if args.algo == "ga":
            X, F, history = run_ga(args.seed, pop, n_gen, ops, norm, runner)
        else:
            X, F, history = run_moo(args.algo, args.seed, pop, n_gen, ops, runner)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    wall = time.time() - t0

    runs = path("runs_dir")
    stem = runs / f"{args.algo}_seed{args.seed}"
    np.savez_compressed(f"{stem}.npz", X=X, F=F)
    meta = {
        "algo": args.algo, "seed": args.seed, "scope": cfg["scope"],
        "pop": pop, "n_gen": n_gen, "n_solutions": int(len(F)),
        "wall_seconds": round(wall, 2),
        "baseline": {"f1": norm[0], "f2": norm[1], "f3": norm[2]},
        "history": history,
    }
    with open(f"{stem}.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\n{args.algo} seed {args.seed}: {len(F)} solutions in {wall:.1f}s -> {stem}.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
