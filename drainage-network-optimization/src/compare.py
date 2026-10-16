"""Phase 6 — Compare GA, NSGA-II and MOEA/D and identify the best.

Builds a reference front from every solution of every run, then scores each algorithm
with Hypervolume (higher better) and IGD to the reference front (lower better), plus
node reduction vs the as-built network, wall-clock, and consistency across seeds. A
Wilcoxon rank-sum test checks whether hypervolume differences are significant.

    python -m src.compare

Writes outputs/figures/*.png and outputs/runs/comparison.csv.

Charts use the Okabe-Ito colorblind-safe categorical palette (fixed hue order, one
axis, legend always present) per the dataviz method.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pymoo.indicators.hv import Hypervolume  # noqa: E402
from pymoo.indicators.igd import IGD  # noqa: E402
from scipy.stats import ranksums  # noqa: E402

from .common import load_config, path  # noqa: E402
from . import encoding  # noqa: E402

# Okabe-Ito: colorblind-safe, fixed order. GA / NSGA-II / MOEA/D.
ALGO_COLOR = {"ga": "#E69F00", "nsga2": "#0072B2", "moead": "#009E73"}
ALGO_LABEL = {"ga": "GA (weighted sum)", "nsga2": "NSGA-II", "moead": "MOEA/D"}
OBJ_LABELS = ["f1  network size", "f2  pipe length", "f3  flooding"]
PENALTY_CUTOFF = 1e8  # rows with any objective above this are infeasible/failed


def load_runs() -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {}
    for npz in sorted(glob.glob(str(path("runs_dir") / "*.npz"))):
        stem = npz[:-4]
        meta = json.load(open(stem + ".json", encoding="utf-8")) if os.path.exists(stem + ".json") else {}
        d = np.load(npz)
        algo = meta.get("algo") or os.path.basename(stem).split("_")[0]
        runs.setdefault(algo, []).append({"X": d["X"], "F": d["F"], "meta": meta})
    return runs


def _feasible(F: np.ndarray) -> np.ndarray:
    return F[(F < PENALTY_CUTOFF).all(axis=1)] if len(F) else F


def non_dominated(F: np.ndarray) -> np.ndarray:
    """Return the non-dominated rows of F (minimisation)."""
    if len(F) == 0:
        return F
    keep = np.ones(len(F), dtype=bool)
    for i in range(len(F)):
        if not keep[i]:
            continue
        dominated = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        keep[dominated] = False
    return F[keep]


def main() -> int:
    cfg = load_config()
    runs = load_runs()
    if not runs:
        print("No runs found. Run e.g. `python -m src.run --algo nsga2 --seed 0` first.")
        return 1

    figdir = path("figures_dir")

    # Combined feasible objectives -> normalization + reference front.
    all_F = np.vstack([r["F"] for rs in runs.values() for r in rs])
    feas = _feasible(all_F)
    lo, hi = feas.min(axis=0), feas.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    norm = lambda F: (F - lo) / span

    ref_front = non_dominated(norm(feas))
    hv = Hypervolume(ref_point=np.array([1.1, 1.1, 1.1]))
    igd = IGD(ref_front)

    g = encoding.graph()
    baseline_nodes = len(set(g.conduits["from_node"]) | set(g.conduits["to_node"]))

    # Per-algorithm, per-seed metrics.
    rows = []
    hv_by_algo: dict[str, list[float]] = {}
    for algo, rs in runs.items():
        for r in rs:
            Ff = _feasible(r["F"])
            if len(Ff) == 0:
                continue
            nF = norm(Ff)
            hv_val = hv(nF)
            igd_val = igd(nF)
            # best node reduction among this run's solutions
            best_red = 0.0
            for x in r["X"]:
                act = len(encoding.active_nodes(x))
                best_red = max(best_red, 100.0 * (1 - act / baseline_nodes))
            rows.append({
                "algo": algo, "seed": r["meta"].get("seed"),
                "hypervolume": hv_val, "igd": igd_val,
                "node_reduction_%": best_red,
                "wall_s": r["meta"].get("wall_seconds"),
                "n_solutions": len(Ff),
            })
            hv_by_algo.setdefault(algo, []).append(hv_val)

    df = pd.DataFrame(rows)
    summary = df.groupby("algo").agg(
        hv_mean=("hypervolume", "mean"), hv_std=("hypervolume", "std"),
        igd_mean=("igd", "mean"), igd_std=("igd", "std"),
        node_red_mean=("node_reduction_%", "mean"),
        node_red_best=("node_reduction_%", "max"),
        wall_mean=("wall_s", "mean"),
    ).reset_index()

    print("\n=== Algorithm comparison (mean ± std over seeds) ===")
    for _, s in summary.iterrows():
        print(f"{ALGO_LABEL[s['algo']]:>18}: "
              f"HV={s['hv_mean']:.4f}±{s['hv_std'] or 0:.4f}  "
              f"IGD={s['igd_mean']:.4f}±{s['igd_std'] or 0:.4f}  "
              f"node_red best={s['node_red_best']:.1f}%  "
              f"wall={s['wall_mean']:.0f}s")

    # Wilcoxon rank-sum on hypervolume between algorithm pairs.
    print("\n=== Wilcoxon rank-sum on hypervolume (p-value) ===")
    algos = list(hv_by_algo)
    for i in range(len(algos)):
        for j in range(i + 1, len(algos)):
            a, b = algos[i], algos[j]
            if len(hv_by_algo[a]) > 1 and len(hv_by_algo[b]) > 1:
                _, p = ranksums(hv_by_algo[a], hv_by_algo[b])
                print(f"  {a} vs {b}: p={p:.4f}")

    df.to_csv(path("runs_dir") / "comparison.csv", index=False)
    summary.to_csv(path("runs_dir") / "comparison_summary.csv", index=False)

    _plot_fronts(runs, figdir)
    _plot_convergence(runs, norm, hv, figdir)
    _plot_node_reduction(summary, figdir)
    print(f"\nFigures -> {figdir}")
    print(f"Tables  -> {path('runs_dir')}/comparison.csv")
    return 0


def _plot_fronts(runs, figdir):
    """Pairwise 2-D projections of the trade-off fronts, colored by algorithm."""
    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (i, j) in zip(axes, pairs):
        for algo, rs in runs.items():
            F = _feasible(np.vstack([r["F"] for r in rs]))
            nd = non_dominated(F)
            ax.scatter(nd[:, i], nd[:, j], s=34, color=ALGO_COLOR[algo],
                       edgecolor="white", linewidth=0.6, label=ALGO_LABEL[algo], zorder=3)
        ax.set_xlabel(OBJ_LABELS[i])
        ax.set_ylabel(OBJ_LABELS[j])
        ax.grid(True, color="#E6E6E6", linewidth=0.8, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Trade-off fronts (non-dominated solutions, 2-D projections)", fontsize=12)
    fig.tight_layout()
    fig.savefig(figdir / "pareto_fronts.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_convergence(runs, norm, hv, figdir):
    """Hypervolume vs evaluations, averaged over seeds (algorithms with history)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    plotted = False
    for algo, rs in runs.items():
        curves = []
        for r in rs:
            hist = r["meta"].get("history") or []
            if not hist:
                continue
            xs = [h["n_eval"] for h in hist]
            ys = [hv(norm(_feasible(np.array(h["F"])))) if len(_feasible(np.array(h["F"]))) else 0.0
                  for h in hist]
            curves.append((xs, ys))
        if not curves:
            continue
        # align on the shortest curve length for averaging
        m = min(len(xs) for xs, _ in curves)
        xs = np.array(curves[0][0][:m])
        ymean = np.mean([ys[:m] for _, ys in curves], axis=0)
        ax.plot(xs, ymean, color=ALGO_COLOR[algo], linewidth=2, label=ALGO_LABEL[algo])
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("SWMM evaluations")
    ax.set_ylabel("hypervolume (higher = better)")
    ax.set_title("Convergence: hypervolume vs evaluation budget")
    ax.grid(True, color="#E6E6E6", linewidth=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "convergence_hypervolume.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_node_reduction(summary, figdir):
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    algos = summary["algo"].tolist()
    vals = summary["node_red_best"].tolist()
    colors = [ALGO_COLOR[a] for a in algos]
    bars = ax.bar([ALGO_LABEL[a] for a in algos], vals, color=colors, width=0.6, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=10, color="#333333")
    ax.set_ylabel("best node reduction vs as-built")
    ax.set_title("Network simplification by algorithm")
    ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.8, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(figdir / "node_reduction.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
