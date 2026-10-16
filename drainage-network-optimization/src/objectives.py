"""Phase 4 (part 2) — The three objectives and the cheap feasibility gate.

All three are MINIMISED:
  f1  feature reduction : active_conduits + active_nodes (+ parallel-pipe penalty)
  f2  pipe material     : total length over active conduits
  f3  hydraulic validity: flood volume + flooded nodes + hours flooded (+ uphill penalty)

The feasibility gate runs the cheap checks first and, if a layout cannot drain even
after repair (a data gap), returns a large penalty WITHOUT paying for a SWMM run.
"""
from __future__ import annotations

import functools

import numpy as np

from .common import load_config
from . import encoding, swmm_eval

# A penalty vector returned for layouts that cannot be simulated meaningfully.
_PENALTY = (1e9, 1e9, 1e12)


@functools.lru_cache(maxsize=1)
def _parallel_groups() -> dict[tuple[str, str], list[int]]:
    """Map each unordered node pair to the gene indices of conduits between them."""
    g = encoding.graph()
    groups: dict[tuple[str, str], list[int]] = {}
    for i, (a, b) in enumerate(g.endpoints):
        key = (a, b) if a <= b else (b, a)
        groups.setdefault(key, []).append(i)
    return {k: v for k, v in groups.items() if len(v) > 1}


def f1_feature_reduction(bits: np.ndarray) -> float:
    g = encoding.graph()
    cfg = load_config()
    active = encoding.active_indices(bits)
    n_conduits = len(active)
    n_nodes = len(encoding.active_nodes(bits))

    active_set = set(active.tolist())
    redundant = 0
    for idxs in _parallel_groups().values():
        on = sum(1 for i in idxs if i in active_set)
        if on > 1:
            redundant += on - 1
    penalty = cfg["objectives"]["parallel_pipe_penalty"] * redundant
    return float(n_conduits + n_nodes + penalty)


def f2_pipe_length(bits: np.ndarray) -> float:
    g = encoding.graph()
    active = encoding.active_indices(bits)
    return float(g.lengths[active].sum())


def f3_hydraulic(bits: np.ndarray) -> float:
    g = encoding.graph()
    cfg = load_config()
    res = swmm_eval.simulate(bits)
    uphill_active = int(np.sum(~g.downhill[encoding.active_indices(bits)]))
    uphill_pen = cfg["objectives"]["uphill_penalty"] * uphill_active
    return float(res.flood_volume + res.flooded_nodes + res.hours_flooded + uphill_pen)


def evaluate(bits: np.ndarray) -> tuple[float, float, float]:
    """Return (f1, f2, f3). Feasibility gate skips the SWMM run for hopeless layouts."""
    bits = np.asarray(bits, dtype=np.int8)

    # Cheap gate: no active conduits, or an inlet unreachable even here -> penalise.
    if bits.sum() == 0 or encoding.disconnected_inlets(bits):
        return _PENALTY

    return f1_feature_reduction(bits), f2_pipe_length(bits), f3_hydraulic(bits)


def evaluate_batch(pop_bits: np.ndarray) -> np.ndarray:
    """Evaluate a 2-D array of chromosomes -> (n, 3) objective matrix (serial)."""
    return np.array([evaluate(row) for row in pop_bits], dtype=float)


@functools.lru_cache(maxsize=1)
def baseline_objectives() -> tuple[float, float, float]:
    """(f1, f2, f3) of the unmodified as-built network (all conduits active).

    This is the reference design every optimized layout is compared against, and the
    per-objective scale used to normalise the weighted-sum GA.
    """
    g = encoding.graph()
    ones = np.ones(g.n_genes, dtype=np.int8)
    f1 = f1_feature_reduction(ones)
    f2 = f2_pipe_length(ones)
    res = swmm_eval.baseline_result()
    uphill_active = int(np.sum(~g.downhill))
    f3 = float(res.flood_volume + res.flooded_nodes + res.hours_flooded
               + load_config()["objectives"]["uphill_penalty"] * uphill_active)
    return f1, f2, f3
