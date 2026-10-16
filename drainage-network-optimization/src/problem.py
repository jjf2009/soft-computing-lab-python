"""Phase 5 (part 1) — pymoo problem, repair operator and shared binary operators.

One Problem, one encoding, one evaluation function, one set of operators: the three
algorithms differ only in the search strategy, so the comparison is fair.
"""
from __future__ import annotations

import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling

from .common import load_config
from . import encoding, objectives


class DrainageProblem(ElementwiseProblem):
    """Multi-objective (f1, f2, f3) binary problem over candidate conduits."""

    def __init__(self, **kwargs):
        n = encoding.n_genes()
        super().__init__(n_var=n, n_obj=3, n_ieq_constr=0, xl=0, xu=1, vtype=bool, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.array(objectives.evaluate(np.asarray(x, dtype=np.int8)), dtype=float)


class WeightedDrainageProblem(ElementwiseProblem):
    """Single-objective view for the baseline GA: a normalised weighted sum of f1,f2,f3."""

    def __init__(self, weights, norm, **kwargs):
        n = encoding.n_genes()
        self.weights = np.asarray(weights, dtype=float)
        self.norm = np.asarray(norm, dtype=float)
        super().__init__(n_var=n, n_obj=1, n_ieq_constr=0, xl=0, xu=1, vtype=bool, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        f = np.array(objectives.evaluate(np.asarray(x, dtype=np.int8)), dtype=float)
        out["F"] = float(np.sum(self.weights * (f / self.norm)))
        out["_raw_F"] = f  # keep the true 3-objective vector for later comparison


class DrainageRepair(Repair):
    """Apply the connectivity repair to every individual so the population stays feasible."""

    def _do(self, problem, X, **kwargs):
        X = np.asarray(X, dtype=np.int8)
        return np.array([encoding.repair(row) for row in X], dtype=np.int8)


def make_operators():
    """Shared sampling / crossover / mutation, configured from config.yaml."""
    cfg = load_config()["optimization"]
    if cfg["crossover"] == "uniform":
        crossover = UniformCrossover(prob=cfg["crossover_prob"])
    else:
        crossover = TwoPointCrossover(prob=cfg["crossover_prob"])
    mut_prob = cfg["mutation_prob"]  # None -> pymoo default (~1/n_var)
    mutation = BitflipMutation(prob=mut_prob) if mut_prob else BitflipMutation()
    return {
        "sampling": BinaryRandomSampling(),
        "crossover": crossover,
        "mutation": mutation,
        "repair": DrainageRepair(),
    }
