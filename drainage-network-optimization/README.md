# Drainage Network Optimization

**Soft Computing mini project — multi-objective feature selection on an urban
drainage network using evolutionary algorithms.**

Each candidate conduit of the real-world **Bellinge** network (EPA SWMM model) is a
binary gene: keep it or drop it. Three evolutionary algorithms search for conduit
subsets that jointly minimise three conflicting objectives, and a real hydrodynamic
simulation (SWMM, via PySWMM) judges every candidate layout.

| Objective | Meaning | Direction |
|---|---|---|
| **f1** | network size — active conduits + active nodes (+ parallel-pipe penalty) | minimise |
| **f2** | total pipe length over active conduits (material/cost) | minimise |
| **f3** | flooding — flood volume + flooded nodes + hours flooded (+ uphill penalty) | minimise |

Three algorithms, one shared encoding / evaluation / budget (fair comparison):

- **GA** — weighted-sum baseline, swept over several weight sets.
- **NSGA-II** — elitist non-dominated sorting + crowding distance.
- **MOEA/D** — Tchebycheff decomposition over a set of weight vectors.

All three, plus the Hypervolume and IGD indicators, come from **pymoo**.

## Why this fixes the "dataset too big" problem

The Colab attempt died on the dataset size. That was really **two** problems, both
fixed here:

1. **Data volume.** The raw/cleaned time-series zips are huge. But the optimizer
   never needs them — only a short *design storm*. [`src/preprocess.py`](src/preprocess.py)
   streams the zips in chunks (memory stays flat) and keeps only the peak window.
   Run once; the search loop then reads nothing but tiny tables.
2. **Compute.** `f3` is a full SWMM simulation *per candidate*. The killers are
   solved by: running **locally** (16 cores, no Colab timeout), **parallel**
   evaluation across cores, **caching** each layout by its bitset, a cheap
   **feasibility gate** + connectivity **repair** so no doomed layout is ever
   simulated, a **shortened** storm window, and a **sub-catchment** demo scope.

Hybrid workflow: heavy/full runs are local; [`notebooks/colab_demo.ipynb`](notebooks/colab_demo.ipynb)
runs a small-budget demo in Colab for submission.

## Setup

> The system Python here is 3.14, which has **no wheels** for `pyswmm` / `swmmio` /
> `pymoo`. Use a **3.11 or 3.12** virtual environment.

```bash
cd drainage-network-optimization
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -c "import pymoo, pyswmm, swmmio; print('ok')"
```

## Run order

```bash
# 1. Fetch the Bellinge dataset (DTU public repo, CC BY 4.0)
python -m src.download_data            # or --list to preview files without downloading

# 2. Preprocess the raw files ONCE -> tiny tables + design storm (streamed, low RAM)
python -m src.preprocess

# 3. Build the candidate-conduit graph (gene order). scope=sub by default (fast demo)
python -m src.build_graph              # --full for the whole 1014-conduit network

# 4. Run each algorithm (repeat over seeds; see config.yaml)
python -m src.run --algo nsga2 --seed 0
python -m src.run --algo moead --seed 0
python -m src.run --algo ga    --seed 0

# 5. Compare + plot (reference front, hypervolume, IGD, node reduction, Wilcoxon)
python -m src.compare
```

A seed sweep, e.g.:

```bash
for a in nsga2 moead ga; do for s in 0 1 2 3 4; do
  python -m src.run --algo $a --seed $s
done; done
python -m src.compare
```

## Configuration

Everything tunable lives in [`config.yaml`](config.yaml): `scope` (sub/full),
sub-catchment carving, design-storm window, population / generations / seeds,
worker count, crossover/mutation, and the GA weight sets.

## Layout

```
src/
  common.py         config + paths
  download_data.py  Phase 1 — fetch dataset from DTU/Figshare (streamed)
  preprocess.py     Phase 2 — raw files -> tables + design storm (run once)
  build_graph.py    Phase 3 — SWMM model -> nodes + candidate_conduits (gene order)
  encoding.py       Phase 3 — decode + connectivity repair
  swmm_eval.py      Phase 4 — decode -> .inp -> PySWMM -> flooding, with caching
  objectives.py     Phase 4 — f1, f2, f3 + feasibility gate
  problem.py        Phase 5 — pymoo Problem, repair operator, shared operators
  run.py            Phase 5 — run one algorithm/seed
  compare.py        Phase 6 — metrics + figures
notebooks/
  colab_demo.ipynb  Phase 7 — small-budget Colab demo
tests/
  selftest.inp      tiny synthetic model for a no-dataset smoke test (see tests/README)
```

Outputs land in `outputs/` (gitignored): `tables/`, `runs/`, `figures/`, `sim_cache/`.

## Smoke test without the dataset

See [tests/README.md](tests/README.md) — copies the tiny `selftest.inp` into `data/`
and runs the full pipeline in seconds, to verify the install and wiring.
