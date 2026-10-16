# Smoke test (no dataset needed)

`selftest.inp` is a tiny synthetic SWMM model (4 junctions, 1 outfall, 6 conduits,
3 subcatchment inlets). It exercises the whole pipeline — model parsing, encoding,
repair, a real PySWMM run, all three algorithms, caching, and the comparison — in a
few seconds, before the full Bellinge dataset is available.

From the project root, with the venv active:

```bash
mkdir -p data && cp tests/selftest.inp data/

python -m src.build_graph                       # -> 5 nodes, 6 genes, 3 inlets, 1 outfall
python -m src.run --algo nsga2 --seed 0 --pop 12 --gen 3 --serial
python -m src.run --algo moead --seed 0 --pop 12 --gen 3 --serial
python -m src.run --algo ga    --seed 0 --pop 12 --gen 3 --serial
python -m src.compare

rm data/selftest.inp                            # remove before downloading real data
```

Expected: each run writes `outputs/runs/<algo>_seed0.npz`, and `compare` writes three
PNGs to `outputs/figures/` plus `outputs/runs/comparison.csv`. On a model this small
the metric values are degenerate — the point is that nothing errors.

Drop `--serial` to also exercise the multiprocessing path.
```
