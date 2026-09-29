# Genetic Algorithm

**Soft Computing Lab — Experiment 7: Genetic Algorithm to solve an optimization
problem.**

`main.py` maximises `f(x) = x²` for `x ∈ [0, 31]` using the classic Goldberg
genetic algorithm with 5-bit binary chromosomes. It evolves a population toward the
optimum `x = 31` (`f = 961`).

## The algorithm

Each generation applies the three genetic operators:

- **Selection** — roulette-wheel: fitter chromosomes are picked with proportionally
  higher probability.
- **Crossover** — single-point: two parents swap tails at a random point.
- **Mutation** — bit-flip: each bit flips with a small probability, maintaining
  diversity.

The best chromosome found is carried across generations and reported at the end.

## Run

```bash
python3 main.py
```

Prints the best `x`, its fitness, and the population average each generation.

## Notes

Tunable parameters (population size, generations, crossover/mutation rates) are
constants at the top of the file. Being stochastic, runs vary but reliably
approach the optimum.
