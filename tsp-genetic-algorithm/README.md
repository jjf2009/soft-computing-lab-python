# Travelling Salesman Problem (Genetic Algorithm)

**Soft Computing Lab — Experiment 9: Travelling Salesman Problem using a Genetic
Algorithm.**

`main.py` finds a short tour visiting every city exactly once and returning to the
start, using a genetic algorithm with **permutation** encoding (a chromosome is an
ordering of the cities).

## The algorithm

- **Fitness** — the total Euclidean tour length (shorter is better).
- **Selection** — tournament: the shortest of a random handful of tours wins.
- **Crossover** — order crossover (OX): copies a slice from one parent and fills
  the rest in the order they appear in the other, keeping every city exactly once.
- **Mutation** — swap: two cities exchange positions.
- **Elitism** — the best tour is copied unchanged into the next generation.

## Run

```bash
python3 main.py
```

Generates 10 random cities (fixed seed) and prints the best tour length as it
improves.

## Notes

TSP is NP-hard, so the GA searches for a good tour, not a proven optimum. Tune the
population size, generations, and mutation rate at the top of the file.
