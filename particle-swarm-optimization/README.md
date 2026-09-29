# Particle Swarm Optimization

**Soft Computing Lab — Experiment 8: Particle Swarm Optimization to solve an
optimization problem.**

`main.py` minimises the sphere function `f(x) = Σ xᵢ²` (optimum `0` at the origin)
with PSO, a swarm-intelligence method inspired by flocking. Each particle is a
candidate solution that flies through the search space guided by its own best
find and the swarm's best find.

## Velocity update

For each particle and dimension:

```
v ← w·v + c1·r1·(pbest − x) + c2·r2·(gbest − x)
x ← x + v
```

- `w` — inertia (keeps momentum)
- `c1` — cognitive pull toward the particle's own best (`pbest`)
- `c2` — social pull toward the global best (`gbest`)
- `r1, r2` — fresh random numbers each step

## Run

```bash
python3 main.py
```

Prints the global best fitness and position periodically as the swarm converges.

## Notes

Swarm size, iterations, and the coefficients `w, c1, c2` are constants at the top
of the file. PSO needs no gradient — only the objective value.
