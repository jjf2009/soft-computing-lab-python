# SC Lab - Experiment 8: Particle Swarm Optimization.
# Minimise the sphere function f(x) = sum(x_i^2) in n dimensions. Each particle
# tracks its own best position (pbest) and follows the swarm best (gbest).

import random

DIM = 2
SWARM_SIZE = 20
ITERATIONS = 30
W = 0.7          # inertia weight
C1 = 1.5         # cognitive (pbest) coefficient
C2 = 1.5         # social (gbest) coefficient
LOW, HIGH = -10.0, 10.0


def fitness(position):
    return sum(x * x for x in position)


class Particle:
    def __init__(self):
        self.position = [random.uniform(LOW, HIGH) for _ in range(DIM)]
        self.velocity = [random.uniform(-1, 1) for _ in range(DIM)]
        self.best_position = self.position[:]
        self.best_fitness = fitness(self.position)


def pso():
    swarm = [Particle() for _ in range(SWARM_SIZE)]
    gbest = min(swarm, key=lambda p: p.best_fitness)
    gbest_position = gbest.best_position[:]
    gbest_fitness = gbest.best_fitness

    for it in range(1, ITERATIONS + 1):
        for p in swarm:
            for d in range(DIM):
                r1, r2 = random.random(), random.random()
                p.velocity[d] = (
                    W * p.velocity[d]
                    + C1 * r1 * (p.best_position[d] - p.position[d])
                    + C2 * r2 * (gbest_position[d] - p.position[d])
                )
                p.position[d] += p.velocity[d]
                p.position[d] = max(LOW, min(HIGH, p.position[d]))

            fit = fitness(p.position)
            if fit < p.best_fitness:
                p.best_fitness = fit
                p.best_position = p.position[:]
                if fit < gbest_fitness:
                    gbest_fitness = fit
                    gbest_position = p.position[:]

        if it % 5 == 0 or it == 1:
            print(f"Iter {it:2d}: gbest f={gbest_fitness:.6f} at "
                  f"({', '.join(f'{x:.4f}' for x in gbest_position)})")

    return gbest_position, gbest_fitness


if __name__ == "__main__":
    position, value = pso()
    print(f"\nBest position: ({', '.join(f'{x:.5f}' for x in position)})")
    print(f"Best fitness:  {value:.6f} (optimum is 0 at the origin)")
