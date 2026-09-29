# SC Lab - Experiment 7: Genetic Algorithm for an optimization problem.
# Maximise f(x) = x^2 for x in [0, 31] using 5-bit binary chromosomes
# (the classic Goldberg example): roulette selection, single-point crossover,
# and bit-flip mutation.

import random

BITS = 5
POP_SIZE = 6
GENERATIONS = 20
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.05


def fitness(chromosome):
    x = int("".join(map(str, chromosome)), 2)
    return x * x


def random_chromosome():
    return [random.randint(0, 1) for _ in range(BITS)]


def roulette_select(population, fits):
    total = sum(fits)
    pick = random.uniform(0, total)
    running = 0
    for chromo, fit in zip(population, fits):
        running += fit
        if running >= pick:
            return chromo[:]
    return population[-1][:]


def crossover(a, b):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, BITS - 1)
        return a[:point] + b[point:], b[:point] + a[point:]
    return a[:], b[:]


def mutate(chromosome):
    return [1 - bit if random.random() < MUTATION_RATE else bit for bit in chromosome]


def genetic_algorithm():
    population = [random_chromosome() for _ in range(POP_SIZE)]
    best = None
    for gen in range(1, GENERATIONS + 1):
        fits = [fitness(c) for c in population]
        gen_best = max(population, key=fitness)
        if best is None or fitness(gen_best) > fitness(best):
            best = gen_best[:]
        x = int("".join(map(str, gen_best)), 2)
        print(f"Gen {gen:2d}: best x={x:2d}, f(x)={fitness(gen_best):4d}, avg={sum(fits) / len(fits):.1f}")

        new_population = []
        while len(new_population) < POP_SIZE:
            p1 = roulette_select(population, fits)
            p2 = roulette_select(population, fits)
            c1, c2 = crossover(p1, p2)
            new_population.append(mutate(c1))
            if len(new_population) < POP_SIZE:
                new_population.append(mutate(c2))
        population = new_population
    return best


if __name__ == "__main__":
    best = genetic_algorithm()
    x = int("".join(map(str, best)), 2)
    print(f"\nBest solution: x={x}, f(x)={fitness(best)} (optimum is x=31, f=961)")
