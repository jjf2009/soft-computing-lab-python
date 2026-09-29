# SC Lab - Experiment 9: Travelling Salesman Problem using a Genetic Algorithm.
# Permutation encoding (a tour visits every city once), tournament selection,
# order crossover (OX), and swap mutation.

import math
import random

POP_SIZE = 100
GENERATIONS = 200
MUTATION_RATE = 0.02
TOURNAMENT = 5


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tour_length(tour, cities):
    return sum(distance(cities[tour[i]], cities[tour[(i + 1) % len(tour)]])
               for i in range(len(tour)))


def tournament_select(population, cities):
    contenders = random.sample(population, TOURNAMENT)
    return min(contenders, key=lambda t: tour_length(t, cities))[:]


def order_crossover(parent1, parent2):
    n = len(parent1)
    a, b = sorted(random.sample(range(n), 2))
    child = [None] * n
    child[a:b] = parent1[a:b]
    fill = [c for c in parent2 if c not in child]
    idx = 0
    for i in range(n):
        if child[i] is None:
            child[i] = fill[idx]
            idx += 1
    return child


def mutate(tour):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(tour)), 2)
        tour[i], tour[j] = tour[j], tour[i]
    return tour


def genetic_tsp(cities):
    n = len(cities)
    population = [random.sample(range(n), n) for _ in range(POP_SIZE)]
    best = min(population, key=lambda t: tour_length(t, cities))

    for gen in range(1, GENERATIONS + 1):
        new_population = [best[:]]  # elitism: keep the best tour
        while len(new_population) < POP_SIZE:
            p1 = tournament_select(population, cities)
            p2 = tournament_select(population, cities)
            child = mutate(order_crossover(p1, p2))
            new_population.append(child)
        population = new_population
        gen_best = min(population, key=lambda t: tour_length(t, cities))
        if tour_length(gen_best, cities) < tour_length(best, cities):
            best = gen_best
        if gen % 50 == 0 or gen == 1:
            print(f"Gen {gen:3d}: best tour length = {tour_length(best, cities):.3f}")

    return best


if __name__ == "__main__":
    random.seed(1)
    cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(10)]
    best = genetic_tsp(cities)
    print("\nBest tour (city order):", best)
    print(f"Tour length: {tour_length(best, cities):.3f}")
