import math
import random

BITS = 10
POPULATION_SIZE = 20
GENERATIONS = 5
CROSSOVER_RATE = 0.8
MUTATION_RATE = 1 / BITS
ELITISM = True


def decode(chromosome):
    value = int("".join(map(str, chromosome)), 2)
    return value / (2**BITS - 1)
                                                                                                                                                                                                                                                                                                                                                                                      

def fitness(chromosome):
    x = decode(chromosome)
    return x * math.sin(10 * math.pi * x) + 2


def random_chromosome():
    return [random.randint(0, 1) for _ in range(BITS)]


def roulette_select(population, fitnesses):
    total_fitness = sum(fitnesses)
    pick = random.uniform(0, total_fitness)
    cumulative = 0.0
    for chromosome, value in zip(population, fitnesses):
        cumulative += value
        if cumulative >= pick:
            return chromosome[:]
    return population[-1][:]


def crossover(parent1, parent2):
    if random.random() >= CROSSOVER_RATE:
        print("Crossover skipped")
        return parent1[:], parent2[:]
    point = random.randint(1, BITS - 1)
    print("CrossOver point:",point)
    return (parent1[:point] + parent2[point:],
            parent2[:point] + parent1[point:])


def mutate(chromosome):
    return [1 - bit if random.random() < MUTATION_RATE else bit for bit in chromosome]


def genetic_algorithm():
    population = [random_chromosome() for _ in range(POPULATION_SIZE)]
    best = None
    for generation in range(1, GENERATIONS + 1):
        fitnesses = [fitness(chromosome) for chromosome in population]
        generation_best = max(population, key=fitness)
        if best is None or fitness(generation_best) > fitness(best):
            best = generation_best[:]

        print(
            f"Generation {generation:2d}: "
            f"best x = {decode(generation_best):.6f}, "
            f"f(x) = {fitness(generation_best):.6f}, "
            f"average = {sum(fitnesses) / POPULATION_SIZE:.6f}"
        )

        new_population = []
        if ELITISM:
            new_population.append(generation_best[:])

        while len(new_population) < POPULATION_SIZE:
            print("                                ")
            parent1 = roulette_select(population, fitnesses)
            print("Parent1 Selected by Roulette:",parent1)
            parent2 = roulette_select(population, fitnesses)
            print("Parent2 Selected by Roulette:",parent2)
            child1, child2 = crossover(parent1, parent2)
            print("Child1 After CrossOver:",child1)
            print("Child1 After CrossOver:",child2)
            new_population.append(mutate(child1))
            if len(new_population) < POPULATION_SIZE:
                new_population.append(mutate(child2))

        population = new_population

    return best


if __name__ == "__main__":
    best = genetic_algorithm()
    print(
        f"\nBest solution: x = {decode(best):.6f}, "
        f"f(x) = {fitness(best):.6f}"
    )
