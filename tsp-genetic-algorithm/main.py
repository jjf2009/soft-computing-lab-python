

import random


POPULATION_SIZE = 30
GENERATIONS = 50
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.80
MUTATION_RATE = 0.10
RANDOM_SEED = 42


CITY_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
DISTANCES = [[0, 14, 12, 18, 25, 31, 9, 15, 22, 19],
[14, 0, 11, 20, 28, 17, 13, 24, 16, 21],
[12, 11, 0, 15, 19, 23, 27, 10, 14, 26],
[18, 20, 15, 0, 13, 16, 22, 29, 12, 17],
[25, 28, 19, 13, 0, 14, 21, 18, 11, 15],
[31, 17, 23, 16, 14, 0, 19, 25, 20, 12],
[9, 13, 27, 22, 21, 19, 0, 16, 24, 18],
[15, 24, 10, 29, 18, 25, 16, 0, 13, 22],
[22, 16, 14, 12, 11, 20, 24, 13, 0, 14],
[19, 21, 26, 17, 15, 12, 18, 22, 14, 0]]

CITIES = list(range(len(CITY_NAMES)))


def route_name(route):
    """Return a readable representation of a route."""
    closed_route = route + [route[0]]
    return " -> ".join(CITY_NAMES[city] for city in closed_route)


def route_distance(route):
    """Calculate the route distance, including the return trip."""
    return sum(
        DISTANCES[route[index]][route[(index + 1) % len(route)]]
        for index in range(len(route))
    )


def create_population():
    """Create the initial population using random routes."""
    population = []

    for _ in range(POPULATION_SIZE):
        route = CITIES.copy()
        random.shuffle(route)
        population.append(route)

    return population


def select_parent(population):
    """Select the shortest route from a random tournament."""
    contestants = random.sample(population, TOURNAMENT_SIZE)
    return min(contestants, key=route_distance).copy()


def order_crossover(parent_one, parent_two):
    """Create a child using order crossover (OX)."""
    if random.random() > CROSSOVER_RATE:
        return parent_one.copy()

    first, second = sorted(random.sample(range(len(CITIES)), 2))
    child = [None] * len(CITIES)
    child[first : second + 1] = parent_one[first : second + 1]

    remaining_cities = [city for city in parent_two if city not in child]
    remaining_positions = [
        position
        for position in range(len(child))
        if child[position] is None
    ]

    for position, city in zip(remaining_positions, remaining_cities):
        child[position] = city

    return child


def mutate(route):
    """Swap two cities with the configured mutation probability."""
    mutated_route = route.copy()

    if random.random() < MUTATION_RATE:
        first, second = random.sample(range(len(mutated_route)), 2)
        mutated_route[first], mutated_route[second] = (
            mutated_route[second],
            mutated_route[first],
        )

    return mutated_route


def create_next_generation(population):
    """Create a new population while preserving the best current route."""
    population = sorted(population, key=route_distance)
    next_generation = [population[0].copy()]  # Elitism

    while len(next_generation) < POPULATION_SIZE:
        parent_one = select_parent(population)
        parent_two = select_parent(population)

        child_one = mutate(order_crossover(parent_one, parent_two))
        next_generation.append(child_one)

        if len(next_generation) < POPULATION_SIZE:
            child_two = mutate(order_crossover(parent_two, parent_one))
            next_generation.append(child_two)

    return next_generation


def print_generation(generation, population, best_distance):
    """Print every route and distance in one generation."""
    print(f"Generation {generation:02}")
    print("-" * 52)

    sorted_population = sorted(population, key=route_distance)
    for index, route in enumerate(sorted_population, start=1):
        print(f"{index:02}. {route_name(route):35} Distance: {route_distance(route)}")

    best_route = sorted_population[0]
    print(f"Best route:     {route_name(best_route)}")
    print(f"Best distance:  {best_distance}")
    print()


def run_genetic_algorithm():
    """Run the genetic algorithm and return the best route and distance."""
    random.seed(RANDOM_SEED)
    population = create_population()

    best_route = min(population, key=route_distance).copy()
    best_distance = route_distance(best_route)

    print("Travelling Salesman Problem - Genetic Algorithm")
    print("=" * 52)
    print(f"Cities: {', '.join(CITY_NAMES)}")
    print(f"Population: {POPULATION_SIZE} | Generations: {GENERATIONS}")
    print()

    for generation in range(1, GENERATIONS + 1):
        population = create_next_generation(population)
        generation_best = min(population, key=route_distance)
        generation_distance = route_distance(generation_best)

        if generation_distance < best_distance:
            best_route = generation_best.copy()
            best_distance = generation_distance

        print_generation(generation, population, generation_distance)

    return best_route, best_distance


def print_result(route, distance):
    """Print the final route and each leg of its journey."""
    print("\nFinal result")
    print("-" * 52)
    print(f"Best route:     {route_name(route)}")
    print(f"Total distance: {distance}")
    print("\nDistance breakdown:")

    closed_route = route + [route[0]]
    for first, second in zip(closed_route, closed_route[1:]):
        print(
            f"{CITY_NAMES[first]} -> {CITY_NAMES[second]}: "
            f"{DISTANCES[first][second]}"
        )


if __name__ == "__main__":
    route, distance = run_genetic_algorithm()
    print_result(route, distance)
