# SC Lab - Experiment 8: Particle Swarm Optimization
# Problem: maximize f(x) = x * sin(10*pi*x) + 2

import random
import math

n_particles = 20
n_iterations = 20
w = 0.7
c1 = 1.5
c2 = 1.5
low, high = -10, 10

def fitness(x):
    return x * math.sin(10 * math.pi * x) + 2


pos_x = [random.uniform(low, high) for _ in range(n_particles)]
vel_x = [random.uniform(-1, 1) for _ in range(n_particles)]

pbest_x = pos_x[:]
pbest_fit = [fitness(pos_x[i]) for i in range(n_particles)]

best_index = pbest_fit.index(max(pbest_fit))
gbest_x = pbest_x[best_index]
gbest_fit = pbest_fit[best_index]


for it in range(1, n_iterations + 1):
    for i in range(n_particles):
        r1 = random.random()
        r2 = random.random()

        vel_x[i] = (w * vel_x[i]
                    + c1 * r1 * (pbest_x[i] - pos_x[i])
                    + c2 * r2 * (gbest_x - pos_x[i]))

        pos_x[i] = pos_x[i] + vel_x[i]
        pos_x[i] = max(low, min(high, pos_x[i]))

        fit = fitness(pos_x[i])

        if fit > pbest_fit[i]:
            pbest_fit[i] = fit
            pbest_x[i] = pos_x[i]

        if fit > gbest_fit:
            gbest_fit = fit
            gbest_x = pos_x[i]


    print(f"\n Iteration {it}")
    print("Particle |  Local best x | Local best f(x)")
    for i in range(n_particles):
        print(f"{i+1:8d} | {pbest_x[i]:13.4f} | {pbest_fit[i]:14.6f}")
    print(f"Global best = {gbest_x:.4f}, f(x) = {gbest_fit:.6f}")


print("\n FINAL RESULT")
print("Global maximum: x =", round(gbest_x, 5), ", f(x) =", round(gbest_fit, 6))