# SC Lab - Experiment 6: Design a fuzzy inference system.
# A Mamdani FIS for the classic "tipping" problem: given food quality and
# service (0-10), infer the tip (0-25%) using triangular membership functions,
# max-min inference, and centroid defuzzification.


def triangular(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


# Input membership functions
def poor(x):    return triangular(x, 0, 0, 5)
def good(x):    return triangular(x, 0, 5, 10)
def excellent(x): return triangular(x, 5, 10, 10)

# Output membership functions (tip %)
TIP_MFS = {
    "low":    lambda t: triangular(t, 0, 0, 13),
    "medium": lambda t: triangular(t, 0, 13, 25),
    "high":   lambda t: triangular(t, 13, 25, 25),
}


def infer(food, service):
    # Fuzzify
    mu = {
        "food_poor": poor(food), "food_good": good(food), "food_excellent": excellent(food),
        "serv_poor": poor(service), "serv_good": good(service), "serv_excellent": excellent(service),
    }
    # Rule base (max over rules firing the same output set = OR)
    low = max(min(mu["food_poor"], mu["serv_poor"]), 0)
    medium = mu["serv_good"]
    high = max(mu["food_excellent"], mu["serv_excellent"])
    strengths = {"low": low, "medium": medium, "high": high}
    print("Rule firing strengths:", {k: round(v, 3) for k, v in strengths.items()})

    # Defuzzify by centroid over the clipped (max-min) aggregate.
    num = den = 0.0
    t = 0.0
    while t <= 25.0:
        agg = max(min(strengths[label], mf(t)) for label, mf in TIP_MFS.items())
        num += t * agg
        den += agg
        t += 0.1
    return num / den if den else 0.0


if __name__ == "__main__":
    food = float(input("Food quality (0-10): "))
    service = float(input("Service (0-10): "))
    tip = infer(food, service)
    print(f"\nRecommended tip: {tip:.2f}%")
