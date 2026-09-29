# SC Lab - Experiment 5: Operations on fuzzy sets.
# A fuzzy set is a dict {element: membership in [0, 1]}.


def union(a, b):
    # max of memberships
    return {x: max(a.get(x, 0), b.get(x, 0)) for x in a.keys() | b.keys()}


def intersection(a, b):
    # min of memberships
    return {x: min(a.get(x, 0), b.get(x, 0)) for x in a.keys() | b.keys()}


def complement(a):
    return {x: round(1 - mu, 3) for x, mu in a.items()}


def difference(a, b):
    # A - B = A ∩ complement(B) = min(mu_A, 1 - mu_B)
    return {x: round(min(a.get(x, 0), 1 - b.get(x, 0)), 3) for x in a.keys() | b.keys()}


def show(name, s):
    body = ", ".join(f"{mu}/{x}" for x, mu in sorted(s.items()))
    print(f"{name} = {{ {body} }}")


if __name__ == "__main__":
    A = {"x1": 0.2, "x2": 0.7, "x3": 1.0, "x4": 0.5}
    B = {"x1": 0.6, "x2": 0.4, "x3": 0.8, "x4": 0.9}

    show("A", A)
    show("B", B)
    print()
    show("A union B", union(A, B))
    show("A intersect B", intersection(A, B))
    show("complement A", complement(A))
    show("complement B", complement(B))
    show("A - B", difference(A, B))
