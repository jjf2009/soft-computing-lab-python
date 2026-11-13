def union(a, b):
    return {x: max(a.get(x, 0), b.get(x, 0)) for x in a.keys() | b.keys()}


def intersection(a, b):
    return {x: min(a.get(x, 0), b.get(x, 0)) for x in a.keys() | b.keys()}


def complement(a):
    return {x: round(1 - mu, 3) for x, mu in a.items()}


def difference(a, b):
    return {x: round(min(a.get(x, 0), 1 - b.get(x, 0)), 3) for x in a.keys() | b.keys()}


def algebraic_sum(a, b):
    return {x: round(a.get(x, 0) + b.get(x, 0) - a.get(x, 0) * b.get(x, 0), 4) for x in a.keys() | b.keys()}


def algebraic_product(a, b):
    return {x: round(a.get(x, 0) * b.get(x, 0), 3)for x in a.keys() | b.keys()}


def bounded_sum(a, b):
    return {x: round(min(1, a.get(x, 0) + b.get(x, 0)), 3)
            for x in a.keys() | b.keys()}


def show(name, s):
    body = ", ".join(f"{mu}/{x}" for x, mu in sorted(s.items()))
    print(f"{name} = {{ {body} }}")

def max_min_composition(r, s):
    if len(r[0]) != len(s):
        raise ValueError("Columns of R must equal rows of S")

    rows = len(r)
    cols = len(s[0])
    common = len(s)

    result = []
    for i in range(rows):
        row = []
        for j in range(cols):
            best = 0
            for k in range(common):
                value = min(r[i][k], s[k][j])
                if value > best:
                    best = value
            row.append(round(best, 3))
        result.append(row)
    return result


def max_product_composition(r, s):
    if len(r[0]) != len(s):
        raise ValueError("Columns of R must equal rows of S")

    rows = len(r)
    cols = len(s[0])
    common = len(s)

    result = []
    for i in range(rows):
        row = []
        for j in range(cols):
            best = 0
            for k in range(common):
                value = r[i][k] * s[k][j]
                if value > best:
                    best = value
            row.append(round(best, 3))
        result.append(row)
    return result

def show_matrix(name, matrix):
    print(name)
    for row in matrix:
        print("  " + "  ".join(f"{value:.3f}" for value in row))


if __name__ == "__main__":
    A = {"x1": 0.2, "x2": 0.7, "x3": 1.0, "x4": 0.5}
    B = {"x1": 0.6, "x2": 0.4, "x3": 0.8, "x4": 0.9}
    R = [
        [0.2, 0.7, 0.5],
        [0.9, 0.4, 0.8],
    ]
    S = [
        [0.6, 0.3],
        [0.8, 0.5],
        [0.4, 1.0],
    ]

    C = max_min_composition(R, S)
    D = max_product_composition(R, S)

    show("A", A)
    show("B", B)
    print()
    show("A union B", union(A, B))
    show("A intersect B", intersection(A, B))
    show("complement A", complement(A))
    show("complement B", complement(B))
    show("A - B", difference(A, B))
    show("B - A", difference(B, A))
    print()
    show("A algebraic sum B", algebraic_sum(A, B))
    show("A algebraic product B", algebraic_product(A, B))
    show("A bounded sum B", bounded_sum(A, B))
    print()
    show_matrix("C = R max-min S", C)
    show_matrix("D = R max-product S", D)
