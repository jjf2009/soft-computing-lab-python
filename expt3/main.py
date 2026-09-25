import numpy as np


def get_vector_input(prompt):
    user_input = input(prompt)
    values = [float(val) for val in user_input.split()]
    return np.array(values).reshape(-1, 1)


def get_matrix_input(name, rows, cols):
    print(f"Enter {name} ({rows}x{cols}), one row per line:")
    matrix = []
    for r in range(rows):
        row = [float(val) for val in input(f"  row {r+1}: ").split()]
        if len(row) != cols:
            raise ValueError(f"Expected {cols} values, got {len(row)}")
        matrix.append(row)
    return np.array(matrix)


def sigmoid(net):
    return 1 / (1 + np.exp(-net))


def sigmoid_derivative(output):
    return output * (1 - output)


def forward(x, V, b_h, W, b_o):
    net_h = V @ x + b_h
    h = sigmoid(net_h)
    net_o = W @ h + b_o
    o = sigmoid(net_o)
    return net_h, h, net_o, o


def train_step(x, d, V, b_h, W, b_o, eta, verbose):
    net_h, h, net_o, o = forward(x, V, b_h, W, b_o)

    # Backward pass
    delta_o = (d - o) * sigmoid_derivative(o)
    delta_h = sigmoid_derivative(h) * (W.T @ delta_o)

    delta_W = eta * delta_o @ h.T
    delta_b_o = eta * delta_o
    delta_V = eta * delta_h @ x.T
    delta_b_h = eta * delta_h

    if verbose:
        print("\n  Forward pass")
        print(f"  net_h = V x + b_h = {np.round(net_h.T[0], 4)}")
        print(f"  h = f(net_h) = {np.round(h.T[0], 4)}")
        print(f"  net_o = W h + b_o = {np.round(net_o.T[0], 4)}")
        print(f"  o = f(net_o) = {np.round(o.T[0], 4)}")
        print(f"  d = {d.T[0]}")
        print(f"  error E = 1/2 * sum(d - o)^2 = {0.5 * np.sum((d - o) ** 2):.6f}")

        print("\n  Backward pass")
        print(f"  delta_o = (d - o) * o(1 - o) = {np.round(delta_o.T[0], 4)}")
        print(f"  delta_h = h(1 - h) * (W^T delta_o) = {np.round(delta_h.T[0], 4)}")

        print("\n  Weight updates")
        print(f"  Delta W = eta * delta_o * h^T = \n{np.round(delta_W, 4)}")
        print(f"  Delta b_o = eta * delta_o = {np.round(delta_b_o.T[0], 4)}")
        print(f"  Delta V = eta * delta_h * x^T = \n{np.round(delta_V, 4)}")
        print(f"  Delta b_h = eta * delta_h = {np.round(delta_b_h.T[0], 4)}")

    W = W + delta_W
    b_o = b_o + delta_b_o
    V = V + delta_V
    b_h = b_h + delta_b_h

    if verbose:
        print("\n  Updated weights")
        print(f"  W = \n{np.round(W, 4)}")
        print(f"  b_o = {np.round(b_o.T[0], 4)}")
        print(f"  V = \n{np.round(V, 4)}")
        print(f"  b_h = {np.round(b_h.T[0], 4)}")

    error = 0.5 * np.sum((d - o) ** 2)
    return V, b_h, W, b_o, error


def main():
    print("--- Back-propagation Algorithm Setup ---")

    n_in = int(input("Number of input neurons: "))
    n_hidden = int(input("Number of hidden neurons: "))
    n_out = int(input("Number of output neurons: "))

    choice = input("Use random initial weights? (y/n): ").strip().lower()
    if choice == "y":
        rng = np.random.default_rng()
        V = rng.uniform(-1, 1, (n_hidden, n_in))
        b_h = rng.uniform(-1, 1, (n_hidden, 1))
        W = rng.uniform(-1, 1, (n_out, n_hidden))
        b_o = rng.uniform(-1, 1, (n_out, 1))
    else:
        V = get_matrix_input("input->hidden weights V", n_hidden, n_in)
        b_h = get_vector_input(f"Enter hidden biases b_h ({n_hidden} values): ")
        W = get_matrix_input("hidden->output weights W", n_out, n_hidden)
        b_o = get_vector_input(f"Enter output biases b_o ({n_out} values): ")

    eta = float(input("Enter learning rate eta: "))
    epochs = int(input("Enter maximum number of epochs: "))
    tolerance = float(input("Enter error tolerance: "))

    n = int(input("How many training patterns (x) are there: "))

    X_list = []
    d_list = []

    for i in range(n):
        x = get_vector_input(f"Enter x{i+1} ({n_in} values): ")
        d = get_vector_input(f"Enter target d{i+1} ({n_out} values): ")
        X_list.append(x)
        d_list.append(d)

    print("\nInitial weights")
    print(f"V = \n{np.round(V, 4)}")
    print(f"b_h = {np.round(b_h.T[0], 4)}")
    print(f"W = \n{np.round(W, 4)}")
    print(f"b_o = {np.round(b_o.T[0], 4)}")

    print("\n---Back-propagation Training---")

    for epoch in range(1, epochs + 1):
        verbose = epoch == 1
        if verbose:
            print(f"\n===== Epoch {epoch} (step by step) =====")

        total_error = 0
        for i in range(n):
            if verbose:
                print(f"\n({i+1}) pattern x{i+1} = {X_list[i].T[0]}")
            V, b_h, W, b_o, error = train_step(
                X_list[i], d_list[i], V, b_h, W, b_o, eta, verbose
            )
            total_error += error

        if verbose or epoch % max(1, epochs // 10) == 0:
            print(f"\nEpoch {epoch}: total error = {total_error:.6f}")

        if total_error < tolerance:
            print(f"\nConverged at epoch {epoch}: total error = {total_error:.6f}")
            break
    else:
        print(f"\nStopped after {epochs} epochs: total error = {total_error:.6f}")

    print("\nFinal weights")
    print(f"V = \n{np.round(V, 4)}")
    print(f"b_h = {np.round(b_h.T[0], 4)}")
    print(f"W = \n{np.round(W, 4)}")
    print(f"b_o = {np.round(b_o.T[0], 4)}")

    print("\n---Back-propagation Testing---")
    for i in range(n):
        _, _, _, o = forward(X_list[i], V, b_h, W, b_o)
        print(f"\n({i+1}) x{i+1} = {X_list[i].T[0]}")
        print(f"o = {np.round(o.T[0], 4)}")
        print(f"rounded o = {np.round(o.T[0]).astype(int)}")
        print(f"d{i+1} = {d_list[i].T[0]}")


if __name__ == "__main__":
    main()
