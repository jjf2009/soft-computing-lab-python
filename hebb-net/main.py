class HebbNet:
    """Hebbian learning: for each pattern, w_i += x_i * t and b += t.
    Works with bipolar inputs/targets (-1, +1), the usual Hebb-rule setting."""

    def __init__(self, n_inputs):
        self.weights = [0] * n_inputs
        self.bias = 0

    def train(self, patterns, targets):
        print("Initial weights:", self.weights, "bias:", self.bias)
        for step, (x, t) in enumerate(zip(patterns, targets), start=1):
            for i in range(len(self.weights)):
                self.weights[i] += x[i] * t
            self.bias += t
            print(f"\nStep {step}: input={x}, target={t}")
            print("  weights:", self.weights, "bias:", self.bias)
        return self.weights, self.bias

    def predict(self, x):
        net = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return 1 if net >= 0 else -1


if __name__ == "__main__":
    # AND gate in bipolar form.
    patterns = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    targets = [1, -1, -1, -1]

    net = HebbNet(n_inputs=2)
    weights, bias = net.train(patterns, targets)

    print("\nFinal weight matrix:", weights)
    print("Final bias:", bias)

    print("\nVerification:")
    for x, t in zip(patterns, targets):
        print(f"  input={x} -> predicted={net.predict(x)} (target {t})")
