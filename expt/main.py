"""
McCulloch-Pitts neuron examples.
The neuron produces 1 when its weighted input sum reaches the threshold;
otherwise it produces 0.
"""


class McCullochPittsNeuron:
    """A binary threshold neuron."""

    def __init__(self, weights, threshold):
        self.weights = tuple(weights)
        self.threshold = threshold

    def predict(self, inputs):
        """Return the neuron's binary output for *inputs*."""
        inputs = tuple(inputs)
        if len(inputs) != len(self.weights):
            raise ValueError("inputs and weights must have the same length")

        weighted_sum = sum(weight * value
                           for weight, value in zip(self.weights, inputs))
        return int(weighted_sum >= self.threshold)


# Single-neuron logic gates.
AND = McCullochPittsNeuron(weights=(1, 1), threshold=2)
OR = McCullochPittsNeuron(weights=(1, 1), threshold=1)
NOT = McCullochPittsNeuron(weights=(-1,), threshold=0)


def XOR(a, b):
    """XOR built from McCulloch-Pitts neurons: (a OR b) AND NOT(a AND b)."""
    return AND.predict((OR.predict((a, b)), NOT.predict((AND.predict((a, b)),))))


if __name__ == "__main__":
    binary_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

    print("A B | AND OR XOR")
    print("----------------")
    for a, b in binary_inputs:
        print(a, b, "| ", AND.predict((a, b)), "  ",
              OR.predict((a, b)), "  ", XOR(a, b))

    print("\nA | NOT")
    print("-------")
    for a in (0, 1):
        print(a, "|", NOT.predict((a,)))
