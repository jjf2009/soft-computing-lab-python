


class McCullochPittsNeuron:
    def __init__(self, weights, threshold):
        self.weights = tuple(weights)
        self.threshold = threshold

    def predict(self, inputs):
        """Return the neuron's binary output for *inputs*."""
        inputs = tuple(inputs)
        if len(inputs) != len(self.weights):
            raise ValueError("inputs and weights must have the same length")

        weighted_sum = sum(weight * value for weight, value in zip(self.weights, inputs))
        return int(weighted_sum >= self.threshold)

AND = McCullochPittsNeuron(weights=(1, 1), threshold=2)
OR = McCullochPittsNeuron(weights=(1, 1), threshold=1)




if __name__ == "__main__":
    while True:
        print("\nMcCulloch-Pitts Neuron Menu")
        print("1. AND model")
        print("2. OR model")
        print("5. Display truth tables")
        print("6. Exit")

        choice = input("Choose a model: ").strip()

        if choice in ("1", "2", "4"):
            model_names = {"1": "AND", "2": "OR"}
            model_name = model_names[choice]
            a = int(input("Enter input A (0 or 1): "))
            b = int(input("Enter input B (0 or 1): "))
            c = int(input("Enter input C (0 or 1): "))
            if choice == "1":
                result = AND.predict((a, b,c))
            else:
                 result = OR.predict((a, b,c))

            print(f"{model_name} output: {result}")

        else:
            print("Invalid choice. Please select 1 to 6.")
