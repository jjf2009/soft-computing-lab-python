class McCullochPittsNeuron:
    def __init__(self, weights, threshold):
        self.weights = tuple(weights)
        self.threshold = threshold

    def predict(self, inputs):
        total = sum(w * x for w, x in zip(self.weights, inputs))
        return int(total >= self.threshold)


def NOT(value):
    """Inverter neuron used as the second layer."""
    return McCullochPittsNeuron((-1,), 0).predict((value,))


def XOR(a, b, c, weights):
    """Three-input XOR using first-layer McCulloch-Pitts weights."""
    inputs = (a, b, c)
    at_least_one = McCullochPittsNeuron(weights, 1).predict(inputs)
    at_least_two = McCullochPittsNeuron(weights, 2).predict(inputs)
    at_least_three = McCullochPittsNeuron(weights, 3).predict(inputs)

    exactly_one = McCullochPittsNeuron((1, -1), 1).predict(
        (at_least_one, at_least_two)
    )

    return McCullochPittsNeuron((1, 1), 1).predict(
        (exactly_one, at_least_three)
    )


if __name__ == "__main__":
    while True:
        print("\nMcCulloch-Pitts Neuron Menu")
        print("1. AND model")
        print("2. OR model")
        print("3. NAND model")
        print("4. NOR model")
        print("5. XOR model")
        print("6. Exit")

        choice = input("Choose a model: ").strip()

        if choice == "6":
            print("Exiting...")
            break

        elif choice in ("1", "2", "3", "4"):
            print("Enter first-layer weights:")
            w1 = int(input("Weight 1: "))
            w2 = int(input("Weight 2: "))
            w3 = int(input("Weight 3: "))
            threshold = int(input("Enter threshold: "))

            a = int(input("Enter A: "))
            b = int(input("Enter B: "))
            c = int(input("Enter C: "))

            first_layer = McCullochPittsNeuron((w1, w2, w3), threshold)
            first_output = first_layer.predict((a, b, c))

            if choice == "1":
                name, result = "AND", first_output
            elif choice == "2":
                name, result = "OR", first_output
            elif choice == "3":
                name, result = "NAND", NOT(first_output)
            else:
                name, result = "NOR", NOT(first_output)

            print(f"{name} output:", result)

        elif choice == "5":
            print("Enter first-layer weights:")
            w1 = int(input("Weight 1: "))
            w2 = int(input("Weight 2: "))
            w3 = int(input("Weight 3: "))

            a = int(input("Enter A: "))
            b = int(input("Enter B: "))
            c = int(input("Enter C: "))

            print("XOR output:", XOR(a, b, c, (w1, w2, w3)))

        else:
            print("Invalid choice.")
