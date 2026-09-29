# Experiment 1 — McCulloch-Pitts Neuron

**Soft Computing Lab — Experiment 1: Implementation of a simple neural network
(McCulloch-Pitts model).**

`main.py` implements the McCulloch-Pitts neuron — the earliest artificial neuron
— and uses it to realise the basic logic gates. A neuron fires (outputs `1`) when
the weighted sum of its inputs reaches its threshold, otherwise it outputs `0`:

```
y = 1 if Σ wᵢ xᵢ ≥ threshold else 0
```

## What it does

The `McCullochPittsNeuron` class stores weights and a threshold and exposes
`predict(inputs)`. A menu-driven program then builds the standard gates:

| Choice | Gate | How |
|---|---|---|
| 1 | AND | single neuron, high threshold |
| 2 | OR | single neuron, low threshold |
| 3 | NAND | AND followed by an inverter neuron (`NOT`) |
| 4 | NOR | OR followed by an inverter neuron (`NOT`) |
| 5 | XOR | two-layer network (XOR is not linearly separable) |

XOR shows the key limitation of a single neuron: it cannot be computed by one
McCulloch-Pitts unit, so it is built from a layered combination of neurons.

## Run

```bash
python3 main.py
```

Choose a gate from the menu, then enter the first-layer weights, threshold, and
the inputs `A B C`.
