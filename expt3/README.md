# Experiment 3 — Back-Propagation

**Soft Computing Lab — Experiment 4: Implement the Back-propagation algorithm.**

`main.py` trains a small feed-forward neural network — 2 inputs, a 2-neuron
hidden layer, and 1 output — on a single pattern using the back-propagation
learning algorithm with sigmoid activations.

## Network

```
x1 ─┐        v11,v12,v21,v22       w1,w2
    ├─► [ h1, h2 ] ──────────────► [ out ]
x2 ─┘   sigmoid                     sigmoid
```

- **Forward pass:** compute hidden activations `h = σ(V·x)`, then the output
  `out = σ(W·h)`, where `σ(net) = 1 / (1 + e⁻ⁿᵉᵗ)`.
- **Error:** `E = ½ (target − out)²`.
- **Backward pass:** propagate the error using the sigmoid derivative
  `σ'(y) = y(1 − y)` to get output and hidden gradients, then update every weight
  by gradient ascent scaled by the learning rate `α`.

Training repeats until the error falls below the given tolerance, printing the
output, error, and all weights each epoch.

## Run

```bash
python3 main.py
```

Enter the inputs, the initial weights `V` (`v11 v12 v21 v22`) and `W` (`w1 w2`),
the learning rate, the target output, and the error threshold.

## Notes

Uses only the standard-library `math` module — no external packages. Because a
single pattern is used, the network simply drives its output toward the target.
