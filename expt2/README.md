# Experiment 2 — Perceptron Learning Rule

**Soft Computing Lab — Experiment 3: Implementation of the Perceptron net.**

`main.py` implements the discrete Perceptron learning rule and shows the weight
update working step by step, exactly as it is solved by hand in class.

## The rule

For each input pattern the net input is `net = wᵀx`, and the output is the sign
activation `y = sgn(net)` (`+1` if `net ≥ 0`, else `-1`). Weights change only when
the output disagrees with the target `d`:

```
if y == d : no change
if d = +1, y = -1 : w ← w + 2c·x
if d = -1, y = +1 : w ← w - 2c·x
```

where `c` is the learning rate. The program prints `net`, `sgn(net)`, the target,
and the resulting weight vector at every step, then re-runs a **testing** pass
over the same patterns with the trained weights.

## Run

```bash
pip install numpy
python3 main.py
```

Enter the initial weight vector, the learning rate `c`, and each input pattern
`x` with its target `d`. Vectors are entered as space-separated numbers.

## Notes

Weights and inputs are handled as NumPy column vectors. The perceptron converges
only when the patterns are **linearly separable**.
