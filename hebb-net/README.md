# Hebb Net

**Soft Computing Lab — Experiment 2: Find the weight matrix and bias of a Hebb
net.**

`main.py` trains a single-layer Hebb network and prints the weight vector and bias
after each training pattern. Hebbian learning is the earliest neural learning rule
— "neurons that fire together wire together".

## The rule

For each input pattern `x` with bipolar target `t` (±1):

```
wᵢ ← wᵢ + xᵢ · t        (for every input i)
b  ← b  + t
```

Starting from zero weights and bias, the updates are applied once per pattern. The
example learns the bipolar **AND** gate; the final weights and bias then classify
all four patterns correctly.

## Run

```bash
python3 main.py
```

Prints the weight/bias after every step, the final weight matrix and bias, and a
verification pass over the training patterns.

## Notes

Hebb learning works cleanly with **bipolar** (−1/+1) data. It has no iterative
convergence loop — each pattern contributes exactly one update.
