# Fuzzy Inference System

**Soft Computing Lab — Experiment 6: Design a fuzzy inference system for a given
problem.**

`main.py` builds a **Mamdani** fuzzy inference system (FIS) for the classic
*tipping* problem: given food quality and service (each 0–10), it infers a
suitable tip (0–25%).

## Pipeline

1. **Fuzzify** — triangular membership functions map each crisp input to `poor`,
   `good`, `excellent` degrees.
2. **Apply rules** — e.g. *if service is excellent or food is excellent then tip is
   high*; rule strengths combine with min (AND) and max (OR).
3. **Aggregate & defuzzify** — the output fuzzy sets (`low`, `medium`, `high`) are
   clipped by their rule strengths (max–min inference) and combined; the crisp tip
   is the **centroid** of the aggregate.

## Run

```bash
python3 main.py
```

Enter food quality and service (0–10); the system prints the rule firing strengths
and the recommended tip percentage.

## Notes

Triangular membership functions and centroid defuzzification keep the whole system
in pure Python — no fuzzy-logic library required.
