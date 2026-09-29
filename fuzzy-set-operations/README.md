# Fuzzy Set Operations

**Soft Computing Lab — Experiment 5: Operations on fuzzy sets.**

`main.py` implements the standard operations on fuzzy sets, where each element has
a **membership degree** in `[0, 1]` rather than a crisp in/out membership. A fuzzy
set is stored as a `{element: membership}` dictionary.

## Operations

| Operation | Definition |
|---|---|
| Union `A ∪ B` | `max(μA(x), μB(x))` |
| Intersection `A ∩ B` | `min(μA(x), μB(x))` |
| Complement `Ā` | `1 − μA(x)` |
| Difference `A − B` | `min(μA(x), 1 − μB(x))` |

## Run

```bash
python3 main.py
```

Runs the operations on two sample fuzzy sets `A` and `B` and prints each result in
`membership/element` notation.

## Notes

These use the classic Zadeh operators (max/min). Fuzzy sets generalise crisp sets:
when every membership is `0` or `1`, the operations reduce to ordinary set union,
intersection, and complement.
