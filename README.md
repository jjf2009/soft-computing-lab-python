# Soft Computing Lab

Python implementations of neural-network and learning algorithms, done as college
lab practicals for **CMP-326 Soft Computing Lab**, B.E. Computer Engineering (Goa
University, NEP 2024-25 curriculum).

Each program is written to mirror the step-by-step hand solution taught in class,
printing intermediate values (net inputs, activations, weight updates, per-epoch
error) rather than just the final answer.

## Contents

| Exp | Folder | Topic |
|---|---|---|
| 1 | [expt1](expt1) | McCulloch-Pitts neuron — logic gates (AND, OR, NAND, NOR, XOR) |
| 2 | [hebb-net](hebb-net) | Hebb net — weight matrix and bias |
| 3 | [expt2](expt2) | Perceptron learning rule (basic logic gates, training + testing) |
| 4 | [expt3](expt3) | Back-propagation on a 2-2-1 feed-forward network |
| 5 | [fuzzy-set-operations](fuzzy-set-operations) | Operations on fuzzy sets (union, intersection, complement, difference) |
| 6 | [fuzzy-inference](fuzzy-inference) | Mamdani fuzzy inference system (tipping problem) |
| 7 | [genetic-algorithm](genetic-algorithm) | Genetic Algorithm for an optimization problem |
| 8 | [particle-swarm-optimization](particle-swarm-optimization) | Particle Swarm Optimization |
| 9 | [tsp-genetic-algorithm](tsp-genetic-algorithm) | Travelling Salesman Problem via Genetic Algorithm |
| 10 | [cnn-classifier](cnn-classifier) | CNN classifier for MNIST digits (TensorFlow/Keras) |

> Note: the earliest experiments live in `expt1`–`expt3` (original folder names);
> the rest use descriptive names. The table lists them in syllabus order.

Each folder has its own README with the model, the update equations, and run
instructions.

## Requirements

- Python 3 for all experiments
- `numpy` — Experiment 3 (Perceptron) — `pip install numpy`
- `tensorflow` — Experiment 10 (CNN) — see its `requirements.txt`

All other experiments use the standard library alone.

## Syllabus reference

Covers all ten CMP-326 experiments and the four course outcomes: artificial
neural network models — McCulloch-Pitts, Hebb, Perceptron, back-propagation (Exp
1–4, CO1); fuzzy sets and inference (Exp 5–6, CO2); genetic algorithms, PSO and
GA-based TSP (Exp 7–9, CO3); and a CNN classifier (Exp 10, CO4).
