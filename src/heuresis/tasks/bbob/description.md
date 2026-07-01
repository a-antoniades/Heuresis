# BBOB Continuous Optimization

BBOB ("Black-Box Optimization Benchmarking") is the standard synthetic test
suite for derivative-free continuous optimization. In this task the agent
designs an optimizer (written in `optimizer.py`) that, given a black-box
callable `f`, a dimensionality `dim`, box bounds, and an FEval budget,
returns the best `x` it finds.

The harness evaluates the optimizer across a small suite of `(function,
instance, seed)` tuples drawn from a 3-function subset — Sphere (unimodal,
separable), Rosenbrock (unimodal, curved valley), Rastrigin (highly
multimodal) — in 5 dimensions with a 1000-FEval budget per tuple. Score is
the mean `log10(best_y − f_opt)` across all tuples; lower is better.

Each function is shifted by a random x_opt and (for Rosenbrock and Rastrigin)
rotated by a random orthogonal matrix, with shift and rotation captured in a
closure so that the optimizer sees only `f(x) → float`. The canonical seed
baseline is uniform random search.

The agent edits `optimizer.py` only. `driver.py` (runs the optimizer on every
tuple, enforces the budget, writes `run.log`) and `problems.py` (function
definitions) are sealed. Typical axes of variation: evolutionary strategies
(CMA-ES family), differential evolution, particle swarms, Bayesian-style
surrogate methods, restart heuristics, trust-region methods, adaptive
step-size schemes.
