"""Seed optimizer: uniform random search.

The agent is expected to REPLACE this with a more interesting algorithm. The
entry point is `optimize(f, dim, bounds, budget, seed) -> dict`. The driver
calls you with a `BudgetedF`-wrapped `f` — treat it as a pure black box:
pass a numpy array of shape (dim,), get back a float. The wrapper raises
`BudgetExhausted` if you exceed the budget, but it is easier to stay under.

Your return value is advisory; the driver reads `best_y` directly from its
own tracker, so you cannot lie about your score.
"""
from __future__ import annotations

import numpy as np

from problems import BudgetExhausted


def optimize(
    f,
    dim: int,
    bounds: tuple[float, float],
    budget: int,
    seed: int,
) -> dict:
    """Uniform random search within the bounds."""
    rng = np.random.default_rng(seed)
    low, high = bounds
    best_x, best_y = None, float("inf")
    try:
        for _ in range(budget):
            x = rng.uniform(low, high, size=dim)
            y = f(x)
            if y < best_y:
                best_y, best_x = y, x
    except BudgetExhausted:
        pass  # driver records best from its own tracker
    return {
        "x_best": best_x,
        "y_best": best_y,
        "notes": "random search baseline (seed)",
    }
