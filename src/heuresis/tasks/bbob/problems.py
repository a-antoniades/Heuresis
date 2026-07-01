"""BBOB problem generators + BudgetedF wrapper.

This module is workspace-local: the driver imports it via `from problems import ...`
when running inside the sandbox. It is also importable as
`heuresis.tasks.bbob.problems` for unit tests.

Functions:
    make_problem(func_id, dim, instance_id) -> (f, f_opt, bounds)
        Returns a callable f(x) -> float with shift/rotation captured in closure,
        the known optimum value f_opt (0.0 for all BBOB functions here), and the
        symmetric bounds.

Classes:
    BudgetedF: wraps f, enforces FEval budget, tracks best_y internally.
    BudgetExhausted: raised when budget is hit.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


class BudgetExhausted(Exception):
    """Raised by BudgetedF when the FEval budget is exhausted."""


class BudgetedF:
    """Wrap a function with a strict FEval budget.

    The driver uses this wrapper as the sole source of truth for scoring:
    an agent's returned y_best is ignored; the driver reads `self.best_y`.
    """

    def __init__(
        self,
        f: Callable[[np.ndarray], float],
        budget: int,
        bounds: tuple[float, float] = (-5.0, 5.0),
    ) -> None:
        self.f = f
        self.budget = int(budget)
        self.low, self.high = bounds
        self.calls = 0
        self.best_y: float = float("inf")
        self.best_x: np.ndarray | None = None
        self.history: list[float] = []

    def __call__(self, x: np.ndarray) -> float:
        if self.calls >= self.budget:
            raise BudgetExhausted(
                f"FEval budget of {self.budget} exhausted"
            )
        self.calls += 1
        x = np.clip(np.asarray(x, dtype=float), self.low, self.high)
        y = float(self.f(x))
        self.history.append(y)
        if np.isfinite(y) and y < self.best_y:
            self.best_y = y
            self.best_x = x.copy()
        return y


def _random_rotation(rng: np.random.Generator, dim: int) -> np.ndarray:
    """Draw a random orthogonal matrix via QR decomposition of a Gaussian.

    Guaranteed proper rotation (det = +1) by sign-fixing against R's diagonal.
    """
    A = rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(A)
    d = np.sign(np.diag(R))
    d[d == 0] = 1.0
    Q = Q * d
    return Q


# ---------- function bodies (take z = Q @ (x - x_opt); minimum at z = 0) ----------

def _sphere(z: np.ndarray) -> float:
    return float(np.sum(z * z))


def _rosenbrock(z: np.ndarray) -> float:
    """Rosenbrock banana. Classical form shifted so f(z=0) = 0.

    Classical: F(u) = sum_i 100*(u_{i+1} - u_i^2)^2 + (u_i - 1)^2; min at u = 1.
    Substitute u = z + 1 so that the min sits at z = 0:
        F(z+1) = sum_i 100*((z_{i+1}+1) - (z_i+1)^2)^2 + z_i^2
    At z = 0: first term = 100*(1 - 1)^2 = 0 and second term = 0. ✓
    """
    u = z + 1.0
    term1 = 100.0 * (u[1:] - u[:-1] ** 2) ** 2
    term2 = (u[:-1] - 1.0) ** 2
    return float(np.sum(term1 + term2))


def _rastrigin(z: np.ndarray) -> float:
    """Classical Rastrigin. f(z=0) = 0 by direct computation."""
    n = len(z)
    return float(10.0 * n + np.sum(z ** 2 - 10.0 * np.cos(2.0 * np.pi * z)))


# ---------- registry + make_problem ----------

# (name, function body, rotate?) — Sphere is separable so rotation is identity.
_FUNC_REGISTRY: dict[int, tuple[str, Callable[[np.ndarray], float], bool]] = {
    1: ("Sphere", _sphere, False),
    8: ("Rosenbrock", _rosenbrock, True),
    15: ("Rastrigin", _rastrigin, True),
}


def make_problem(
    func_id: int,
    dim: int,
    instance_id: int,
) -> tuple[Callable[[np.ndarray], float], float, tuple[float, float]]:
    """Instantiate a BBOB-style problem.

    Returns (f, f_opt, bounds). f is a closure capturing the shift x_opt and
    (if applicable) the rotation matrix Q; neither is exposed on f itself.
    All functions are constructed so f_opt = 0.0 at the shifted/rotated origin.
    """
    if func_id not in _FUNC_REGISTRY:
        raise ValueError(
            f"Unknown func_id {func_id}. Supported: {sorted(_FUNC_REGISTRY)}"
        )
    name, fn, rotate = _FUNC_REGISTRY[func_id]
    rng = np.random.default_rng(seed=func_id * 1000 + instance_id)
    x_opt = rng.uniform(-4.0, 4.0, size=dim)
    Q = _random_rotation(rng, dim) if rotate else np.eye(dim)
    bounds = (-5.0, 5.0)
    f_opt = 0.0

    def f(x: np.ndarray) -> float:
        z = Q @ (np.asarray(x, dtype=float) - x_opt)
        return fn(z)

    return f, f_opt, bounds
