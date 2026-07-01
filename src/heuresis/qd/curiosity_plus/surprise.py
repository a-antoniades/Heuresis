"""Prediction dataclass and surprise computation (§4.3 of proposal).

Surprise measures how wrong the LLM's prediction was:
  - Both predicted and actual invalid → 0 (no information)
  - Validity mismatch → λ (large fixed surprise)
  - Both valid → |actual_fitness - predicted_fitness| / σ_f

Curiosity-plus Change E (memory-aware surprise): when a memory
consensus has its own prediction for the outcome, ``apply_memory_discount``
shrinks raw surprise by the fraction the memory's prediction beat the
LLM's. Idea: if memory could already predict this idea would behave a
particular way, the LLM's "surprise" was just LLM ignorance, not
genuine novelty in the search.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass
class Prediction:
    """Structured prediction from the LLM before execution."""

    predicted_valid: bool
    predicted_fitness: float | None  # None if predicted invalid
    reasoning: str = ""
    confidence: float | None = None


def surprise(
    prediction: Prediction | None,
    actual_valid: bool,
    actual_fitness: float | None,
    *,
    sigma_tracker: SigmaTracker | None = None,
    lambda_mismatch: float = 1.0,
) -> float | None:
    """Compute combined surprise from a prediction-outcome pair.

    Returns None if prediction is None (parse failure — idea still runs
    but produces no curiosity signal).
    """
    if prediction is None:
        return None

    # Both predicted and actual invalid → no information
    if not prediction.predicted_valid and not actual_valid:
        return 0.0

    # Validity mismatch → fixed surprise
    if prediction.predicted_valid != actual_valid:
        return lambda_mismatch

    # Both valid → normalized fitness error
    if actual_fitness is not None and prediction.predicted_fitness is not None:
        raw_error = abs(actual_fitness - prediction.predicted_fitness)
        if sigma_tracker is not None:
            sigma_tracker.observe(actual_fitness)
            sigma = sigma_tracker.sigma
            if sigma > 0:
                return raw_error / sigma
        return raw_error

    return lambda_mismatch


def _residual(
    predicted_valid: bool,
    predicted_fitness: float | None,
    actual_valid: bool,
    actual_fitness: float | None,
    *,
    sigma: float,
    lambda_mismatch: float,
) -> float:
    """Side-effect-free version of the surprise formula.

    Mirrors ``surprise()`` but never updates a SigmaTracker. Used to
    compute memory's "would-have-been-surprise" without observing the
    same actual_fitness twice into σ_f.
    """
    if not predicted_valid and not actual_valid:
        return 0.0
    if predicted_valid != actual_valid:
        return lambda_mismatch
    if actual_fitness is None or predicted_fitness is None:
        return lambda_mismatch
    raw_error = abs(actual_fitness - predicted_fitness)
    if sigma > 0:
        return raw_error / sigma
    return raw_error


def apply_memory_discount(
    raw_surprise: float,
    *,
    prediction: Prediction | None,
    mem_pred_score: float | None,
    mem_pred_valid: bool,
    actual_fitness: float | None,
    actual_valid: bool,
    sigma: float,
    alpha: float,
    lambda_mismatch: float = 1.0,
) -> tuple[float, float]:
    """Discount raw surprise by how much memory beat the LLM's prediction.

    Returns ``(discounted_surprise, explanation)`` where ``explanation``
    ∈ [0, 1] is the fraction of LLM error that memory captured.

    No-op when α=0, when prediction is missing, or when the LLM's residual
    is ~0 (nothing to discount).
    """
    if alpha <= 0.0 or prediction is None:
        return raw_surprise, 0.0

    llm_residual = _residual(
        predicted_valid=prediction.predicted_valid,
        predicted_fitness=prediction.predicted_fitness,
        actual_valid=actual_valid,
        actual_fitness=actual_fitness,
        sigma=sigma,
        lambda_mismatch=lambda_mismatch,
    )
    if llm_residual <= 1e-9:
        return raw_surprise, 0.0

    mem_residual = _residual(
        predicted_valid=mem_pred_valid,
        predicted_fitness=mem_pred_score,
        actual_valid=actual_valid,
        actual_fitness=actual_fitness,
        sigma=sigma,
        lambda_mismatch=lambda_mismatch,
    )

    explanation = max(0.0, llm_residual - mem_residual) / llm_residual
    explanation = min(explanation, 1.0)
    factor = 1.0 - max(0.0, min(alpha, 1.0)) * explanation
    return raw_surprise * factor, explanation


class SigmaTracker:
    """Running estimate of fitness standard deviation.

    Uses a sliding window of recent fitness values. Welford's online
    algorithm would also work but a window adapts to non-stationarity.
    """

    def __init__(self, window: int = 50) -> None:
        self._values: deque[float] = deque(maxlen=window)

    def observe(self, fitness: float) -> None:
        if math.isfinite(fitness):
            self._values.append(fitness)

    @property
    def sigma(self) -> float:
        if len(self._values) < 2:
            return 1.0  # default until we have data
        arr = list(self._values)
        mean = sum(arr) / len(arr)
        var = sum((x - mean) ** 2 for x in arr) / (len(arr) - 1)
        return max(math.sqrt(var), 1e-8)

    @property
    def count(self) -> int:
        return len(self._values)
