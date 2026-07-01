"""Prediction dataclass and surprise computation (§4.3 of proposal).

Surprise measures how wrong the LLM's prediction was:
  - Both predicted and actual invalid → 0 (no information)
  - Validity mismatch → λ (large fixed surprise)
  - Both valid → |actual_fitness - predicted_fitness| / σ_f
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
