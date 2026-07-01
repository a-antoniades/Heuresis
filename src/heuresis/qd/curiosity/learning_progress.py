"""Learning progress via split-halves over kNN neighborhoods (§4.5).

LP = mean_surprise(older half) - mean_surprise(recent half)

Positive LP → predictions improving in this region → explore here.
Zero LP → region understood or inherently noisy → low priority.
Negative LP → predictions getting worse → unexpected complexity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from heuresis.qd.curiosity.embedding_store import EmbeddingStore

import numpy as np


def compute_lp(
    run_id: str,
    store: EmbeddingStore,
    *,
    k: int = 10,
) -> tuple[float, bool]:
    """Compute learning progress for an idea's kNN neighborhood.

    Returns (lp_value, confident) where confident=True when the
    neighborhood has enough data for a reliable LP estimate (≥ k
    neighbors with surprise values).
    """
    neighbors = store.knn_by_run_id(run_id, k, include_self=True)

    # Collect surprise values sorted by iteration (time order)
    timed: list[tuple[int, float]] = []
    for entry, _sim in neighbors:
        if entry.surprise is not None:
            timed.append((entry.iteration, entry.surprise))

    if len(timed) < 4:
        # Too few data points for a meaningful split
        return 0.0, False

    timed.sort(key=lambda t: t[0])
    surprises = [s for _, s in timed]

    mid = len(surprises) // 2
    older = surprises[:mid]
    recent = surprises[mid:]

    lp = _mean(older) - _mean(recent)
    confident = len(timed) >= k
    return lp, confident


def compute_lp_from_vector(
    vector: np.ndarray,
    store: EmbeddingStore,
    *,
    k: int = 10,
) -> tuple[float, bool]:
    """Compute LP from a raw embedding vector (for ideas not yet stored)."""
    neighbors = store.knn(vector, k)

    timed: list[tuple[int, float]] = []
    for entry, _sim in neighbors:
        if entry.surprise is not None:
            timed.append((entry.iteration, entry.surprise))

    if len(timed) < 4:
        return 0.0, False

    timed.sort(key=lambda t: t[0])
    surprises = [s for _, s in timed]

    mid = len(surprises) // 2
    older = surprises[:mid]
    recent = surprises[mid:]

    lp = _mean(older) - _mean(recent)
    confident = len(timed) >= k
    return lp, confident


def compute_all_lp(
    candidate_run_ids: list[str],
    store: EmbeddingStore,
    *,
    k: int = 10,
) -> list[tuple[str, float, bool]]:
    """Batch LP computation for a candidate window.

    Returns list of (run_id, lp, confident) triples.
    """
    return [
        (rid, *compute_lp(rid, store, k=k))
        for rid in candidate_run_ids
    ]


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
