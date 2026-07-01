"""Anchor selection: softmax over LP with cosine repetition penalty (§5.2).

w_i = exp(signal_i / τ) · (1 - mean_cos(e_i, last M anchors))

The signal per entry is learning progress where the kNN neighborhood has
enough data (``compute_lp`` returns ``confident=True``), otherwise raw
surprise. This per-entry fallback means early iterations naturally steer
on raw surprise until neighborhoods fill in, without a separate global
phase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from heuresis.qd.curiosity.embedding_store import EmbeddingStore

import numpy as np

from heuresis.qd.curiosity.learning_progress import compute_lp


def select_anchor(
    store: EmbeddingStore,
    *,
    candidate_window: int = 20,
    tau: float = 1.0,
    anchor_history: list[str] | None = None,
    M: int = 5,
    k: int = 10,
    rng: np.random.Generator | None = None,
) -> str | None:
    """Select an anchor idea via softmax over LP with repetition penalty.

    Signal per entry is LP when its kNN neighborhood has ``>= k`` surprise
    values (``confident=True``), otherwise raw surprise. Ideas with no
    surprise value contribute signal 0.

    Returns a run_id, or ``None`` if the store has no valid entries in the
    window.

    Entries with ``valid != True`` are excluded:
      - ``valid=False``: executor failed or training produced no score —
        anchoring on them just steers future ideators at dead ends.
      - ``valid=None``: seed reservation that hasn't finished executing
        yet (no surprise/LP signal available).
    """
    entries = store.recent_entries(candidate_window)
    entries = [e for e in entries if e.valid is True]
    if not entries:
        return None

    rng = rng or np.random.default_rng()
    history = anchor_history or []
    history_window = history[-M:]

    signals = np.zeros(len(entries), dtype=np.float64)
    vectors = np.zeros((len(entries), store.dim), dtype=np.float32)

    for i, entry in enumerate(entries):
        lp, confident = compute_lp(entry.run_id, store, k=k)
        if confident:
            signals[i] = lp
        elif entry.surprise is not None:
            signals[i] = entry.surprise
        vectors[i] = entry.vector

    # Softmax over signal / τ (shift for numerical stability)
    if tau <= 0:
        tau = 1.0
    exp_weights = np.exp((signals - signals.max()) / tau)

    # Repetition penalty: (1 - mean_cos(e_i, recent anchors))
    penalties = np.ones(len(entries), dtype=np.float64)
    history_vecs = []
    for rid in history_window:
        try:
            history_vecs.append(store.get_vector(rid))
        except KeyError:
            continue
    if history_vecs:
        h_matrix = np.stack(history_vecs)  # (H, D)
        sims = vectors @ h_matrix.T        # (N, H)
        penalties = np.clip(1.0 - sims.mean(axis=1), 0.01, 1.0)

    weights = exp_weights * penalties
    total = weights.sum()
    if total <= 0:
        idx = rng.integers(0, len(entries))
    else:
        idx = rng.choice(len(entries), p=weights / total)

    return entries[int(idx)].run_id
