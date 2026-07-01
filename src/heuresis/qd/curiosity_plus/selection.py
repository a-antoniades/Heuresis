"""Anchor selection: softmax over LP with cosine repetition penalty (§5.2).

w_i = exp(signal_i / τ) · (1 - mean_cos(e_i, last M anchors))

The signal per entry is learning progress where the kNN neighborhood has
enough data (``compute_lp`` returns ``confident=True``), otherwise raw
surprise. This per-entry fallback means early iterations naturally steer
on raw surprise until neighborhoods fill in, without a separate global
phase.

Optional quality pressure (Change A in curiosity-plus): when
``score_weight > 0``, the final anchor weight blends the LP-based weight
with a score-rank component so the search visibly prefers higher-scoring
anchors. ``score_weight=0`` preserves the pre-curiosity-plus behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from heuresis.qd.curiosity_plus.embedding_store import EmbeddingStore

import numpy as np

from heuresis.qd.curiosity_plus.learning_progress import compute_lp


def select_anchor(
    store: EmbeddingStore,
    *,
    candidate_window: int = 20,
    tau: float = 1.0,
    anchor_history: list[str] | None = None,
    M: int = 5,
    k: int = 10,
    rng: np.random.Generator | None = None,
    score_weight: float = 0.0,
    lower_is_better: bool = True,
    tag_repetition: bool = False,
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

    # Repetition penalty: by default cosine over text embeddings.
    # When tag_repetition=True and recent anchors carry tags, switch to
    # mean Jaccard distance over tag sets — this prevents the search from
    # treating "Soft MoE + butterfly" and "Locally-Connected + Householder"
    # as far apart when they edit the same line of train.py (Change B).
    penalties = np.ones(len(entries), dtype=np.float64)
    if tag_repetition:
        history_tags = [
            ts for ts in (store.get_tag_set(rid) for rid in history_window) if ts
        ]
        # Pre-compute history vectors for the per-candidate cosine fallback.
        history_vecs_for_fallback: list[np.ndarray] = []
        for rid in history_window:
            try:
                history_vecs_for_fallback.append(store.get_vector(rid))
            except KeyError:
                continue
        if history_tags:
            for i, entry in enumerate(entries):
                cand_tags = store.get_tag_set(entry.run_id)
                if cand_tags:
                    dists = [
                        1.0 - (len(cand_tags & h) / max(len(cand_tags | h), 1))
                        for h in history_tags
                    ]
                    penalties[i] = float(np.clip(np.mean(dists), 0.01, 1.0))
                elif history_vecs_for_fallback:
                    # No tags on this candidate — fall back to cosine for it.
                    h_arr = np.stack(history_vecs_for_fallback)
                    sims = entry.vector @ h_arr.T
                    penalties[i] = float(np.clip(1.0 - sims.mean(), 0.01, 1.0))
        # If history has no tagged anchors, leave penalties at 1 — no
        # repetition info to apply yet (steady-state warmup).
    else:
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

    lp_weights = exp_weights * penalties

    # Optional quality pressure: blend score-rank into the final weight.
    # Convex mix: w = (1-β) · lp_norm + β · score_norm
    if score_weight > 0.0:
        scores = np.array(
            [e.score if e.score is not None else np.nan for e in entries],
            dtype=np.float64,
        )
        valid_mask = ~np.isnan(scores)
        if valid_mask.any():
            valid_scores = scores[valid_mask]
            score_w = np.zeros(len(entries))
            if valid_mask.sum() == 1:
                # Single scored entry: it gets the full score-rank weight.
                score_w[valid_mask] = 1.0
            else:
                spread = valid_scores.max() - valid_scores.min()
                if spread > 1e-9:
                    if lower_is_better:
                        score_pref = (valid_scores.max() - valid_scores) / spread
                    else:
                        score_pref = (valid_scores - valid_scores.min()) / spread
                else:
                    score_pref = np.ones_like(valid_scores)
                score_w[valid_mask] = score_pref
            # Both normalized, then convex-mixed.
            lp_norm = lp_weights / max(lp_weights.sum(), 1e-9)
            score_norm = score_w / max(score_w.sum(), 1e-9)
            beta = float(np.clip(score_weight, 0.0, 1.0))
            weights = (1.0 - beta) * lp_norm + beta * score_norm
        else:
            # Fall back to pure LP if no entry has a score yet.
            weights = lp_weights
    else:
        weights = lp_weights

    total = weights.sum()
    if total <= 0:
        idx = rng.integers(0, len(entries))
    else:
        idx = rng.choice(len(entries), p=weights / total)

    return entries[int(idx)].run_id
