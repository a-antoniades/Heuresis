from __future__ import annotations

from heuresis.loops.curiosity import _run_curiosity


def run_curiosity_plus(task_name: str, *, argv: list[str] | None = None) -> None:
    """Curiosity-plus (curiosity + score weighting / tag novelty / memory).

    Shares the prediction-error loop with ``curiosity``; differs only in the
    SearchStrategy (``CuriosityPlusSearch``)."""
    def _make(embedder, adapter, settings, cfg):
        from heuresis.qd import CuriosityPlusSearch
        return CuriosityPlusSearch(
            embedder, k_neighbors=cfg.k_neighbors, candidate_window=cfg.candidate_window,
            softmax_temperature=cfg.softmax_tau, anchor_history=cfg.anchor_history,
            novelty_threshold=cfg.novelty_threshold, n_seed=cfg.n_seed,
            lower_is_better=adapter.lower_is_better, memory=settings.memory,
            score_weight=cfg.score_weight, memory_strength=cfg.memory_strength,
            memory_k=cfg.memory_k, memory_min_k=cfg.memory_min_k)
    _run_curiosity(task_name, "curiosity_plus", _make, argv=argv)
