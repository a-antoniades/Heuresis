from __future__ import annotations

from heuresis.loops.map_elites import _run_cell_search
from heuresis.qd import GoExploreSearch


def run_go_explore(task_name: str, *, argv: list[str] | None = None) -> None:
    """Go-Explore search (cell-targeted + score/visit-weighted cell sampling).

    Shares the cell-targeted loop with ``map_elites``; differs only in the
    SearchStrategy (``GoExploreSearch``)."""
    def _make(features, name_fn, adapter, settings, cfg):
        baseline_score = adapter.baseline if adapter.baseline is not None else 1.0
        return GoExploreSearch(
            features,
            maximize=not adapter.lower_is_better,
            baseline_score=baseline_score,
            alpha=cfg.go_explore_alpha,
            crossover_rate=cfg.crossover_rate,
            feature_name_fn=name_fn,
            memory=settings.memory,
        )
    _run_cell_search(task_name, "go_explore", _make, argv=argv)
