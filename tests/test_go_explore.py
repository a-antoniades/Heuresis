"""Tests for GoExploreSearch."""
from __future__ import annotations

from heuresis.qd import CellTargetedMapElitesSearch, Feature, GoExploreSearch


FEATURES = [Feature("x", 0, 2, num_bins=3), Feature("y", 0, 0, num_bins=1)]


def _add_elite(
    strategy: GoExploreSearch,
    run_id: str,
    score: float,
    x: float,
) -> None:
    meta = strategy.on_result(
        run_id,
        score,
        features={"x": x, "y": 0.0},
        idea=f"## Strategy\n{run_id}",
    )
    assert meta["archive_status"] == "elite"


def test_go_explore_inherits_cell_targeted_behavior():
    assert issubclass(GoExploreSearch, CellTargetedMapElitesSearch)


def test_empty_archive_returns_no_parents_and_cell_target_context():
    s = GoExploreSearch(FEATURES, maximize=False, baseline_score=1.0, seed=0)

    parents = s.select_parents()
    ctx = s.context()

    assert parents == []
    assert "Target cell" in ctx
    assert "EMPTY" in ctx


def test_lower_visit_cell_is_sampled_more_often():
    s = GoExploreSearch(
        FEATURES,
        maximize=False,
        baseline_score=1.0,
        alpha=0.0,
        crossover_rate=0.0,
        seed=0,
    )
    for i in range(20):
        s.on_result(
            f"flooded_{i}",
            0.8 + i * 0.001,
            {"x": 0.0, "y": 0.0},
            idea="idea",
            parent_ids=[],
        )
    _add_elite(s, "fresh", 0.8, x=1.0)

    counts = {"flooded": 0, "fresh": 0}
    for _ in range(1000):
        parents = s.select_parents()
        if parents == ["fresh"]:
            counts["fresh"] += 1
        elif parents:
            counts["flooded"] += 1

    assert counts["fresh"] > counts["flooded"]


def test_better_score_cell_is_sampled_more_often_when_visits_match():
    s = GoExploreSearch(
        FEATURES,
        maximize=False,
        baseline_score=1.0,
        alpha=0.0,
        crossover_rate=0.0,
        seed=0,
    )
    _add_elite(s, "better", 0.8, x=0.0)
    _add_elite(s, "worse", 0.95, x=1.0)

    counts = {"better": 0, "worse": 0}
    for _ in range(1000):
        parents = s.select_parents()
        if parents == ["better"]:
            counts["better"] += 1
        elif parents == ["worse"]:
            counts["worse"] += 1

    assert counts["better"] > counts["worse"]


def test_crossover_samples_both_parent_cells_with_go_explore_weighting():
    s = GoExploreSearch(
        FEATURES,
        maximize=False,
        baseline_score=1.0,
        alpha=0.0,
        crossover_rate=1.0,
        seed=0,
    )
    _add_elite(s, "strong_a", 0.8, x=0.0)
    _add_elite(s, "strong_b", 0.81, x=1.0)
    _add_elite(s, "weak", 0.99, x=2.0)

    included = {"strong_a": 0, "strong_b": 0, "weak": 0}
    for _ in range(1000):
        parents = s.select_parents()
        for parent in parents:
            included[parent] += 1

    assert included["strong_a"] > included["weak"]
    assert included["strong_b"] > included["weak"]


def test_summary_contains_visit_stats():
    s = GoExploreSearch(FEATURES, maximize=False, baseline_score=1.0, seed=0)
    _add_elite(s, "a", 0.8, x=0.0)
    s.on_result("loser", 0.9, {"x": 0.0, "y": 0.0}, idea="idea", parent_ids=[])

    assert "Visits (occupied min/max/mean)" in s.summary()
