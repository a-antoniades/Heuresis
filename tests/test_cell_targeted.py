"""Tests for CellTargetedMapElitesSearch."""
from __future__ import annotations

from heuresis.qd import CellTargetedMapElitesSearch, Feature


FEATURES = [Feature("x", 0, 2, num_bins=3), Feature("y", 0, 1, num_bins=2)]


def _add_elite(
    strategy: CellTargetedMapElitesSearch,
    run_id: str,
    score: float,
    x: float,
    y: float,
) -> None:
    meta = strategy.on_result(
        run_id,
        score,
        features={"x": x, "y": y},
        idea=f"## Strategy\n{run_id}",
    )
    assert meta["archive_status"] == "elite"


def test_empty_target_returns_no_parents_and_explore_context():
    s = CellTargetedMapElitesSearch(FEATURES, maximize=False, seed=0)

    parents = s.select_parents()
    ctx = s.context()

    assert parents == []
    assert "Target cell" in ctx
    assert "EMPTY" in ctx
    assert "unexplored region" in ctx
    assert "Operator: CROSSOVER" not in ctx
    assert "parent" not in ctx.lower()


def test_occupied_target_with_no_crossover_mutates_target_elite():
    s = CellTargetedMapElitesSearch(
        FEATURES,
        maximize=False,
        empty_weight=0.0,
        crossover_rate=0.0,
        seed=0,
    )
    _add_elite(s, "a", 0.5, x=1.0, y=0.0)

    parents = s.select_parents()
    ctx = s.context()

    assert parents == ["a"]
    assert "Operator: MUTATE" in ctx
    assert "load-bearing mechanism" in ctx
    assert "does not need to stay in the same archive cell" in ctx
    assert "staying in the same region" not in ctx


def test_occupied_target_with_crossover_selects_target_and_donor_elites():
    s = CellTargetedMapElitesSearch(
        FEATURES,
        maximize=False,
        empty_weight=0.0,
        crossover_rate=1.0,
        seed=0,
    )
    _add_elite(s, "a", 0.5, x=1.0, y=0.0)
    _add_elite(s, "b", 0.4, x=2.0, y=1.0)

    parents = s.select_parents()
    ctx = s.context()

    assert len(parents) == 2
    assert len(set(parents)) == 2
    assert "Operator: CROSSOVER" in ctx
    assert "Parent A is the target-cell elite" in ctx
    assert "Parent B is another archive elite" in ctx
    assert "Avoid simply stacking incompatible changes" in ctx


def test_crossover_uses_second_sampled_target_cell():
    class ScriptedCellTargetedSearch(CellTargetedMapElitesSearch):
        def _sample_target_cells(self, k: int) -> list[tuple[int, ...]]:
            return [(1, 0), (2, 1)][:k]

    s = ScriptedCellTargetedSearch(
        FEATURES,
        maximize=False,
        empty_weight=0.0,
        crossover_rate=1.0,
        seed=0,
    )
    _add_elite(s, "a", 0.5, x=1.0, y=0.0)
    _add_elite(s, "b", 0.4, x=2.0, y=1.0)

    assert s.select_parents() == ["a", "b"]


def test_crossover_falls_back_to_mutation_when_second_sampled_cell_is_empty():
    class ScriptedCellTargetedSearch(CellTargetedMapElitesSearch):
        def _sample_target_cells(self, k: int) -> list[tuple[int, ...]]:
            return [(1, 0), (0, 0)][:k]

    s = ScriptedCellTargetedSearch(
        FEATURES,
        maximize=False,
        empty_weight=0.0,
        crossover_rate=1.0,
        seed=0,
    )
    _add_elite(s, "a", 0.5, x=1.0, y=0.0)
    _add_elite(s, "b", 0.4, x=2.0, y=1.0)

    parents = s.select_parents()
    ctx = s.context()

    assert parents == ["a"]
    assert "Operator: MUTATE" in ctx


def test_crossover_falls_back_to_mutation_when_only_one_elite_exists():
    s = CellTargetedMapElitesSearch(
        FEATURES,
        maximize=False,
        empty_weight=0.0,
        crossover_rate=1.0,
        seed=0,
    )
    _add_elite(s, "a", 0.5, x=1.0, y=0.0)

    parents = s.select_parents()
    ctx = s.context()

    assert parents == ["a"]
    assert "Operator: MUTATE" in ctx
    assert "Operator: CROSSOVER" not in ctx


def test_strategy_preserves_memory_flag():
    s = CellTargetedMapElitesSearch(FEATURES, maximize=False, memory=True)

    assert s.memory is True


def test_context_uses_target_for_matching_ideator():
    s = CellTargetedMapElitesSearch(FEATURES, maximize=False, seed=42)

    s.select_parents(ideator_id=0)
    ctx0_before = s.context(ideator_id=0)
    s.select_parents(ideator_id=1)
    ctx0_after = s.context(ideator_id=0)

    assert ctx0_after == ctx0_before


def test_targeted_cell_is_deterministic_given_seed():
    s1 = CellTargetedMapElitesSearch(FEATURES, maximize=False, seed=42)
    s2 = CellTargetedMapElitesSearch(FEATURES, maximize=False, seed=42)
    ctx1 = s1.context()
    ctx2 = s2.context()
    assert ctx1 == ctx2
