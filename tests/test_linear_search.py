"""Tests for LinearSearch strategy."""
from __future__ import annotations

from heuresis.qd.linear.search import LinearSearch


def test_empty_returns_no_parents():
    ls = LinearSearch(max_parents=5, maximize=False)
    assert ls.select_parents() == []


def test_selects_top_k_by_score_lower_is_better():
    ls = LinearSearch(max_parents=3, maximize=False)
    for rid, s in [("a", 0.9), ("b", 0.7), ("c", 0.8), ("d", 0.6), ("e", 0.95)]:
        ls.on_result(rid, s, features=None)
    parents = ls.select_parents()
    assert parents == ["d", "b", "c"]  # 0.6, 0.7, 0.8


def test_selects_top_k_by_score_higher_is_better():
    ls = LinearSearch(max_parents=2, maximize=True)
    for rid, s in [("a", 0.1), ("b", 0.9), ("c", 0.5)]:
        ls.on_result(rid, s, features=None)
    parents = ls.select_parents()
    assert parents == ["b", "c"]


def test_unscored_runs_ignored():
    ls = LinearSearch(max_parents=3, maximize=False)
    ls.on_result("a", 0.5)
    ls.on_result("b", None)  # invalid
    ls.on_result("c", 0.7)
    assert set(ls.select_parents()) == {"a", "c"}


def test_should_reset_session():
    ls = LinearSearch(max_parents=5, session_reset_every=3, maximize=False)
    assert ls.should_reset_session(0) is False
    assert ls.should_reset_session(1) is False
    assert ls.should_reset_session(3) is True
    assert ls.should_reset_session(6) is True


def test_should_reset_session_disabled():
    ls = LinearSearch(max_parents=5, session_reset_every=None, maximize=False)
    for i in range(20):
        assert ls.should_reset_session(i) is False


def test_rebuild():
    ls = LinearSearch(max_parents=5, maximize=False)
    records = [
        ("a", 0.5, {"generation": 0, "idea": "## Strategy\nUse RoPE\n"}),
        ("b", 0.3, {"generation": 1, "idea": "## Strategy\nAdd SwiGLU"}),
        ("c", None, {"generation": 1}),
    ]
    ls.rebuild(records)
    parents = ls.select_parents()
    assert parents == ["b", "a"]


def test_on_result_returns_lineage_metadata():
    ls = LinearSearch(maximize=False)
    meta = ls.on_result("a", 0.5, features=None, idea="## Strategy\nX",
                        parent_ids=["parent"], ideator_id=0)
    assert "parent_ids" in meta
    assert meta["parent_ids"] == ["parent"]
