from __future__ import annotations

from pathlib import Path

import pytest

from heuresis.models import RunResult
from heuresis.store import ResultStore


def _save_run(
    exp,
    run_id: str,
    *,
    tmp_path: Path,
    iteration: int,
    score: float | None,
    parent_ids: list[str] | None = None,
    generation: int = 0,
    metadata: dict | None = None,
) -> None:
    meta = dict(metadata or {})
    if score is not None:
        meta["best_score"] = score
    exp.save(
        run_id,
        result=RunResult(workspace=tmp_path / run_id, exit_code=0),
        iteration=iteration,
        run_type="executor",
        valid=score is not None,
        parent_ids=parent_ids or [],
        generation=generation,
        metadata=meta,
    )


def test_build_lineage_graph_reads_parents_and_infers_edge_kinds(tmp_path):
    from analysis.utils.lineage_tree import build_lineage_graph, load_lineage_nodes

    store = ResultStore(tmp_path / "store.db")
    exp = store.experiment("lineage", root=tmp_path / "runs")
    _save_run(
        exp,
        "exec_000",
        tmp_path=tmp_path,
        iteration=0,
        score=1.0,
        generation=0,
        metadata={"island_id": 0},
    )
    _save_run(
        exp,
        "exec_001",
        tmp_path=tmp_path,
        iteration=1,
        score=0.9,
        parent_ids=["exec_000"],
        generation=1,
        metadata={"island_id": 0, "operator": "mutation"},
    )
    _save_run(
        exp,
        "exec_002",
        tmp_path=tmp_path,
        iteration=2,
        score=0.8,
        parent_ids=["exec_000", "exec_001"],
        generation=2,
        metadata={"island_id": 0, "operator": "crossover"},
    )
    _save_run(
        exp,
        "exec_003",
        tmp_path=tmp_path,
        iteration=3,
        score=0.7,
        parent_ids=["exec_001"],
        generation=2,
        metadata={"island_id": 1, "operator": "mutation"},
    )

    nodes = load_lineage_nodes(tmp_path / "store.db", exp.id)
    graph = build_lineage_graph(nodes)

    assert set(graph.nodes) == {"exec_000", "exec_001", "exec_002", "exec_003"}
    edge_kinds = {(edge.parent_id, edge.child_id): edge.kind for edge in graph.edges}
    assert edge_kinds[("exec_000", "exec_001")] == "mutation"
    assert edge_kinds[("exec_000", "exec_002")] == "crossover_a"
    assert edge_kinds[("exec_001", "exec_002")] == "crossover_b"
    assert edge_kinds[("exec_001", "exec_003")] == "migration"


def test_build_lineage_graph_adds_missing_parent_as_ghost(tmp_path):
    from analysis.utils.lineage_tree import build_lineage_graph, load_lineage_nodes

    store = ResultStore(tmp_path / "store.db")
    exp = store.experiment("lineage", root=tmp_path / "runs")
    _save_run(
        exp,
        "exec_001",
        tmp_path=tmp_path,
        iteration=1,
        score=0.9,
        parent_ids=["exec_missing"],
        generation=1,
        metadata={"operator": "mutation"},
    )

    graph = build_lineage_graph(load_lineage_nodes(tmp_path / "store.db", exp.id))

    assert "exec_missing" in graph.nodes
    assert graph.nodes["exec_missing"].is_ghost is True
    assert graph.nodes["exec_missing"].score is None
    assert [(edge.parent_id, edge.child_id, edge.kind) for edge in graph.edges] == [
        ("exec_missing", "exec_001", "mutation")
    ]


def test_compute_layout_layers_by_generation_and_group():
    from analysis.utils.lineage_tree import (
        LineageEdge,
        LineageGraph,
        LineageNode,
        compute_layered_layout,
    )

    nodes = {
        "a": LineageNode("a", score=1.0, iteration=0, generation=0, group="0"),
        "b": LineageNode("b", score=0.9, iteration=1, generation=1, group="0"),
        "c": LineageNode("c", score=0.8, iteration=2, generation=1, group="1"),
    }
    graph = LineageGraph(nodes=nodes, edges=[LineageEdge("a", "b", "mutation")])

    layout = compute_layered_layout(graph)

    assert list(layout.groups) == ["0", "1"]
    assert layout.positions["a"][1] == pytest.approx(0.0)
    assert layout.positions["b"][1] == pytest.approx(-1.0)
    assert layout.positions["c"][1] == pytest.approx(-1.0)
