"""Tests for judge_reviews persistence in ResultStore."""
from __future__ import annotations

from pathlib import Path

import pytest

from heuresis.judge import HackerVerdict
from heuresis.store import ResultStore


def test_save_judge_review_creates_row(tmp_path: Path) -> None:
    store = ResultStore(db_path=tmp_path / "s.db")
    exp = store.experiment("t", task="fake", root=tmp_path / "runs")

    verdict = HackerVerdict(
        decision="suspicious_evidence",
        reasoning="fake block at line 1432",
        evidence_refs=["run.log:1432", "agent.log:94"],
        raw_response='{"decision": "suspicious_evidence"}',
        errored=False,
        duration_s=12.3,
        input_tokens=1500,
        output_tokens=80,
    )
    exp.save_judge_review("exec_007", verdict)

    rows = store.query(
        "SELECT * FROM judge_reviews WHERE experiment_id = ? AND run_id = ?",
        (exp.id, "exec_007"),
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["decision"] == "suspicious_evidence"
    assert row["reasoning"] == "fake block at line 1432"
    assert row["errored"] == 0
    assert row["duration_s"] == pytest.approx(12.3)
    assert row["input_tokens"] == 1500
    assert row["output_tokens"] == 80
    # evidence_refs is JSON-encoded
    import json
    assert json.loads(row["evidence_refs"]) == ["run.log:1432", "agent.log:94"]


def test_save_judge_review_errored_round_trips(tmp_path: Path) -> None:
    store = ResultStore(db_path=tmp_path / "s.db")
    exp = store.experiment("t", task="fake", root=tmp_path / "runs")
    v = HackerVerdict(errored=True, decision="valid", raw_response="")
    exp.save_judge_review("exec_000", v)
    rows = store.query(
        "SELECT errored, decision FROM judge_reviews WHERE experiment_id = ?",
        (exp.id,),
    )
    assert rows[0]["errored"] == 1
    assert rows[0]["decision"] == "valid"


def test_save_judge_review_replaces_on_duplicate_run_id(tmp_path: Path) -> None:
    """PRIMARY KEY is (experiment_id, run_id); a second save overwrites."""
    store = ResultStore(db_path=tmp_path / "s.db")
    exp = store.experiment("t", task="fake", root=tmp_path / "runs")

    exp.save_judge_review("exec_001", HackerVerdict(
        decision="valid", reasoning="first", evidence_refs=[],
    ))
    exp.save_judge_review("exec_001", HackerVerdict(
        decision="invalid_idea", reasoning="second", evidence_refs=[],
    ))
    rows = store.query(
        "SELECT decision, reasoning FROM judge_reviews WHERE run_id = ?",
        ("exec_001",),
    )
    assert len(rows) == 1
    assert rows[0]["decision"] == "invalid_idea"
    assert rows[0]["reasoning"] == "second"
