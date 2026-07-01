"""Tests for ResultStore schema + widened save/query surface."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from heuresis.models import RunResult
from heuresis.store import ResultStore


@pytest.fixture
def tmp_store(tmp_path):
    return ResultStore(db_path=tmp_path / "store.db")


def test_save_persists_iteration_and_run_type(tmp_store, tmp_path):
    exp = tmp_store.experiment("test", root=tmp_path / "runs")
    result = RunResult(workspace=tmp_path, exit_code=0, stats={"duration": 12.3})
    exp.save(
        "exec_000",
        result=result,
        iteration=0,
        run_type="executor",
        valid=True,
        idea="Use RoPE with base 50000",
        parent_ids=["exec_prev"],
        generation=1,
        metadata={"best_score": 0.95, "input_tokens": 1234, "total_cost": 0.002},
    )
    runs = exp.runs()
    assert len(runs) == 1
    r = runs[0]
    assert r.iteration == 0
    assert r.run_type == "executor"
    assert r.valid is True
    assert r.idea.startswith("Use RoPE")
    assert r.parent_ids == ["exec_prev"]
    assert r.generation == 1
    assert r.duration_s == pytest.approx(12.3)
    assert r.input_tokens == 1234
    assert r.total_cost == pytest.approx(0.002)
    assert r.score == pytest.approx(0.95)


def test_sqlite_timeout_set(tmp_store):
    # Not directly observable; just verify concurrent writes don't deadlock.
    import threading

    exp = tmp_store.experiment("concurrent", root=Path(tempfile.mkdtemp()))
    errors = []

    def writer(i):
        try:
            exp.save(f"exec_{i:03d}", result=RunResult(workspace=Path("."), exit_code=0),
                     iteration=i, run_type="executor", metadata={"best_score": 1.0 * i})
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert len(exp.runs()) == 8


def test_save_file_snapshot(tmp_store, tmp_path):
    exp = tmp_store.experiment("test", root=tmp_path / "runs")
    exp.save_file("exec_000", "train.py", "print('hello')")
    exp.save_file("exec_000", "run.log", "val_bpb: 0.95\n")
    files = exp.files("exec_000")
    assert files == {"train.py": "print('hello')", "run.log": "val_bpb: 0.95\n"}


def test_log_archive_event(tmp_store, tmp_path):
    exp = tmp_store.experiment("test", root=tmp_path / "runs")
    exp.log_archive_event(
        cell_key="0,0",
        new_id="exec_000", new_fitness=0.95,
        old_id=None, old_fitness=None,
    )
    events = exp.archive_events()
    assert len(events) == 1
    assert events[0]["new_id"] == "exec_000"


def test_save_novelty_review(tmp_store, tmp_path):
    exp = tmp_store.experiment("test", root=tmp_path / "runs")
    exp.save_review(
        review_id="iter0_attempt0",
        run_id=None,  # rejected — no executor run
        iteration=0, attempt=0,
        novelty_score=1, accepted=False,
        explanation="similar to FNet",
        raw_response='{"novelty": 1, "explanation": "similar to FNet"}',
        duration_s=45.0,
        input_tokens=500, output_tokens=120, total_cost=0.003,
    )
    reviews = exp.reviews()
    assert len(reviews) == 1
    assert reviews[0]["accepted"] is False
    assert reviews[0]["explanation"] == "similar to FNet"
