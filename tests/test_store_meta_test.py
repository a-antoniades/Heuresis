"""Tests for meta_test_results table."""
import json
from pathlib import Path

import pytest

from heuresis.store import ResultStore


@pytest.fixture
def store(tmp_path: Path):
    db = tmp_path / "test.db"
    return ResultStore(db_path=db)


@pytest.fixture
def experiment(store, tmp_path):
    exp_dir = tmp_path / "runs" / "test_exp"
    exp_dir.mkdir(parents=True)
    return store.experiment("test-exp", task="discogen", root=tmp_path / "runs")


def test_save_meta_test_result(experiment):
    """Save and retrieve a meta-test result."""
    per_dataset = {
        "MinAtar/Asterix": {"return_mean": 20.5, "normalized": 1.2},
        "MinAtar/SpaceInvaders": {"return_mean": 190.0, "normalized": 1.05},
    }
    experiment.save_meta_test_result(
        run_id="exec_042",
        score=1.125,
        per_dataset=per_dataset,
    )

    results = experiment.meta_test_results()
    assert len(results) == 1
    assert results[0]["run_id"] == "exec_042"
    assert results[0]["score"] == 1.125
    assert json.loads(results[0]["per_dataset"]) == per_dataset


def test_save_meta_test_result_upsert(experiment):
    """Second save for same run_id should replace."""
    experiment.save_meta_test_result(run_id="exec_001", score=0.9, per_dataset={})
    experiment.save_meta_test_result(run_id="exec_001", score=1.1, per_dataset={"a": 1})

    results = experiment.meta_test_results()
    assert len(results) == 1
    assert results[0]["score"] == 1.1


def test_meta_test_results_empty(experiment):
    """No results when none saved."""
    results = experiment.meta_test_results()
    assert results == []


def test_meta_test_results_multiple_elites(experiment):
    """Multiple elites stored and retrieved in order."""
    experiment.save_meta_test_result(run_id="exec_010", score=1.1, per_dataset={})
    experiment.save_meta_test_result(run_id="exec_005", score=0.95, per_dataset={})
    experiment.save_meta_test_result(run_id="exec_020", score=1.3, per_dataset={})

    results = experiment.meta_test_results()
    assert len(results) == 3
    assert [r["run_id"] for r in results] == ["exec_005", "exec_010", "exec_020"]
