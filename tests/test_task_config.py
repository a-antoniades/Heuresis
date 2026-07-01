"""Tests for tasks/config.py helpers."""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from heuresis.tasks import (
    baseline_scores,
    load_yaml,
    lower_is_better,
    task_dir,
)


def test_task_dir_resolves_known_task() -> None:
    p = task_dir("nanogpt")
    assert p.is_dir()
    assert p.name == "nanogpt"
    assert (p / "task_config.yaml").is_file()


def test_task_dir_raises_on_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown task"):
        task_dir("does_not_exist_xyz")


def test_baseline_scores_returns_dict_for_nanogpt() -> None:
    scores = baseline_scores(task_dir("nanogpt"))
    assert scores["metric"] == "val_bpb"
    assert scores["objective"] == "min"
    assert scores["baseline"] == pytest.approx(0.992)


def test_lower_is_better_nanogpt_true() -> None:
    assert lower_is_better(task_dir("nanogpt")) is True


def test_load_yaml_reads_arbitrary_file(tmp_path: Path) -> None:
    (tmp_path / "thing.yaml").write_text("a: 1\nb: [x, y]\n")
    out = load_yaml(tmp_path, "thing.yaml")
    assert out == {"a": 1, "b": ["x", "y"]}


def test_load_yaml_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path, "no_such_file.yaml")


def test_baseline_scores_synthetic_max(tmp_path: Path) -> None:
    (tmp_path / "baseline_scores.yaml").write_text(
        dedent(
            """
            metric: accuracy
            objective: max
            baseline: 0.85
            """
        )
    )
    assert lower_is_better(tmp_path) is False
    assert baseline_scores(tmp_path)["metric"] == "accuracy"
