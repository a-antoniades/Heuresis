"""Tests for DiscoGenGrader."""
import json
from pathlib import Path

import pytest

from heuresis.tasks.discogen.grader import DiscoGenGrader


BASELINES = {
    "./MinAtar/Breakout": 69.90625,
    "./MinAtar/Freeway": 61.82031,
}


class TestDiscoGenGrader:

    def test_input_files_declares_run_log(self, tmp_path: Path):
        """Grader declares run.log as input_file for host-side fallback scoring."""
        grader = DiscoGenGrader(
            tmp_path / ".grade.sock", baselines=BASELINES, objective="max",
        )
        assert grader.input_files == ["run.log"]

    def test_grade_valid_result(self, tmp_path: Path):
        """Parses JSON, normalizes per-dataset, returns aggregated score."""
        grader = DiscoGenGrader(
            tmp_path / ".grade.sock",
            baselines=BASELINES,
            objective="max",
        )
        run_output = json.dumps({
            "./MinAtar/Breakout": {"return_mean": 72.5, "return_std": 3.1},
            "./MinAtar/Freeway": {"return_mean": 60.0, "return_std": 2.5},
        })
        log_content = f"Running: main.py\nsome output\n{run_output}\n"
        files = {"run.log": log_content.encode()}
        result = grader.grade(files)

        assert result["valid"] is True
        assert result["score"] is not None
        expected_score = (72.5 / 69.90625 + 60.0 / 61.82031) / 2
        assert abs(result["score"] - expected_score) < 1e-6
        assert "per_dataset" in result["details"]
        assert "./MinAtar/Breakout" in result["details"]["per_dataset"]

    def test_grade_missing_run_log(self, tmp_path: Path):
        """Returns invalid when run.log is missing."""
        grader = DiscoGenGrader(tmp_path / ".grade.sock", baselines=BASELINES, objective="max")
        result = grader.grade({})
        assert result["valid"] is False
        assert result["score"] is None

    def test_grade_no_json_in_log(self, tmp_path: Path):
        """Returns invalid when log has no JSON."""
        grader = DiscoGenGrader(tmp_path / ".grade.sock", baselines=BASELINES, objective="max")
        result = grader.grade({"run.log": b"no json here\n"})
        assert result["valid"] is False

    def test_grade_missing_dataset(self, tmp_path: Path):
        """Returns invalid when a baseline dataset is missing from output."""
        grader = DiscoGenGrader(tmp_path / ".grade.sock", baselines=BASELINES, objective="max")
        partial = json.dumps({"./MinAtar/Breakout": {"return_mean": 72.5}})
        result = grader.grade({"run.log": f"{partial}\n".encode()})
        assert result["valid"] is False
        assert "MinAtar/Freeway" in result["details"]["error"]

    def test_grade_min_objective(self, tmp_path: Path):
        """For min objective, lower return_mean is better (inverted normalization)."""
        baselines_min = {"./task_a": 100.0}
        grader = DiscoGenGrader(
            tmp_path / ".grade.sock", baselines=baselines_min, objective="min",
        )
        output = json.dumps({"./task_a": {"return_mean": 80.0}})
        result = grader.grade({"run.log": f"{output}\n".encode()})
        assert result["valid"] is True
        assert result["score"] == pytest.approx(100.0 / 80.0)

    def test_grade_nan_return_mean(self, tmp_path: Path):
        """Returns invalid when return_mean is not a number."""
        grader = DiscoGenGrader(tmp_path / ".grade.sock", baselines=BASELINES, objective="max")
        output = json.dumps({
            "./MinAtar/Breakout": {"return_mean": "nan"},
            "./MinAtar/Freeway": {"return_mean": 60.0},
        })
        result = grader.grade({"run.log": f"{output}\n".encode()})
        assert result["valid"] is False

    def test_grade_per_dataset_details(self, tmp_path: Path):
        """Per-dataset details include both raw and normalized scores."""
        grader = DiscoGenGrader(
            tmp_path / ".grade.sock",
            baselines={"./ds1": 50.0, "./ds2": 100.0},
            objective="max",
        )
        output = json.dumps({
            "./ds1": {"return_mean": 55.0},
            "./ds2": {"return_mean": 110.0},
        })
        result = grader.grade({"run.log": f"{output}\n".encode()})
        details = result["details"]["per_dataset"]
        assert details["./ds1"]["return_mean"] == 55.0
        assert details["./ds1"]["normalized"] == pytest.approx(55.0 / 50.0)
        assert details["./ds2"]["normalized"] == pytest.approx(110.0 / 100.0)

    def test_grade_zero_baseline(self, tmp_path: Path):
        """Zero baseline should return normalized=0, not raise ZeroDivisionError."""
        grader = DiscoGenGrader(
            tmp_path / ".grade.sock",
            baselines={"./ds": 0.0},
            objective="max",
        )
        output = json.dumps({"./ds": {"return_mean": 50.0}})
        result = grader.grade({"run.log": f"{output}\n".encode()})
        assert result["valid"] is True
        assert result["score"] == 0.0

    def test_grade_zero_return_mean_min_objective(self, tmp_path: Path):
        """Zero return_mean with min objective should return normalized=0."""
        grader = DiscoGenGrader(
            tmp_path / ".grade.sock",
            baselines={"./ds": 100.0},
            objective="min",
        )
        output = json.dumps({"./ds": {"return_mean": 0.0}})
        result = grader.grade({"run.log": f"{output}\n".encode()})
        assert result["valid"] is True
        assert result["score"] == 0.0
