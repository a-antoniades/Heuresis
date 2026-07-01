"""Tests for experiment.judge_and_maybe_regrade."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from heuresis.experiment import judge_and_maybe_regrade
from heuresis.judge import HackerVerdict


class _FakeJudge:
    def __init__(self, verdict: HackerVerdict) -> None:
        self._verdict = verdict
        self.calls = 0

    def review(self, **kw: Any) -> HackerVerdict:
        self.calls += 1
        return self._verdict


class _FakeGrader:
    input_files = ["run.log"]

    def __init__(self, new_score: float | None, valid: bool = True) -> None:
        self._new = {"score": new_score, "valid": valid}
        self.calls = 0

    def grade(self, files: dict[str, bytes]) -> dict[str, Any]:
        self.calls += 1
        return self._new


def _make_dirs(tmp_path: Path) -> tuple[Path, Path, Path]:
    task = tmp_path / "task"
    task.mkdir()
    ws = tmp_path / "exec"
    ws.mkdir()
    jd = tmp_path / "judge"
    return task, ws, jd


def test_passthrough_when_judge_is_none(tmp_path: Path) -> None:
    task, ws, jd = _make_dirs(tmp_path)
    info = {"best_score": 0.9, "valid": True}
    new_info, verdict = judge_and_maybe_regrade(
        judge=None, task_dir=task, grader=_FakeGrader(None),
        exec_workspace=ws, judge_dir=jd, idea="x", info=info, gpu_ids=[0],
    )
    assert new_info is info
    assert verdict is None


def test_short_circuits_when_score_is_none(tmp_path: Path) -> None:
    task, ws, jd = _make_dirs(tmp_path)
    judge = _FakeJudge(HackerVerdict(decision="valid", reasoning="x", evidence_refs=[]))
    new_info, verdict = judge_and_maybe_regrade(
        judge=judge, task_dir=task, grader=_FakeGrader(None),
        exec_workspace=ws, judge_dir=jd, idea="x",
        info={"best_score": None, "valid": False}, gpu_ids=[0],
    )
    assert verdict is None
    assert judge.calls == 0


def test_valid_verdict_passes_through_info(tmp_path: Path) -> None:
    task, ws, jd = _make_dirs(tmp_path)
    judge = _FakeJudge(HackerVerdict(
        decision="valid", reasoning="clean", evidence_refs=["run.log:1"],
    ))
    info = {"best_score": 0.9, "valid": True}
    new_info, verdict = judge_and_maybe_regrade(
        judge=judge, task_dir=task, grader=_FakeGrader(None),
        exec_workspace=ws, judge_dir=jd, idea="x", info=info, gpu_ids=[0],
    )
    assert verdict is not None and verdict.decision == "valid"
    assert new_info["best_score"] == 0.9
    assert new_info["valid"] is True
    assert new_info["judge_verdict"] == "valid"


def test_invalid_idea_nullifies_score_and_marks_invalid(tmp_path: Path) -> None:
    task, ws, jd = _make_dirs(tmp_path)
    judge = _FakeJudge(HackerVerdict(
        decision="invalid_idea", reasoning="breaks causality", evidence_refs=["train.py:240"],
    ))
    info = {"best_score": 0.85, "valid": True}
    new_info, verdict = judge_and_maybe_regrade(
        judge=judge, task_dir=task, grader=_FakeGrader(None),
        exec_workspace=ws, judge_dir=jd, idea="x", info=info, gpu_ids=[0],
    )
    assert new_info["best_score"] is None
    assert new_info["valid"] is False
    assert new_info["judge_rejection"] == "breaks causality"
    assert new_info["judge_verdict"] == "invalid_idea"


def test_suspicious_evidence_triggers_regenerate_and_regrade(tmp_path: Path) -> None:
    task, ws, jd = _make_dirs(tmp_path)
    # Write a verify section so regenerate() returns True.
    import yaml
    (task / "task_config.yaml").write_text(yaml.safe_dump({
        "name": "t", "verify": {"command": "echo x"},
    }))

    # Fake the _bwrap call to simulate regeneration succeeding and writing run.log.
    def fake_run_command(*, workspace, command, gpu_ids, timeout, stdout_to):
        out = workspace / stdout_to
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("real output\n")

    grader = _FakeGrader(new_score=0.91, valid=True)
    judge = _FakeJudge(HackerVerdict(
        decision="suspicious_evidence", reasoning="fabricated", evidence_refs=["run.log:1"],
    ))

    with patch("heuresis.experiment._bwrap_run_command", side_effect=fake_run_command):
        new_info, verdict = judge_and_maybe_regrade(
            judge=judge, task_dir=task, grader=grader,
            exec_workspace=ws, judge_dir=jd, idea="x",
            info={"best_score": 0.85, "valid": True}, gpu_ids=[0],
        )

    assert grader.calls == 1
    assert new_info["best_score"] == 0.91
    assert new_info["valid"] is True
    assert new_info["regenerated"] is True
    assert new_info["judge_verdict"] == "suspicious_evidence"


def test_suspicious_evidence_without_verify_marks_invalid(tmp_path: Path) -> None:
    task, ws, jd = _make_dirs(tmp_path)
    # No verify section
    import yaml
    (task / "task_config.yaml").write_text(yaml.safe_dump({"name": "t"}))

    grader = _FakeGrader(new_score=0.91)
    judge = _FakeJudge(HackerVerdict(
        decision="suspicious_evidence", reasoning="fake", evidence_refs=["run.log:1"],
    ))
    new_info, verdict = judge_and_maybe_regrade(
        judge=judge, task_dir=task, grader=grader,
        exec_workspace=ws, judge_dir=jd, idea="x",
        info={"best_score": 0.85, "valid": True}, gpu_ids=[0],
    )
    assert grader.calls == 0
    assert new_info["best_score"] is None
    assert new_info["valid"] is False
    assert new_info["regenerate_unavailable"] is True


def test_errored_fails_closed_by_default(tmp_path: Path) -> None:
    task, ws, jd = _make_dirs(tmp_path)
    judge = _FakeJudge(HackerVerdict(errored=True, decision="valid", reasoning="", evidence_refs=[]))
    info = {"best_score": 0.9, "valid": True}
    new_info, verdict = judge_and_maybe_regrade(
        judge=judge, task_dir=task, grader=_FakeGrader(None),
        exec_workspace=ws, judge_dir=jd, idea="x", info=info, gpu_ids=[0],
    )
    assert new_info["best_score"] is None
    assert new_info["valid"] is False
    assert new_info["judge_errored"] is True


def test_errored_with_fail_open_preserves_score(tmp_path: Path) -> None:
    task, ws, jd = _make_dirs(tmp_path)
    judge = _FakeJudge(HackerVerdict(errored=True, decision="valid", reasoning="", evidence_refs=[]))
    info = {"best_score": 0.9, "valid": True}
    new_info, verdict = judge_and_maybe_regrade(
        judge=judge, task_dir=task, grader=_FakeGrader(None),
        exec_workspace=ws, judge_dir=jd, idea="x", info=info, gpu_ids=[0],
        fail_open=True,
    )
    assert new_info["best_score"] == 0.9
    assert new_info["valid"] is True


def test_suspicious_evidence_warns_when_no_regenerated_files_match_grader(
    tmp_path: Path, caplog,
) -> None:
    """If verify.stdout doesn't match grader.input_files, the read path yields
    an empty files dict. Helper should log a warning and mark invalid."""
    import logging
    import yaml
    task, ws, jd = _make_dirs(tmp_path)
    (task / "task_config.yaml").write_text(yaml.safe_dump({
        "name": "t",
        "verify": {"command": "echo x", "stdout": "out.log"},
    }))

    def fake_run_command(*, workspace, command, gpu_ids, timeout, stdout_to):
        # Regenerate writes to regenerated/out.log
        out = workspace / stdout_to
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("would be out.log\n")

    # Grader reads a different filename
    grader = _FakeGrader(new_score=0.91, valid=True)
    assert grader.input_files == ["run.log"]  # mismatch with verify.stdout="out.log"

    judge = _FakeJudge(HackerVerdict(
        decision="suspicious_evidence", reasoning="fake", evidence_refs=["run.log:1"],
    ))

    with patch("heuresis.experiment._bwrap_run_command", side_effect=fake_run_command):
        with caplog.at_level(logging.WARNING, logger="heuresis.experiment"):
            new_info, verdict = judge_and_maybe_regrade(
                judge=judge, task_dir=task, grader=grader,
                exec_workspace=ws, judge_dir=jd, idea="x",
                info={"best_score": 0.85, "valid": True}, gpu_ids=[0],
            )

    # Warning was logged
    assert any("regenerate produced no files" in rec.message for rec in caplog.records)
    # Grader was NOT called (no files to grade)
    assert grader.calls == 0
    # Run marked invalid (since no files matched)
    assert new_info["best_score"] is None
    assert new_info["valid"] is False
    # regenerated flag still set because regenerate() ran (returned True)
    assert new_info["regenerated"] is True
