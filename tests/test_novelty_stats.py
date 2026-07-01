"""Tests for NoveltyAssessment stats fields."""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

from heuresis.novelty import NoveltyAssessment, NoveltyReviewer


def test_novelty_assessment_has_stats_fields():
    a = NoveltyAssessment(
        novelty=2,
        explanation="test",
        duration_s=42.0,
        input_tokens=100,
        output_tokens=30,
        total_cost=0.005,
    )
    assert a.duration_s == 42.0
    assert a.input_tokens == 100
    assert a.total_cost == 0.005


def test_assess_populates_stats(tmp_path):
    # Mock harness.run() returning a RunFuture whose result() has stats + workspace
    from heuresis.models import RunResult

    ws = tmp_path / "review"
    ws.mkdir()
    (ws / "novelty.json").write_text('{"novelty": 2, "explanation": "test"}')

    result = RunResult(
        workspace=ws,
        exit_code=0,
        stats={
            "duration": 33.0,
            "input_tokens": 150,
            "output_tokens": 40,
            "total_cost": 0.007,
        },
    )
    future = MagicMock()
    future.result.return_value = result
    harness = MagicMock()
    harness.run.return_value = future

    reviewer = NoveltyReviewer(harness, timeout=300)
    # Inject Workspace that doesn't try to materialize venv
    reviewer._workspace = MagicMock()
    with _no_real_hf_mount():
        assessment = reviewer.assess("test idea", workspace_path=ws)

    assert assessment.novelty == 2
    assert assessment.duration_s == 33.0
    assert assessment.input_tokens == 150


def test_reviewer_uses_novelty_project_extra():
    harness = MagicMock()
    reviewer = NoveltyReviewer(harness, timeout=300)
    assert reviewer._workspace.project_extra == "novelty"
    assert reviewer._workspace.requirements is None


@contextmanager
def _no_real_hf_mount():
    yield
