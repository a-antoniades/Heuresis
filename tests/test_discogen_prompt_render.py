"""Render the modified discogen prompt templates with memory=True and
memory=False to confirm both branches render cleanly under
StrictUndefined and have the right content.
"""
from __future__ import annotations

from pathlib import Path

from heuresis.workspace import Workspace

REPO = Path(__file__).resolve().parent.parent
DISCOGEN_TASK = REPO / "src" / "heuresis" / "tasks" / "discogen"
DISCOGEN_LINEAR = DISCOGEN_TASK / "prompts"


def _render(template_path: Path, **vars):
    return Workspace(prompt=template_path).render_prompt(vars)


# Common ideator template inputs (mirrors what `ideate()` passes after Task 7).
_IDEATOR_VARS_BASE = dict(
    description="DISCOGEN PROBLEM DESC",
    timeout_minutes=20,
    is_lower_better=True,
    tools=[],
    past_results=[
        {"run_id": "exec_001", "executor_id": "abc123", "score": 0.42, "idea": "loss clip"},
    ],
    new_since_last_turn=[
        {"run_id": "exec_002", "executor_id": "def456", "score": 0.40, "idea": "advantage smoothing"},
    ],
)


# Common executor template inputs (mirrors what `execute()` passes).
_EXEC_VARS_BASE = dict(
    description="DISCOGEN PROBLEM DESC",
    idea="STRATEGY: do X",
    timeout_minutes=20,
    gpu_info="1x A100 40GB",
    tools=[],
)


def test_discogen_linear_ideator_renders_memory_off():
    out = _render(
        DISCOGEN_LINEAR / "linear_ideator_prompt.j2",
        memory=False,
        **_IDEATOR_VARS_BASE,
    )
    # Local memory.md guidance survives the off-branch.
    assert "memory.md" in out
    # Shared CLI must be absent.
    assert "memory search" not in out
    assert "memory append" not in out
    assert "executor_id" not in out
    # past_results must still render.
    assert "exec_001" in out


def test_discogen_linear_ideator_renders_memory_on():
    out = _render(
        DISCOGEN_LINEAR / "linear_ideator_prompt.j2",
        memory=True,
        **_IDEATOR_VARS_BASE,
    )
    # Local memory.md guidance survives.
    assert "memory.md" in out
    # Shared CLI is now visible.
    assert "memory search" in out
    assert "memory append" in out
    # executor_id is rendered alongside run_id in parent lists.
    assert "abc123" in out
    # The composed prompt still includes shared-memory lookup guidance.
    assert "Shared Campaign Memory" in out


def test_discogen_linear_ideator_renders_memory_on_higher_better():
    out = _render(
        DISCOGEN_LINEAR / "linear_ideator_prompt.j2",
        memory=True,
        **{**_IDEATOR_VARS_BASE, "is_lower_better": False},
    )
    assert "score=0.42" in out


def test_discogen_linear_ideator_renders_with_empty_results():
    """Empty past_results / new_since_last_turn must render the
    "(none yet)" / "(nothing new)" cases without crashing.
    """
    out = _render(
        DISCOGEN_LINEAR / "linear_ideator_prompt.j2",
        memory=True,
        **{**_IDEATOR_VARS_BASE, "past_results": [], "new_since_last_turn": []},
    )
    assert "none yet" in out or "nothing new" in out


def test_discogen_executor_prompt_renders_memory_off():
    out = _render(DISCOGEN_TASK / "executor_prompt.j2", memory=False, **_EXEC_VARS_BASE)
    assert "memory append" not in out
    assert "memory search" not in out
    assert "grade run.log" in out


def test_discogen_executor_prompt_renders_memory_on():
    out = _render(DISCOGEN_TASK / "executor_prompt.j2", memory=True, **_EXEC_VARS_BASE)
    assert "memory append" in out
    assert "memory search" in out
    # Local-memory guidance should survive both branches.
    assert "notes.md" in out


def test_discogen_executor_prompt_omits_shared_memory_when_key_missing():
    out = _render(DISCOGEN_TASK / "executor_prompt.j2", **_EXEC_VARS_BASE)
    assert "memory append" not in out
    assert "memory search" not in out
    assert "grade run.log" in out
