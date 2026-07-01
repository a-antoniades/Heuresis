"""Render tests for composed prompt templates."""

from __future__ import annotations

from pathlib import Path

from heuresis.workspace import Workspace


ROOT = Path(__file__).resolve().parents[1]


def render_template(relative: str, **vars) -> str:
    defaults = {
        "tools": [],
        "memory": False,
        "past_results": [],
        "new_since_last_turn": [],
        "timeout_minutes": 35,
        "problem": "PROBLEM TEXT",
        "description": "DESCRIPTION TEXT",
        "idea": "IDEA TEXT",
        "gpu_info": "1x A100 40GB",
        "is_lower_better": True,
        "search_context": "SEARCH CONTEXT",
        "curiosity_context": "CURIOSITY CONTEXT",
        "existing_summaries": [],
        "prediction_context": "PREDICTION CONTEXT",
        "archive_context": "ARCHIVE CONTEXT",
        "task_name": "test-task",
        "metric": "score",
        "metric_direction": "lower is better",
        "metrics_table": [
            {"name": "forget_acc", "objective": "min", "baseline": 0.5},
            {"name": "utility_acc", "objective": "max", "baseline": 0.7},
        ],
        "baseline": 1.0,
        "training_budget_minutes": 3,
        "editable_file": "train.py",
        "idea_schema": "IDEA SCHEMA",
        "task_prompt_template": "heuresis/tasks/nanogpt/prompts/ideator_task.j2",
    }
    return Workspace(prompt=ROOT / relative).render_prompt({**defaults, **vars})


def test_nanogpt_executor_composed_prompt_keeps_hard_constraints() -> None:
    out = render_template("src/heuresis/tasks/nanogpt/executor_prompt.j2")
    assert "The ONLY file you modify" in out
    assert "python train.py > run.log 2>&1" in out
    assert "grade run.log" in out
    assert "Shared memory" not in out


def test_nanogpt_executor_composed_prompt_memory_branch() -> None:
    out = render_template(
        "src/heuresis/tasks/nanogpt/executor_prompt.j2",
        memory=True,
    )
    assert "memory search" in out
    assert "memory append" in out


def test_nanogpt_linear_ideator_has_three_section_model() -> None:
    out = render_template(
        "src/heuresis/tasks/nanogpt/prompts/linear_ideator_prompt.j2",
        past_results=[{"run_id": "exec_001", "score": 0.99, "idea": "depth"}],
        new_since_last_turn=[],
    )
    assert "## General Agent Guidelines" in out
    assert "## Task-Specific Context" in out
    assert "## Search-Specific Context" in out
    assert "exec_001" in out
    assert "Update `memory.md` first" in out


def test_nanogpt_cell_targeted_prompt_keeps_operator_semantics() -> None:
    out = render_template("src/heuresis/tasks/nanogpt/prompts/cell_ideator_prompt.j2")
    assert "## General Agent Guidelines" in out
    assert "## Search-Specific Context" in out
    assert "Empty target cell:" in out
    assert "MUTATE:" in out
    assert "CROSSOVER:" in out


def test_nanogpt_islands_prompt_keeps_operator_semantics() -> None:
    out = render_template("src/heuresis/tasks/nanogpt/prompts/islands_ideator_prompt.j2")
    assert "## General Agent Guidelines" in out
    assert "## Search-Specific Context" in out
    assert "island search" in out
    assert "MUTATE:" in out
    assert "COMBINE:" in out


REPRESENTATIVE_TEMPLATES = [
    "src/heuresis/tasks/nanogpt/prompts/linear_ideator_prompt.j2",
    "src/heuresis/tasks/nanogpt/prompts/cell_ideator_prompt.j2",
    "src/heuresis/tasks/discogen/prompts/linear_ideator_prompt.j2",
    "src/heuresis/tasks/discogen/prompts/cell_ideator_prompt.j2",
    "src/heuresis/tasks/discogen/prompts/modelunlearning_linear_ideator_prompt.j2",
    "src/heuresis/tasks/discogen/prompts/modelunlearning_cell_ideator_prompt.j2",
    "src/heuresis/tasks/discogen/prompts/modelunlearning_islands_ideator_prompt.j2",
    "src/heuresis/tasks/discogen/prompts/modelunlearning_curiosity_ideator_prompt.j2",
    "src/heuresis/tasks/discogen/prompts/modelunlearning_curiosity_prediction_prompt.j2",
    "src/heuresis/tasks/discogen/prompts/modelunlearning_curiosity_seeding_prompt.j2",
    "src/heuresis/tasks/bbob/prompts/linear_ideator_prompt.j2",
    "src/heuresis/qd/omni_epic/ideator_prompt.j2",
    "src/heuresis/tasks/nanogpt/executor_prompt.j2",
    "src/heuresis/tasks/discogen/executor_prompt.j2",
    "src/heuresis/tasks/discogen/modelunlearning_executor_prompt.j2",
    "src/heuresis/tasks/bbob/executor_prompt.j2",
]


def test_representative_prompts_render_memory_on_and_off() -> None:
    for relative in REPRESENTATIVE_TEMPLATES:
        render_template(relative, memory=False)
        render_template(relative, memory=True)


def test_memory_off_omits_shared_memory_cli_text() -> None:
    for relative in REPRESENTATIVE_TEMPLATES:
        out = render_template(relative, memory=False)
        assert "memory search" not in out, relative
        assert "memory append" not in out, relative
        assert "Shared Campaign Memory" not in out, relative
        assert "Shared Memory" not in out, relative


def test_omni_epic_prompt_uses_task_owned_fragment() -> None:
    out = render_template(
        "src/heuresis/qd/omni_epic/ideator_prompt.j2",
        task_prompt_template="heuresis/tasks/bbob/prompts/ideator_task.j2",
    )
    assert "BBOB runs" in out
    assert "This run uses a Model-of-Interestingness gate" in out


# The per-(task x strategy) run.py shims were removed; every strategy now runs
# through heuresis.loops, and the executor-prompt wiring lives entirely in each
# task's TaskAdapter.
TASK_EXECUTOR_ADAPTERS = [
    "src/heuresis/tasks/nanogpt/adapter.py",
    "src/heuresis/tasks/discogen/adapter.py",
    "src/heuresis/tasks/bbob/adapter.py",
]


def test_task_adapters_use_task_executor_prompt() -> None:
    for relative in TASK_EXECUTOR_ADAPTERS:
        source = (ROOT / relative).read_text()
        assert '_TASK_DIR / "executor_prompt.j2"' in source, relative


def test_modelunlearning_adapter_uses_task_executor_prompt() -> None:
    source = (ROOT / "src/heuresis/tasks/discogen/adapter_modelunlearning.py").read_text()
    assert '_TASK_DIR / "modelunlearning_executor_prompt.j2"' in source
