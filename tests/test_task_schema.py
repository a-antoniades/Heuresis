"""Contract tests for public task directory layouts."""

from __future__ import annotations

import pytest

from heuresis.tasks import baseline_scores, load_yaml, task_dir


TASKS = ["nanogpt", "bbob", "discogen"]


@pytest.mark.parametrize("name", TASKS)
def test_task_has_required_yaml(name: str) -> None:
    td = task_dir(name)
    cfg = load_yaml(td, "task_config.yaml")
    scores = baseline_scores(td)

    assert cfg["name"]
    assert cfg["description"]
    assert scores["metric"]
    assert scores["objective"] in {"min", "max"}


@pytest.mark.parametrize("name", ["nanogpt", "bbob"])
def test_static_tasks_have_public_prompt_files(name: str) -> None:
    td = task_dir(name)
    for filename in ("description.md", "problem.j2", "idea_schema.md"):
        assert (td / filename).is_file(), f"{name} missing {filename}"


def test_discogen_documents_runtime_prompt_layout() -> None:
    td = task_dir("discogen")
    assert (td / "prompts" / "ideator_task.j2").is_file()
    assert (td / "prompts" / "executor_task.j2").is_file()
    assert (td / "prompts" / "modelunlearning_idea_schema.md").is_file()
