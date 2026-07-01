from __future__ import annotations

from pathlib import Path
from typing import Any

from heuresis import Workspace
from heuresis.tasks.adapter import TaskAdapter
from heuresis.tasks.bbob import BBOBGrader, check_bbob
from heuresis.tools.defaults import MEMORY

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_TASK_DIR = Path(__file__).resolve().parent

_SEED_FILES = {
    "optimizer.py": _TASK_DIR / "optimizer.py",
    "driver.py": _TASK_DIR / "driver.py",
    "problems.py": _TASK_DIR / "problems.py",
    "problem_spec.json": _TASK_DIR / "problem_spec.json",
}


class BBOBAdapter(TaskAdapter):
    """Deprioritized (not a paper task); kept for a fast CPU test path."""

    name = "bbob"
    metric_label = "mean_log_gap"
    lower_is_better = True
    uses_gpu = False
    always_inherits_parent = False

    def __init__(self) -> None:
        self._problem = ""

    @property
    def task_dir(self) -> Path:
        return _TASK_DIR

    @property
    def runs_root(self) -> Path:
        return _PROJECT_ROOT / "runs" / "bbob"

    @property
    def problem_text(self) -> str:
        return self._problem

    @property
    def task_prompt_template(self) -> str | None:
        return "heuresis/tasks/bbob/prompts/ideator_task.j2"

    def normalize_settings(self, settings: Any) -> None:
        if settings.num_ideators == 1:
            settings.num_ideators = 4

    def preflight(self, settings: Any) -> list[str]:
        return check_bbob()

    def on_experiment(self, exp: Any, state: Any, settings: Any) -> None:
        self._problem = (_TASK_DIR / "problem.j2").read_text()

    def _memory_tools(self, memory_on: bool) -> list:
        return [MEMORY] if memory_on else []

    def seed_files(self) -> dict[str, Path]:
        return dict(_SEED_FILES)

    def ideator_workspace(self, settings: Any, *, prompt: Path) -> Workspace:
        return Workspace(
            tools=self._memory_tools(settings.memory),
            files=dict(_SEED_FILES),
            prompt=prompt,
            role="ideator" if settings.memory else None,
        )

    def executor_workspace(
        self, *, files: dict[str, Path], memory_on: bool, prompt: Path | None = None
    ) -> Workspace:
        return Workspace(
            tools=self._memory_tools(memory_on),
            files=files,
            prompt=prompt or (_TASK_DIR / "executor_prompt.j2"),
            venv=_PROJECT_ROOT / "venvs" / "bbob",
            requirements=_TASK_DIR / "requirements.txt",
            role="executor" if memory_on else None,
        )

    def make_grader(self, exec_dir: Path) -> BBOBGrader:
        return BBOBGrader(exec_dir / ".grade.sock")
