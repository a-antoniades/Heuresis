from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from heuresis import HackerJudge, Harness, Mount, Workspace
from heuresis.tasks.adapter import TaskAdapter

if TYPE_CHECKING:
    from heuresis.qd import FeatureClassifier
from heuresis.tasks import baseline_scores
from heuresis.tasks.nanogpt import NanoGPTGrader, check_nanogpt
from heuresis.tools.defaults import MEMORY

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_TASK_DIR = Path(__file__).resolve().parent
_CACHE_DIR = Path.home() / ".cache" / "autoresearch"

_SEED_FILES = {
    "train.py": _TASK_DIR / "train.py",
    "prepare.py": _TASK_DIR / "prepare.py",
}


class NanoGPTAdapter(TaskAdapter):
    name = "nanogpt"
    metric_label = "val_bpb"
    lower_is_better = True
    uses_gpu = True
    always_inherits_parent = False

    def __init__(self) -> None:
        self._problem = ""
        self._idea_schema: str | None = None
        self._metric: str | None = None
        self._baseline: float | None = None

    @property
    def task_dir(self) -> Path:
        return _TASK_DIR

    @property
    def runs_root(self) -> Path:
        return _PROJECT_ROOT / "runs" / "nanogpt"

    @property
    def problem_text(self) -> str:
        return self._problem

    @property
    def idea_schema_text(self) -> str | None:
        return self._idea_schema

    @property
    def metric(self) -> str | None:
        return self._metric

    @property
    def baseline(self) -> float | None:
        return self._baseline

    @property
    def task_prompt_template(self) -> str | None:
        return "heuresis/tasks/nanogpt/prompts/ideator_task.j2"

    # ---- cell-search capability (SupportsCellSearch) ---------------------
    @property
    def cell_ideator_prompt(self) -> Path:
        return self.strategy_prompt("cell")

    def make_classifier(self) -> "FeatureClassifier":
        from heuresis.tasks.nanogpt.features import make_classifier
        return make_classifier(use_llm=True)

    def normalize_settings(self, settings: Any) -> None:
        if not settings.gpus:
            settings.gpus = list(range(8))
        if settings.num_ideators == 1 and len(settings.gpus) > 1:
            settings.num_ideators = len(settings.gpus)

    def preflight(self, settings: Any) -> list[str]:
        return check_nanogpt(gpus=settings.gpus)

    def setup_objective(self, settings: Any) -> None:
        scores = baseline_scores(_TASK_DIR)
        self._metric = scores.get("metric")
        self._baseline = scores.get("baseline")
        self.lower_is_better = scores.get("objective", "min") == "min"

    def on_experiment(self, exp: Any, state: Any, settings: Any) -> None:
        self._problem = (_TASK_DIR / "problem.j2").read_text()
        schema = _TASK_DIR / "idea_schema.md"
        if schema.exists():
            self._idea_schema = schema.read_text()

    def make_judge(self, settings: Any) -> HackerJudge | None:
        if not settings.enable_judge:
            return None
        jh = Harness(settings.judge_agent, model=settings.judge_model, gpus=[])
        return HackerJudge(jh, _TASK_DIR, timeout=settings.judge_timeout)

    def _memory_tools(self, memory_on: bool) -> list:
        return [MEMORY] if memory_on else []

    def seed_files(self) -> dict[str, Path]:
        return dict(_SEED_FILES)

    def parent_files(self, parent_run: Any) -> dict[str, Path]:
        # inherit the evolved train.py from the parent; prepare.py stays canonical
        src = Path(parent_run.workspace)
        return {"train.py": src / "train.py", "prepare.py": _TASK_DIR / "prepare.py"}

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
            venv=_PROJECT_ROOT / "venvs" / "nanogpt",
            requirements=_TASK_DIR / "requirements.txt",
            role="executor" if memory_on else None,
        )

    def make_grader(self, exec_dir: Path) -> NanoGPTGrader:
        return NanoGPTGrader(exec_dir / ".grade.sock")

    def mounts(self) -> list[Mount]:
        return [
            Mount(source=_CACHE_DIR / "data",
                  target="/workspace/.cache/autoresearch/data"),
            Mount(source=_CACHE_DIR / "tokenizer",
                  target="/workspace/.cache/autoresearch/tokenizer"),
        ]
