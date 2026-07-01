"""TaskAdapter: a THIN, task-only runtime façade over a task's existing assets.

A task is a directory layout + config files (see tasks/config.py — "there is
no Task class, the layout is the contract"). TaskAdapter is the runtime
counterpart: it exposes that task's data (problem/description/schema/metric/
baseline loaded from the task's own .j2/.yaml assets) and task-only behaviors
(preflight, objective, grader, judge, workspaces, mounts, snapshots, post-loop).

It is strategy-agnostic by construction: it has NO strategy methods. Each
``run_<strategy>`` loop owns all search machinery (archive/reviewer/retry,
classifiers, seeding/prediction, migration, cell-targeting), chooses prompt
paths, and assembles prompt vars by merging its strategy ``extra`` into the
task-data dicts the adapter provides. Adding a new task = one thin adapter
pointing at that task's assets.
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from heuresis import HackerJudge, Mount, Workspace
    from heuresis.experiment import ExperimentState, Settings
    from heuresis.grading import GradingServer
    from heuresis.qd import FeatureClassifier
    from heuresis.store import Experiment


@runtime_checkable
class SupportsCellSearch(Protocol):
    """Capability a task must provide to run cell-based strategies
    (map_elites, go_explore): a feature classifier (which carries the labeled
    archive axes) and a cell ideator prompt. Tasks that can't be binned onto a
    grid simply don't implement these, and the cell loops reject them with a
    clear error instead of failing deep inside the archive.
    """

    def make_classifier(self) -> "FeatureClassifier": ...

    @property
    def cell_ideator_prompt(self) -> Path: ...


class TaskAdapter(abc.ABC):
    """Per-task runtime data + behaviors consumed by every per-strategy loop."""

    name: str
    metric_label: str                 # "val_bpb" / "mean_log_gap" / "score"
    lower_is_better: bool = True      # finalized in setup_objective()
    uses_gpu: bool = True
    always_inherits_parent: bool = False  # discogen domains start from parent/baseline
    prompt_prefix: str = ""           # disambiguates tasks sharing a task_dir
                                      # (discogen_modelunlearning sets "modelunlearning_")

    # ---- identity / paths ------------------------------------------------
    @property
    @abc.abstractmethod
    def task_dir(self) -> Path: ...

    @property
    @abc.abstractmethod
    def runs_root(self) -> Path: ...

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    @property
    def store_path(self) -> Path:
        return self.runs_root / "store.db"

    def strategy_prompt(self, strategy: str, kind: str = "ideator") -> Path:
        """Path to a per-(task x strategy) wrapper prompt under the task's own
        ``prompts/`` dir. ``kind`` is ``ideator`` / ``seeding`` / ``prediction``.
        ``prompt_prefix`` disambiguates tasks that share a ``task_dir``."""
        return (self.task_dir / "prompts"
                / f"{self.prompt_prefix}{strategy}_{kind}_prompt.j2")

    @property
    def store_task(self) -> str:
        return self.name

    # ---- task metadata (loaded from the task's own assets) ---------------
    @property
    def problem_text(self) -> str:
        return ""

    @property
    def description_text(self) -> str:
        return self.problem_text

    @property
    def idea_schema_text(self) -> str | None:
        return None

    @property
    def metric(self) -> str | None:
        return None

    @property
    def baseline(self) -> float | None:
        return None

    @property
    def metrics_table(self) -> list[dict[str, Any]] | None:
        return None

    @property
    def task_prompt_template(self) -> str | None:
        return None

    @property
    def editable(self) -> str:
        """The task's editable path (for the omni ideator prompt). Reads the
        task's own task_config.yaml; '' if absent."""
        from heuresis.tasks.config import load_yaml
        try:
            return load_yaml(self.task_dir, "task_config.yaml").get("editable", "") or ""
        except Exception:
            return ""

    def moi_context(self) -> Any | None:
        """Optional MoIContext for the omni reviewer (discogen overrides)."""
        return None

    # ---- lifecycle (loops call in this order) ----------------------------
    def normalize_settings(self, settings: "Settings") -> None: ...

    @abc.abstractmethod
    def preflight(self, settings: "Settings") -> list[str]: ...

    def setup_objective(self, settings: "Settings") -> None: ...

    def on_experiment(
        self, exp: "Experiment", state: "ExperimentState", settings: "Settings"
    ) -> None: ...

    def make_judge(self, settings: "Settings") -> "HackerJudge | None":
        return None

    # ---- workspaces (loop supplies the strategy-chosen prompt + files) ----
    @abc.abstractmethod
    def seed_files(self) -> dict[str, Path]:
        """Files an executor starts from when NOT inheriting a parent."""

    def parent_files(self, parent_run: Any) -> dict[str, Path]:
        """Files an executor inherits from a parent run. Default: parent's seed-named
        files resolved under the parent workspace (tasks override as needed)."""
        src = Path(parent_run.workspace)
        return {name: src / Path(p).name for name, p in self.seed_files().items()}

    @abc.abstractmethod
    def ideator_workspace(self, settings: "Settings", *, prompt: Path) -> "Workspace": ...

    @abc.abstractmethod
    def executor_workspace(
        self, *, files: dict[str, Path], memory_on: bool, prompt: Path | None = None
    ) -> "Workspace": ...

    @abc.abstractmethod
    def make_grader(self, exec_dir: Path) -> "GradingServer": ...

    def mounts(self) -> "list[Mount]":
        return []

    def snapshot_files(self, exec_dir: Path) -> tuple[str, ...] | None:
        return None

    # ---- task prompt data (loops merge strategy `extra`) -----------------
    def ideator_task_vars(self, *, timeout_minutes: int, memory_on: bool) -> dict[str, Any]:
        base: dict[str, Any] = {
            "problem": self.problem_text,
            "timeout_minutes": timeout_minutes,
            "memory": memory_on,
        }
        if self.description_text != self.problem_text:
            base["description"] = self.description_text
        for k, v in (("idea_schema", self.idea_schema_text), ("metric", self.metric),
                     ("baseline", self.baseline), ("metrics_table", self.metrics_table),
                     ("task_prompt_template", self.task_prompt_template)):
            if v is not None:
                base[k] = v
        return base

    def executor_task_vars(
        self, *, idea: str, timeout_minutes: int, gpu_count: int, memory_on: bool
    ) -> dict[str, Any]:
        base: dict[str, Any] = {
            "problem": self.problem_text,
            "idea": idea,
            "timeout_minutes": timeout_minutes,
            "gpu_info": (f"{gpu_count}x A100 40GB" if gpu_count else "1x A100 40GB"),
            "memory": memory_on,
        }
        if self.description_text != self.problem_text:
            base["description"] = self.description_text
        if self.metrics_table is not None:
            base["metrics_table"] = self.metrics_table
        return base

    # ---- post ------------------------------------------------------------
    def post_loop(self, exp: "Experiment", settings: "Settings") -> None: ...

    def report_best(self, exp: "Experiment") -> None:
        best = exp.best(lower_is_better=self.lower_is_better, run_type="executor")
        if best:
            print(f"Best: {best.run_id} {self.metric_label}={best.score}")


def get_task_adapter(task: str, task_cfg: Any, settings: Any) -> TaskAdapter:
    """Resolve a task name to its concrete adapter (lazy imports keep heavy
    task deps off the import path until the task is actually selected)."""
    if task == "nanogpt":
        from heuresis.tasks.nanogpt.adapter import NanoGPTAdapter
        return NanoGPTAdapter()
    if task in {"discogen_onpolicyrl", "discogen"}:
        from heuresis.tasks.discogen.adapter import OnPolicyRLAdapter
        return OnPolicyRLAdapter(task_cfg)
    if task == "discogen_modelunlearning":
        from heuresis.tasks.discogen.adapter_modelunlearning import ModelUnlearningAdapter
        return ModelUnlearningAdapter(task_cfg)
    if task == "bbob":
        from heuresis.tasks.bbob.adapter import BBOBAdapter
        return BBOBAdapter()
    raise ValueError(f"no TaskAdapter for task {task!r}")
