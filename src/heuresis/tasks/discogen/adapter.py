from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from heuresis.grading import GradingServer
    from heuresis.qd import FeatureClassifier

import yaml

from heuresis import HackerJudge, Harness, Workspace
from heuresis.tasks.adapter import TaskAdapter
from heuresis.tasks.config import load_yaml
from heuresis.tasks.discogen import (
    DiscoGenGrader,
    check_discogen_config,
    check_discogen_gpus,
    load_baselines,
    patch_run_main_walk,
    setup_meta_test_workspace,
)
from heuresis.tools.defaults import MEMORY
from heuresis.workspace import ensure_venv

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_TASK_DIR = Path(__file__).resolve().parent
_EXCLUDE = {"requirements.txt", "install.sh"}
_META_TEST_TOP_K = 16


def _read_discovered_seed_code(baseline_dir: Path, *, max_chars: int = 60000) -> str:
    """Reviewer seed context from generated discovered/*.py files (ported)."""
    discovered_dir = baseline_dir / "discovered"
    if not discovered_dir.is_dir():
        return ""
    chunks: list[str] = []
    used = 0
    for path in sorted(discovered_dir.glob("*.py")):
        text = path.read_text(errors="replace")
        header = f"\n# {path.relative_to(baseline_dir)}\n"
        budget = max_chars - used - len(header)
        if budget <= 0:
            break
        if len(text) > budget:
            text = text[:budget] + "\n# ... truncated ...\n"
        chunks.append(header + text)
        used += len(header) + len(text)
    return "\n".join(chunks)


def _parse_last_json(text: str) -> dict | None:
    for line in reversed(text.strip().split("\n")):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


class _DiscoGenBase(TaskAdapter):
    """Shared discogen runtime: per-DOMAIN venv, generated baseline, parent/baseline
    file sourcing, editable+lockdown sandbox, discovered/* snapshots. OnPolicyRL uses
    it directly; ModelUnlearning subclasses it."""

    name = "discogen"
    metric_label = "score"
    uses_gpu = True
    always_inherits_parent = True

    def __init__(self, task_cfg: Any) -> None:
        self._config_path = Path(task_cfg.config)
        self._domain = task_cfg.domain
        self._install_args = ["--prerelease", "allow"]
        self._config: dict = {}
        self._venv_path = Path()
        self._baseline_dir = Path()
        self._description = ""
        self._idea_schema: str | None = None
        self._baseline_names: set[str] = set()
        self._requirements_path = Path()
        self._train_baselines: dict = {}
        self._test_baselines: dict = {}
        self._objective = "min"
        self._metric: str | None = None
        self._baseline: float | None = None

    @property
    def task_dir(self) -> Path:
        return _TASK_DIR

    @property
    def runs_root(self) -> Path:
        return _PROJECT_ROOT / "runs" / "discogen" / (self._domain or "")

    @property
    def store_task(self) -> str:
        return self._domain or "discogen"

    @property
    def task_prompt_template(self) -> str | None:
        return "heuresis/tasks/discogen/prompts/ideator_task.j2"

    # ---- cell-search capability (SupportsCellSearch) ---------------------
    @property
    def cell_ideator_prompt(self) -> Path:
        return self.strategy_prompt("cell")

    def make_classifier(self) -> "FeatureClassifier":
        from heuresis.tasks.discogen.features import make_classifier
        return make_classifier(self._domain, config=self._config, use_llm=True)

    def moi_context(self) -> Any:
        from heuresis.qd import MoIContext
        return MoIContext(
            task_name=f"discogen-{self._domain}",
            task_description=("DiscoGen runtime-generated algorithm discovery task. "
                              "The executor edits only files under discovered/."),
            domain_description=self._description,
            problem_text="",
            seed_code=_read_discovered_seed_code(self._baseline_dir),
            metric="baseline_normalized_score",
            baseline=1.0,
            lower_is_better=self.lower_is_better,
        )

    @property
    def description_text(self) -> str:
        # discogen drives the ideator with `description`; problem_text stays ""
        # (base default) so ideator_task_vars emits the `description` key.
        return self._description

    @property
    def idea_schema_text(self) -> str | None:
        return self._idea_schema

    @property
    def metric(self) -> str | None:
        return self._metric

    @property
    def baseline(self) -> float | None:
        return self._baseline

    def normalize_settings(self, settings: Any) -> None:
        if not self._domain:
            raise SystemExit("--domain must be set (typically by the shell wrapper).")
        # Rewrite the generic task/strategy name to include the DiscoGen DOMAIN.
        if settings.experiment_name and settings.experiment_name.startswith("discogen-onpolicyrl-"):
            tail = settings.experiment_name[len("discogen-onpolicyrl-"):]
            settings.experiment_name = f"discogen-{self._domain}-{tail}"
        elif settings.experiment_name and settings.experiment_name.startswith("discogen-"):
            tail = settings.experiment_name[len("discogen-"):]
            settings.experiment_name = f"discogen-{self._domain}-{tail}"
        if settings.num_ideators == 1 and len(settings.gpus) > 1:
            settings.num_ideators = len(settings.gpus)

    def preflight(self, settings: Any) -> list[str]:
        if not self._config_path.is_file():
            return [f"--config must point to a valid YAML file. Got: {self._config_path}"]
        with open(self._config_path) as f:
            self._config = yaml.safe_load(f)
        self._config["_domain"] = self._domain
        if not settings.gpus:
            return ["--gpus must be set (wrapper derives it from CUDA_VISIBLE_DEVICES)."]
        errs = check_discogen_config(self._config)
        if errs:
            return errs
        self._venv_path = _PROJECT_ROOT / "venvs" / "discogen" / self._domain
        import discogen as _discogen

        req = (Path(_discogen.__file__).parent / "domains" / self._domain
               / "utils" / "requirements.txt")
        ensure_venv(self._venv_path, requirements=req, install_args=self._install_args)
        return check_discogen_gpus(settings.gpus,
                                   venv_python=self._venv_path / "bin" / "python")

    def setup_objective(self, settings: Any) -> None:
        backend = self._config.get("template_backend", "default")
        self._train_baselines, self._test_baselines, self._objective = load_baselines(
            self._domain, backend,
            train_task_ids=self._config["train_task_id"],
            test_task_ids=self._config["test_task_id"],
        )
        self.lower_is_better = self._objective == "min"

    def on_experiment(self, exp: Any, state: Any, settings: Any) -> None:
        self._baseline_dir = exp.dir / "src"
        if not state.is_resume:
            train_config = dict(self._config)
            train_config["source_path"] = str(self._baseline_dir)
            from discogen import create_task

            create_task(task_domain=self._domain, test=False,
                        config_dict=train_config, no_data=False,
                        use_base=self._config.get("use_base", True))
            patch_run_main_walk(self._baseline_dir)
            print(f"Baseline generated at {self._baseline_dir}")
        self._description = (self._baseline_dir / "description.md").read_text()
        self._baseline_names = {p.name for p in self._baseline_dir.iterdir()
                                if p.name not in _EXCLUDE}
        self._requirements_path = self._baseline_dir / "requirements.txt"
        schema = _TASK_DIR / "idea_schema.md"
        if schema.exists():
            self._idea_schema = schema.read_text()

    def make_judge(self, settings: Any) -> HackerJudge | None:
        if not settings.enable_judge:
            return None
        jh = Harness(settings.judge_agent, model=settings.judge_model, gpus=[])
        return HackerJudge(jh, _TASK_DIR, baseline_dir=self._baseline_dir,
                           timeout=settings.judge_timeout)

    def _memory_tools(self, memory_on: bool) -> list:
        return [MEMORY] if memory_on else []

    def _files_from(self, source: Path) -> dict[str, Path]:
        return {name: source / name for name in self._baseline_names
                if (source / name).exists()}

    def seed_files(self) -> dict[str, Path]:
        return self._files_from(self._baseline_dir)

    def parent_files(self, parent_run: Any) -> dict[str, Path]:
        return self._files_from(Path(parent_run.workspace))

    def ideator_workspace(self, settings: Any, *, prompt: Path) -> Workspace:
        return Workspace(
            tools=self._memory_tools(settings.memory),
            prompt=prompt,
            role="ideator" if settings.memory else None,
        )

    def executor_workspace(
        self, *, files: dict[str, Path], memory_on: bool, prompt: Path | None = None
    ) -> Workspace:
        return Workspace(
            files=files,
            prompt=prompt or (_TASK_DIR / "executor_prompt.j2"),
            venv=self._venv_path,
            requirements=self._requirements_path,
            install_args=self._install_args,
            editable=load_yaml(_TASK_DIR, "task_config.yaml").get("editable"),
            lock_down_edits=True,
            tools=self._memory_tools(memory_on),
            role="executor" if memory_on else None,
        )

    def make_grader(self, exec_dir: Path) -> "GradingServer":
        return DiscoGenGrader(exec_dir / ".grade.sock",
                              baselines=self._train_baselines,
                              objective=self._objective)

    def snapshot_files(self, exec_dir: Path) -> tuple[str, ...]:
        discovered = exec_dir / "discovered"
        files = [f"discovered/{f.name}" for f in discovered.iterdir()
                 if f.is_file()] if discovered.is_dir() else []
        return tuple(files) + ("run.log", "notes.md", "novelty.json")

    def ideator_task_vars(self, *, timeout_minutes: int, memory_on: bool) -> dict[str, Any]:
        base = super().ideator_task_vars(timeout_minutes=timeout_minutes, memory_on=memory_on)
        base["is_lower_better"] = self.lower_is_better
        return base

    def post_loop(self, exp: Any, settings: Any) -> None:
        # Meta-test on held-out datasets (top-K). Verbatim port; sys.exit -> return.
        print()
        print("=" * 60)
        print(f"Meta-test evaluation on held-out datasets (top-{_META_TEST_TOP_K} runs)")
        executor_runs = [r for r in exp.runs(run_type="executor")
                         if r.valid and r.score is not None]
        executor_runs.sort(key=lambda r: r.score, reverse=not self.lower_is_better)
        top_runs = executor_runs[:_META_TEST_TOP_K]
        if not top_runs:
            print("No valid executor runs to evaluate")
            return
        if not self._test_baselines:
            print("No test dataset baselines found; skipping meta-test.")
            return
        for run in top_runs:
            elite_dir = Path(run.workspace)
            test_dir = exp.dir / f"{run.run_id}_test"
            print(f"  Evaluating {run.run_id} (train score={run.score:.4f})...", flush=True)
            try:
                setup_meta_test_workspace(
                    elite_dir, test_dir, self._config,
                    venv_path=self._venv_path,
                    requirements_path=self._requirements_path,
                    install_args=self._install_args,
                )
                proc = subprocess.run(
                    [str(self._venv_path / "bin" / "python"), "run_main.py"],
                    cwd=test_dir, capture_output=True, text=True,
                    timeout=settings.executor_timeout,
                )
                if proc.returncode != 0:
                    print(f"    FAIL: {proc.stderr[:300]}", flush=True)
                    continue
                metrics = _parse_last_json(proc.stdout)
                if not metrics:
                    print("    FAIL: no JSON output in run_main.py stdout", flush=True)
                    continue
                per_dataset: dict[str, dict[str, float]] = {}
                normalized_scores: list[float] = []
                for ds_path, baseline in self._test_baselines.items():
                    if ds_path not in metrics:
                        print(f"    WARN: missing {ds_path} in output", flush=True)
                        continue
                    rm = metrics[ds_path].get("return_mean")
                    if rm is None or not isinstance(rm, (int, float)) or math.isnan(rm):
                        continue
                    if self._objective == "max":
                        norm = rm / baseline if baseline != 0 else 0.0
                    else:
                        norm = baseline / rm if rm != 0 else 0.0
                    if math.isnan(norm) or math.isinf(norm):
                        continue
                    per_dataset[ds_path] = {"return_mean": rm, "normalized": norm}
                    normalized_scores.append(norm)
                if not normalized_scores:
                    print("    FAIL: no valid per-dataset scores", flush=True)
                    continue
                test_score = sum(normalized_scores) / len(normalized_scores)
                exp.save_meta_test_result(run.run_id, score=test_score, per_dataset=per_dataset)
                print(f"    test_score={test_score:.4f}", flush=True)
            except subprocess.TimeoutExpired:
                print(f"    TIMEOUT after {settings.executor_timeout}s", flush=True)
            except Exception as e:
                print(f"    ERROR: {e}", flush=True)
        print("Meta-test complete.")


class OnPolicyRLAdapter(_DiscoGenBase):
    """DiscoGen with the OnPolicyRL domain (the generic discogen path)."""

    name = "discogen_onpolicyrl"
