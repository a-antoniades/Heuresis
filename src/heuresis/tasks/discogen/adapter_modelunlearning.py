from __future__ import annotations

from pathlib import Path
from typing import Any

from heuresis import HackerJudge, Harness, Workspace
from heuresis.tasks.discogen.adapter import _DiscoGenBase, _TASK_DIR, _PROJECT_ROOT
from heuresis.tasks.discogen import (
    ModelUnlearningGrader,
    apply_fast_eval_patches,
    check_discogen_config,
    check_discogen_gpus_torch,
    clone_baseline_template,
    ensure_modelunlearning_baseline_template,
    load_unlearning_baselines,
)
from heuresis.workspace import ensure_venv


class ModelUnlearningAdapter(_DiscoGenBase):
    """DiscoGen ModelUnlearning domain. Distinct family: multi-metric grader,
    torch GPU preflight, cached baseline-template clone, optional fast-eval,
    maximize objective, no meta-test."""

    name = "discogen_modelunlearning"
    metric_label = "score"
    uses_gpu = True
    always_inherits_parent = True
    prompt_prefix = "modelunlearning_"  # shares discogen task_dir with OnPolicyRL

    def __init__(self, task_cfg: Any) -> None:
        super().__init__(task_cfg)
        self._fast_eval = getattr(task_cfg, "mu_fast_eval", False)
        self._train_pairs: list = []
        self._test_pairs: list | None = None
        self._metrics_table: list[dict[str, Any]] | None = None

    @property
    def metrics_table(self) -> list[dict[str, Any]] | None:
        return self._metrics_table

    @property
    def task_prompt_template(self) -> str | None:
        return "heuresis/tasks/discogen/prompts/modelunlearning_ideator_task.j2"

    def normalize_settings(self, settings: Any) -> None:
        if not self._domain:
            raise SystemExit("--domain must be set (typically by the shell wrapper).")
        if settings.experiment_name and settings.experiment_name.startswith("discogen_modelunlearning-"):
            tail = settings.experiment_name.split("-", 1)[1]
            settings.experiment_name = f"discogen-{self._domain}-{tail}"
        if settings.num_ideators == 1 and len(settings.gpus) > 1:
            settings.num_ideators = len(settings.gpus)

    def _pairs(self) -> None:
        c = self._config
        self._train_pairs = list(zip(c["train_task_id"], c["train_model_id"]))
        test_models = c.get("test_model_id")
        self._test_pairs = (list(zip(c["test_task_id"], test_models))
                            if test_models else None)

    def preflight(self, settings: Any) -> list[str]:
        if not self._config_path.is_file():
            return [f"--config must point to a valid YAML file. Got: {self._config_path}"]
        import yaml
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
        return check_discogen_gpus_torch(settings.gpus,
                                         venv_python=self._venv_path / "bin" / "python")

    def setup_objective(self, settings: Any) -> None:
        self._pairs()
        backend = self._config.get("template_backend", "default")
        self._train_baselines, self._test_baselines = load_unlearning_baselines(
            self._domain, backend,
            train_pairs=self._train_pairs, test_pairs=self._test_pairs,
        )
        self.lower_is_better = False  # composite score: >1.0 beats baseline
        only = next(iter(self._train_baselines.values())) if self._train_baselines else {}
        self._metrics_table = [
            {"name": name, "baseline": baseline, "objective": objective}
            for name, (baseline, objective) in only.items()
        ]

    def on_experiment(self, exp: Any, state: Any, settings: Any) -> None:
        backend = self._config.get("template_backend", "default")
        template_dir = ensure_modelunlearning_baseline_template(
            domain=self._domain,
            train_pairs=self._train_pairs,
            test_pairs=self._test_pairs,
            template_backend=backend,
            use_base=self._config.get("use_base", True),
            template_root=self._venv_path / "baseline_template",
            venv_python=self._venv_path / "bin" / "python",
        )
        self._baseline_dir = exp.dir / "src"
        if not state.is_resume:
            clone_baseline_template(template_dir, self._baseline_dir)
            print(f"Baseline cloned at {self._baseline_dir}")
        if self._fast_eval:
            apply_fast_eval_patches(self._baseline_dir, limit=200, max_steps=10)
            print("MU_FAST_EVAL=1: applied infrastructure-smoke patches")
        self._description = (self._baseline_dir / "description.md").read_text()
        self._baseline_names = {p.name for p in self._baseline_dir.iterdir()
                                if p.name not in {"requirements.txt", "install.sh"}}
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

    def executor_workspace(
        self, *, files: dict[str, Path], memory_on: bool, prompt: Path | None = None
    ) -> Workspace:
        # MU uses a domain-specific executor prompt.
        return super().executor_workspace(
            files=files, memory_on=memory_on,
            prompt=prompt or (_TASK_DIR / "modelunlearning_executor_prompt.j2"),
        )

    def make_grader(self, exec_dir: Path) -> ModelUnlearningGrader:
        return ModelUnlearningGrader(exec_dir / ".grade.sock",
                                     baselines=self._train_baselines)

    def post_loop(self, exp: Any, settings: Any) -> None:
        # No meta-test: WMDP+MMLU evaluators already split forget/retain.
        pass
