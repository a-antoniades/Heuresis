"""Strict CLI parsing for task x strategy experiment entry points."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from heuresis.experiment import Settings, add_settings_args


@dataclass(frozen=True)
class EmptyConfig:
    pass


@dataclass(frozen=True)
class CuriosityConfig:
    n_seed: int = 10
    k_neighbors: int = 10
    softmax_tau: float = 1.0
    anchor_history: int = 5
    candidate_window: int = 20
    seed_batch: int = 5
    novelty_threshold: float = 0.90
    prediction_timeout: int = 60
    ideators_per_gpu: int = 1


@dataclass(frozen=True)
class CuriosityPlusConfig(CuriosityConfig):
    score_weight: float = 0.0
    tag_novelty: bool = False
    memory_strength: float = 0.0
    memory_k: int = 10
    memory_min_k: int = 3


@dataclass(frozen=True)
class OmniEpicConfig:
    min_archive_size: int = 5
    seed_source: str = ""
    seed_count: int = 10
    skip_meta_test: bool = False
    ideators_per_gpu: int = 1


@dataclass(frozen=True)
class CellConfig:
    empty_weight: float = 3.0
    crossover_rate: float = 0.5
    go_explore_alpha: float = 0.01


@dataclass(frozen=True)
class IslandConfig:
    migration_interval: int = 24
    crossover_rate: float = 0.4


@dataclass(frozen=True)
class NanoGPTTaskConfig:
    pass


@dataclass(frozen=True)
class BBOBTaskConfig:
    pass


@dataclass(frozen=True)
class DiscoGenTaskConfig:
    config: Path
    domain: str | None = None
    mu_fast_eval: bool = False


@dataclass(frozen=True)
class ExperimentDefinition:
    task: str
    strategy: str
    settings_defaults: dict[str, Any]
    task_defaults: dict[str, Any] | None = None
    strategy_defaults: dict[str, Any] | None = None
    task_config: type = EmptyConfig
    strategy_config: type = EmptyConfig

    @property
    def key(self) -> str:
        return f"{self.task}:{self.strategy}"


@dataclass(frozen=True)
class ParsedExperiment:
    definition: ExperimentDefinition
    settings: Settings
    task: Any
    strategy: Any


def _add_bool_pair(
    parser: argparse.ArgumentParser,
    *,
    dest: str,
    enabled: str,
    disabled: str,
    default: bool,
) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(enabled, dest=dest, action="store_true")
    group.add_argument(disabled, dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def _namespace_values(ns: argparse.Namespace, keys: tuple[str, ...]) -> dict[str, Any]:
    data = vars(ns)
    return {key: data[key] for key in keys}


def _add_task_args(parser: argparse.ArgumentParser, cfg_type: type, defaults: dict[str, Any]) -> None:
    if cfg_type is DiscoGenTaskConfig:
        parser.add_argument("--config", default=defaults.get("config", ""))
        parser.add_argument("--domain", default=defaults.get("domain"))
        _add_bool_pair(
            parser,
            dest="mu_fast_eval",
            enabled="--mu-fast-eval",
            disabled="--no-mu-fast-eval",
            default=defaults.get("mu_fast_eval", False),
        )


def _task_from_args(cfg_type: type, ns: argparse.Namespace) -> Any:
    if cfg_type is DiscoGenTaskConfig:
        raw_config = ns.config
        config = Path(raw_config).resolve() if raw_config else Path()
        return DiscoGenTaskConfig(
            config=config,
            domain=ns.domain or None,
            mu_fast_eval=ns.mu_fast_eval,
        )
    return cfg_type()


def _add_strategy_args(
    parser: argparse.ArgumentParser,
    cfg_type: type,
    defaults: dict[str, Any],
) -> None:
    if issubclass(cfg_type, CuriosityConfig):
        base = CuriosityConfig(
            **{k: v for k, v in defaults.items() if k in CuriosityConfig.__annotations__}
        )
        parser.add_argument("--n-seed", type=int, default=base.n_seed)
        parser.add_argument("--k-neighbors", type=int, default=base.k_neighbors)
        parser.add_argument("--softmax-tau", type=float, default=base.softmax_tau)
        parser.add_argument("--anchor-history", type=int, default=base.anchor_history)
        parser.add_argument("--candidate-window", type=int, default=base.candidate_window)
        parser.add_argument("--seed-batch", type=int, default=base.seed_batch)
        parser.add_argument(
            "--curiosity-novelty-threshold",
            dest="curiosity_novelty_threshold",
            type=float,
            default=base.novelty_threshold,
        )
        parser.add_argument("--prediction-timeout", type=int, default=base.prediction_timeout)
        parser.add_argument("--ideators-per-gpu", type=int, default=base.ideators_per_gpu)

    if issubclass(cfg_type, CuriosityPlusConfig):
        parser.add_argument(
            "--curiosity-score-weight",
            type=float,
            default=defaults.get("score_weight", 0.0),
        )
        _add_bool_pair(
            parser,
            dest="curiosity_tag_novelty",
            enabled="--curiosity-tag-novelty",
            disabled="--no-curiosity-tag-novelty",
            default=defaults.get("tag_novelty", False),
        )
        parser.add_argument(
            "--curiosity-memory-strength",
            type=float,
            default=defaults.get("memory_strength", 0.0),
        )
        parser.add_argument("--curiosity-memory-k", type=int, default=defaults.get("memory_k", 10))
        parser.add_argument(
            "--curiosity-memory-min-k",
            type=int,
            default=defaults.get("memory_min_k", 3),
        )

    if cfg_type is OmniEpicConfig:
        base = OmniEpicConfig(**defaults)
        parser.add_argument("--min-archive-size", type=int, default=base.min_archive_size)
        parser.add_argument("--seed-source", default=base.seed_source)
        parser.add_argument("--seed-count", type=int, default=base.seed_count)
        _add_bool_pair(
            parser,
            dest="skip_meta_test",
            enabled="--skip-meta-test",
            disabled="--run-meta-test",
            default=base.skip_meta_test,
        )
        parser.add_argument("--ideators-per-gpu", type=int, default=base.ideators_per_gpu)

    if cfg_type is CellConfig:
        base = CellConfig(**defaults)
        parser.add_argument("--cell-empty-weight", type=float, default=base.empty_weight)
        parser.add_argument("--cell-crossover-rate", type=float, default=base.crossover_rate)
        parser.add_argument("--go-explore-alpha", type=float, default=base.go_explore_alpha)

    if cfg_type is IslandConfig:
        base = IslandConfig(**defaults)
        parser.add_argument("--migration-interval", type=int, default=base.migration_interval)
        parser.add_argument("--crossover-rate", type=float, default=base.crossover_rate)


def _strategy_from_args(cfg_type: type, ns: argparse.Namespace) -> Any:
    if cfg_type is CuriosityConfig:
        return CuriosityConfig(
            n_seed=ns.n_seed,
            k_neighbors=ns.k_neighbors,
            softmax_tau=ns.softmax_tau,
            anchor_history=ns.anchor_history,
            candidate_window=ns.candidate_window,
            seed_batch=ns.seed_batch,
            novelty_threshold=ns.curiosity_novelty_threshold,
            prediction_timeout=ns.prediction_timeout,
            ideators_per_gpu=ns.ideators_per_gpu,
        )
    if cfg_type is CuriosityPlusConfig:
        return CuriosityPlusConfig(
            n_seed=ns.n_seed,
            k_neighbors=ns.k_neighbors,
            softmax_tau=ns.softmax_tau,
            anchor_history=ns.anchor_history,
            candidate_window=ns.candidate_window,
            seed_batch=ns.seed_batch,
            novelty_threshold=ns.curiosity_novelty_threshold,
            prediction_timeout=ns.prediction_timeout,
            ideators_per_gpu=ns.ideators_per_gpu,
            score_weight=ns.curiosity_score_weight,
            tag_novelty=ns.curiosity_tag_novelty,
            memory_strength=ns.curiosity_memory_strength,
            memory_k=ns.curiosity_memory_k,
            memory_min_k=ns.curiosity_memory_min_k,
        )
    if cfg_type is OmniEpicConfig:
        return OmniEpicConfig(
            min_archive_size=ns.min_archive_size,
            seed_source=ns.seed_source,
            seed_count=ns.seed_count,
            skip_meta_test=ns.skip_meta_test,
            ideators_per_gpu=ns.ideators_per_gpu,
        )
    if cfg_type is CellConfig:
        return CellConfig(
            empty_weight=ns.cell_empty_weight,
            crossover_rate=ns.cell_crossover_rate,
            go_explore_alpha=ns.go_explore_alpha,
        )
    if cfg_type is IslandConfig:
        return IslandConfig(
            migration_interval=ns.migration_interval,
            crossover_rate=ns.crossover_rate,
        )
    return cfg_type()


_SETTINGS_KEYS = tuple(Settings.__dataclass_fields__.keys())


def build_parser(definition: ExperimentDefinition) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Run {definition.task} / {definition.strategy} experiment."
    )
    add_settings_args(parser, definition.settings_defaults)
    _add_task_args(parser, definition.task_config, definition.task_defaults or {})
    _add_strategy_args(parser, definition.strategy_config, definition.strategy_defaults or {})
    return parser


def parse_experiment(
    definition: ExperimentDefinition | str,
    strategy: str | None = None,
    argv: list[str] | None = None,
    task_defaults: dict[str, Any] | None = None,
    strategy_defaults: dict[str, Any] | None = None,
    **settings_defaults: Any,
) -> ParsedExperiment:
    if isinstance(definition, str):
        if strategy is None:
            definition = get_definition(definition)
        else:
            # Start from the registered definition (so its gpus/iterations/
            # config/domain/... defaults apply) and overlay any explicit
            # overrides passed here. Falls back to a bare definition for
            # unregistered task x strategy pairs.
            base = EXPERIMENT_REGISTRY.get(f"{definition}:{strategy}")
            if base is not None:
                settings_defaults = {**base.settings_defaults, **settings_defaults}
                task_defaults = {**(base.task_defaults or {}), **(task_defaults or {})} or None
                strategy_defaults = (
                    {**(base.strategy_defaults or {}), **(strategy_defaults or {})} or None
                )
            definition = make_definition(
                definition,
                strategy,
                task_defaults=task_defaults,
                strategy_defaults=strategy_defaults,
                **settings_defaults,
            )
    parser = build_parser(definition)
    ns = parser.parse_args(argv)
    settings = Settings(**_namespace_values(ns, _SETTINGS_KEYS))
    task = _task_from_args(definition.task_config, ns)
    strategy_cfg = _strategy_from_args(definition.strategy_config, ns)
    return ParsedExperiment(definition, settings, task, strategy_cfg)


def _task_config_for(task: str) -> type:
    if task in {"discogen_onpolicyrl", "discogen", "discogen_modelunlearning"}:
        return DiscoGenTaskConfig
    if task == "nanogpt":
        return NanoGPTTaskConfig
    if task == "bbob":
        return BBOBTaskConfig
    return EmptyConfig


def _strategy_config_for(strategy: str) -> type:
    if strategy == "curiosity":
        return CuriosityConfig
    if strategy == "curiosity_plus":
        return CuriosityPlusConfig
    if strategy == "omni_epic":
        return OmniEpicConfig
    if strategy in {"map_elites", "go_explore"}:
        return CellConfig
    if strategy == "islands":
        return IslandConfig
    return EmptyConfig


def make_definition(
    task: str,
    strategy: str,
    *,
    task_defaults: dict[str, Any] | None = None,
    strategy_defaults: dict[str, Any] | None = None,
    **settings_defaults: Any,
) -> ExperimentDefinition:
    return ExperimentDefinition(
        task=task,
        strategy=strategy,
        settings_defaults=settings_defaults,
        task_defaults=task_defaults,
        strategy_defaults=strategy_defaults,
        task_config=_task_config_for(task),
        strategy_config=_strategy_config_for(strategy),
    )


EXPERIMENT_REGISTRY: dict[str, ExperimentDefinition] = {}


def register_definition(definition: ExperimentDefinition) -> ExperimentDefinition:
    EXPERIMENT_REGISTRY[definition.key] = definition
    return definition


def get_definition(key: str) -> ExperimentDefinition:
    try:
        return EXPERIMENT_REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"Unknown experiment definition: {key}") from exc


def _register_matrix() -> None:
    common_gpu: dict[str, Any] = {
        "gpus": list(range(8)),
        "num_iterations": 100,
        "num_ideators": 8,
    }
    nanogpt_common: dict[str, Any] = {
        **common_gpu,
        "agent": "opencode",
        "model": "google/gemini-3.1-pro-preview",
        "judge_agent": "claude",
        "judge_model": "claude-sonnet-4-6",
        "enable_judge": True,
        "count_valid": True,
        "executor_timeout": 2100,
        "ideator_timeout": 600,
        "judge_timeout": 300,
        "reviewer_timeout": 300,
    }
    discogen_common: dict[str, Any] = {
        **common_gpu,
        "agent": "opencode",
        "model": "google/gemini-3.1-pro-preview",
        "judge_agent": "claude",
        "judge_model": "claude-sonnet-4-6",
        "enable_judge": True,
        "count_valid": True,
        "max_parents": 5,
        "session_reset_every": 10,
        "executor_timeout": 1200,
        "ideator_timeout": 600,
        "judge_timeout": 300,
        "reviewer_timeout": 300,
        "memory": True,
    }
    bbob_common: dict[str, Any] = {
        "agent": "opencode",
        "model": "google/gemini-3.1-pro-preview",
        "count_valid": True,
        "executor_timeout": 600,
        "ideator_timeout": 300,
        "num_ideators": 4,
    }

    def merged(base: dict[str, Any], **overrides: Any) -> dict[str, Any]:
        return {**base, **overrides}

    definitions = [
        make_definition("nanogpt", "linear", **merged(nanogpt_common, experiment_name="nanogpt-linear", memory=True, max_parents=5, session_reset_every=10)),
        make_definition("nanogpt", "map_elites", **merged(nanogpt_common, experiment_name="nanogpt-map-elites", memory=True)),
        make_definition("nanogpt", "go_explore", **merged(nanogpt_common, experiment_name="nanogpt-go-explore", memory=True)),
        make_definition("nanogpt", "islands", **merged(nanogpt_common, experiment_name="nanogpt-islands-judge", memory=False)),
        make_definition("nanogpt", "curiosity", **merged(nanogpt_common, experiment_name="nanogpt-curiosity", ideator_timeout=240, memory=False)),
        make_definition("nanogpt", "curiosity_plus", **merged(nanogpt_common, experiment_name="nanogpt-curiosity-plus", ideator_timeout=240, memory=False)),
        make_definition(
            "nanogpt",
            "omni_epic",
            strategy_defaults={"seed_source": "2026-04-14_104048_nanogpt-linear"},
            **merged(nanogpt_common, experiment_name="nanogpt-omni-epic", num_iterations=400, memory=True),
        ),
        make_definition("bbob", "linear", **merged(bbob_common, experiment_name="bbob-linear", num_iterations=10, num_ideators=1)),
        make_definition("bbob", "islands", **merged(bbob_common, experiment_name="bbob-islands", num_iterations=20, memory=False)),
        make_definition("bbob", "curiosity", strategy_defaults={"n_seed": 6, "prediction_timeout": 180}, **merged(bbob_common, experiment_name="bbob-curiosity", num_iterations=20, memory=False)),
        make_definition("bbob", "curiosity_plus", strategy_defaults={"n_seed": 6, "prediction_timeout": 180}, **merged(bbob_common, experiment_name="bbob-curiosity-plus", num_iterations=20, memory=False)),
        make_definition("bbob", "omni_epic", **merged(bbob_common, experiment_name="bbob-omni-epic", num_iterations=40)),
    ]

    onpolicyrl_strategies = [
        "linear",
        "map_elites",
        "go_explore",
        "islands",
        "curiosity",
        "curiosity_plus",
        "omni_epic",
    ]
    for task_name, name_prefix in [
        ("discogen_onpolicyrl", "discogen-onpolicyrl"),
        ("discogen", "discogen"),
    ]:
        for strategy in onpolicyrl_strategies:
            settings: dict[str, Any] = dict(discogen_common)
            strategy_defaults: dict[str, Any] | None = None
            if strategy == "go_explore":
                strategy_defaults = {"go_explore_alpha": 0.004}
            if strategy in {"curiosity", "curiosity_plus", "omni_epic"}:
                settings["gpus"] = [4, 5, 6, 7]
                settings["num_ideators"] = 4
            definitions.append(
                make_definition(
                    task_name,
                    strategy,
                    experiment_name=f"{name_prefix}-{strategy}",
                    strategy_defaults=strategy_defaults,
                    task_defaults={
                        "config": "configs/discogen/onpolicy_rl_breakout_all_editable.yaml",
                        "domain": "OnPolicyRL",
                    },
                    **settings,
                )
            )

    for strategy in [
        "linear",
        "map_elites",
        "go_explore",
        "islands",
        "curiosity",
        "omni_epic",
    ]:
        settings: dict[str, Any] = dict(discogen_common)
        settings["num_iterations"] = 100 if strategy == "curiosity" else 300
        strategy_defaults: dict[str, Any] | None = None
        if strategy == "go_explore":
            strategy_defaults = {"go_explore_alpha": 0.001}  # ModelUnlearning-tuned
        if strategy == "omni_epic":
            settings["gpus"] = [4, 5, 6, 7]
            settings["num_ideators"] = 4
        definitions.append(
            make_definition(
                "discogen_modelunlearning",
                strategy,
                experiment_name=f"discogen-modelunlearning-{strategy}",
                strategy_defaults=strategy_defaults,
                task_defaults={
                    "config": "configs/discogen/modelunlearning_wmdp_cyber.yaml",
                    "domain": "ModelUnlearning",
                },
                **settings,
            )
        )

    for definition in definitions:
        register_definition(definition)


_register_matrix()
