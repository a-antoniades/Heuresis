"""The ``heuresis`` launcher: ``heuresis <task> <strategy> [flags]``.

It resolves the experiment arguments (environment-variable defaults from shell
wrappers, an optional launch-config YAML, and explicit flags — all validated by
the registered parser for the selected task and strategy in
:mod:`heuresis.experiment_cli`), then dispatches to that strategy's loop in
:data:`heuresis.loops.LOOPS` and runs the experiment end-to-end. No per-pairing
``experiments/<task>_<strategy>/run.py`` is needed.

Usage::

    heuresis nanogpt islands --gpus 0,1 --num-iterations 100
    heuresis --launch-config configs/experiments/nanogpt/islands.yaml

Pass ``--print-args`` to print the resolved arguments instead of running (the
mode shell wrappers historically used); ``--format {nul,lines}`` selects the
encoding for that output.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from heuresis.experiment_cli import ExperimentDefinition, build_parser, get_definition


TRUTHY = {"1", "true", "TRUE", "yes", "YES", "on", "ON"}


@dataclass(frozen=True)
class ValueEnv:
    name: str
    flag: str


@dataclass(frozen=True)
class BoolEnv:
    name: str
    enabled_flag: str
    disabled_flag: str


VALUE_ENVS: tuple[ValueEnv, ...] = (
    ValueEnv("EXP_NAME", "--experiment-name"),
    ValueEnv("RESUME_EXP_ID", "--resume-exp-id"),
    ValueEnv("AGENT", "--agent"),
    ValueEnv("MODEL", "--model"),
    ValueEnv("GPUS", "--gpus"),
    ValueEnv("NUM_ITERATIONS", "--num-iterations"),
    ValueEnv("NUM_IDEATORS", "--num-ideators"),
    ValueEnv("EXECUTOR_TIMEOUT", "--executor-timeout"),
    ValueEnv("IDEATOR_TIMEOUT", "--ideator-timeout"),
    ValueEnv("REVIEWER_TIMEOUT", "--reviewer-timeout"),
    ValueEnv("JUDGE_TIMEOUT", "--judge-timeout"),
    ValueEnv("JUDGE_AGENT", "--judge-agent"),
    ValueEnv("JUDGE_MODEL", "--judge-model"),
    ValueEnv("NOVELTY_THRESHOLD", "--novelty-threshold"),
    ValueEnv("NOVELTY_MAX_ROUNDS", "--novelty-max-rounds"),
    ValueEnv("MAX_PARENTS", "--max-parents"),
    ValueEnv("SESSION_RESET_EVERY", "--session-reset-every"),
    ValueEnv("N_SEED", "--n-seed"),
    ValueEnv("K_NEIGHBORS", "--k-neighbors"),
    ValueEnv("SOFTMAX_TAU", "--softmax-tau"),
    ValueEnv("ANCHOR_HISTORY", "--anchor-history"),
    ValueEnv("CANDIDATE_WINDOW", "--candidate-window"),
    ValueEnv("SEED_BATCH", "--seed-batch"),
    ValueEnv("CURIOSITY_NOVELTY_THRESHOLD", "--curiosity-novelty-threshold"),
    ValueEnv("PREDICTION_TIMEOUT", "--prediction-timeout"),
    ValueEnv("IDEATORS_PER_GPU", "--ideators-per-gpu"),
    ValueEnv("CURIOSITY_SCORE_WEIGHT", "--curiosity-score-weight"),
    ValueEnv("CURIOSITY_MEMORY_STRENGTH", "--curiosity-memory-strength"),
    ValueEnv("CURIOSITY_MEMORY_K", "--curiosity-memory-k"),
    ValueEnv("CURIOSITY_MEMORY_MIN_K", "--curiosity-memory-min-k"),
    ValueEnv("MIN_ARCHIVE_SIZE", "--min-archive-size"),
    ValueEnv("SEED_SOURCE", "--seed-source"),
    ValueEnv("SEED_COUNT", "--seed-count"),
    ValueEnv("CELL_EMPTY_WEIGHT", "--cell-empty-weight"),
    ValueEnv("CELL_CROSSOVER_RATE", "--cell-crossover-rate"),
    ValueEnv("GO_EXPLORE_ALPHA", "--go-explore-alpha"),
    ValueEnv("MIGRATION_INTERVAL", "--migration-interval"),
    ValueEnv("CROSSOVER_RATE", "--crossover-rate"),
    ValueEnv("CONFIG", "--config"),
    ValueEnv("DOMAIN", "--domain"),
)


BOOL_ENVS: tuple[BoolEnv, ...] = (
    BoolEnv("COUNT_VALID", "--count-valid", "--count-total"),
    BoolEnv("ENABLE_JUDGE", "--enable-judge", "--disable-judge"),
    BoolEnv("MEMORY", "--memory", "--no-memory"),
    BoolEnv("CURIOSITY_TAG_NOVELTY", "--curiosity-tag-novelty", "--no-curiosity-tag-novelty"),
    BoolEnv("SKIP_META_TEST", "--skip-meta-test", "--run-meta-test"),
    BoolEnv("MU_FAST_EVAL", "--mu-fast-eval", "--no-mu-fast-eval"),
)


def _option_strings(parser: argparse.ArgumentParser) -> set[str]:
    return {
        option
        for action in parser._actions
        for option in action.option_strings
    }


def _is_truthy(raw: str) -> bool:
    return raw in TRUTHY


def build_env_args(
    task: str,
    strategy: str,
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    """Return CLI args supported by the registered task x strategy parser."""
    env = os.environ if environ is None else environ
    definition = get_definition(f"{task}:{strategy}")
    options = _option_strings(build_parser(definition))
    args: list[str] = []

    for item in VALUE_ENVS:
        value = env.get(item.name)
        if value and item.flag in options:
            args.extend([item.flag, value])

    for item in BOOL_ENVS:
        value = env.get(item.name)
        if value is None or value == "":
            continue
        flag = item.enabled_flag if _is_truthy(value) else item.disabled_flag
        if flag in options:
            args.append(flag)

    return args


def _flag_name(arg: str) -> str:
    return arg.split("=", 1)[0]


def _looks_like_config_path(raw: str) -> bool:
    return raw.endswith((".yaml", ".yml")) or Path(raw).is_file()


def _explicit_value_flags(args: list[str]) -> set[str]:
    value_flags = {item.flag for item in VALUE_ENVS}
    return {_flag_name(arg) for arg in args if _flag_name(arg) in value_flags}


def _explicit_bool_flags(args: list[str]) -> set[str]:
    bool_flags = {
        flag
        for item in BOOL_ENVS
        for flag in (item.enabled_flag, item.disabled_flag)
    }
    return {_flag_name(arg) for arg in args if _flag_name(arg) in bool_flags}


def merge_args(default_args: list[str], explicit_args: list[str]) -> list[str]:
    """Merge env-derived defaults with explicit CLI args.

    Explicit args come last and override defaults. Boolean pairs need special
    handling because argparse rejects both sides of a mutually exclusive pair.
    """
    value_overrides = _explicit_value_flags(explicit_args)
    bool_overrides = _explicit_bool_flags(explicit_args)
    bool_groups = {
        item.enabled_flag: {item.enabled_flag, item.disabled_flag}
        for item in BOOL_ENVS
    } | {
        item.disabled_flag: {item.enabled_flag, item.disabled_flag}
        for item in BOOL_ENVS
    }

    merged: list[str] = []
    index = 0
    while index < len(default_args):
        arg = default_args[index]
        flag = _flag_name(arg)
        if flag in value_overrides:
            index += 2 if "=" not in arg else 1
            continue
        if flag in bool_groups and bool_groups[flag] & bool_overrides:
            index += 1
            continue
        merged.append(arg)
        if flag in {item.flag for item in VALUE_ENVS} and "=" not in arg and index + 1 < len(default_args):
            merged.append(default_args[index + 1])
            index += 2
        else:
            index += 1

    return merged + explicit_args


def _long_option(action: argparse.Action) -> str:
    for option in action.option_strings:
        if option.startswith("--"):
            return option
    return action.option_strings[0]


def _parser_actions(parser: argparse.ArgumentParser) -> list[argparse.Action]:
    return [
        action
        for action in parser._actions
        if action.dest != "help" and action.option_strings
    ]


def _value_to_arg(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def _flatten_config_mapping(raw: Mapping[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for section in ("settings", "task_config", "strategy_config"):
        values = raw.get(section)
        if values is None:
            continue
        if not isinstance(values, Mapping):
            raise ValueError(f"{section} must be a mapping")
        flat.update(values)

    strategy_section = raw.get("strategy")
    if isinstance(strategy_section, Mapping):
        flat.update(strategy_section)

    args = raw.get("args")
    if args is None:
        return flat
    if isinstance(args, Mapping):
        flat.update(args)
        return flat
    raise ValueError("args must be a mapping when present")


def _config_mapping_to_args(
    definition: ExperimentDefinition,
    raw: Mapping[str, Any],
) -> list[str]:
    parser = build_parser(definition)
    flat = _flatten_config_mapping(raw)
    actions = _parser_actions(parser)
    dests = {action.dest for action in actions}
    unknown = sorted(set(flat) - dests)
    if unknown:
        raise ValueError(f"unknown config keys for {definition.key}: {', '.join(unknown)}")

    bool_actions: dict[str, dict[bool, str]] = {}
    value_actions: dict[str, argparse.Action] = {}
    for action in actions:
        if isinstance(action, argparse._StoreTrueAction):
            bool_actions.setdefault(action.dest, {})[True] = _long_option(action)
        elif isinstance(action, argparse._StoreFalseAction):
            bool_actions.setdefault(action.dest, {})[False] = _long_option(action)
        elif action.dest not in value_actions:
            value_actions[action.dest] = action

    args: list[str] = []
    for key, value in flat.items():
        if value is None:
            continue
        if key in bool_actions:
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be a boolean")
            args.append(bool_actions[key][value])
            continue
        action = value_actions.get(key)
        if action is None:
            continue
        args.extend([_long_option(action), _value_to_arg(value)])
    return args


def load_config_args(path: str | Path) -> tuple[str, str, list[str]]:
    """Load a launch YAML and return ``(task, strategy, cli_args)``."""
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text()) or {}
    except OSError as exc:
        raise ValueError(f"config file {config_path} is not readable") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("launch config must be a mapping")

    task = raw.get("task")
    strategy = raw.get("strategy")
    if not isinstance(task, str) or not isinstance(strategy, str):
        raise ValueError("launch config must include string task and strategy")

    definition = get_definition(f"{task}:{strategy}")
    return task, strategy, _config_mapping_to_args(definition, raw)


def build_experiment_args(
    task: str,
    strategy: str,
    explicit_args: list[str] | None = None,
    environ: Mapping[str, str] | None = None,
    config_args: list[str] | None = None,
    include_env: bool = True,
) -> list[str]:
    """Return validated CLI args for *task* x *strategy*.

    Args from the environment are treated as defaults. ``explicit_args`` are
    parsed through the registered experiment parser and take precedence.
    """
    definition = get_definition(f"{task}:{strategy}")
    parser = build_parser(definition)
    args = config_args or []
    if include_env:
        args = merge_args(args, build_env_args(task, strategy, environ))
    args = merge_args(args, explicit_args or [])
    parser.parse_args(args)
    return args


def _extract_format(argv: list[str]) -> tuple[str, list[str]]:
    output_format = "nul"
    remaining: list[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--format":
            try:
                output_format = argv[index + 1]
            except IndexError:
                raise SystemExit("--format requires one of: nul, lines")
            index += 2
            continue
        if arg.startswith("--format="):
            output_format = arg.split("=", 1)[1]
            index += 1
            continue
        remaining.append(arg)
        index += 1

    if output_format not in {"nul", "lines"}:
        raise SystemExit("--format requires one of: nul, lines")
    return output_format, remaining


def _extract_launch_config(argv: list[str]) -> tuple[str | None, list[str]]:
    remaining: list[str] = []
    config_path: str | None = None
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--launch-config":
            try:
                config_path = argv[index + 1]
            except IndexError:
                raise SystemExit("--launch-config requires a YAML path")
            index += 2
            continue
        if arg.startswith("--launch-config="):
            config_path = arg.split("=", 1)[1]
            index += 1
            continue
        remaining.append(arg)
        index += 1
    return config_path, remaining


def _print_top_level_help() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_or_config")
    parser.add_argument("strategy", nargs="?")
    parser.add_argument("experiment_args", nargs=argparse.REMAINDER)
    parser.add_argument("--launch-config", help="YAML launch config path.")
    parser.add_argument(
        "--format",
        choices=("nul", "lines"),
        default="nul",
        help="Output format for generated arguments.",
    )
    parser.print_help()


def main(argv: list[str] | None = None) -> int:
    raw_args = sys.argv[1:] if argv is None else argv
    print_args = "--print-args" in raw_args
    if print_args:
        raw_args = [a for a in raw_args if a != "--print-args"]
    output_format, remaining = _extract_format(raw_args)
    config_path, remaining = _extract_launch_config(remaining)
    if not remaining and config_path is None:
        _print_top_level_help()
        return 0
    if remaining and remaining[0] in {"-h", "--help"}:
        _print_top_level_help()
        return 0

    config_args: list[str] = []
    include_env = True
    if config_path is not None:
        task, strategy, config_args = load_config_args(config_path)
        explicit_args = remaining
        include_env = False
    elif _looks_like_config_path(remaining[0]):
        task, strategy, config_args = load_config_args(remaining[0])
        explicit_args = remaining[1:]
        include_env = False
    elif len(remaining) >= 2:
        task, strategy, *explicit_args = remaining
    else:
        _print_top_level_help()
        return 0

    definition = get_definition(f"{task}:{strategy}")
    if any(arg in {"-h", "--help"} for arg in explicit_args):
        build_parser(definition).print_help()
        return 0

    args = build_experiment_args(
        task,
        strategy,
        explicit_args,
        config_args=config_args,
        include_env=include_env,
    )

    if print_args:
        if output_format == "lines":
            sys.stdout.write("\n".join(args))
            if args:
                sys.stdout.write("\n")
        else:
            sys.stdout.buffer.write(b"".join(arg.encode() + b"\0" for arg in args))
        return 0

    # Default: dispatch to the strategy loop and run the experiment end-to-end.
    from heuresis.loops import LOOPS

    loop = LOOPS.get(strategy)
    if loop is None:
        raise SystemExit(
            f"no loop registered for strategy {strategy!r}; "
            f"known strategies: {', '.join(sorted(LOOPS))}"
        )
    loop(task, argv=args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
