"""Tests for heuresis.experiment module (Settings + loop helpers)."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from heuresis.experiment import (
    Settings,
    resume_or_new,
    iterate_until_valid,
    novelty_retry_loop,
    parse_gpus,
)
from heuresis.experiment_cli import (
    CuriosityPlusConfig,
    EXPERIMENT_REGISTRY,
    build_parser,
    parse_experiment,
)
from heuresis.models import RunResult
from heuresis.store import ResultStore


# --- Settings ---

def test_settings_defaults():
    s = Settings()
    assert s.agent == "opencode"
    assert s.num_iterations == 100
    assert s.count_valid is True
    assert s.max_parents == 5


def test_composed_parser_shared_settings():
    parsed = parse_experiment(
        "nanogpt",
        "linear",
        [
            "--num-iterations",
            "42",
            "--gpus",
            "0,1,4",
            "--num-ideators",
            "3",
            "--resume-exp-id",
            "old_exp_id",
        ],
        experiment_name="nanogpt-linear",
    )
    s = parsed.settings
    assert s.num_iterations == 42
    assert s.gpus == [0, 1, 4]
    assert s.num_ideators == 3
    assert s.resume_exp_id == "old_exp_id"


def test_composed_parser_defaults():
    s = parse_experiment(
        "nanogpt", "linear", [], experiment_name="nanogpt-linear", num_iterations=10, agent="claude"
    ).settings
    assert s.num_iterations == 10  # explicit wins over env
    assert s.agent == "claude"


def test_settings_count_total_flag():
    s = parse_experiment("nanogpt", "linear", ["--count-total"], experiment_name="nanogpt-linear").settings
    assert s.count_valid is False

    s = parse_experiment("nanogpt", "linear", ["--count-valid"], experiment_name="nanogpt-linear").settings
    assert s.count_valid is True


def test_settings_boolean_pairs():
    assert parse_experiment("nanogpt", "linear", ["--memory"], experiment_name="nanogpt-linear").settings.memory is True
    assert parse_experiment("nanogpt", "linear", ["--no-memory"], experiment_name="nanogpt-linear").settings.memory is False
    assert parse_experiment(
        "nanogpt", "linear", ["--disable-judge"], experiment_name="nanogpt-linear"
    ).settings.enable_judge is False


def test_composed_parser_parses_strategy_values():
    parsed = parse_experiment(
        "nanogpt",
        "curiosity_plus",
        [
            "--curiosity-score-weight",
            "0.25",
            "--curiosity-tag-novelty",
            "--curiosity-memory-k=7",
        ],
        experiment_name="nanogpt-curiosity-plus",
    )
    assert isinstance(parsed.strategy, CuriosityPlusConfig)
    assert parsed.strategy.score_weight == 0.25
    assert parsed.strategy.tag_novelty is True
    assert parsed.strategy.memory_k == 7


def test_composed_parser_rejects_unknown_flags():
    with pytest.raises(SystemExit):
        parse_experiment(
            "nanogpt",
            "curiosity",
            ["--curiosity-score-weght", "0.25"],
            experiment_name="nanogpt-curiosity",
        )


def test_registered_parsers_build_with_defaults():
    assert EXPERIMENT_REGISTRY
    for definition in EXPERIMENT_REGISTRY.values():
        build_parser(definition).parse_args([])


def test_discogen_onpolicyrl_registered_as_canonical_task():
    definition = EXPERIMENT_REGISTRY["discogen_onpolicyrl:linear"]

    parsed = parse_experiment(
        definition.task,
        definition.strategy,
        [],
        **definition.settings_defaults,
        task_defaults=definition.task_defaults,
        strategy_defaults=definition.strategy_defaults,
    )

    assert definition.task == "discogen_onpolicyrl"
    assert parsed.task.domain == "OnPolicyRL"
    assert parsed.settings.experiment_name == "discogen-onpolicyrl-linear"


def test_legacy_discogen_alias_still_registered_for_onpolicyrl():
    definition = EXPERIMENT_REGISTRY["discogen:linear"]

    assert definition.task == "discogen"
    assert definition.task_defaults["domain"] == "OnPolicyRL"
    assert definition.settings_defaults["experiment_name"] == "discogen-linear"


def test_strategy_flags_are_scoped():
    argv = [
        "--curiosity-score-weight",
        "0.25",
    ]
    with pytest.raises(SystemExit):
        parse_experiment("nanogpt", "linear", argv, experiment_name="nanogpt-linear")


def test_parse_gpus():
    assert parse_gpus("0,1,4") == [0, 1, 4]
    assert parse_gpus("") == []


# --- resume_or_new ---

def test_resume_or_new_creates_when_no_resume(tmp_path):
    store = ResultStore(db_path=tmp_path / "store.db")
    strategy = MagicMock()
    strategy.rebuild = MagicMock()
    settings = Settings(resume_exp_id=None)
    state = resume_or_new(store, "test", strategy, settings, root=tmp_path / "runs")
    assert state.is_resume is False
    assert state.next_iter_idx == 0
    assert state.valid_count == 0
    strategy.rebuild.assert_not_called()


def test_resume_or_new_loads_existing(tmp_path):
    store = ResultStore(db_path=tmp_path / "store.db")
    exp = store.experiment("test", root=tmp_path / "runs")
    result = RunResult(workspace=tmp_path, exit_code=0)
    exp.save("exec_000", result=result, iteration=0, run_type="executor",
             valid=True, metadata={"best_score": 0.5})
    exp.save("exec_001", result=result, iteration=1, run_type="executor",
             valid=False, metadata={})

    strategy = MagicMock()
    settings = Settings(resume_exp_id=exp.id)
    state = resume_or_new(store, "test", strategy, settings, root=tmp_path / "runs")
    assert state.is_resume is True
    assert state.next_iter_idx == 2
    assert state.valid_count == 1
    strategy.rebuild.assert_called_once()


# --- iterate_until_valid ---

def test_iterate_until_valid_count_valid_mode():
    settings = Settings(num_iterations=3, count_valid=True)
    class FakeState:
        next_iter_idx = 0
        valid_count = 0
    loop = iterate_until_valid(FakeState(), settings)
    n = 0
    for i in loop:
        n += 1
        loop.mark_done(valid=(n >= 2))  # first invalid, rest valid
        if n > 10:
            break  # safety
    assert n == 4  # 1 invalid + 3 valid = 4 iterations for target=3


def test_iterate_until_valid_iteration_mode():
    settings = Settings(num_iterations=3, count_valid=False)
    class FakeState:
        next_iter_idx = 0
        valid_count = 0
    loop = iterate_until_valid(FakeState(), settings)
    n = 0
    for i in loop:
        n += 1
        loop.mark_done(valid=False)
        if n > 10:
            break
    assert n == 3


def test_iterate_until_valid_resumes_at_nonzero():
    """ExperimentLoop must start at state.next_iter_idx and continue counting from valid_count."""
    settings = Settings(num_iterations=5, count_valid=True)
    class FakeState:
        next_iter_idx = 5
        valid_count = 3
    loop = iterate_until_valid(FakeState(), settings)
    first_idx = next(loop)
    assert first_idx == 5
    loop.mark_done(valid=True)  # valid_count = 4
    second_idx = next(loop)
    assert second_idx == 6
    loop.mark_done(valid=True)  # valid_count = 5 — should stop
    with pytest.raises(StopIteration):
        next(loop)


# --- novelty_retry_loop ---

def test_novelty_retry_accepts_first():
    ideas = iter(["idea_1"])
    class FakeReview:
        accepted = True
        assessment = MagicMock(novelty=3, explanation="novel")
    reviews = iter([FakeReview()])

    result = novelty_retry_loop(
        ideate=lambda fb: next(ideas),
        review=lambda i, a: next(reviews),
        max_rounds=3,
    )
    assert result is not None
    idea, rev = result
    assert idea == "idea_1"


def test_novelty_retry_accepts_third():
    ideas = iter(["idea_0", "idea_1", "idea_2"])
    reviews = iter([
        MagicMock(accepted=False, assessment=MagicMock(novelty=1, explanation="a")),
        MagicMock(accepted=False, assessment=MagicMock(novelty=1, explanation="b")),
        MagicMock(accepted=True, assessment=MagicMock(novelty=3, explanation="c")),
    ])
    result = novelty_retry_loop(
        ideate=lambda fb: next(ideas),
        review=lambda i, a: next(reviews),
        max_rounds=3,
    )
    assert result is not None
    idea, _ = result
    assert idea == "idea_2"


def test_novelty_retry_all_rejected():
    ideas = iter(["i0", "i1", "i2"])
    reviews = iter([
        MagicMock(accepted=False, assessment=MagicMock(novelty=1, explanation="x")),
        MagicMock(accepted=False, assessment=MagicMock(novelty=1, explanation="y")),
        MagicMock(accepted=False, assessment=MagicMock(novelty=1, explanation="z")),
    ])
    result = novelty_retry_loop(
        ideate=lambda fb: next(ideas),
        review=lambda i, a: next(reviews),
        max_rounds=3,
    )
    assert result is None


def test_novelty_retry_persists_each_attempt():
    ideas = iter(["i0", "i1"])
    reviews = iter([
        MagicMock(accepted=False, assessment=MagicMock(novelty=1, explanation="rej")),
        MagicMock(accepted=True, assessment=MagicMock(novelty=2, explanation="acc")),
    ])
    persisted: list = []
    novelty_retry_loop(
        ideate=lambda fb: next(ideas),
        review=lambda i, a: next(reviews),
        max_rounds=3,
        persist=lambda rev, attempt: persisted.append((rev.accepted, attempt)),
    )
    assert persisted == [(False, 0), (True, 1)]


# --- judge settings ---------------------------------------------------------

def test_settings_judge_defaults() -> None:
    from heuresis.experiment import Settings
    s = Settings()
    assert s.judge_timeout == 300
    assert s.judge_agent == "claude"
    assert s.judge_model == "claude-sonnet-4-6"


def test_settings_judge_cli_overrides() -> None:
    s = parse_experiment(
        "nanogpt",
        "linear",
        [
            "--judge-timeout",
            "450",
            "--judge-agent",
            "opencode",
            "--judge-model",
            "google/gemini-3.1-pro-preview",
        ],
        experiment_name="nanogpt-linear",
    ).settings
    assert s.judge_timeout == 450
    assert s.judge_agent == "opencode"
    assert s.judge_model == "google/gemini-3.1-pro-preview"


def test_settings_enable_judge_default() -> None:
    from heuresis.experiment import Settings
    s = Settings()
    assert s.enable_judge is True


def test_settings_enable_judge_cli_false() -> None:
    s = parse_experiment(
        "nanogpt", "linear", ["--disable-judge"], experiment_name="nanogpt-linear"
    ).settings
    assert s.enable_judge is False


def test_settings_enable_judge_cli_count_total_independent() -> None:
    s = parse_experiment(
        "nanogpt",
        "linear",
        ["--disable-judge", "--count-total"],
        experiment_name="nanogpt-linear",
    ).settings
    assert s.enable_judge is False
    assert s.count_valid is False


def test_settings_enable_judge_cli_on() -> None:
    s = parse_experiment(
        "nanogpt", "linear", ["--enable-judge"], experiment_name="nanogpt-linear"
    ).settings
    assert s.enable_judge is True
