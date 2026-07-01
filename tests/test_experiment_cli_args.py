import pytest

from heuresis.experiment_cli_args import build_env_args, build_experiment_args, load_config_args


def test_env_args_include_shared_and_strategy_flags() -> None:
    args = build_env_args(
        "nanogpt",
        "curiosity_plus",
        {
            "GPUS": "0,1",
            "MEMORY": "true",
            "CURIOSITY_SCORE_WEIGHT": "0.25",
            "CURIOSITY_TAG_NOVELTY": "false",
        },
    )

    assert args == [
        "--gpus",
        "0,1",
        "--curiosity-score-weight",
        "0.25",
        "--memory",
        "--no-curiosity-tag-novelty",
    ]


def test_env_args_filter_unsupported_strategy_flags() -> None:
    args = build_env_args(
        "nanogpt",
        "linear",
        {
            "GPUS": "0",
            "CURIOSITY_SCORE_WEIGHT": "0.25",
            "SKIP_META_TEST": "1",
        },
    )

    assert args == ["--gpus", "0"]


def test_env_args_include_task_flags_only_for_supported_tasks() -> None:
    onpolicyrl_args = build_env_args(
        "discogen_onpolicyrl",
        "linear",
        {"CONFIG": "config.yaml", "DOMAIN": "OnPolicyRL", "MU_FAST_EVAL": "1"},
    )
    discogen_args = build_env_args(
        "discogen_modelunlearning",
        "linear",
        {"CONFIG": "config.yaml", "DOMAIN": "ModelUnlearning", "MU_FAST_EVAL": "1"},
    )
    nanogpt_args = build_env_args(
        "nanogpt",
        "linear",
        {"CONFIG": "config.yaml", "DOMAIN": "ModelUnlearning", "MU_FAST_EVAL": "1"},
    )

    assert onpolicyrl_args == [
        "--config",
        "config.yaml",
        "--domain",
        "OnPolicyRL",
        "--mu-fast-eval",
    ]
    assert discogen_args == [
        "--config",
        "config.yaml",
        "--domain",
        "ModelUnlearning",
        "--mu-fast-eval",
    ]
    assert nanogpt_args == []


def test_explicit_args_override_env_defaults() -> None:
    args = build_experiment_args(
        "nanogpt",
        "curiosity_plus",
        ["--gpus", "2", "--memory", "--curiosity-score-weight", "0.5"],
        {
            "GPUS": "0,1",
            "MEMORY": "false",
            "CURIOSITY_SCORE_WEIGHT": "0.25",
            "N_SEED": "10",
        },
    )

    assert args == [
        "--n-seed",
        "10",
        "--gpus",
        "2",
        "--memory",
        "--curiosity-score-weight",
        "0.5",
    ]


def test_experiment_args_reject_unsupported_strategy_flags() -> None:
    with pytest.raises(SystemExit):
        build_experiment_args("nanogpt", "linear", ["--n-seed", "10"], {})


def test_load_config_args_supports_yaml_launch_config(tmp_path) -> None:
    config = tmp_path / "curiosity.yaml"
    config.write_text(
        "task: nanogpt\n"
        "strategy: curiosity\n"
        "settings:\n"
        "  gpus: [0, 1]\n"
        "  memory: false\n"
        "strategy_config:\n"
        "  n_seed: 12\n"
    )

    task, strategy, args = load_config_args(config)

    assert (task, strategy) == ("nanogpt", "curiosity")
    assert args == ["--gpus", "0,1", "--no-memory", "--n-seed", "12"]


def test_load_config_args_supports_discogen_onpolicyrl_launch_config() -> None:
    task, strategy, args = load_config_args(
        "configs/experiments/discogen_onpolicyrl/linear.yaml"
    )

    assert (task, strategy) == ("discogen_onpolicyrl", "linear")
    assert "--config" in args
    assert "configs/discogen/onpolicy_rl_breakout_all_editable.yaml" in args
    assert "--domain" in args
    assert "OnPolicyRL" in args


def test_yaml_config_args_can_be_overridden_by_explicit_args(tmp_path) -> None:
    config = tmp_path / "curiosity.yaml"
    config.write_text(
        "task: nanogpt\n"
        "strategy: curiosity\n"
        "settings:\n"
        "  gpus: [0, 1]\n"
        "strategy_config:\n"
        "  n_seed: 12\n"
    )
    task, strategy, config_args = load_config_args(config)

    args = build_experiment_args(
        task,
        strategy,
        ["--gpus", "2", "--n-seed", "20"],
        config_args=config_args,
        include_env=False,
    )

    assert args == ["--gpus", "2", "--n-seed", "20"]


def test_yaml_config_rejects_unknown_keys(tmp_path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text(
        "task: nanogpt\n"
        "strategy: linear\n"
        "settings:\n"
        "  does_not_exist: true\n"
    )

    with pytest.raises(ValueError, match="unknown config keys"):
        load_config_args(config)
