"""Tests for discogen preflight checks."""
from unittest.mock import patch

import pytest

from heuresis.tasks.discogen.preflight import (
    check_all_modules_editable,
    check_discogen,
)


_MOCK_TARGETS = [
    "heuresis.tasks.discogen.preflight._check_discogen_importable",
    "heuresis.tasks.discogen.preflight._check_domain_valid",
    "heuresis.tasks.discogen.preflight._check_baselines",
    "heuresis.tasks.discogen.preflight._check_gpus",
]


def _all_ok():
    """Return a context manager stack that patches every check to return []."""
    return [patch(t, return_value=[]) for t in _MOCK_TARGETS]


def test_check_discogen_missing_discogen():
    """Should error if discogen is not importable."""
    patches = _all_ok()
    patches[0] = patch(
        "heuresis.tasks.discogen.preflight._check_discogen_importable",
        return_value=["discogen package not installed. Run: uv add discogen"],
    )
    with patches[0], patches[1], patches[2], patches[3]:
        errors = check_discogen(gpus=[0], config={})
    assert any("discogen" in e.lower() for e in errors)


def test_check_all_modules_editable_errors_on_false_flag():
    """MAP-Elites-specific: any change_* flag set to False (or missing) yields an error."""
    config = {
        "_domain": "OnPolicyRL",
        "template_backend": "default",
        "change_loss": True,
        "change_optim": True,
        "change_networks": False,
        "change_train": True,
        "change_activation": True,
        "change_targets": True,
    }
    errors = check_all_modules_editable(config)
    assert any("change_networks" in e for e in errors)


def test_check_all_modules_editable_errors_on_missing_flag():
    """MAP-Elites-specific: missing change_* flag is also an error."""
    config = {
        "_domain": "OnPolicyRL",
        "change_loss": True,  # only loss declared; others missing
    }
    errors = check_all_modules_editable(config)
    missing = {"change_optim", "change_networks", "change_train",
               "change_activation", "change_targets"}
    assert missing.issubset({e.split()[0] for e in errors})


def test_check_all_modules_editable_all_true_ok():
    """All six change_* True -> no errors."""
    config = {f"change_{m}": True for m in
              ("loss", "optim", "networks", "train", "activation", "targets")}
    assert check_all_modules_editable(config) == []


def test_check_discogen_does_not_require_all_editable():
    """check_discogen (linear/islands path) should not enforce modules-editable.

    Regression test: previously check_discogen_config included
    _check_all_modules_editable, which blocked configs intended for
    non-MAP-Elites strategies (e.g. loss-only editable for a linear run).
    """
    config = {
        "_domain": "OnPolicyRL",
        "template_backend": "default",
        "change_loss": True,  # everything else missing
    }
    patches = _all_ok()
    with patches[0], patches[1], patches[2], patches[3]:
        errors = check_discogen(gpus=[0], config=config)
    assert errors == []


def test_check_discogen_all_ok():
    """No errors when every check passes."""
    config = {
        "_domain": "OnPolicyRL",
        "template_backend": "default",
        "change_loss": True, "change_optim": True, "change_networks": True,
        "change_train": True, "change_activation": True, "change_targets": True,
    }
    patches = _all_ok()
    with patches[0], patches[1], patches[2], patches[3]:
        errors = check_discogen(gpus=[0], config=config)
    assert errors == []


def test_check_discogen_no_gpus():
    """Empty gpus list triggers GPU error."""
    patches = _all_ok()
    # Un-mock _check_gpus to let the real function run
    patches[3] = patch(
        "heuresis.tasks.discogen.preflight._check_gpus",
        wraps=__import__(
            "heuresis.tasks.discogen.preflight",
            fromlist=["_check_gpus"],
        )._check_gpus,
    )
    config = {
        "_domain": "OnPolicyRL",
        "change_loss": True, "change_optim": True, "change_networks": True,
        "change_train": True, "change_activation": True, "change_targets": True,
    }
    with patches[0], patches[1], patches[2], patches[3]:
        errors = check_discogen(gpus=[], config=config)
    assert any("gpu" in e.lower() for e in errors)


def test_check_discogen_missing_domain():
    """Missing _domain in config triggers validation error."""
    patches = _all_ok()
    # Un-mock _check_domain_valid to run real logic
    patches[1] = patch(
        "heuresis.tasks.discogen.preflight._check_domain_valid",
        wraps=__import__(
            "heuresis.tasks.discogen.preflight",
            fromlist=["_check_domain_valid"],
        )._check_domain_valid,
    )
    with patches[0], patches[1], patches[2], patches[3]:
        errors = check_discogen(gpus=[0], config={})
    assert any("_domain" in e for e in errors)


def test_check_discogen_invalid_domain():
    """Unknown domain in config triggers validation error."""
    patches = _all_ok()
    # Un-mock _check_domain_valid to run real logic
    patches[1] = patch(
        "heuresis.tasks.discogen.preflight._check_domain_valid",
        wraps=__import__(
            "heuresis.tasks.discogen.preflight",
            fromlist=["_check_domain_valid"],
        )._check_domain_valid,
    )
    with patches[0], patches[1], patches[2], patches[3], \
         patch("discogen.get_domains", return_value=["OnPolicyRL", "OffPolicyRL"]):
        errors = check_discogen(
            gpus=[0],
            config={
                "_domain": "NotARealDomain",
                "change_loss": True, "change_optim": True, "change_networks": True,
                "change_train": True, "change_activation": True, "change_targets": True,
            },
        )
    assert any("Unknown domain" in e and "NotARealDomain" in e for e in errors)


def test_check_discogen_gpu_id_out_of_range():
    """Requesting GPU id higher than available GPUs triggers error."""
    pytest.importorskip("jax")
    patches = _all_ok()
    # Un-mock _check_gpus
    patches[3] = patch(
        "heuresis.tasks.discogen.preflight._check_gpus",
        wraps=__import__(
            "heuresis.tasks.discogen.preflight",
            fromlist=["_check_gpus"],
        )._check_gpus,
    )
    # Also mock jax.devices to return only 2 devices
    with patches[0], patches[1], patches[2], patches[3], \
         patch("jax.devices", return_value=[object(), object()]):
        errors = check_discogen(
            gpus=[0, 1, 7],  # 7 is out of range for 2 devices
            config={
                "_domain": "OnPolicyRL",
                "change_loss": True, "change_optim": True, "change_networks": True,
                "change_train": True, "change_activation": True, "change_targets": True,
            },
        )
    assert any("GPU id" in e or "device" in e.lower() for e in errors)


# --- task_config.yaml verification tests ----------------------------------


def _grep_matches(pattern: str, text: str) -> bool:
    """Mirror the production consumer: agent.log is grep'd with `grep -nE`.

    Python's `re` module does NOT support POSIX character classes (`[[:space:]]`)
    — so we validate via the actual tool that runs at runtime.
    """
    import subprocess
    result = subprocess.run(
        ["grep", "-nE", pattern],
        input=text,
        capture_output=True,
        text=True,
    )
    # grep returns 0 on match, 1 on no match, 2 on error
    return result.returncode == 0


def test_discogen_task_config_loads():
    """Discogen task_config.yaml has editable=discovered + verify.{diff_scope, invariants, fabrication_patterns}."""
    import yaml
    from heuresis.tasks import task_dir

    cfg_path = task_dir("discogen") / "task_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())

    assert cfg.get("editable") == "discovered"
    verify = cfg.get("verify") or {}
    assert verify.get("diff_scope") == "."
    assert verify.get("invariants"), "verify.invariants must be non-empty"
    pat = verify.get("fabrication_patterns")
    assert pat, "verify.fabrication_patterns must be non-empty"
    # Verify the regex is valid POSIX ERE by running grep against a sentinel.
    # If the pattern is malformed, grep exits with code 2.
    import subprocess
    result = subprocess.run(
        ["grep", "-nE", pat], input="sentinel\n",
        capture_output=True, text=True,
    )
    assert result.returncode in (0, 1), (
        f"fabrication_patterns is not a valid POSIX ERE regex; "
        f"grep stderr: {result.stderr}"
    )


def test_discogen_fabrication_pattern_matches_known_hacks():
    """Strengthened regex catches all known write-to-run.log mechanisms; not the canonical pipeline.

    Validates via `grep -nE` (the production consumer) to test POSIX ERE semantics.
    """
    import yaml
    from heuresis.tasks import task_dir

    cfg = yaml.safe_load((task_dir("discogen") / "task_config.yaml").read_text())
    pattern = cfg["verify"]["fabrication_patterns"]

    hostile = [
        "echo '{}' > run.log",
        "printf '%s\\n' '{}' > run.log",
        "cat <<EOF > run.log\n{}\nEOF",
        "tee run.log < something",
        "open('run.log', 'w')",
        "with open('run.log', 'w') as f:",
        'Write("/workspace/run/run.log", "fake")',
        'Edit("/workspace/exec_001/run.log", ...)',
    ]
    for s in hostile:
        assert _grep_matches(pattern, s), f"regex should match hostile line: {s!r}"

    legit = [
        "python run_main.py > run.log 2>&1",
        "uv run python run_main.py > run.log",
    ]
    for s in legit:
        assert not _grep_matches(pattern, s), f"regex should NOT match legit line: {s!r}"
