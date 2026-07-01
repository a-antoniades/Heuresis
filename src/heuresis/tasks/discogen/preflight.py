"""DiscoGen task-specific preflight checks.

Validates discogen installation, config (domain, editable modules,
backend baselines), and GPU visibility BEFORE the first harness.run()
so we fail fast with clear messages instead of crashing inside bwrap
or after the venv build.

GPU visibility must be checked in the task venv (which has JAX+CUDA),
not the main project venv (which has CPU-only jax). Pass
``venv_python`` to route ``_check_gpus`` through a subprocess.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

_MODULE_FLAGS = [
    "change_loss", "change_optim", "change_networks",
    "change_train", "change_activation", "change_targets",
]


def check_discogen(
    gpus: list[int],
    config: dict[str, Any],
    *,
    venv_python: Path | None = None,
) -> list[str]:
    """Run the full preflight: config checks + GPU visibility check.

    Args:
        gpus: List of GPU ids the experiment will use.
        config: Loaded discogen YAML config (with ``_domain`` and
            ``template_backend`` populated).
        venv_python: Path to the task venv's python. When provided,
            GPU visibility is checked via subprocess in the task venv
            (required because JAX+CUDA lives there, not in main venv).
            When ``None``, the GPU check imports jax in the current
            interpreter (used by unit tests).

    Returns:
        List of error strings. Empty list means all checks passed.
    """
    errors = check_discogen_config(config)
    errors.extend(check_discogen_gpus(gpus, venv_python=venv_python))
    return errors


def check_discogen_config(config: dict[str, Any]) -> list[str]:
    """Config-only preflight: discogen import, domain, baselines.

    Safe to run in any python interpreter. Does not touch jax/GPUs.

    Does NOT check that all ``change_<module>`` flags are True -- that
    restriction only applies to strategies whose archive shape depends
    on the number of editable modules (MAP-Elites's feature grid).
    Strategies that need it should call ``check_all_modules_editable``
    explicitly from their run.py.
    """
    errors: list[str] = []
    import_errors = _check_discogen_importable()
    errors.extend(import_errors)
    if not import_errors:
        errors.extend(_check_domain_valid(config))
        errors.extend(_check_baselines(config))
    return errors


def check_discogen_gpus(
    gpus: list[int],
    *,
    venv_python: Path | None = None,
) -> list[str]:
    """GPU visibility preflight.

    When ``venv_python`` is provided, shells out to that interpreter to
    import jax and count devices. Required when the main project venv
    does not have JAX+CUDA installed.
    """
    return _check_gpus(gpus, venv_python=venv_python)


def _check_discogen_importable() -> list[str]:
    try:
        import discogen  # noqa: F401
        return []
    except ImportError:
        return ["discogen package not installed. Run: uv add discogen"]


def _check_domain_valid(config: dict[str, Any]) -> list[str]:
    domain = config.get("_domain", "")
    if not domain:
        return ["_domain not set in config"]
    from discogen import get_domains
    valid = get_domains()
    if domain not in valid:
        return [f"Unknown domain '{domain}'. Valid domains: {', '.join(sorted(valid))}"]
    return []


def check_all_modules_editable(config: dict[str, Any]) -> list[str]:
    """MAP-Elites-specific check: every ``change_<module>`` flag must be True.

    The MAP-Elites archive uses a fixed-dimension feature grid where each
    dimension indexes one editable module. Missing or explicitly False
    flags would change the grid shape mid-run, which the archive cannot
    currently handle. Linear / Islands / OMNI-EPIC strategies do not use
    this grid and should not call this check.
    """
    errors = []
    for flag in _MODULE_FLAGS:
        if not config.get(flag, False):
            errors.append(
                f"{flag} is False or missing in config. All modules must be editable "
                f"for MAP-Elites (limitation: numeric feature grid cannot resize dynamically)."
            )
    return errors


def _check_baselines(config: dict[str, Any]) -> list[str]:
    domain = config.get("_domain", "")
    if not domain:
        return []
    import discogen
    path = (
        Path(discogen.__file__).parent
        / "domains" / domain / "utils" / "baseline_scores.yaml"
    )
    if not path.exists():
        return [f"baseline_scores.yaml not found for domain {domain} at {path}"]
    return []


def _check_gpus(gpus: list[int], *, venv_python: Path | None = None) -> list[str]:
    if not gpus:
        return ["No GPUs specified"]
    if venv_python is None:
        return _check_gpus_inproc(gpus)
    return _check_gpus_subprocess(gpus, venv_python)


def _check_gpus_inproc(gpus: list[int]) -> list[str]:
    try:
        import jax
        devices = jax.devices("gpu")
        if not devices:
            return ["JAX found no GPU devices"]
        if max(gpus) >= len(devices):
            return [
                f"Requested GPU id {max(gpus)} but JAX only sees {len(devices)} "
                f"device(s). Set CUDA_VISIBLE_DEVICES or reduce requested GPU ids."
            ]
    except Exception as e:
        return [f"JAX GPU check failed: {e}"]
    return []


_GPU_PROBE = """
import json
try:
    import jax
    devices = jax.devices('gpu')
    print(json.dumps({'ok': True, 'count': len(devices)}))
except Exception as e:
    print(json.dumps({'ok': False, 'error': str(e)}))
"""


_TORCH_GPU_PROBE = """
import json
try:
    import torch
    if not torch.cuda.is_available():
        print(json.dumps({'ok': False, 'error': 'torch.cuda.is_available() is False'}))
    else:
        print(json.dumps({'ok': True, 'count': torch.cuda.device_count()}))
except Exception as e:
    print(json.dumps({'ok': False, 'error': str(e)}))
"""


def check_discogen_gpus_torch(
    gpus: list[int],
    *,
    venv_python: Path | None = None,
) -> list[str]:
    """PyTorch-based GPU visibility preflight (for non-JAX domains).

    OnPolicyRL/MARL/etc use JAX, but ModelUnlearning (and other LLM-based
    discogen domains) only need a working torch+CUDA. Probe via torch
    instead of JAX so we don't false-fail when the task venv lacks JAX.

    Otherwise mirrors :func:`check_discogen_gpus`.
    """
    if not gpus:
        return ["No GPUs specified"]
    if venv_python is None:
        try:
            import torch  # noqa: F401
            if not torch.cuda.is_available():
                return ["torch.cuda.is_available() returned False"]
            count = torch.cuda.device_count()
            if count == 0:
                return ["torch found no CUDA devices"]
            if max(gpus) >= count:
                return [
                    f"Requested GPU id {max(gpus)} but torch sees {count} device(s). "
                    f"Set CUDA_VISIBLE_DEVICES or reduce requested GPU ids."
                ]
            return []
        except Exception as e:
            return [f"torch GPU check failed: {e}"]
    if not venv_python.is_file():
        return [f"Task venv python not found at {venv_python}"]
    try:
        result = subprocess.run(
            [str(venv_python), "-c", _TORCH_GPU_PROBE],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return ["torch GPU probe timed out (120s) in task venv"]
    if result.returncode != 0:
        return [
            f"torch GPU probe exited {result.returncode}: "
            f"{(result.stderr.strip() or result.stdout.strip())[:300]}"
        ]
    payload = None
    for line in reversed(result.stdout.strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if payload is None:
        return [f"torch GPU probe produced no JSON: stdout={result.stdout[:300]!r}"]
    if not payload.get("ok"):
        return [f"torch GPU check failed: {payload.get('error', 'unknown')}"]
    count = int(payload.get("count", 0))
    if count == 0:
        return ["torch found no CUDA devices"]
    if max(gpus) >= count:
        return [
            f"Requested GPU id {max(gpus)} but torch sees {count} device(s). "
            f"Set CUDA_VISIBLE_DEVICES or reduce requested GPU ids."
        ]
    return []


def _check_gpus_subprocess(gpus: list[int], venv_python: Path) -> list[str]:
    if not venv_python.is_file():
        return [f"Task venv python not found at {venv_python}"]
    try:
        result = subprocess.run(
            [str(venv_python), "-c", _GPU_PROBE],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return ["JAX GPU probe timed out (120s) in task venv"]
    if result.returncode != 0:
        return [
            f"JAX GPU probe exited {result.returncode}: "
            f"{(result.stderr.strip() or result.stdout.strip())[:300]}"
        ]
    payload = None
    for line in reversed(result.stdout.strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if payload is None:
        return [f"JAX GPU probe produced no JSON: stdout={result.stdout[:300]!r}"]
    if not payload.get("ok"):
        return [f"JAX GPU check failed: {payload.get('error', 'unknown')}"]
    count = int(payload.get("count", 0))
    if count == 0:
        return ["JAX found no GPU devices"]
    if max(gpus) >= count:
        return [
            f"Requested GPU id {max(gpus)} but JAX only sees {count} "
            f"device(s). Set CUDA_VISIBLE_DEVICES or reduce requested GPU ids."
        ]
    return []
