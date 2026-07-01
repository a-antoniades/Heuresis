"""BBOB task preflight. Task runs on CPU only — no GPU checks."""
from __future__ import annotations

import platform


def check_bbob() -> list[str]:
    """Return a list of error strings. Empty list = OK."""
    errors: list[str] = []
    if platform.system() != "Linux":
        errors.append(
            f"BBOB task requires Linux (uses signal.alarm for wallclock); "
            f"got {platform.system()}"
        )
    try:
        import numpy  # noqa: F401
    except ImportError:
        errors.append("numpy not importable in the harness venv")
    return errors
