"""Environment loading for local credentials and runtime controls."""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

_LOADED = False


def load_environment(
    dotenv_path: str | Path | None = None,
    *,
    force: bool = False,
) -> Path | None:
    """Load a local ``.env`` file without overriding existing environment vars."""
    global _LOADED

    if _LOADED and dotenv_path is None and not force:
        _derive_single_gemini_keys()
        return None

    raw_path = str(dotenv_path) if dotenv_path is not None else find_dotenv(usecwd=True)
    if raw_path:
        load_dotenv(raw_path, override=False)
        loaded_path = Path(raw_path)
    else:
        loaded_path = None

    _LOADED = True
    _derive_single_gemini_keys()
    return loaded_path


def _derive_single_gemini_keys() -> None:
    """Populate single-key Gemini vars for tools that do not read lists."""
    multi = os.environ.get("GEMINI_API_KEYS")
    if not multi or not multi.strip():
        return

    first = re.split(r"[,\n:]+", multi.strip())[0].strip()
    if not first:
        return

    os.environ.setdefault("GEMINI_API_KEY", first)
    os.environ.setdefault("GOOGLE_GENERATIVE_AI_API_KEY", first)
