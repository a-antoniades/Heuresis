"""Central API-key loading from environment variables."""

from __future__ import annotations

import os
import re
from pathlib import Path

from heuresis.env import load_environment

_PROVIDERS: dict[str, tuple[str, ...]] = {
    "gemini": (
        "GEMINI_API_KEYS",
        "GEMINI_API_KEY",
        "GOOGLE_GENERATIVE_AI_API_KEY",
    ),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "cursor": ("CURSOR_API_KEY",),
    "huggingface": ("HF_TOKEN",),
}

def _split_key_list(raw: str) -> list[str]:
    """Split a multi-key env value on comma, newline, or colon."""
    parts = re.split(r"[,:\n]+", raw)
    return [p.strip() for p in parts if p.strip()]


def read_keys_file(path: Path) -> list[str]:
    """Read one API key per line from *path* (# comments and blanks skipped)."""
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ValueError(f"keys file {path} not readable") from exc
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]


def load_api_keys(provider: str) -> list[str]:
    """Return all configured keys for *provider* (may be empty).

    For ``gemini``, precedence is:

    1. ``GEMINI_API_KEYS`` (multi-key list)
    2. ``GEMINI_API_KEY``
    3. ``GOOGLE_GENERATIVE_AI_API_KEY``
    """
    load_environment()
    provider = provider.lower()
    if provider not in _PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}")

    if provider == "gemini":
        return _load_gemini_keys()

    for var in _PROVIDERS[provider]:
        val = os.environ.get(var)
        if val and val.strip():
            return [val.strip()]
    return []


def load_api_key(provider: str) -> str | None:
    """Return the first configured key for *provider*, or ``None``."""
    keys = load_api_keys(provider)
    return keys[0] if keys else None


def _load_gemini_keys() -> list[str]:
    multi = os.environ.get("GEMINI_API_KEYS")
    if multi and multi.strip():
        return _split_key_list(multi)

    for var in ("GEMINI_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY"):
        val = os.environ.get(var)
        if val and val.strip():
            return [val.strip()]

    return []
