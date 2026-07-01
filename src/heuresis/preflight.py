"""Fail-fast preflight checks for sandbox and agent dependencies."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def check_binary(name: str, hint: str = "") -> str | None:
    """Return an error string if *name* is not on PATH, else None."""
    if shutil.which(name):
        return None
    msg = f"Required binary not found: {name}"
    if hint:
        msg += f"\n  {hint}"
    return msg


def check_bwrap() -> str | None:
    return check_binary(
        "bwrap",
        hint="Install bubblewrap: https://github.com/containers/bubblewrap",
    )


def check_taskset() -> str | None:
    return check_binary(
        "taskset",
        hint="Part of util-linux, should be pre-installed on most Linux systems.",
    )


def check_gpu_devices(gpu_ids: list[int]) -> list[str]:
    """Verify /dev/nvidia* and /dev/dri/* device files exist for each GPU."""
    errors: list[str] = []
    for gid in gpu_ids:
        dev = Path(f"/dev/nvidia{gid}")
        if not dev.exists():
            errors.append(f"GPU device file missing: {dev}")
        card = Path(f"/dev/dri/card{gid}")
        if not card.exists():
            errors.append(f"DRI card device missing: {card}")
        render = Path(f"/dev/dri/renderD{128 + gid}")
        if not render.exists():
            errors.append(f"DRI render device missing: {render}")

    if gpu_ids:
        for shared in ("/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools"):
            if not Path(shared).exists():
                errors.append(f"Shared NVIDIA device missing: {shared}")
    return errors


def check_opencode(binary: str = "opencode") -> list[str]:
    """Check opencode binary and LLM credentials."""
    return check_agent(binary)


def check_agent(binary: str = "opencode") -> list[str]:
    """Check that an agent binary exists and LLM credentials are available."""
    errors: list[str] = []
    if not shutil.which(binary):
        errors.append(f"Agent binary not found: {binary}")
        return errors

    has_env_key = any(
        os.environ.get(k)
        for k in (
            "GEMINI_API_KEYS",
            "GEMINI_API_KEY",
            "GOOGLE_GENERATIVE_AI_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENROUTER_API_KEY",
            "CURSOR_API_KEY",
        )
    )
    home = Path.home()
    has_auth_file = any(
        p.exists()
        for p in (
            home / ".local" / "share" / "opencode" / "auth.json",
            home / ".claude" / ".credentials.json",
            home / ".claude.json",
            home / ".cursor-agent",
            home / ".codex" / "auth.json",
        )
    )
    if not has_auth_file and not has_env_key:
        errors.append(
            "No LLM credentials found.\n"
            "  Fix: set an API key env var (e.g. ANTHROPIC_API_KEY) "
            "or run your agent's auth command "
            "(opencode auth login / claude auth / agent login)."
        )
    return errors


def run_all_checks(
    gpu_ids: list[int] | None = None,
    agent_binary: str = "opencode",
    opencode_binary: str | None = None,
) -> list[str]:
    """Run all preflight checks, return list of error strings (empty = OK)."""
    errors: list[str] = []

    for check in (check_bwrap, check_taskset):
        err = check()
        if err:
            errors.append(err)

    if gpu_ids:
        errors.extend(check_gpu_devices(gpu_ids))

    binary = opencode_binary or agent_binary
    errors.extend(check_agent(binary))
    return errors
