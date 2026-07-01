"""Internal: wrap a command with taskset for CPU pinning."""

from __future__ import annotations

import shutil


def wrap_command(cmd: list[str], *, cpu_cores: list[int] | None = None) -> list[str]:
    """Prepend taskset to *cmd* if CPU cores are specified and taskset is available."""
    if not cpu_cores:
        return cmd

    taskset = shutil.which("taskset")
    if not taskset:
        return cmd

    cores_str = ",".join(str(c) for c in cpu_cores)
    return [taskset, "-c", cores_str] + cmd
