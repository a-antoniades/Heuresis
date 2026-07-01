"""Self-describing CLI tools for agent workspaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Tool:
    """A CLI tool that can be installed into an agent workspace.

    Each tool is an executable file paired with documentation that gets
    automatically included in the agent's prompt via Jinja rendering.
    Tools are the access-control boundary: if it's not in the baseline
    (linux CLIs + uv) and not declared as a Tool, the agent can't use it.

    The ``binary`` field points to an executable file on the host — a
    Python Click script, a shell script, or any other executable.

    By default the binary is copied to ``.bin/<name>`` inside the
    workspace at setup time. This keeps everything self-contained but
    leaves a plain-text copy agents can ``cat`` and — as we learned the
    hard way — reimplement instead of invoking.

    Set ``system_install=True`` and the tool is *not* copied. Instead
    the host path is recorded in ``.system_tools.json`` and the harness
    bind-mounts it at ``/opt/qd/bin/<name>`` outside ``/workspace``, on
    PATH. Agents see it the way they see ``ls`` or ``cat``: an opaque
    system command. Use this for shared framework primitives (memory,
    future global services). Keep task-specific tools (grade, …) as
    workspace tools.
    """

    name: str
    binary: Path
    docs: str
    env: list[str] = field(default_factory=list)
    system_install: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "binary", Path(self.binary))
