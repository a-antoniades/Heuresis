"""Agent profiles: declarative descriptions of CLI coding agents."""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path

_OPENCODE_LOG_LEVEL = os.environ.get("OPENCODE_LOG_LEVEL", "WARN")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_HOST_OPENCODE_CACHE_DIR = Path.home() / ".cache" / "opencode"
_PROJECT_OPENCODE_CACHE_DIR = _PROJECT_ROOT / ".deps" / "opencode-cache"
_OPENCODE_CACHE_DIR = Path(
    os.environ.get(
        "QD_OPENCODE_CACHE_DIR",
        str(_HOST_OPENCODE_CACHE_DIR if _HOST_OPENCODE_CACHE_DIR.is_dir() else _PROJECT_OPENCODE_CACHE_DIR),
    )
)


@dataclass(frozen=True)
class DataMount:
    """A host directory to mount into the bwrap sandbox for the agent."""

    host_path: str
    sandbox_path: str
    readonly: bool = False


@dataclass(frozen=True)
class AgentProfile:
    """Describes how to invoke a CLI coding agent.

    The Harness uses this to build commands instead of hardcoding
    agent-specific flags.  Built-in profiles are provided for OpenCode,
    Claude Code, and Cursor Agent.
    """

    name: str
    run_cmd: list[str]
    format_args: list[str]
    model_flag: str
    session_flag: str
    continue_flag: str
    permission_args: list[str] = field(default_factory=list)
    data_mounts: list[DataMount] = field(default_factory=list)
    extra_args: list[str] = field(default_factory=list)
    session_id_flag: str | None = None
    create_session_cmd: list[str] | None = None


def _create_cursor_session(binary: str) -> str:
    """Run ``agent create-chat`` and return the new chat ID."""
    result = subprocess.run(
        [binary, "create-chat"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def _parse_session_id_from_log(log_path: Path) -> str | None:
    """Extract a session/chat ID from a JSONL agent log.

    Scans for common event fields that carry the session identifier.
    """
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for key in (
                    "session_id",
                    "sessionId",
                    "session",
                    "chatId",
                    "thread_id",
                    "threadId",
                    "id",
                ):
                    val = event.get(key)
                    if isinstance(val, str) and len(val) >= 8:
                        return val
    except OSError:
        pass
    return None


OPENCODE = AgentProfile(
    name="opencode",
    run_cmd=["run"],
    format_args=["--format", "json"],
    model_flag="-m",
    session_flag="-s",
    continue_flag="-c",
    permission_args=[],
    extra_args=["--print-logs", "--log-level", _OPENCODE_LOG_LEVEL],
    # Only the plugin bin/ subdir is bind-mounted (read-only) so every
    # sandbox shares plugins without contending on the host's opencode.db.
    # Each sandbox writes its own db/log/storage into the workspace-local
    # /workspace/.local/share/opencode/ (fresh per run).
    data_mounts=[
        DataMount("~/.local/share/opencode/bin", "/workspace/.local/share/opencode/bin", readonly=True),
        DataMount("~/.config/opencode", "/workspace/.config/opencode", readonly=True),
    ],
)

CLAUDE_CODE = AgentProfile(
    name="claude",
    run_cmd=["-p"],
    format_args=["--output-format", "json"],
    model_flag="--model",
    session_flag="-r",
    continue_flag="--continue",
    permission_args=["--dangerously-skip-permissions"],
    data_mounts=[
        DataMount("~/.claude", "/workspace/.claude"),
    ],
    session_id_flag="--session-id",
)

CURSOR_AGENT = AgentProfile(
    name="agent",
    run_cmd=["-p"],
    format_args=["--output-format", "json"],
    model_flag="--model",
    session_flag="--resume",
    continue_flag="--continue",
    permission_args=["--force", "--trust"],
    data_mounts=[
        DataMount("~/.cursor-agent", "/workspace/.cursor-agent"),
    ],
    create_session_cmd=["create-chat"],
)

# Codex CLI (OpenAI). Invocation shape:
#
#   fresh:   codex exec --model X --dangerously-... --skip-git-repo-check <prompt> --json
#   resume:  codex exec resume --model X --dangerously-... --skip-git-repo-check <id> <prompt> --json
#
# Note: `resume` is a positional sub-subcommand that Harness._build_inner_cmd
# handles specially for codex (run_cmd + "resume" + options + positional id +
# prompt). The session UUID is not specifiable up front; it is recovered from
# the --json event stream after the first run via _parse_session_id_from_log
# (which scans for session_id / thread_id etc.).
#
# `uses_host_execution=True` means this profile bypasses bwrap entirely. Codex
# manages its own sandboxing + process tree and doesn't compose with bwrap;
# running it inside a bwrap jail tended to break its tool spawning. The host
# run sets up PATH/VIRTUAL_ENV etc. so the agent still sees the workspace venv.
CODEX = AgentProfile(
    name="codex",
    run_cmd=["exec"],
    format_args=["--json"],
    model_flag="--model",
    session_flag="resume",
    continue_flag="--last",
    # Already inside bwrap — no need for codex's own sandboxing. bwrap does
    # the filesystem/network/pid isolation; codex is allowed full access
    # within that jail.
    permission_args=["--dangerously-bypass-approvals-and-sandbox"],
    extra_args=["--skip-git-repo-check"],
    data_mounts=[
        DataMount("~/.codex", "/workspace/.codex"),
    ],
)

_BUILTIN_PROFILES: dict[str, AgentProfile] = {
    "opencode": OPENCODE,
    "claude": CLAUDE_CODE,
    "agent": CURSOR_AGENT,
    "codex": CODEX,
}


def detect_profile(agent_name: str) -> AgentProfile | None:
    """Return a built-in profile matching *agent_name*, or ``None``."""
    return _BUILTIN_PROFILES.get(agent_name)


def generate_session_id(profile: AgentProfile, binary: str) -> str | None:
    """Pre-generate a session ID if the profile supports it.

    Returns a dashed UUID (str(uuid.uuid4())) for profiles with session_id_flag —
    Claude Code's --session-id requires the dashed format; opencode accepts both.
    - Cursor Agent: runs ``agent create-chat`` to obtain a chat ID.
    - OpenCode: returns ``None`` (ID is parsed from output after first run).
    """
    if profile.session_id_flag:
        return str(uuid.uuid4())  # dashed UUID (Claude Code requirement)
    if profile.create_session_cmd:
        return _create_cursor_session(binary)
    return None
