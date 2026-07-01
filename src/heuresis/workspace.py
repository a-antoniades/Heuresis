"""Workspace configuration and setup."""

from __future__ import annotations

import logging
import os
import shutil
import stat
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import ChoiceLoader, Environment, FileSystemLoader, StrictUndefined

from heuresis.tool import Tool

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_VENV = _PROJECT_ROOT / "venvs" / "base"
_DEFAULT_PROJECT_EXTRA = "sandbox"

_UV_WRAPPER = """\
#!/usr/bin/env bash
set -euo pipefail
# uv lives in the agent's host PATH, mounted into the sandbox at /workspace/.agent-bin/
# (see _mount_agent_binary in _bwrap.py). The task venv itself doesn't include uv.
REAL_UV="/workspace/.agent-bin/uv"
if [[ ! -x "$REAL_UV" ]]; then
    REAL_UV="/workspace/.venv/bin/uv"  # fallback for venvs that bundle uv
fi
if [[ "${1:-}" == "pip" && "${2:-}" == "install" ]]; then
    echo "ERROR: Cannot install packages. All dependencies are pre-installed in /workspace/.venv. If you need a package, the run is misconfigured -- write notes.md and stop." >&2
    exit 1
fi
exec "$REAL_UV" "$@"
"""

_PIP_WRAPPER = """\
#!/usr/bin/env bash
if [[ "${1:-}" == "install" ]]; then
    echo "ERROR: Cannot install packages. All dependencies are pre-installed in /workspace/.venv. If you need a package, the run is misconfigured -- write notes.md and stop." >&2
    exit 1
fi
exec /workspace/.venv/bin/python -m pip "$@"
"""


@dataclass(frozen=True)
class Mount:
    """An explicit bind-mount directive for harness.run()."""

    source: Path
    target: str
    readonly: bool = True


@dataclass
class Workspace:
    """Configuration for an agent workspace.

    Defines what a workspace looks like before the agent touches it:
    which CLI tools are available, what files are seeded, how the
    prompt is constructed. This is a config object, not a directory —
    the actual directory is a ``Path`` passed at run time.

    Workspace.setup(path) materializes the config into a directory:
    installs tools to ``.bin/``, copies seed files, links the venv,
    and installs the uv wrapper.
    """

    tools: list[Tool] = field(default_factory=list)
    files: dict[str, Path] = field(default_factory=dict)
    prompt: str | Path = ""
    venv: Path | None = None
    requirements: Path | None = None
    project_extra: str | None = None
    install_args: list[str] = field(default_factory=list)
    # Optional role tag ("ideator" | "executor"). Written to .workspace_role
    # on setup; the memory CLI reads it to stamp author_role on appends.
    # None = marker not written (memory ops for this workspace will default
    # to "executor" on the tool side).
    role: str | None = None
    # Optional path to a MemoryStore Unix socket. When set, Workspace.setup
    # writes .memory_socket_path alongside the workspace so the in-sandbox
    # `memory` CLI can find the server. Harmless when unset.
    memory_socket: Path | None = None
    # Name of the writable entry (file or directory) at the workspace root,
    # mirroring the `editable:` field in task_config.yaml. Mostly informational
    # for the framework — the value is consumed by bwrap when the lockdown is
    # enabled (see ``lock_down_edits`` below) to decide which top-level entry
    # stays writable.
    editable: str | None = None
    # Whether bwrap should enforce the lockdown for this workspace: ro-bind
    # every top-level entry except ``editable`` and dotfiles (sandbox internals)
    # on top of the workspace bind, so the agent can only modify files inside
    # ``editable`` and at the workspace root itself. Off by default; set True
    # for discogen executors. Has no effect unless ``editable`` is also set.
    lock_down_edits: bool = False

    def setup(self, path: Path) -> None:
        """Materialize this workspace config at *path*.

        Idempotent: skips steps that are already done (venv linked,
        tools installed). Safe to call on every run — stateful
        workspaces call this repeatedly with no extra work.

        A base venv (with click for the grade tool) is always linked
        unless the user provides their own.

        If a venv doesn't exist, ``requirements`` builds it from a task-local
        requirements file. The default base venv builds from the package's
        ``sandbox`` extra in ``pyproject.toml``.
        """
        path.mkdir(parents=True, exist_ok=True)
        self._write_identity(path)
        self._install_tools(path)
        self._copy_files(path)
        if os.environ.get("QD_SEED_OPENCODE_CACHE", "1") == "0":
            _seed_opencode_cache_root(path)
        else:
            _seed_opencode_cache(path)
        venv = self.venv or _DEFAULT_VENV
        req = self.requirements
        project_extra = self.project_extra
        if project_extra is None and req is None and venv == _DEFAULT_VENV:
            project_extra = _DEFAULT_PROJECT_EXTRA
        ensure_venv(venv, req, install_args=self.install_args, project_extra=project_extra)
        self._link_venv(path, venv)

    def _write_identity(self, path: Path) -> None:
        """Write .workspace_id, .workspace_role, .memory_socket_path markers.

        `.workspace_id` is unconditional and idempotent: any caller that
        needs a stable per-workspace identity (memory, telemetry, etc.)
        can read it. First setup on a fresh dir generates a UUID; later
        setups (stateful ideator dirs) read the existing one.

        `.workspace_role` and `.memory_socket_path` are gated on the
        corresponding fields being set. Both are harmless if the memory
        primitive isn't in use — no process reads them.
        """
        id_marker = path / ".workspace_id"
        if not id_marker.exists():
            id_marker.write_text(uuid.uuid4().hex[:12])
        if self.role is not None:
            (path / ".workspace_role").write_text(self.role)
        if self.memory_socket is not None:
            (path / ".memory_socket_path").write_text(str(self.memory_socket))

    def env_vars(self) -> list[str]:
        """Collect all env var requirements from composed tools."""
        seen: set[str] = set()
        result: list[str] = []
        for tool in self._all_tools():
            for var in tool.env:
                if var not in seen:
                    seen.add(var)
                    result.append(var)
        return result

    def render_prompt(self, variables: dict[str, Any]) -> str:
        """Render the Jinja prompt template with tool docs auto-injected.

        Missing variables raise ``jinja2.UndefinedError`` (StrictUndefined)
        so template typos surface at run time instead of silently producing
        empty output.

        The template receives ``tools`` (all tools including defaults)
        plus any user-provided variables.
        """
        template_str = self._load_template()
        if not template_str:
            return ""

        search_paths = [_PROJECT_ROOT, _PROJECT_ROOT / "src"]
        if isinstance(self.prompt, Path):
            search_paths.insert(0, self.prompt.parent)

        env = Environment(
            loader=ChoiceLoader([FileSystemLoader(str(p)) for p in search_paths]),
            autoescape=False,
            undefined=StrictUndefined,
        )
        template = env.from_string(template_str)
        context = {"tools": self._all_tools(), **variables}
        return template.render(context)

    def _load_template(self) -> str:
        if isinstance(self.prompt, Path):
            return self.prompt.read_text()
        return self.prompt

    def _all_tools(self) -> list[Tool]:
        """User tools + default tools (grade)."""
        from heuresis.tools.defaults import GRADE
        defaults = [GRADE]
        user_names = {t.name for t in self.tools}
        merged = list(self.tools)
        for d in defaults:
            if d.name not in user_names:
                merged.append(d)
        return merged

    def _install_tools(self, path: Path) -> None:
        bin_dir = path / ".bin"
        bin_dir.mkdir(exist_ok=True)

        uv_script = bin_dir / "uv"
        if not uv_script.exists():
            uv_script.write_text(_UV_WRAPPER)
            _make_executable(uv_script)

        pip_script = bin_dir / "pip"
        if not pip_script.exists():
            pip_script.write_text(_PIP_WRAPPER)
            _make_executable(pip_script)

        system_tools: dict[str, str] = {}
        for tool in self._all_tools():
            if not tool.binary.exists():
                logger.warning("Tool binary not found: %s (%s)", tool.name, tool.binary)
                continue
            if tool.system_install:
                # Skip copying. The harness will bind-mount this at a
                # system-looking path (/opt/qd/bin/<name>) outside /workspace,
                # and add that dir to PATH. Agents see only the command name.
                system_tools[tool.name] = str(tool.binary.resolve())
                continue
            dest = bin_dir / tool.name
            shutil.copy2(tool.binary, dest)
            _make_executable(dest)

        sidecar = path / ".system_tools.json"
        if system_tools:
            import json as _json
            sidecar.write_text(_json.dumps(system_tools, indent=2, sort_keys=True))
        elif sidecar.exists():
            sidecar.unlink()

    def _copy_files(self, path: Path) -> None:
        for dest_rel, source in self.files.items():
            dest = path / dest_rel
            if dest.exists():
                continue
            source = Path(source)
            if not source.exists():
                logger.warning("Seed file not found: %s", source)
                continue
            if source.is_dir():
                shutil.copytree(source, dest, symlinks=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)

    def _link_venv(self, path: Path, base: Path) -> None:
        target = path / ".venv"
        source_marker = path / ".venv_source"
        if source_marker.exists():
            return
        if not base.exists():
            logger.warning("Venv not found and no install source to auto-create: %s", base)
            return
        target.mkdir(exist_ok=True)
        source_marker.write_text(str(base.resolve()))
        (path / ".venv_extra").mkdir(exist_ok=True)


_OPENCODE_CACHE_TARBALL = _PROJECT_ROOT / ".deps" / "opencode-cache.tar.gz"
_HOST_OPENCODE_CACHE_DIR = Path.home() / ".cache" / "opencode"
_PROJECT_OPENCODE_CACHE_DIR = _PROJECT_ROOT / ".deps" / "opencode-cache"
_OPENCODE_CACHE_DIR = Path(
    os.environ.get(
        "QD_OPENCODE_CACHE_DIR",
        str(_HOST_OPENCODE_CACHE_DIR if _HOST_OPENCODE_CACHE_DIR.is_dir() else _PROJECT_OPENCODE_CACHE_DIR),
    )
)


def _seed_opencode_cache(workspace: Path) -> None:
    """Pre-populate .cache/opencode with the auth plugin so opencode doesn't
    re-run ``bun add opencode-anthropic-auth`` on every session.

    Without this, 6 parallel opencode sandboxes all hit the registry at
    once for the same plugin, which rate-limits / hangs / silently fails
    and contributes to the 85-260s SIGTERM pattern.
    """
    oc_cache = workspace / ".cache" / "opencode"
    marker = oc_cache / "package.json"
    if marker.exists():
        try:
            import json as _json
            deps = _json.loads(marker.read_text()).get("dependencies", {})
            if "opencode-anthropic-auth" in deps:
                return
        except (OSError, ValueError):
            pass
    shutil.rmtree(oc_cache, ignore_errors=True)
    if _OPENCODE_CACHE_DIR.is_dir():
        _hardlink_tree(_OPENCODE_CACHE_DIR, oc_cache)
    elif _OPENCODE_CACHE_TARBALL.is_file():
        oc_cache.mkdir(parents=True, exist_ok=True)
        import tarfile
        with tarfile.open(_OPENCODE_CACHE_TARBALL) as tf:
            tf.extractall(oc_cache)
    else:
        return
    logger.info("Seeded opencode cache into %s", oc_cache)


def _hardlink_or_copy(src: str, dst: str) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _seed_opencode_cache_root(workspace: Path) -> None:
    oc_cache = workspace / ".cache" / "opencode"
    oc_cache.mkdir(parents=True, exist_ok=True)
    if not _OPENCODE_CACHE_DIR.is_dir():
        return
    for item in _OPENCODE_CACHE_DIR.iterdir():
        if item.is_dir():
            continue
        dest = oc_cache / item.name
        if not dest.exists():
            _hardlink_or_copy(str(item), str(dest))


def _hardlink_tree(source: Path, target: Path) -> None:
    """Materialize *source* at *target* without copying directory metadata."""
    target.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(source):
        root_path = Path(root)
        rel_root = root_path.relative_to(source)
        target_root = target / rel_root
        for dirname in dirs:
            src_dir = root_path / dirname
            dst_dir = target_root / dirname
            if src_dir.is_symlink():
                dst_dir.symlink_to(os.readlink(src_dir))
            else:
                dst_dir.mkdir(exist_ok=True)
        for filename in files:
            src_file = root_path / filename
            dst_file = target_root / filename
            if src_file.is_symlink():
                dst_file.symlink_to(os.readlink(src_file))
            else:
                _hardlink_or_copy(str(src_file), str(dst_file))


def ensure_venv(
    venv_path: Path,
    requirements: Path | None = None,
    *,
    install_args: list[str] | None = None,
    project_extra: str | None = None,
) -> None:
    """Idempotently ensure a sandbox venv exists, building it if needed.

    Called internally by ``Workspace.setup()`` when materializing a workspace, and
    externally by experiment ``run.py`` scripts that need the task venv staged
    before preflight (e.g. to shell a GPU-visibility check into the venv's python
    before any harness launches).

    No-op when ``venv_path/bin/python`` already exists. Otherwise creates the venv
    and installs from either a task-local requirements file or a package extra.

    Args:
        venv_path: Target venv directory.
        requirements: Requirements file (defaults to venv_path/requirements.txt).
            If absent on disk, the function returns without error (nothing to build).
        install_args: Extra args passed to ``uv pip install`` (e.g. --prerelease allow).
        project_extra: Optional package extra to install from this repository,
            such as ``sandbox`` or ``novelty``.
    """
    if (venv_path / "bin" / "python").exists():
        return
    req_file = requirements or (venv_path / "requirements.txt")
    if project_extra is not None:
        install_source = f"{_PROJECT_ROOT}[{project_extra}]"
        install_args_base = ["pip", "install", str(install_source)]
        python = "python3.12"
    elif req_file.exists():
        install_source = req_file
        install_args_base = ["pip", "install", "-r", str(req_file)]
        python = "python3.11"
    else:
        return

    logger.info("Auto-creating venv at %s from %s", venv_path, install_source)
    venv_path.mkdir(parents=True, exist_ok=True)

    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is required to auto-create venvs but was not found on PATH")

    subprocess.run(
        [uv, "venv", str(venv_path), "--python", python],
        check=True,
    )
    subprocess.run(
        [uv, *install_args_base, "--python", str(venv_path / "bin" / "python"), *(install_args or [])],
        check=True,
    )
    logger.info("Venv created: %s", venv_path)


def _make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
