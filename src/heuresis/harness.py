"""Agent harness: run coding agents inside bwrap sandboxes."""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from heuresis import _bwrap, _limits, preflight
from heuresis.agent import AgentProfile, detect_profile, _parse_session_id_from_log
from heuresis.models import RunResult
from heuresis.workspace import Mount, Workspace

logger = logging.getLogger(__name__)

_DEFAULT_STRIP_ENV: dict[str, list[str]] = {
    "claude": ["ANTHROPIC_API_KEY"],
    "opencode": [],
    "agent": [],
}


class RunFuture:
    """Handle to an in-flight agent run. Always returned by Harness.run()."""

    def __init__(self, future: Future[RunResult]) -> None:
        self._future = future

    def done(self) -> bool:
        return self._future.done()

    def result(self, timeout: float | None = None) -> RunResult:
        """Block until the run finishes and return the result."""
        return self._future.result(timeout=timeout)


class Harness:
    """Runs coding agents inside bwrap sandboxes.

    Configured once with agent identity and model.  Single method:
    ``run()``, which always returns a ``RunFuture``.

    Session state lives on disk (``<workspace>/.session_id``), not in
    the Harness object.  The Harness itself is stateless across runs.
    """

    def __init__(
        self,
        agent: str = "opencode",
        model: str | None = None,
        *,
        binary: str | None = None,
        profile: AgentProfile | None = None,
        gpus: list[int] | None = None,
        bash_timeout_ms: int = 14_400_000,
        max_workers: int = 8,
        strip_env: list[str] | None = None,
    ) -> None:
        self.agent = agent
        self.model = model
        self.bash_timeout_ms = bash_timeout_ms
        self._gpus = gpus or []

        self.profile = profile or detect_profile(agent)
        if self.profile is None:
            raise ValueError(
                f"No built-in profile for agent {agent!r}. "
                f"Pass an explicit profile=AgentProfile(...)."
            )

        resolved = binary or agent
        self._binary = shutil.which(resolved) or resolved
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._strip_env = (
            list(strip_env) if strip_env is not None
            else list(_DEFAULT_STRIP_ENV.get(agent, []))
        )

    @property
    def gpus(self) -> list[int]:
        """Public read-only view of the pinned GPU list."""
        return list(self._gpus)

    def preflight(self) -> list[str]:
        """Run fail-fast checks. Returns list of errors (empty = OK)."""
        errors = preflight.check_agent(self._binary)
        if self._gpus:
            errors.extend(preflight.check_gpu_devices(self._gpus))
        err = preflight.check_bwrap()
        if err:
            errors.append(err)
        err = preflight.check_taskset()
        if err:
            errors.append(err)
        return errors

    def run(
        self,
        workspace: Workspace,
        prompt: dict[str, Any] | str,
        *,
        mounts: list[Path | Mount] | None = None,
        stateful: bool = False,
        timeout: int | None = None,
        path: Path | None = None,
        session_id: str | None = None,
        extra_cmd_args: list[str] | None = None,
    ) -> RunFuture:
        """Launch an agent run. Always returns immediately with a RunFuture.

        Parameters
        ----------
        workspace:
            Workspace config (tools, venv, files, prompt template).
        prompt:
            If ``dict``, renders the Workspace's Jinja template with
            these variables.  If ``str``, sent directly to the agent.
        mounts:
            Additional workspace dirs to mount read-only.  Accepts plain
            ``Path`` (auto-mounted at ``/workspace/refs/<name>/``) or
            explicit ``Mount(source, target)`` objects.
        stateful:
            If True, continue an existing session at this path.  Reads
            ``.session_id`` from the workspace dir; writes it after the
            first run.
        timeout:
            Timeout in seconds for this run.
        path:
            Directory to run in.  If omitted, a temporary directory is
            created under the current working directory.
        """
        if path is None:
            path = Path(f"run_{uuid.uuid4().hex[:8]}")

        future = self._pool.submit(
            self._execute,
            workspace=workspace,
            prompt=prompt,
            mounts=mounts or [],
            stateful=stateful,
            timeout=timeout,
            path=path,
            explicit_session_id=session_id,
            extra_cmd_args=extra_cmd_args or [],
        )
        return RunFuture(future)

    def session(self, *, workspace: Path, sandbox: Any | None = None) -> Any:
        """Create a compatibility session wrapper around this harness."""
        from heuresis.session import Session

        return Session(self, workspace, sandbox)

    # -- Internal ----------------------------------------------------------

    def _execute(
        self,
        *,
        workspace: Workspace,
        prompt: dict[str, Any] | str,
        mounts: list[Path | Mount],
        stateful: bool,
        timeout: int | None,
        path: Path,
        explicit_session_id: str | None = None,
        extra_cmd_args: list[str] | None = None,
    ) -> RunResult:
        path = path.resolve()
        workspace.setup(path)

        if isinstance(prompt, dict):
            prompt_text = workspace.render_prompt(prompt)
        else:
            prompt_text = prompt

        prompt_file = path / ".prompt.txt"
        prompt_file.write_text(prompt_text)

        session_id, is_continuation = (
            (explicit_session_id, explicit_session_id is not None)
            if explicit_session_id is not None
            else self._resolve_session(path, stateful)
        )

        inner_cmd = self._build_inner_cmd(
            prompt_text,
            session_id=session_id,
            is_continuation=is_continuation,
            extra_cmd_args=extra_cmd_args or [],
        )

        ref_runs, explicit_mounts = _normalize_mounts(mounts)
        env_passthrough = workspace.env_vars()
        # Always expose WORKSPACE_PATH for in-sandbox tools (e.g., grade fallback).
        env_passthrough = list(env_passthrough) + ["WORKSPACE_PATH"]
        # WORKSPACE_ID / WORKSPACE_ROLE are the stable per-workspace identity
        # (see Workspace.setup). We pass the values through a per-subprocess
        # env dict below so concurrent runs never race on os.environ.
        workspace_id = (path / ".workspace_id").read_text().strip()
        env_passthrough = env_passthrough + ["WORKSPACE_ID"]
        workspace_role = ""
        role_marker = path / ".workspace_role"
        if role_marker.exists():
            workspace_role = role_marker.read_text().strip()
            env_passthrough = env_passthrough + ["WORKSPACE_ROLE"]
        # GRADE_SOCKET is deliberately NOT passed via env: with parallel graders,
        # os.environ races. The per-workspace .grade_socket_path marker is the
        # authoritative source; the in-sandbox `grade` tool reads it directly.
        # Filter out env vars the harness is configured to strip (after additions).
        env_passthrough = [v for v in env_passthrough if v not in self._strip_env]

        bwrap_cmd = _bwrap.build_command(
            workspace=path,
            inner_cmd=inner_cmd,
            gpu_ids=self._gpus or None,
            reference_runs=ref_runs,
            extra_mounts=explicit_mounts,
            env_passthrough=env_passthrough,
            bash_timeout_ms=self.bash_timeout_ms,
            profile=self.profile,
            session_mode=session_id is not None,
            strip_env=self._strip_env,
            editable=workspace.editable,
            lock_down_edits=workspace.lock_down_edits,
        )

        full_cmd = _limits.wrap_command(bwrap_cmd, cpu_cores=None)

        logger.info(
            "Launching agent in %s (timeout=%s, model=%s, stateful=%s)",
            path, timeout, self.model, stateful,
        )

        extra_env = {"WORKSPACE_ID": workspace_id}
        if workspace_role:
            extra_env["WORKSPACE_ROLE"] = workspace_role
        duration, exit_code, log_path = self._run_subprocess(
            full_cmd, workspace=path, timeout=timeout,
            extra_env=extra_env,
        )

        if stateful and not (path / ".session_id").exists():
            parsed = _parse_session_id_from_log(log_path)
            if parsed:
                (path / ".session_id").write_text(parsed)
                logger.info("Captured session ID: %s", parsed)

        return RunResult(
            workspace=path,
            exit_code=exit_code,
            stats={"duration": duration},
        )

    def _resolve_session(self, path: Path, stateful: bool) -> tuple[str | None, bool]:
        """Read or create a session ID for stateful runs.

        Returns (session_id, is_continuation). is_continuation is True
        when resuming an existing session, False when creating a new one.
        """
        if not stateful:
            return None, False

        session_file = path / ".session_id"
        if session_file.exists():
            return session_file.read_text().strip(), True

        profile = self.profile
        if profile.session_id_flag:
            sid = str(uuid.uuid4())  # dashed UUID for Claude Code compatibility
            session_file.write_text(sid)
            return sid, False
        if profile.create_session_cmd:
            from heuresis.agent import _create_cursor_session
            sid = _create_cursor_session(self._binary)
            session_file.write_text(sid)
            return sid, False

        return None, False

    def _build_inner_cmd(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        is_continuation: bool = False,
        extra_cmd_args: list[str] | None = None,
    ) -> list[str]:
        p = self.profile

        # Codex has a non-standard shape: session resumption is a positional
        # sub-subcommand (`codex exec resume <id> <prompt>`), so we assemble
        # it separately rather than forcing it through the shared template.
        if p.name == "codex":
            # --search is a *top-level* codex flag (not an exec flag), so it
            # must come before `exec`. Enables the native Responses
            # web_search tool for the model. On by default for codex because
            # ideation workflows (BBOB, nanogpt, etc.) benefit from being
            # able to look up algorithms / papers.
            cmd = [self._binary, "--search"]
            cmd.extend(p.run_cmd)  # exec
            if session_id:
                cmd.append(p.session_flag)  # resume
            if self.model:
                cmd.extend([p.model_flag, self.model])
            cmd.extend(p.permission_args)
            cmd.extend(p.extra_args)
            if session_id:
                cmd.append(session_id)  # positional SESSION_ID for `resume`
            cmd.append(prompt)
            cmd.extend(p.format_args)
            return cmd

        cmd = [self._binary] + list(p.run_cmd)

        if self.model:
            cmd.extend([p.model_flag, self.model])

        if extra_cmd_args:
            cmd.extend(extra_cmd_args)

        if session_id:
            if is_continuation:
                cmd.extend([p.session_flag, session_id])
            elif p.session_id_flag:
                cmd.extend([p.session_id_flag, session_id])
            else:
                cmd.extend([p.session_flag, session_id])

        cmd.extend(p.permission_args)
        cmd.extend(p.extra_args)
        cmd.append(prompt)
        cmd.extend(p.format_args)

        return cmd

    def _run_subprocess(
        self,
        cmd: list[str],
        *,
        workspace: Path,
        timeout: int | None,
        extra_env: dict[str, str] | None = None,
    ) -> tuple[float, int, Path]:
        log_path = workspace / "agent.log"
        t0 = time.monotonic()
        env = {**os.environ, "WORKSPACE_PATH": "/workspace"}  # bwrap bind point
        if extra_env:
            env.update(extra_env)
        with open(log_path, "w") as log_fh:
            proc = subprocess.Popen(
                cmd,
                cwd=workspace,
                stdin=subprocess.DEVNULL,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env,
            )
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning("Agent timed out after %ss, killing", timeout)
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()

        duration = time.monotonic() - t0
        exit_code = proc.returncode if proc.returncode is not None else -1
        logger.info("Agent finished: exit=%d duration=%.1fs", exit_code, duration)
        return duration, exit_code, log_path


def _normalize_mounts(mounts: list[Path | Mount]) -> tuple[
    list[tuple[str, Path]], list[tuple[Path, str]]
]:
    """Split mounts into reference runs and explicit bind mounts.

    Returns (reference_runs, explicit_mounts):
      - reference_runs: [(run_id, path)] mounted at /workspace/runs/<run_id>
      - explicit_mounts: [(source, target)] mounted at exact target path
    """
    refs: list[tuple[str, Path]] = []
    explicit: list[tuple[Path, str]] = []
    for m in mounts:
        if isinstance(m, Mount):
            explicit.append((Path(m.source), m.target))
        elif isinstance(m, Path):
            refs.append((m.name, m))
        else:
            refs.append((str(m), Path(m)))
    return refs, explicit
