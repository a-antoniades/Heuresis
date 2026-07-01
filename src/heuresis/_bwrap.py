"""Internal: build the bwrap command line for sandbox isolation."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from heuresis.agent import AgentProfile

logger = logging.getLogger(__name__)

_SYSTEM_RO_BINDS = [
    "/usr",
    "/lib",
    "/lib64",
    "/etc",
]

# /etc is bind-mounted into the sandbox, but on hosts where
# /etc/resolv.conf is a symlink into /run/... (systemd-resolved on GCE,
# other resolver setups elsewhere) the symlink target is absent inside
# the sandbox and DNS resolution silently fails. Resolve the symlink
# at command-construction time and bind the actual target's parent dir
# when it lives outside /etc.
_DNS_CONFIG = Path("/etc/resolv.conf")
_ETC_DIR = Path("/etc")

_SYSTEM_SYMLINKS = [
    ("/bin", "usr/bin"),
    ("/sbin", "usr/sbin"),
    ("/lib", "usr/lib"),
    ("/lib64", "usr/lib64"),
]

_SHARED_NVIDIA_DEVICES = [
    "/dev/nvidiactl",
    "/dev/nvidia-uvm",
    "/dev/nvidia-uvm-tools",
    "/dev/nvidia-modeset",
]

_LLM_ENV_VARS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEYS",
    "GEMINI_API_KEY",
    "GOOGLE_GENERATIVE_AI_API_KEY",
    "OPENROUTER_API_KEY",
    "CURSOR_API_KEY",
    "HF_TOKEN",
]

# Task-tunable env vars forwarded into the sandbox when set on the host.
# Lets a task's seed code read a knob (e.g. a reduced training budget for
# smoke runs) without the executor agent seeing a hardcoded value to "fix".
_TASK_ENV_VARS = [
    "NANOGPT_TIME_BUDGET",
]

_HF_CACHE_SANDBOX = "/tmp/huggingface"


def build_command(
    *,
    workspace: Path,
    inner_cmd: list[str],
    gpu_ids: list[int] | None = None,
    reference_runs: list[tuple[str, Path]] | None = None,
    extra_mounts: list[tuple[Path, str]] | None = None,
    extra_env: dict[str, str] | None = None,
    env_passthrough: list[str] | None = None,
    bash_timeout_ms: int = 14_400_000,
    profile: "AgentProfile | None" = None,
    session_mode: bool = False,
    strip_env: list[str] | None = None,
    editable: str | None = None,
    lock_down_edits: bool = False,
) -> list[str]:
    """Assemble a full bwrap command from config pieces."""
    inner_cmd = list(inner_cmd)
    original_binary = inner_cmd[0] if inner_cmd else ""
    inner_cmd = _rewrite_binary_for_sandbox(inner_cmd)

    cmd: list[str] = ["bwrap"]

    cmd.extend(["--unshare-pid", "--die-with-parent"])

    for path in _SYSTEM_RO_BINDS:
        p = Path(path)
        if p.exists() and not p.is_symlink():
            cmd.extend(["--ro-bind", path, path])

    _mount_dns_resolver(cmd)

    for link_path, link_target in _SYSTEM_SYMLINKS:
        p = Path(link_path)
        if p.is_symlink():
            cmd.extend(["--symlink", link_target, link_path])
        elif p.is_dir():
            cmd.extend(["--ro-bind", link_path, link_path])

    cmd.extend(["--proc", "/proc"])
    cmd.extend(["--dev", "/dev"])
    cmd.extend(["--tmpfs", "/tmp"])
    cmd.extend(["--tmpfs", "/dev/shm"])

    _add_gpu_mounts(cmd, gpu_ids)

    cmd.extend(["--bind", str(workspace), "/workspace"])

    if lock_down_edits:
        if editable is None:
            logger.warning(
                "lock_down_edits=True but editable is unset — lockdown skipped for %s",
                workspace,
            )
        else:
            for host, target in _lockdown_binds(workspace, editable=editable):
                cmd.extend(["--ro-bind", str(host), target])

    venv_dir = workspace / ".venv"
    if venv_dir.is_dir():
        venv_source = workspace / ".venv_source"
        venv_host = Path(venv_source.read_text().strip()) if venv_source.exists() else venv_dir
        cmd.extend(["--ro-bind", str(venv_host), "/workspace/.venv"])

    venv_extra = workspace / ".venv_extra"
    if venv_extra.is_dir():
        cmd.extend(["--bind", str(venv_extra), "/workspace/.venv_extra"])

    if reference_runs:
        for run_id, run_path in reference_runs:
            cmd.extend(["--ro-bind", str(run_path), f"/workspace/runs/{run_id}"])

    if extra_mounts:
        for source, target in extra_mounts:
            if source.exists():
                cmd.extend(["--ro-bind", str(source), target])

    _mount_python_includes(cmd)
    _mount_hf_cache(cmd, gpu_ids)
    _mount_grade_socket(cmd, workspace)
    _mount_memory_socket(cmd, workspace)
    _mount_system_tools(cmd, workspace)
    _mount_editable_deps(cmd, workspace)
    _mount_agent_binary(cmd, original_binary)

    if profile:
        _add_agent_mounts(cmd, profile, session_mode)
    else:
        _add_opencode_mounts(cmd)

    env_vars = _build_env_vars(
        gpu_ids=gpu_ids,
        workspace=workspace,
        extra_env=extra_env,
        bash_timeout_ms=bash_timeout_ms,
    )
    for key, val in env_vars.items():
        cmd.extend(["--setenv", key, val])

    if env_passthrough:
        for key in env_passthrough:
            val = os.environ.get(key)
            if val is not None:
                cmd.extend(["--setenv", key, val])

    for key in _LLM_ENV_VARS:
        val = os.environ.get(key)
        if val is not None:
            cmd.extend(["--setenv", key, val])

    for key in _TASK_ENV_VARS:
        val = os.environ.get(key)
        if val is not None:
            cmd.extend(["--setenv", key, val])

    for var in strip_env or []:
        cmd.extend(["--unsetenv", var])

    cmd.extend(["--chdir", "/workspace"])
    cmd.append("--")
    cmd.extend(inner_cmd)

    return cmd


def _add_gpu_mounts(cmd: list[str], gpu_ids: list[int] | None) -> None:
    if not gpu_ids:
        return

    cmd.extend(["--dir", "/dev/dri"])

    if Path("/sys").is_dir():
        cmd.extend(["--ro-bind", "/sys", "/sys"])

    for dev in _SHARED_NVIDIA_DEVICES:
        if Path(dev).exists():
            cmd.extend(["--dev-bind", dev, dev])

    nvidia_caps = Path("/dev/nvidia-caps")
    if nvidia_caps.is_dir():
        cmd.extend(["--dev-bind", str(nvidia_caps), str(nvidia_caps)])

    for gid in gpu_ids:
        nvidia_dev = f"/dev/nvidia{gid}"
        if Path(nvidia_dev).exists():
            cmd.extend(["--dev-bind", nvidia_dev, nvidia_dev])

        card = f"/dev/dri/card{gid}"
        if Path(card).exists():
            cmd.extend(["--dev-bind", card, card])

        render = f"/dev/dri/renderD{128 + gid}"
        if Path(render).exists():
            cmd.extend(["--dev-bind", render, render])


_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_GRADE_SOCK_SANDBOX = "/workspace/.grade.sock"
_MEMORY_SOCK_SANDBOX = "/workspace/.memory.sock"
_SYSTEM_TOOLS_SANDBOX_DIR = "/opt/qd/bin"


def _mount_python_includes(cmd: list[str]) -> None:
    """Mount Python dev headers so Triton JIT (torch.compile) can build C extensions.

    Triton inside the task venv calls sysconfig which points at /usr/include/pythonX.Y.
    System may lack those headers; search conda envs for matching python3.{10..13}
    include dirs and overlay-mount at /usr/include/pythonX.Y.
    """
    conda_roots = [
        Path.home() / ".conda" / "envs",
        Path("/opt/conda/envs"),
    ]
    home_candidates = [Path.home()]
    for raw_root in os.environ.get("QD_CONDA_HOME_ROOTS", "").split(os.pathsep):
        if raw_root:
            home_candidates.append(Path(raw_root).expanduser())
    for home in home_candidates:
        for ana in sorted(home.glob("anaconda*")) + sorted(home.glob("miniconda*")):
            conda_roots.append(ana / "envs")
            conda_roots.append(ana / "pkgs")
    for minor in range(10, 14):
        py_ver = f"python3.{minor}"
        target = Path(f"/usr/include/{py_ver}")
        if (target / "Python.h").is_file():
            continue
        if not target.is_dir():
            # bwrap can't create a mount point under the RO /usr bind; skip.
            continue
        for root in conda_roots:
            if not root.is_dir():
                continue
            found = False
            for env_dir in root.iterdir():
                candidate = env_dir / "include" / py_ver
                if (candidate / "Python.h").is_file():
                    cmd.extend(["--ro-bind", str(candidate), str(target)])
                    found = True
                    break
            if found:
                break


def _mount_hf_cache(cmd: list[str], gpu_ids: list[int] | None) -> None:
    """Mount the host HuggingFace cache so pre-downloaded kernels (FA3, etc.)
    are available offline inside the sandbox.

    Only mounted for GPU runs; without the cache + HF_HUB_OFFLINE=1, train.py's
    `kernels.get_kernel(...)` call makes unauthenticated HTTP requests that
    rate-limit and fail mid-run.
    """
    if not gpu_ids:
        return
    hf_cache = Path(
        os.environ.get("HF_HOME")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or str(Path.home() / ".cache" / "huggingface")
    )
    if hf_cache.is_dir():
        # Writable bind: kernels>=0.13 calls check_status → hf_hub_download,
        # which writes refs/<revision> commit-hash markers into the cache.
        # Read-only bind made those writes fatal (OSError: [Errno 30]).
        # Sandboxed writes here are small metadata updates; trade-off accepted.
        cmd.extend(["--bind", str(hf_cache), _HF_CACHE_SANDBOX])


def _mount_grade_socket(cmd: list[str], workspace: Path) -> None:
    """Bind-mount the grading server socket into the sandbox.

    The ``GradingServer`` may place the socket under ``/tmp`` (to avoid
    the 108-char Unix socket path limit).  It writes the actual path to
    ``workspace/.grade_socket_path``.  We mount that socket at a fixed
    location inside the sandbox so the grade script can find it.
    """
    marker = workspace / ".grade_socket_path"
    if not marker.exists():
        return
    host_sock = Path(marker.read_text().strip())
    if host_sock.exists():
        cmd.extend(["--bind", str(host_sock), _GRADE_SOCK_SANDBOX])


def _mount_memory_socket(cmd: list[str], workspace: Path) -> None:
    """Bind-mount the memory server socket into the sandbox.

    Mirror of ``_mount_grade_socket``. MemoryStore writes the host socket
    path to ``workspace/.memory_socket_path``; we bind-mount that socket
    at ``/workspace/.memory.sock`` inside the sandbox because the host
    ``/tmp`` is shadowed by ``--tmpfs /tmp`` and the CLI can't reach
    the original path otherwise.
    """
    marker = workspace / ".memory_socket_path"
    if not marker.exists():
        return
    host_sock = Path(marker.read_text().strip())
    if host_sock.exists():
        cmd.extend(["--bind", str(host_sock), _MEMORY_SOCK_SANDBOX])


def _mount_system_tools(cmd: list[str], workspace: Path) -> None:
    """Bind-mount system-installed tools at ``/opt/qd/bin/<name>``.

    Workspace.setup writes ``.system_tools.json`` mapping tool name →
    absolute host path for every tool with ``system_install=True``.
    We bind-mount each as a read-only file inside the sandbox at a
    path that looks like a standard install, then rely on
    ``_build_env_vars`` to prepend ``/opt/qd/bin`` to ``PATH``.

    Rationale: copying the source into ``.bin/`` makes it trivial for
    agents to ``cat`` the script and reimplement it (we watched them
    open sockets by hand). A bind-mount outside ``/workspace`` both
    hides it from casual ``ls`` and makes it feel like a system
    command they shouldn't touch.
    """
    import json as _json
    marker = workspace / ".system_tools.json"
    if not marker.exists():
        return
    try:
        tools = _json.loads(marker.read_text())
    except (OSError, ValueError):
        return
    for name, host_path in tools.items():
        host = Path(host_path)
        if not host.is_file():
            continue
        cmd.extend(["--ro-bind", str(host), f"{_SYSTEM_TOOLS_SANDBOX_DIR}/{name}"])


def _mount_editable_deps(cmd: list[str], workspace: Path) -> None:
    """Mount editable-installed dependencies so imports work in the sandbox.

    Editable installs (``pip install -e``) create path finders that
    reference the source directory on the host.  We mount the ``.deps/``
    tree read-only at the same host path so the finder resolves.
    """
    deps_dir = _PROJECT_ROOT / ".deps"
    if deps_dir.is_dir():
        cmd.extend(["--ro-bind", str(deps_dir), str(deps_dir)])


def _lockdown_binds(workspace: Path, *, editable: str) -> list[tuple[Path, str]]:
    """Walk ``workspace`` top level and pick paths to ro-bind.

    Returns ``(host_path, sandbox_path)`` pairs intended to be appended
    after the workspace ``--bind`` so they override it for those paths.
    Skipped:

    * The entry named ``editable`` (file or directory) — agent edits land there.
    * Anything starting with ``.`` — sandbox-internal infrastructure
      (``.workspace_id``, ``.memory.sock``, ``.bin/``, ``.cache/``,
      ``.config/``, ``.local/``, ``.venv*``, ``.prompt.txt``, etc.).

    The workspace root directory itself stays writable, so the agent
    can still create new top-level files (notes.md, attempts/,
    probe scripts) at runtime; only entries that exist at bwrap-build
    time get locked.

    Caveat: ro-binds are path-based, not inode-based. A symlink under a
    locked entry (e.g. ``MinAtar/Breakout/loss.py -> ../../{editable}/loss.py``)
    is itself read-only, but the kernel resolves it and writes land in
    the editable target — which is intended for symlinks that point back
    into ``editable`` but does not block tampering through any symlink
    that escapes ``editable``.
    """
    pairs: list[tuple[Path, str]] = []
    for entry in sorted(workspace.iterdir()):
        name = entry.name
        if name == editable:
            continue
        if name.startswith("."):
            continue
        pairs.append((entry, f"/workspace/{name}"))
    return pairs


_AGENT_BIN_SANDBOX = "/workspace/.agent-bin"

_SYSTEM_BIN_PREFIXES = ("/usr/", "/bin/", "/sbin/")


def _rewrite_binary_for_sandbox(inner_cmd: list[str]) -> list[str]:
    """Rewrite the agent binary path for use inside the sandbox.

    If the binary lives outside standard system paths (e.g.
    ``/local/home/user/.opencode/bin/opencode``), rewrite it to
    a sandbox mount point.  The corresponding directory is mounted
    by :func:`_mount_agent_binary`.

    Symlinks are resolved before rewriting so that the bwrap mount covers
    the symlink's actual target directory — otherwise the in-sandbox
    symlink dereferences to a path that isn't mounted. Example: Claude
    Code installs ``~/.local/bin/claude`` as a symlink to
    ``~/.local/share/claude/versions/X.Y.Z``; mounting only ``.local/bin``
    makes the symlink unusable in the sandbox.
    """
    if not inner_cmd:
        return inner_cmd
    binary = inner_cmd[0]
    if binary.startswith(_SYSTEM_BIN_PREFIXES):
        return inner_cmd
    binary_path = Path(binary).resolve()
    if not binary_path.is_absolute() or not binary_path.exists():
        return inner_cmd
    return [f"{_AGENT_BIN_SANDBOX}/{binary_path.name}"] + inner_cmd[1:]


def _mount_agent_binary(cmd: list[str], original_binary: str) -> None:
    """Bind-mount the agent binary's directory into the sandbox.

    Resolves symlinks so the mounted directory contains the actual binary
    (see :func:`_rewrite_binary_for_sandbox` for the reason).
    """
    if original_binary.startswith(_SYSTEM_BIN_PREFIXES):
        return
    binary_path = Path(original_binary).resolve()
    if not binary_path.is_absolute() or not binary_path.exists():
        return
    cmd.extend(["--ro-bind", str(binary_path.parent), _AGENT_BIN_SANDBOX])


def _add_agent_mounts(
    cmd: list[str],
    profile: "AgentProfile",
    session_mode: bool = False,
) -> None:
    """Mount agent data/config dirs based on the profile's ``data_mounts``.

    Mounts are always read-write because agents (OpenCode, Claude Code,
    Cursor Agent) need to write to their data dirs even for basic runs
    (database updates, session logs, etc.).

    For each directory mount, also resolve direct-child symlinks pointing
    outside the bound directory and bind their targets at the same absolute
    host paths inside the sandbox. This is required for setups where the
    agent's home directory (e.g. ``~/.claude``) is a symlink farm into a
    separate runtime/config tree — without these extra binds, internal
    symlinks dangle in-sandbox and any tool that follows them fails with
    ENOENT (Claude Code's Bash tool ``mkdir session-env/<uuid>/`` is the
    forcing case). See :func:`_resolve_outward_symlinks`.
    """
    for mount in profile.data_mounts:
        host = Path(mount.host_path).expanduser()
        if not host.exists():
            continue
        flag = "--ro-bind" if mount.readonly else "--bind"
        cmd.extend([flag, str(host), mount.sandbox_path])
        for target in _resolve_outward_symlinks(host):
            cmd.extend([flag, str(target), str(target)])


def _resolve_outward_symlinks(host_dir: Path) -> list[Path]:
    """Find direct-child symlinks of *host_dir* whose targets live outside it.

    Returns a deduplicated list of resolved target paths, in iteration order.
    Each needs an additional bwrap bind at its absolute host path so the
    symlink resolves inside the sandbox.

    Only depth-1 children are scanned: deeper-nested outward symlinks are
    rare and recursing risks binding large unrelated subtrees. Dangling
    symlinks are skipped silently (logged at DEBUG) — the brain-vault layout
    has many depth-1 symlinks pointing at per-machine state that may not
    exist yet (telemetry, debug, settings.local.json), and these are not
    actionable. Symlinks that resolve back inside the bound directory are
    skipped (bwrap handles them automatically).
    """
    if not host_dir.is_dir():
        return []
    host_real = host_dir.resolve()
    seen: dict[str, Path] = {}
    for entry in host_dir.iterdir():
        if not entry.is_symlink():
            continue
        try:
            target = entry.resolve(strict=True)
        except (FileNotFoundError, RuntimeError):
            logger.debug(
                "Skipping dangling symlink %s -> %s",
                entry, os.readlink(entry),
            )
            continue
        try:
            target.relative_to(host_real)
            continue  # target is inside the bound dir; bwrap handles it
        except ValueError:
            pass
        seen.setdefault(str(target), target)
    return list(seen.values())


def _mount_dns_resolver(cmd: list[str]) -> None:
    """Ensure /etc/resolv.conf resolves to a real file inside the sandbox.

    /etc is already bind-mounted, so a regular-file resolv.conf is covered
    automatically. When resolv.conf is a single-hop symlink (typical on
    systemd-resolved and similar setups), bind the immediate target's
    parent directory so the kernel can follow the symlink inside the
    sandbox.

    Multi-hop chains, where a parent component of the symlink target is
    itself a symlink (e.g. ``/etc/resolv.conf -> /var/run/X/resolv.conf``
    and ``/var/run -> /run``), are intentionally not supported. The kernel
    walks each path component, so binding only the canonical target's
    parent leaves the intermediate symlink-parent unreachable inside the
    sandbox. Refuse to launch instead of silently breaking DNS.

    Raises:
        RuntimeError: if the symlink target does not exist on the host, or
            if any parent component of the immediate symlink target is
            itself a symlink (multi-hop chain).
    """
    if not _DNS_CONFIG.is_symlink():
        return
    link_value = os.readlink(_DNS_CONFIG)
    if link_value.startswith("/"):
        target = Path(link_value)
    else:
        target = _DNS_CONFIG.parent / link_value
    if not target.exists():
        raise RuntimeError(
            f"/etc/resolv.conf is a symlink to {target} which does not "
            "exist on the host. DNS resolution will fail inside the agent "
            "sandbox. Fix the host resolver configuration first."
        )
    try:
        target.relative_to(_ETC_DIR)
        return
    except ValueError:
        pass
    ancestor = target.parent
    while ancestor != ancestor.parent:
        if ancestor.is_symlink():
            raise RuntimeError(
                f"/etc/resolv.conf points to {target} but ancestor {ancestor} "
                "is a symlink (multi-hop chain). The sandbox does not yet "
                "support multi-hop resolver layouts. Bind the chain manually "
                "or simplify the host resolver configuration."
            )
        ancestor = ancestor.parent
    cmd.extend(["--ro-bind", str(target.parent), str(target.parent)])


def _add_opencode_mounts(cmd: list[str]) -> None:
    """Legacy fallback: mount opencode plugin dir when no profile is set.

    Only the bin/ subdir is bind-mounted (read-only) — each sandbox writes
    its own opencode.db / storage / logs into the workspace-local path,
    preventing SQLite lock contention across parallel sandboxes on the
    shared host db.
    """
    home = Path.home()

    bin_dir = home / ".local" / "share" / "opencode" / "bin"
    if bin_dir.is_dir():
        cmd.extend(["--ro-bind", str(bin_dir), "/workspace/.local/share/opencode/bin"])

    config_dir = home / ".config" / "opencode"
    if config_dir.is_dir():
        cmd.extend(["--ro-bind", str(config_dir), "/workspace/.config/opencode"])


def _build_env_vars(
    *,
    gpu_ids: list[int] | None,
    workspace: Path,
    extra_env: dict[str, str] | None,
    bash_timeout_ms: int,
) -> dict[str, str]:
    env: dict[str, str] = {}

    env["HOME"] = "/workspace"
    env["XDG_DATA_HOME"] = "/workspace/.local/share"
    env["XDG_CONFIG_HOME"] = "/workspace/.config"
    env["XDG_CACHE_HOME"] = "/workspace/.cache"

    path_parts = [
        "/workspace/.bin",
        "/workspace/.venv/bin",
        _SYSTEM_TOOLS_SANDBOX_DIR,
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
    ]
    env["PATH"] = ":".join(path_parts)

    if gpu_ids is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(len(gpu_ids)))
        env["HF_HOME"] = _HF_CACHE_SANDBOX
        # kernels>=0.13 calls list_repo_tree at startup which is not offline-safe;
        # rely on HF_HOME cache for file reads, allow the metadata round-trip.

    ncpu = "1"
    env["OMP_NUM_THREADS"] = ncpu
    env["MKL_NUM_THREADS"] = ncpu
    env["OPENBLAS_NUM_THREADS"] = ncpu
    env["LOKY_MAX_CPU_COUNT"] = ncpu

    venv_dir = workspace / ".venv"
    if venv_dir.is_dir():
        env["VIRTUAL_ENV"] = "/workspace/.venv"
        env["UV_PROJECT_ENVIRONMENT"] = "/workspace/.venv"

    venv_extra = workspace / ".venv_extra"
    if venv_extra.is_dir():
        env["PYTHONPATH"] = "/workspace/.venv_extra"
        env["WORKSPACE_VENV_EXTRA"] = "/workspace/.venv_extra"

    env["OPENCODE_EXPERIMENTAL_BASH_DEFAULT_TIMEOUT_MS"] = str(bash_timeout_ms)

    # Session start time so the agent can track elapsed time
    env["SESSION_START_EPOCH"] = str(int(time.time()))

    if extra_env:
        env.update(extra_env)

    return env


# --- Agent-less command runner --------------------------------------------

import os as _os  # noqa: E402
import signal as _signal  # noqa: E402
import subprocess as _subprocess  # noqa: E402
import time as _time  # noqa: E402
from dataclasses import dataclass as _dataclass  # noqa: E402

# Re-import to avoid shadowing if the module is partially imported during tests.
from pathlib import Path as _Path  # noqa: E402


@_dataclass
class SandboxResult:
    """Result of a no-agent sandbox command."""

    exit_code: int
    duration_s: float


def run_command(
    *,
    workspace: _Path,
    command: list[str],
    gpu_ids: list[int] | None = None,
    mounts: "list[object] | None" = None,  # list[Mount]; imported lazily below
    env_passthrough: list[str] | None = None,
    timeout: int | None = None,
    stdout_to: str | None = None,
    strip_env: list[str] | None = None,
) -> SandboxResult:
    """Run ``command`` inside bwrap over ``workspace``. No agent, no grade socket.

    Reuses :func:`build_command` (with ``profile=None``) so GPU mounts, HF cache,
    venv binds, and env passthrough match the executor path exactly.

    ``stdout_to`` is a workspace-relative path. When set, both stdout and stderr
    are redirected to ``workspace/<stdout_to>`` (parent dirs are created). When
    unset, stdout/stderr are redirected to /dev/null.
    """
    from heuresis.workspace import Mount  # lazy: avoid circular import

    explicit_mounts: list[tuple[_Path, str]] = []
    for m in mounts or []:
        if isinstance(m, Mount):
            explicit_mounts.append((_Path(m.source), m.target))

    bwrap_cmd = build_command(
        workspace=workspace,
        inner_cmd=command,
        gpu_ids=gpu_ids,
        extra_mounts=explicit_mounts,
        env_passthrough=env_passthrough,
        profile=None,
        session_mode=False,
        strip_env=strip_env,
    )

    if stdout_to is not None:
        stdout_file = workspace / stdout_to
        stdout_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        stdout_file = _Path("/dev/null")

    duration, exit_code = _run_subprocess(
        bwrap_cmd, stdout_file=stdout_file, timeout=timeout,
    )
    return SandboxResult(exit_code=exit_code, duration_s=duration)


def _run_subprocess(
    cmd: list[str],
    *,
    stdout_file: _Path,
    timeout: int | None,
) -> tuple[float, int]:
    """Run a subprocess with SIGTERM → SIGKILL escalation on timeout.

    Redirects stdout and stderr to ``stdout_file``. Returns (duration_s, exit_code).
    """
    t0 = _time.monotonic()
    # open() treats "/dev/null" specially via the kernel; no tmpfile needed.
    with open(stdout_file, "w") as fh:
        proc = _subprocess.Popen(
            cmd,
            stdout=fh,
            stderr=_subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=timeout)
        except _subprocess.TimeoutExpired:
            _os.killpg(proc.pid, _signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except _subprocess.TimeoutExpired:
                _os.killpg(proc.pid, _signal.SIGKILL)
                proc.wait()

    duration = _time.monotonic() - t0
    exit_code = proc.returncode if proc.returncode is not None else -1
    return duration, exit_code
