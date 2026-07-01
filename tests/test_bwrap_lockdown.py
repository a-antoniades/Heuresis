"""Tests for the workspace-lockdown bind walker in _bwrap.

The walker enforces task-declared editable scope at the bwrap layer:
everything at the workspace top level except the editable subdir and
sandbox-internal dotfiles is ro-bound on top of the workspace bind.
The agent can still create new files at the workspace root (the dir
itself stays writable) and inside the editable subdir.
"""
from __future__ import annotations

from pathlib import Path

from heuresis import _bwrap


def _targets(pairs: list[tuple[Path, str]]) -> list[str]:
    return sorted(t for _, t in pairs)


def test_empty_workspace_returns_nothing(tmp_path: Path) -> None:
    """An empty workspace has nothing to lock down."""
    assert _bwrap._lockdown_binds(tmp_path, editable="discovered") == []


def test_only_editable_and_dotfiles_returns_nothing(tmp_path: Path) -> None:
    """A workspace consisting of only the editable subdir and dotfile internals
    needs no ro-binds — the editable dir is intentionally writable, dotfiles are
    sandbox-internal infrastructure (memory socket, venv, etc.)."""
    (tmp_path / "discovered").mkdir()
    (tmp_path / "discovered" / "loss.py").touch()
    (tmp_path / ".workspace_id").write_text("abc123def456")
    (tmp_path / ".memory.sock").touch()
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv_extra").mkdir()
    (tmp_path / ".bin").mkdir()
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".config").mkdir()
    (tmp_path / ".prompt.txt").write_text("...")

    assert _bwrap._lockdown_binds(tmp_path, editable="discovered") == []


def test_locks_non_editable_top_level_files_and_dirs(tmp_path: Path) -> None:
    """Top-level non-dotfile entries that aren't the editable subdir get ro-bound.
    Files and directories alike. A whole non-editable dir (MinAtar/) is bound as
    one entry, not walked recursively. The bind makes the path read-only;
    symlinks under it that point back into the editable subdir still resolve
    through the kernel and writes land on the editable target (intended)."""
    (tmp_path / "discovered").mkdir()
    (tmp_path / "run_main.py").write_text("# harness")
    (tmp_path / "description.md").write_text("# task")
    (tmp_path / "requirements.txt").write_text("jax")
    (tmp_path / "install.sh").write_text("#!/bin/sh")
    (tmp_path / "MinAtar").mkdir()
    (tmp_path / "MinAtar" / "Breakout").mkdir()
    (tmp_path / "MinAtar" / "Breakout" / "main.py").touch()
    (tmp_path / "MinAtar" / "Asterix").mkdir()

    pairs = _bwrap._lockdown_binds(tmp_path, editable="discovered")

    assert _targets(pairs) == [
        "/workspace/MinAtar",
        "/workspace/description.md",
        "/workspace/install.sh",
        "/workspace/requirements.txt",
        "/workspace/run_main.py",
    ]


def test_returns_host_path_to_sandbox_path_pairs(tmp_path: Path) -> None:
    """Each pair is (host absolute path, sandbox path under /workspace/)."""
    (tmp_path / "discovered").mkdir()
    seed = tmp_path / "run_main.py"
    seed.write_text("# harness")

    pairs = _bwrap._lockdown_binds(tmp_path, editable="discovered")

    assert len(pairs) == 1
    host, target = pairs[0]
    assert host == seed
    assert target == "/workspace/run_main.py"


def test_runtime_creations_absent_at_setup_are_not_locked(tmp_path: Path) -> None:
    """The walker is called at bwrap-build time, before the agent runs.
    Files the agent creates later (notes.md, attempts/, run.log) don't exist
    yet, so they aren't in the lock list — and the agent can freely write
    them inside the writable workspace root dir."""
    # Simulate workspace state at setup time: only seed files are present.
    (tmp_path / "discovered").mkdir()
    (tmp_path / "run_main.py").touch()

    pairs = _bwrap._lockdown_binds(tmp_path, editable="discovered")

    targets = _targets(pairs)
    assert "/workspace/notes.md" not in targets
    assert "/workspace/attempts" not in targets
    assert "/workspace/run.log" not in targets
    assert targets == ["/workspace/run_main.py"]


def test_editable_subdir_name_is_configurable(tmp_path: Path) -> None:
    """The walker accepts any editable subdir name, not just 'discovered'.
    A different task with editable: src would lock everything else and
    leave src/ writable."""
    (tmp_path / "src").mkdir()
    (tmp_path / "harness.py").touch()
    (tmp_path / "discovered").mkdir()  # would be the editable for discogen
    (tmp_path / "discovered" / "extra.py").touch()

    pairs = _bwrap._lockdown_binds(tmp_path, editable="src")

    targets = _targets(pairs)
    # 'discovered' is no longer special — it gets locked because src/ is the
    # editable target. harness.py also gets locked.
    assert targets == ["/workspace/discovered", "/workspace/harness.py"]


def test_build_command_applies_lockdown_when_configured(tmp_path: Path) -> None:
    """build_command appends ro-bind directives after the workspace bind
    when lock_down_edits=True and editable is set; omits them otherwise."""
    (tmp_path / "discovered").mkdir()
    (tmp_path / "run_main.py").touch()
    (tmp_path / "MinAtar").mkdir()

    cmd_default = _bwrap.build_command(
        workspace=tmp_path,
        inner_cmd=["true"],
    )
    cmd_locked = _bwrap.build_command(
        workspace=tmp_path,
        inner_cmd=["true"],
        editable="discovered",
        lock_down_edits=True,
    )

    # Without the flag, no extra ro-binds for the seed files.
    assert "/workspace/run_main.py" not in cmd_default
    assert "/workspace/MinAtar" not in cmd_default

    # With the flag, the ro-binds appear AFTER the workspace --bind. Bwrap
    # processes args in order; later directives override earlier ones for
    # the same path, so a per-path ro-bind on top of a writable parent bind
    # makes that path read-only inside the sandbox.
    ws_bind_pos = next(
        i for i, arg in enumerate(cmd_locked)
        if arg == "/workspace" and cmd_locked[i - 1] == str(tmp_path)
    )
    run_main_pos = cmd_locked.index("/workspace/run_main.py")
    minatar_pos = cmd_locked.index("/workspace/MinAtar")
    assert run_main_pos > ws_bind_pos
    assert minatar_pos > ws_bind_pos
    assert cmd_locked[run_main_pos - 1] == str(tmp_path / "run_main.py")
    assert cmd_locked[run_main_pos - 2] == "--ro-bind"


def test_build_command_skips_lockdown_when_bool_false(tmp_path: Path) -> None:
    """The bool gates lockdown enforcement: editable can be set (it's also
    informational for non-bwrap consumers like the judge), but unless
    lock_down_edits=True, bwrap emits no ro-binds for the seed entries."""
    (tmp_path / "discovered").mkdir()
    (tmp_path / "run_main.py").touch()
    (tmp_path / "MinAtar").mkdir()

    cmd = _bwrap.build_command(
        workspace=tmp_path,
        inner_cmd=["true"],
        editable="discovered",
        lock_down_edits=False,
    )

    assert "/workspace/run_main.py" not in cmd
    assert "/workspace/MinAtar" not in cmd


def test_build_command_skips_lockdown_when_editable_unset(tmp_path: Path, caplog) -> None:
    """If lock_down_edits=True is set but editable is None (misconfiguration),
    bwrap fail-opens: skips the lockdown so the agent can still do anything,
    but logs a warning so the misconfiguration is visible in run logs.
    A workspace with no declared editable region shouldn't be locked entirely
    by accident, since that would prevent the agent from doing anything — yet
    silent no-op also hides bugs, hence the warning."""
    (tmp_path / "discovered").mkdir()
    (tmp_path / "run_main.py").touch()

    import logging
    with caplog.at_level(logging.WARNING, logger="heuresis._bwrap"):
        cmd = _bwrap.build_command(
            workspace=tmp_path,
            inner_cmd=["true"],
            editable=None,
            lock_down_edits=True,
        )

    assert "/workspace/run_main.py" not in cmd
    assert "/workspace/discovered" not in cmd
    assert any(
        "lock_down_edits=True but editable is unset" in rec.message
        for rec in caplog.records
    ), f"expected warning about misconfig, got {[r.message for r in caplog.records]}"


def test_gpu_hf_cache_mount_stays_outside_workspace(tmp_path: Path, monkeypatch) -> None:
    """HF cache should not live under /workspace.

    Agents often run broad commands like ``find .`` from the workspace root.
    Keeping the large HF cache outside /workspace prevents those searches from
    traversing unrelated model/kernel cache contents during GPU runs.
    """
    hf_cache = tmp_path / "hf-cache"
    hf_cache.mkdir()
    monkeypatch.setenv("HF_HOME", str(hf_cache))

    cmd = _bwrap.build_command(
        workspace=tmp_path,
        inner_cmd=["true"],
        gpu_ids=[4],
    )

    assert "/tmp/huggingface" in cmd
    assert "/workspace/.cache/huggingface" not in cmd
    hf_home_pos = cmd.index("HF_HOME")
    assert cmd[hf_home_pos + 1] == "/tmp/huggingface"
