"""Outward-symlink resolution for agent data mounts.

Background: ``~/.claude`` (and similar agent home dirs) can be a symlink
farm where direct children point to absolute paths outside the bound dir.
bwrap binds the dir's realpath but doesn't follow internal symlinks, so
those targets dangle in-sandbox — Claude Code's Bash tool then fails
``mkdir session-env/<uuid>/`` with ENOENT on every call.

These tests pin the helper that walks each data_mount and adds extra
binds for outward symlink targets.
"""
from __future__ import annotations

from pathlib import Path

from heuresis import _bwrap
from heuresis.agent import AgentProfile, DataMount


def _make_claude_like_layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create the brain-vault-style layout: bound dir is itself a symlink,
    its children are symlinks into separate runtime/config trees.

    Returns (claude_link, runtime_dir, config_dir).
    """
    claude_real = tmp_path / "agent-settings" / ".claude-home"
    claude_real.mkdir(parents=True)
    claude_link = tmp_path / ".claude"
    claude_link.symlink_to(claude_real)

    runtime = tmp_path / ".agent-runtime" / "claude"
    (runtime / "session-env").mkdir(parents=True)
    (runtime / "projects").mkdir()
    (runtime / "history.jsonl").write_text("")

    config = tmp_path / "agent-settings" / "08-Agents"
    (config / "Shared" / "Skills").mkdir(parents=True)
    (config / "Claude").mkdir()
    (config / "Claude" / "settings.json").write_text("{}")

    (claude_real / "session-env").symlink_to(runtime / "session-env")
    (claude_real / "projects").symlink_to(runtime / "projects")
    (claude_real / "history.jsonl").symlink_to(runtime / "history.jsonl")
    (claude_real / "skills").symlink_to(config / "Shared" / "Skills")
    (claude_real / "settings.json").symlink_to(config / "Claude" / "settings.json")
    (claude_real / ".credentials.json").write_text("{}")  # real file, not a symlink

    return claude_link, runtime, config


def test_outward_symlinks_collected(tmp_path: Path) -> None:
    claude_link, runtime, config = _make_claude_like_layout(tmp_path)

    targets = _bwrap._resolve_outward_symlinks(claude_link)

    target_strs = {str(t) for t in targets}
    assert str(runtime / "session-env") in target_strs
    assert str(runtime / "projects") in target_strs
    assert str(runtime / "history.jsonl") in target_strs
    assert str(config / "Shared" / "Skills") in target_strs
    assert str(config / "Claude" / "settings.json") in target_strs


def test_real_files_and_inside_targets_skipped(tmp_path: Path) -> None:
    claude_link, _runtime, _config = _make_claude_like_layout(tmp_path)
    inside = claude_link.resolve() / "subdir"
    inside.mkdir()
    (claude_link.resolve() / "self_link").symlink_to(inside)

    targets = _bwrap._resolve_outward_symlinks(claude_link)

    target_strs = {str(t) for t in targets}
    assert str(claude_link.resolve() / ".credentials.json") not in target_strs  # real file
    assert str(inside) not in target_strs  # symlink resolves inside the bound dir


def test_dangling_symlinks_skipped(tmp_path: Path) -> None:
    claude_link, _runtime, _config = _make_claude_like_layout(tmp_path)
    (claude_link.resolve() / "broken").symlink_to(tmp_path / "does_not_exist")

    targets = _bwrap._resolve_outward_symlinks(claude_link)

    target_strs = {str(t) for t in targets}
    assert str(tmp_path / "does_not_exist") not in target_strs


def test_dedupes_targets(tmp_path: Path) -> None:
    claude_link, runtime, _config = _make_claude_like_layout(tmp_path)
    (claude_link.resolve() / "alias_session_env").symlink_to(runtime / "session-env")

    targets = _bwrap._resolve_outward_symlinks(claude_link)

    occurrences = sum(1 for t in targets if str(t) == str(runtime / "session-env"))
    assert occurrences == 1


def test_add_agent_mounts_emits_extra_binds(tmp_path: Path) -> None:
    claude_link, runtime, config = _make_claude_like_layout(tmp_path)

    profile = AgentProfile(
        name="claude",
        run_cmd=["-p"],
        format_args=[],
        model_flag="--model",
        session_flag="-r",
        continue_flag="--continue",
        data_mounts=[DataMount(str(claude_link), "/workspace/.claude")],
    )

    cmd: list[str] = []
    _bwrap._add_agent_mounts(cmd, profile, session_mode=False)

    # Original bind for /workspace/.claude.
    assert "--bind" in cmd
    assert str(claude_link) in cmd
    assert "/workspace/.claude" in cmd

    # Each outward target bound at its absolute host path inside the sandbox.
    for target in (
        runtime / "session-env",
        runtime / "projects",
        runtime / "history.jsonl",
        config / "Shared" / "Skills",
        config / "Claude" / "settings.json",
    ):
        idx = cmd.index(str(target))
        # Both source and target args at the same path — bind absolute → absolute.
        assert cmd[idx + 1] == str(target)
        assert cmd[idx - 1] == "--bind"


def test_readonly_flag_inherits_to_symlink_targets(tmp_path: Path) -> None:
    claude_link, runtime, _config = _make_claude_like_layout(tmp_path)

    profile = AgentProfile(
        name="claude",
        run_cmd=["-p"],
        format_args=[],
        model_flag="--model",
        session_flag="-r",
        continue_flag="--continue",
        data_mounts=[DataMount(str(claude_link), "/workspace/.claude", readonly=True)],
    )

    cmd: list[str] = []
    _bwrap._add_agent_mounts(cmd, profile, session_mode=False)

    assert "--bind" not in cmd
    assert "--ro-bind" in cmd
    # Symlink target is also bound read-only.
    idx = cmd.index(str(runtime / "session-env"))
    assert cmd[idx - 1] == "--ro-bind"


def test_nonexistent_mount_no_resolution(tmp_path: Path) -> None:
    profile = AgentProfile(
        name="claude",
        run_cmd=["-p"],
        format_args=[],
        model_flag="--model",
        session_flag="-r",
        continue_flag="--continue",
        data_mounts=[DataMount(str(tmp_path / "missing"), "/workspace/.claude")],
    )

    cmd: list[str] = []
    _bwrap._add_agent_mounts(cmd, profile, session_mode=False)

    assert cmd == []
