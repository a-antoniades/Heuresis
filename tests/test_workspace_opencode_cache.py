"""Tests for seeding the opencode auth cache into workspaces."""

from __future__ import annotations

from pathlib import Path

from heuresis import Workspace
from heuresis.agent import OPENCODE
import heuresis.workspace as workspace


def test_seed_opencode_cache_prefers_preextracted_hardlink_tree(
    tmp_path: Path,
    monkeypatch,
) -> None:
    extracted = tmp_path / "preextracted"
    package = extracted / "node_modules" / "opencode-anthropic-auth" / "package.json"
    package.parent.mkdir(parents=True)
    package.write_text('{"name":"opencode-anthropic-auth"}')
    (extracted / "package.json").write_text(
        '{"dependencies":{"opencode-anthropic-auth":"1.0.0"}}'
    )

    monkeypatch.setattr(workspace, "_OPENCODE_CACHE_DIR", extracted, raising=False)
    monkeypatch.setattr(
        workspace,
        "_OPENCODE_CACHE_TARBALL",
        tmp_path / "missing-cache.tar.gz",
    )

    ws = tmp_path / "workspace"
    workspace._seed_opencode_cache(ws)

    copied = ws / ".cache" / "opencode" / "node_modules" / "opencode-anthropic-auth" / "package.json"
    assert copied.read_text() == '{"name":"opencode-anthropic-auth"}'
    assert copied.stat().st_ino == package.stat().st_ino


def test_workspace_can_skip_opencode_cache_seeding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    extracted = tmp_path / "preextracted"
    package = extracted / "node_modules" / "opencode-anthropic-auth" / "package.json"
    package.parent.mkdir(parents=True)
    package.write_text('{"name":"opencode-anthropic-auth"}')
    (extracted / "package.json").write_text(
        '{"dependencies":{"opencode-anthropic-auth":"1.0.0"}}'
    )
    (extracted / "version").write_text("1")

    monkeypatch.setenv("QD_SEED_OPENCODE_CACHE", "0")
    monkeypatch.setattr(workspace, "_OPENCODE_CACHE_DIR", extracted, raising=False)
    monkeypatch.setattr(
        workspace,
        "_OPENCODE_CACHE_TARBALL",
        tmp_path / "missing-cache.tar.gz",
    )

    ws = tmp_path / "workspace"
    Workspace().setup(ws)

    assert (ws / ".cache").is_dir()
    assert (ws / ".cache" / "opencode" / "package.json").exists()
    assert (ws / ".cache" / "opencode" / "version").exists()
    assert not (
        ws
        / ".cache"
        / "opencode"
        / "node_modules"
        / "opencode-anthropic-auth"
        / "package.json"
    ).exists()


def test_opencode_profile_does_not_mount_shared_writable_cache() -> None:
    # Regression: a shared writable bind of ~/.cache/opencode across all
    # sandboxes causes intra-run contention at higher parallelism (8+
    # ideators). Each sandbox keeps its own per-workspace cache copy via
    # _seed_opencode_cache instead.
    assert not any(
        mount.sandbox_path == "/workspace/.cache/opencode"
        for mount in OPENCODE.data_mounts
    )
