"""Tests for the harness -> sandbox identity plumbing.

Covers how Harness._execute reads ``.workspace_id`` / ``.workspace_role``
and threads them into:
  - ``env_passthrough`` (so bwrap lets those names into the sandbox)
  - ``_run_subprocess(extra_env=...)`` (so the names have values)

We mock bwrap / subprocess so no real agent runs, and intercept
``_run_subprocess`` to capture the final arguments.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from heuresis.harness import Harness
from heuresis.workspace import Workspace


def _run_harness(tmp_path: Path, workspace: Workspace) -> dict:
    """Drive Harness._execute through mocks. Returns the captured kwargs."""
    harness = Harness("opencode", model="fake-model")
    captured: dict = {}

    def _fake_subprocess(cmd, *, workspace, timeout, extra_env=None):
        captured["cmd"] = cmd
        captured["workspace"] = workspace
        captured["extra_env"] = extra_env or {}
        # Capture env_passthrough out of the fake bwrap call too.
        return 0.01, 0, Path(workspace) / "agent.log"

    # _run_subprocess writes a log file to the workspace — short-circuit it.
    with patch.object(Harness, "_run_subprocess", side_effect=_fake_subprocess), \
         patch("heuresis.harness._bwrap.build_command") as bwrap_mock, \
         patch("heuresis.harness._limits.wrap_command",
               side_effect=lambda cmd, **_: cmd):
        bwrap_mock.return_value = ["echo", "fake-bwrap"]
        harness._execute(
            workspace=workspace,
            prompt="noop",
            mounts=[],
            stateful=False,
            timeout=None,
            path=tmp_path / "run_dir",
        )
        captured["bwrap_kwargs"] = bwrap_mock.call_args.kwargs
    return captured


def test_workspace_id_env_passthrough_for_default_workspace(tmp_path: Path):
    captured = _run_harness(tmp_path, Workspace())
    env_passthrough = captured["bwrap_kwargs"]["env_passthrough"]
    assert "WORKSPACE_PATH" in env_passthrough
    assert "WORKSPACE_ID" in env_passthrough
    # No role marker -> no WORKSPACE_ROLE pass-through.
    assert "WORKSPACE_ROLE" not in env_passthrough


def test_workspace_id_value_threaded_into_extra_env(tmp_path: Path):
    captured = _run_harness(tmp_path, Workspace())
    extra = captured["extra_env"]
    assert "WORKSPACE_ID" in extra
    # Must match the marker on disk.
    wsid = (captured["workspace"] / ".workspace_id").read_text().strip()
    assert extra["WORKSPACE_ID"] == wsid
    assert len(wsid) == 12


def test_role_marker_enables_role_env_passthrough(tmp_path: Path):
    captured = _run_harness(tmp_path, Workspace(role="ideator"))
    env_passthrough = captured["bwrap_kwargs"]["env_passthrough"]
    assert "WORKSPACE_ROLE" in env_passthrough
    assert captured["extra_env"]["WORKSPACE_ROLE"] == "ideator"


def test_executor_role_propagated(tmp_path: Path):
    captured = _run_harness(tmp_path, Workspace(role="executor"))
    assert captured["extra_env"]["WORKSPACE_ROLE"] == "executor"


def test_role_env_absent_when_role_unset(tmp_path: Path):
    captured = _run_harness(tmp_path, Workspace())
    assert "WORKSPACE_ROLE" not in captured["extra_env"]


def test_workspace_id_is_stable_across_reruns(tmp_path: Path):
    """Two _execute calls against the same path (stateful ideator pattern)
    must report the same WORKSPACE_ID."""
    ws = Workspace()
    a = _run_harness(tmp_path, ws)["extra_env"]["WORKSPACE_ID"]
    b = _run_harness(tmp_path, ws)["extra_env"]["WORKSPACE_ID"]
    assert a == b
