"""Tests for _bwrap.run_command (no-agent sandbox execution)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from heuresis._bwrap import SandboxResult, run_command


def test_sandbox_result_has_expected_fields() -> None:
    r = SandboxResult(exit_code=0, duration_s=1.5)
    assert r.exit_code == 0
    assert r.duration_s == 1.5


def test_run_command_builds_bwrap_cmd_and_redirects_stdout(tmp_path: Path) -> None:
    """run_command must invoke build_command and redirect stdout to the
    workspace-relative path provided via stdout_to."""
    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured_cmd: list[list[str]] = []

    def fake_run_subprocess(cmd: list[str], *, stdout_file: Path, timeout: int | None) -> tuple[float, int]:
        captured_cmd.append(cmd)
        # Simulate redirect: touch the stdout file to prove the wrapper wires it.
        stdout_file.parent.mkdir(parents=True, exist_ok=True)
        stdout_file.write_text("fake output\n")
        return 0.25, 0

    with patch("heuresis._bwrap._run_subprocess", side_effect=fake_run_subprocess):
        result = run_command(
            workspace=workspace,
            command=["echo", "hello"],
            stdout_to="sub/out.log",
            timeout=10,
        )

    assert isinstance(result, SandboxResult)
    assert result.exit_code == 0
    assert result.duration_s == 0.25
    assert (workspace / "sub" / "out.log").read_text() == "fake output\n"
    # bwrap prefix present, inner command at the end.
    assert captured_cmd[0][0] == "bwrap"
    assert captured_cmd[0][-2:] == ["echo", "hello"]


def test_run_command_without_stdout_to_still_runs(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()

    def fake_run_subprocess(cmd: list[str], *, stdout_file: Path, timeout: int | None) -> tuple[float, int]:
        # stdout_file is /dev/null-equivalent when stdout_to is None
        assert stdout_file == Path("/dev/null")
        return 0.1, 0

    with patch("heuresis._bwrap._run_subprocess", side_effect=fake_run_subprocess):
        result = run_command(
            workspace=workspace,
            command=["true"],
            timeout=5,
        )
    assert result.exit_code == 0
