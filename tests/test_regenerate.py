"""Tests for experiment.regenerate helper."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from heuresis.experiment import regenerate


def _write_task(td: Path, verify: dict | None) -> None:
    import yaml
    cfg: dict = {
        "name": "fake",
        "description": "test task",
        "templates": {},
        "seed_files": [],
    }
    if verify is not None:
        cfg["verify"] = verify
    (td / "task_config.yaml").write_text(yaml.safe_dump(cfg))


def test_regenerate_returns_false_when_verify_section_missing(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    _write_task(task_dir, verify=None)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    with patch("heuresis.experiment._bwrap_run_command") as mock_run:
        ran = regenerate(task_dir, workspace, gpu_ids=[0])

    assert ran is False
    mock_run.assert_not_called()


def test_regenerate_invokes_run_command_with_parsed_verify_fields(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    _write_task(task_dir, verify={
        "command": "uv run python train.py",
        "stdout": "run.log",
        "timeout": 900,
    })
    workspace = tmp_path / "ws"
    workspace.mkdir()

    with patch("heuresis.experiment._bwrap_run_command") as mock_run:
        ran = regenerate(task_dir, workspace, gpu_ids=[3])

    assert ran is True
    mock_run.assert_called_once()
    kwargs = mock_run.call_args.kwargs
    assert kwargs["workspace"] == workspace
    assert kwargs["command"] == ["uv", "run", "python", "train.py"]
    assert kwargs["gpu_ids"] == [3]
    assert kwargs["timeout"] == 900
    assert kwargs["stdout_to"] == "regenerated/run.log"


def test_regenerate_uses_default_stdout_when_unspecified(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    _write_task(task_dir, verify={"command": "python x.py"})
    workspace = tmp_path / "ws"
    workspace.mkdir()

    with patch("heuresis.experiment._bwrap_run_command") as mock_run:
        regenerate(task_dir, workspace, gpu_ids=[])

    kwargs = mock_run.call_args.kwargs
    assert kwargs["stdout_to"] == "regenerated/run.log"
    assert kwargs["timeout"] == 1400   # default
