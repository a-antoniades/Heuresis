"""Verify install_args flows through Workspace -> ensure_venv."""

from unittest.mock import patch
from pathlib import Path

from heuresis.workspace import Workspace, ensure_venv


def test_ensure_venv_passes_install_args(tmp_path: Path):
    """Extra install args should be appended to the uv pip install command."""
    venv_path = tmp_path / "venv"
    req = tmp_path / "requirements.txt"
    req.write_text("jax==0.6.2\n")

    with patch("heuresis.workspace.subprocess.run") as mock_run, \
         patch("heuresis.workspace.shutil.which", return_value="/usr/bin/uv"):
        ensure_venv(venv_path, req, install_args=["--prerelease", "allow"])

    pip_call = mock_run.call_args_list[1]
    assert "--prerelease" in pip_call[0][0]
    assert "allow" in pip_call[0][0]


def test_ensure_venv_empty_install_args(tmp_path: Path):
    """Empty install_args should not change the command."""
    venv_path = tmp_path / "venv"
    req = tmp_path / "requirements.txt"
    req.write_text("requests\n")

    with patch("heuresis.workspace.subprocess.run") as mock_run, \
         patch("heuresis.workspace.shutil.which", return_value="/usr/bin/uv"):
        ensure_venv(venv_path, req, install_args=[])

    pip_call = mock_run.call_args_list[1]
    cmd = pip_call[0][0]
    assert "--prerelease" not in cmd


def test_ensure_venv_can_install_project_extra(tmp_path: Path):
    """Project extras should install from pyproject instead of a requirements file."""
    venv_path = tmp_path / "venv"

    with patch("heuresis.workspace.subprocess.run") as mock_run, \
         patch("heuresis.workspace.shutil.which", return_value="/usr/bin/uv"):
        ensure_venv(venv_path, project_extra="sandbox")

    pip_call = mock_run.call_args_list[1]
    cmd = pip_call[0][0]
    assert cmd[:3] == ["/usr/bin/uv", "pip", "install"]
    assert mock_run.call_args_list[0][0][0][-1] == "python3.12"
    assert cmd[3].endswith("[sandbox]")
    assert "-r" not in cmd


def test_workspace_install_args_default():
    """Workspace should default install_args to empty list."""
    ws = Workspace()
    assert ws.install_args == []
