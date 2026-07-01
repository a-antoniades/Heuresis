"""Regression tests for the project preflight shell script."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_check_script_accepts_nested_task_venv_directories(tmp_path: Path) -> None:
    """Discogen stores per-domain venvs under venvs/discogen/<DOMAIN>."""
    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "check.sh", scripts / "check.sh")

    nested_python = project / "venvs" / "discogen" / "OnPolicyRL" / "bin" / "python"
    nested_python.parent.mkdir(parents=True)
    _write_executable(nested_python, "#!/usr/bin/env bash\necho 'Python 3.11.13'\n")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "python3.11", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(fake_bin / "uv", "#!/usr/bin/env bash\necho 'uv 0.0.0'\n")
    _write_executable(fake_bin / "taskset", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(fake_bin / "opencode", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(fake_bin / "bwrap", "#!/usr/bin/env bash\necho bwrap_ok\n")

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    result = subprocess.run(
        ["bash", str(scripts / "check.sh")],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "venvs/discogen/OnPolicyRL ready (Python 3.11.13)" in result.stdout
    assert "venvs/discogen exists but has no bin/python" not in result.stdout


def test_check_script_checks_sqlite_vec_in_uv_environment(tmp_path: Path) -> None:
    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "check.sh", scripts / "check.sh")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "python3.12",
        """#!/usr/bin/env bash
if [ "$1" = "--version" ]; then
  echo "Python 3.12.0"
  exit 0
fi
if [[ "$*" == *"import sqlite_vec"* ]]; then
  exit 1
fi
exit 0
""",
    )
    _write_executable(
        fake_bin / "uv",
        """#!/usr/bin/env bash
if [ "$1" = "--version" ]; then
  echo "uv 0.0.0"
  exit 0
fi
if [ "$1" = "run" ] && [ "$2" = "python" ]; then
  exit 0
fi
exit 2
""",
    )
    _write_executable(fake_bin / "taskset", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(fake_bin / "opencode", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(fake_bin / "bwrap", "#!/usr/bin/env bash\necho bwrap_ok\n")

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    result = subprocess.run(
        ["bash", str(scripts / "check.sh")],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "sqlite-vec Python package importable in uv environment" in result.stdout
    assert "sqlite-vec not installed in this Python" not in result.stdout


def test_setup_nanogpt_creates_task_venv_when_data_is_cached(tmp_path: Path) -> None:
    project = tmp_path / "project"
    scripts = project / "scripts"
    task_dir = project / "src" / "heuresis" / "tasks" / "nanogpt"
    scripts.mkdir(parents=True)
    task_dir.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "setup.sh", scripts / "setup.sh")
    _write_executable(scripts / "check.sh", "#!/usr/bin/env bash\nexit 0\n")
    (task_dir / "requirements.txt").write_text("torch\n")
    (task_dir / "prepare.py").write_text("raise SystemExit('prepare should not run')\n")

    home = tmp_path / "home"
    cache_data = home / ".cache" / "autoresearch" / "data"
    cache_tokenizer = home / ".cache" / "autoresearch" / "tokenizer"
    cache_data.mkdir(parents=True)
    cache_tokenizer.mkdir(parents=True)
    (cache_data / "shard.parquet").write_text("cached")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "uv",
        """#!/usr/bin/env bash
if [ "$1" = "venv" ]; then
  mkdir -p "$2/bin"
  cat > "$2/bin/python" <<'PY'
#!/usr/bin/env bash
echo "Python 3.12.0"
PY
  chmod +x "$2/bin/python"
  exit 0
fi
if [ "$1" = "pip" ] && [ "$2" = "install" ]; then
  exit 0
fi
exit 2
""",
    )

    env = os.environ.copy()
    env["HOME"] = str(home)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    result = subprocess.run(
        ["bash", str(scripts / "setup.sh"), "nanogpt"],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (project / "venvs" / "nanogpt" / "bin" / "python").exists()
    assert "Creating nanogpt venv from requirements.txt" in result.stdout
    assert "Data already downloaded" in result.stdout


def test_setup_nanogpt_rebuilds_wrong_python_task_venv(tmp_path: Path) -> None:
    project = tmp_path / "project"
    scripts = project / "scripts"
    task_dir = project / "src" / "heuresis" / "tasks" / "nanogpt"
    stale_venv = project / "venvs" / "nanogpt"
    scripts.mkdir(parents=True)
    task_dir.mkdir(parents=True)
    (stale_venv / "bin").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "setup.sh", scripts / "setup.sh")
    _write_executable(scripts / "check.sh", "#!/usr/bin/env bash\nexit 0\n")
    (task_dir / "requirements.txt").write_text("torch\n")
    (task_dir / "prepare.py").write_text("raise SystemExit('prepare should not run')\n")
    _write_executable(
        stale_venv / "bin" / "python",
        "#!/usr/bin/env bash\nif [ \"$1\" = \"-c\" ]; then echo 3.11; else echo 'Python 3.11.13'; fi\n",
    )

    home = tmp_path / "home"
    cache_data = home / ".cache" / "autoresearch" / "data"
    cache_tokenizer = home / ".cache" / "autoresearch" / "tokenizer"
    cache_data.mkdir(parents=True)
    cache_tokenizer.mkdir(parents=True)
    (cache_data / "shard.parquet").write_text("cached")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "uv",
        """#!/usr/bin/env bash
if [ "$1" = "venv" ]; then
  rm -rf "$2"
  mkdir -p "$2/bin"
  cat > "$2/bin/python" <<'PY'
#!/usr/bin/env bash
if [ "$1" = "-c" ]; then echo 3.12; else echo "Python 3.12.0"; fi
PY
  chmod +x "$2/bin/python"
  exit 0
fi
if [ "$1" = "pip" ] && [ "$2" = "install" ]; then
  exit 0
fi
exit 2
""",
    )

    env = os.environ.copy()
    env["HOME"] = str(home)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    result = subprocess.run(
        ["bash", str(scripts / "setup.sh"), "nanogpt"],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Nanogpt venv uses Python 3.11; rebuilding with Python 3.12" in result.stdout
    assert "Data already downloaded" in result.stdout


def test_setup_rebuilds_wrong_python_base_venv(tmp_path: Path) -> None:
    project = tmp_path / "project"
    scripts = project / "scripts"
    stale_venv = project / "venvs" / "base"
    scripts.mkdir(parents=True)
    (stale_venv / "bin").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "setup.sh", scripts / "setup.sh")
    _write_executable(scripts / "check.sh", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        stale_venv / "bin" / "python",
        "#!/usr/bin/env bash\nif [ \"$1\" = \"-c\" ]; then echo 3.11; else echo 'Python 3.11.13'; fi\n",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "uv",
        """#!/usr/bin/env bash
if [ "$1" = "venv" ]; then
  rm -rf "$2"
  mkdir -p "$2/bin"
  cat > "$2/bin/python" <<'PY'
#!/usr/bin/env bash
if [ "$1" = "-c" ]; then echo 3.12; else echo "Python 3.12.0"; fi
PY
  chmod +x "$2/bin/python"
  exit 0
fi
if [ "$1" = "pip" ] && [ "$2" = "install" ]; then
  exit 0
fi
exit 2
""",
    )

    env = os.environ.copy()
    env["HOME"] = str(tmp_path / "home")
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    result = subprocess.run(
        ["bash", str(scripts / "setup.sh")],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Base venv uses Python 3.11; rebuilding with Python 3.12" in result.stdout
    assert "Base venv created" in result.stdout
