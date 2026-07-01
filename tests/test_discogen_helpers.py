"""Tests for discogen task helpers."""
from pathlib import Path

import pytest

from heuresis.tasks.discogen.helpers import (
    _filter_baselines_by_task_ids,
    build_workspace_files,
    patch_run_main_walk,
)


def test_build_workspace_files(tmp_path: Path):
    """build_workspace_files returns top-level entries minus excluded files."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "discovered").mkdir()
    (src / "MinAtar").mkdir()
    (src / "run_main.py").write_text("# runner")
    (src / "description.md").write_text("# desc")
    (src / "requirements.txt").write_text("jax")
    (src / "install.sh").write_text("#!/bin/bash")

    files = build_workspace_files(src)

    assert "discovered" in files
    assert "MinAtar" in files
    assert "run_main.py" in files
    assert "description.md" in files
    assert "requirements.txt" not in files
    assert "install.sh" not in files
    assert files["discovered"] == src / "discovered"


def test_build_workspace_files_custom_exclude(tmp_path: Path):
    """Custom exclude set overrides the default."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("a")
    (src / "b.txt").write_text("b")

    files = build_workspace_files(src, exclude={"b.txt"})
    assert "a.txt" in files
    assert "b.txt" not in files


def test_build_workspace_files_empty_dir(tmp_path: Path):
    """Empty directory returns empty dict."""
    src = tmp_path / "empty"
    src.mkdir()
    assert build_workspace_files(src) == {}


_UPSTREAM_RUN_MAIN = '''import os
import sys


def run_all_main_py(start_dir: str = "."):
    results = {}
    for root, dirs, files in os.walk(start_dir):
        dirs[:] = [d for d in dirs if d != "data"]
        if "main.py" in files:
            pass
    return results
'''


def test_patch_run_main_walk_replaces_filter(tmp_path: Path):
    """Patches the os.walk filter to also exclude hidden directories."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "run_main.py").write_text(_UPSTREAM_RUN_MAIN)

    patch_run_main_walk(src)

    patched = (src / "run_main.py").read_text()
    assert 'd != "data" and not d.startswith(".")' in patched
    assert 'd for d in dirs if d != "data"]' not in patched


def test_patch_run_main_walk_idempotent(tmp_path: Path):
    """Re-running on an already-patched file is a no-op."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "run_main.py").write_text(_UPSTREAM_RUN_MAIN)

    patch_run_main_walk(src)
    after_first = (src / "run_main.py").read_text()
    patch_run_main_walk(src)
    after_second = (src / "run_main.py").read_text()

    assert after_first == after_second


def test_patch_run_main_walk_raises_when_substring_absent(tmp_path: Path):
    """Raises if upstream format diverged so the patch can't be applied."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "run_main.py").write_text("# unrelated content with no walk\n")

    with pytest.raises(RuntimeError, match="expected substring not found"):
        patch_run_main_walk(src)


_BASELINES_FIXTURE = {
    "./MinAtar/Breakout": 69.9,
    "./MinAtar/Asterix": 17.0,
    "./Brax/Humanoid": 6309.0,
    "./Brax/Pusher": -338.5,
}


def test_filter_baselines_keeps_only_requested_task_ids():
    """Filter returns exactly the entries matching the requested ids."""
    out = _filter_baselines_by_task_ids(
        _BASELINES_FIXTURE, ["MinAtar/Breakout", "Brax/Humanoid"]
    )
    assert set(out) == {"./MinAtar/Breakout", "./Brax/Humanoid"}
    assert out["./MinAtar/Breakout"] == 69.9
    assert out["./Brax/Humanoid"] == 6309.0


def test_filter_baselines_raises_on_unknown_task_id():
    """Raises with the offending id and the available set when an id is unknown."""
    with pytest.raises(ValueError, match="not present in upstream baselines"):
        _filter_baselines_by_task_ids(
            _BASELINES_FIXTURE, ["MinAtar/Breakout", "Typo/NotADomain"]
        )


def test_filter_baselines_empty_request_returns_empty_dict():
    """An empty list returns an empty dict (no entries pass the filter)."""
    assert _filter_baselines_by_task_ids(_BASELINES_FIXTURE, []) == {}
