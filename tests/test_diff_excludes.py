"""Diff-exclude regression tests.

The hacker judge runs `diff -ur --exclude-from=<file>` between the
baseline and the executor workspace. Any inert workspace marker that
shows up only on the executor side (because the baseline is generated
without `Workspace.setup`) becomes a noisy diff line that can confuse
the judge. `_DIFF_EXCLUDES` is the universal exclude list that hides
those markers.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from heuresis.judge.hacker import _DIFF_EXCLUDES


def test_system_tools_json_is_excluded():
    """`.system_tools.json` is written by Workspace._install_tools when a
    system-installed tool is present (e.g. the MEMORY tool). It must not
    appear in the judge diff or the judge will see "Only in run:
    .system_tools.json" on every MEMORY-enabled discogen run.
    """
    assert ".system_tools.json" in _DIFF_EXCLUDES.splitlines()


def test_diff_excludes_filters_system_tools_json(tmp_path: Path):
    """End-to-end: `diff -ur --exclude-from=<excludes>` does not surface
    `.system_tools.json` when only the run side has it.
    """
    baseline = tmp_path / "baseline"
    run = tmp_path / "run"
    baseline.mkdir()
    run.mkdir()

    # Baseline lacks the marker; run has it (mirrors a discogen executor
    # workspace built with Workspace.setup() and a system-installed tool).
    (run / ".system_tools.json").write_text('{"memory": "/opt/qd/bin/memory"}\n')

    excludes_file = tmp_path / "excludes"
    excludes_file.write_text(_DIFF_EXCLUDES + "\n")

    proc = subprocess.run(
        [
            "diff",
            "-ur",
            f"--exclude-from={excludes_file}",
            str(baseline),
            str(run),
        ],
        capture_output=True,
        text=True,
    )
    assert ".system_tools.json" not in proc.stdout
    assert ".system_tools.json" not in proc.stderr
