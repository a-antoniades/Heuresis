"""Tests for ``system_install`` tools.

These tools (e.g. the shared ``memory`` command) must NOT be copied into
``/workspace/.bin/<name>``. Instead Workspace.setup writes a sidecar
``.system_tools.json`` that the harness uses to bind-mount the tool at a
system-looking path (``/opt/qd/bin/<name>``) outside the workspace.

Covered:
- regular tools still land in ``.bin/``
- ``system_install=True`` tools do NOT end up in ``.bin/``
- ``.system_tools.json`` lists the right tools with absolute host paths
- sidecar is removed when no system tools remain (e.g. memory disabled)
"""
from __future__ import annotations

import json
import stat
from pathlib import Path

from heuresis.tool import Tool
from heuresis.workspace import Workspace


def _mk_tool(tmp_path: Path, name: str, *, system: bool = False) -> Tool:
    binary = tmp_path / f"{name}.py"
    binary.write_text("#!/usr/bin/env python3\nprint('hi')\n")
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return Tool(name=name, binary=binary, docs="x", system_install=system)


def test_regular_tool_copied_to_bin(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    ws_path = tmp_path / "ws"
    tool = _mk_tool(src, "mytool", system=False)
    Workspace(tools=[tool]).setup(ws_path)
    assert (ws_path / ".bin" / "mytool").is_file()


def test_system_install_tool_not_copied(tmp_path):
    """The whole point: no plain-text copy inside the workspace."""
    src = tmp_path / "src"
    src.mkdir()
    ws_path = tmp_path / "ws"
    tool = _mk_tool(src, "memory", system=True)
    Workspace(tools=[tool]).setup(ws_path)
    assert not (ws_path / ".bin" / "memory").exists()


def test_system_install_tool_recorded_in_sidecar(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    ws_path = tmp_path / "ws"
    tool = _mk_tool(src, "memory", system=True)
    Workspace(tools=[tool]).setup(ws_path)

    sidecar = ws_path / ".system_tools.json"
    assert sidecar.is_file()
    mapping = json.loads(sidecar.read_text())
    assert set(mapping.keys()) == {"memory"}
    assert Path(mapping["memory"]).is_file()
    assert mapping["memory"] == str(tool.binary.resolve())


def test_sidecar_omits_regular_tools(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    ws_path = tmp_path / "ws"
    regular = _mk_tool(src, "regular", system=False)
    sysi = _mk_tool(src, "sysi", system=True)
    Workspace(tools=[regular, sysi]).setup(ws_path)

    mapping = json.loads((ws_path / ".system_tools.json").read_text())
    assert set(mapping.keys()) == {"sysi"}
    assert (ws_path / ".bin" / "regular").is_file()


def test_sidecar_absent_when_no_system_tools(tmp_path):
    ws_path = tmp_path / "ws"
    Workspace().setup(ws_path)
    assert not (ws_path / ".system_tools.json").exists()


def test_resetup_clears_sidecar_when_tools_drop(tmp_path):
    """Experiment toggles memory off -> next setup mustn't leave a stale sidecar."""
    src = tmp_path / "src"
    src.mkdir()
    ws_path = tmp_path / "ws"

    sysi = _mk_tool(src, "sysi", system=True)
    Workspace(tools=[sysi]).setup(ws_path)
    assert (ws_path / ".system_tools.json").exists()

    Workspace(tools=[]).setup(ws_path)
    assert not (ws_path / ".system_tools.json").exists()


def test_default_memory_tool_is_system_installed():
    """Regression guard: ``MEMORY`` flips from workspace-install to
    system-install. Tests + prompts + bwrap rely on this."""
    from heuresis.tools.defaults import MEMORY
    assert MEMORY.system_install is True
