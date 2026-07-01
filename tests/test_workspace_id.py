"""Tests for the workspace-identity primitive.

`.workspace_id`, `.workspace_role`, and `.memory_socket_path` markers are
all written by Workspace.setup(). These are the contract the memory CLI
and harness depend on, so we cover them explicitly.
"""
from __future__ import annotations

import re
from pathlib import Path

from heuresis.workspace import Workspace


# -- .workspace_id -----------------------------------------------------------


def test_setup_writes_workspace_id(tmp_path: Path):
    """Fresh setup must produce a 12-char hex UUID marker."""
    Workspace().setup(tmp_path)
    marker = tmp_path / ".workspace_id"
    assert marker.exists()
    wsid = marker.read_text()
    assert len(wsid) == 12
    assert re.fullmatch(r"[0-9a-f]{12}", wsid), wsid


def test_setup_is_idempotent_on_workspace_id(tmp_path: Path):
    """Re-setup on the same dir must preserve the existing id.

    This is the whole point of making the id file-backed: stateful
    ideator workspaces get the same identity across iterations.
    """
    ws = Workspace()
    ws.setup(tmp_path)
    first = (tmp_path / ".workspace_id").read_text()

    ws.setup(tmp_path)
    second = (tmp_path / ".workspace_id").read_text()
    assert first == second


def test_setup_respects_pre_existing_workspace_id(tmp_path: Path):
    """A pre-populated .workspace_id (e.g. from resume) must survive setup."""
    (tmp_path / ".workspace_id").write_text("deadbeef1234")
    Workspace().setup(tmp_path)
    assert (tmp_path / ".workspace_id").read_text() == "deadbeef1234"


def test_distinct_workspaces_get_distinct_ids(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    Workspace().setup(a)
    Workspace().setup(b)
    assert (a / ".workspace_id").read_text() != (b / ".workspace_id").read_text()


# -- .workspace_role ---------------------------------------------------------


def test_setup_writes_role_when_provided(tmp_path: Path):
    Workspace(role="ideator").setup(tmp_path)
    assert (tmp_path / ".workspace_role").read_text() == "ideator"


def test_setup_writes_executor_role(tmp_path: Path):
    Workspace(role="executor").setup(tmp_path)
    assert (tmp_path / ".workspace_role").read_text() == "executor"


def test_setup_skips_role_marker_when_unset(tmp_path: Path):
    Workspace().setup(tmp_path)
    assert not (tmp_path / ".workspace_role").exists()


def test_setup_overwrites_role_on_re_setup(tmp_path: Path):
    """If someone swaps the role (rare, but possible), setup should follow."""
    Workspace(role="ideator").setup(tmp_path)
    assert (tmp_path / ".workspace_role").read_text() == "ideator"
    Workspace(role="executor").setup(tmp_path)
    assert (tmp_path / ".workspace_role").read_text() == "executor"


# -- .memory_socket_path -----------------------------------------------------


def test_setup_writes_memory_socket_marker(tmp_path: Path):
    sock = Path("/tmp/fake-memory.sock")
    Workspace(memory_socket=sock).setup(tmp_path)
    assert (tmp_path / ".memory_socket_path").read_text() == str(sock)


def test_setup_skips_memory_socket_marker_when_unset(tmp_path: Path):
    Workspace().setup(tmp_path)
    assert not (tmp_path / ".memory_socket_path").exists()


# -- defaults ---------------------------------------------------------------


def test_workspace_defaults_have_no_identity_fields():
    """By default, role and memory_socket are None — opt-in per experiment."""
    ws = Workspace()
    assert ws.role is None
    assert ws.memory_socket is None
