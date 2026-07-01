"""End-to-end tests for the in-sandbox memory CLI.

These run a real MemoryStore (with a fake embedder so no network) and
subprocess-invoke the CLI the same way an in-sandbox agent would:
passing WORKSPACE_ID / WORKSPACE_ROLE / MEMORY_SOCKET via env.

Contracts covered:

- append stamps author_id from the env marker.
- append with no WORKSPACE_ID is refused client-side (exit 1).
- search / read flow through to the store and return structured JSON.
- read rejects underlying-table access at the server level (defence
  in depth: even if the CLI lets it through, the store doesn't).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from heuresis.memory.embeddings import DIM
from heuresis.memory.store import MemoryStore


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "src" / "heuresis" / "tools" / "memory.py"


class _Fake:
    model = "fake"
    dim = DIM

    def embed(self, texts):
        return [self.embed_one(t) for t in texts]

    def embed_one(self, text):
        # shake_256 yields arbitrary-length deterministic output.
        h = hashlib.shake_256(text.encode()).digest(DIM // 8)
        bits: list[float] = []
        for b in h:
            for i in range(8):
                bits.append(1.0 if (b >> i) & 1 else 0.0)
        return bits


def _run_cli(env: dict, *args: str) -> subprocess.CompletedProcess:
    base = os.environ.copy()
    base.update(env)
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        env=base, capture_output=True, text=True, timeout=60,
    )


def _store_env(store: MemoryStore, workspace_id: str, role: str) -> dict:
    return {
        "MEMORY_SOCKET": str(store.socket_path),
        "WORKSPACE_ID": workspace_id,
        "WORKSPACE_ROLE": role,
    }


@pytest.fixture
def running_store(tmp_path):
    store = MemoryStore(tmp_path / "memory.db", embedder=_Fake())
    store.start()
    try:
        yield store
    finally:
        store.stop()


# -- append -----------------------------------------------------------------


def test_cli_append_succeeds(running_store):
    proc = _run_cli(
        _store_env(running_store, "aaaaaaaaaaaa", "ideator"),
        "append", "simplex collapses on rastrigin",
        "--tags", "rastrigin,nelder-mead",
    )
    assert proc.returncode == 0, proc.stderr
    body = json.loads(proc.stdout.strip())
    assert body["ok"] is True
    assert isinstance(body["learning_id"], int)


def test_cli_append_stamps_author_from_env(running_store):
    _run_cli(
        _store_env(running_store, "id0000000000", "executor"),
        "append", "executor note",
    )
    # Query the DB directly to confirm the row carries the env author.
    conn = running_store._connect(readonly=True)
    try:
        row = conn.execute(
            "SELECT author_id, author_role FROM learnings ORDER BY learning_id DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    assert row["author_id"] == "id0000000000"
    assert row["author_role"] == "executor"


def test_cli_append_missing_workspace_id_fails(running_store):
    """No env var, no marker in /workspace -> exit 1 with clear error."""
    env = {"MEMORY_SOCKET": str(running_store.socket_path)}
    proc = _run_cli(env, "append", "hi")
    assert proc.returncode == 1
    body = json.loads(proc.stdout.strip())
    assert body["ok"] is False
    assert "workspace_id" in body["error"].lower()


def test_cli_append_role_defaults_to_executor(running_store):
    """If WORKSPACE_ROLE is unset and no marker, the role defaults to executor.

    This prevents agents from impersonating an ideator by omitting the hint.
    """
    env = {
        "MEMORY_SOCKET": str(running_store.socket_path),
        "WORKSPACE_ID": "id0000000001",
    }
    proc = _run_cli(env, "append", "defaulted-role")
    assert proc.returncode == 0, proc.stderr
    conn = running_store._connect(readonly=True)
    try:
        row = conn.execute(
            "SELECT author_role FROM learnings ORDER BY learning_id DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    assert row["author_role"] == "executor"


# -- search -----------------------------------------------------------------


def test_cli_search_experiments(running_store):
    running_store.ingest_experiment(
        ideator_id="i1", executor_id="e1",
        valid=True, score=1.0, features=None, parent_ids=None,
        generation=0, idea_md="apple banana cherry", notes_md=None,
    )
    proc = _run_cli(
        _store_env(running_store, "i1", "ideator"),
        "search", "apple banana cherry", "--table", "experiments", "--k", "1",
    )
    assert proc.returncode == 0, proc.stderr
    body = json.loads(proc.stdout.strip())
    assert body["ok"] is True
    rows = body["rows"]
    assert rows and rows[0]["executor_id"] == "e1"


def test_cli_search_learnings(running_store):
    env = _store_env(running_store, "author-a", "executor")
    append_body = json.loads(_run_cli(env, "append", "unique marker text").stdout)
    lid = append_body["learning_id"]

    proc = _run_cli(env, "search", "unique marker text",
                    "--table", "learnings", "--k", "1")
    assert proc.returncode == 0, proc.stderr
    body = json.loads(proc.stdout.strip())
    assert body["rows"][0]["learning_id"] == lid


# -- read -------------------------------------------------------------------


def test_cli_read_select(running_store):
    running_store.ingest_experiment(
        ideator_id="ia", executor_id="ea",
        valid=True, score=2.0, features=None, parent_ids=None,
        generation=0, idea_md="x", notes_md=None,
    )
    proc = _run_cli(
        _store_env(running_store, "ia", "ideator"),
        "read", "SELECT executor_id, score FROM memory_experiments_v",
    )
    assert proc.returncode == 0, proc.stderr
    body = json.loads(proc.stdout.strip())
    assert body["ok"] is True
    assert body["rows"] == [{"executor_id": "ea", "score": 2.0}]


def test_cli_read_rejects_underlying_table(running_store):
    proc = _run_cli(
        _store_env(running_store, "x", "ideator"),
        "read", "SELECT * FROM experiments",
    )
    # Server rejects; CLI exits 1.
    assert proc.returncode == 1
    body = json.loads(proc.stdout.strip())
    assert body["ok"] is False


def test_cli_read_rejects_delete(running_store):
    proc = _run_cli(
        _store_env(running_store, "x", "ideator"),
        "read", "DELETE FROM experiments",
    )
    assert proc.returncode == 1
    body = json.loads(proc.stdout.strip())
    assert body["ok"] is False
