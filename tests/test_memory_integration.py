"""End-to-end integration test for the memory primitive.

Wires together the real production classes and drives them the way
``heuresis.loops.run_linear`` does for the bbob task:

- Real ``Workspace`` + ``Workspace.setup()`` — markers on disk, venv links,
  tool installation
- Real ``ResultStore`` + ``Experiment`` — SQLite store.db
- Real ``MemoryStore`` — SQLite memory.db, Unix-socket server thread,
  real sqlite-vec
- Real ``record_run()`` — the framework orchestrator

The only stand-in is a ``FakeEmbedder`` injected into MemoryStore, which
skips the Gemini API but produces real DIM-wide vectors that round-trip
through sqlite-vec. Everything else is exactly what runs in production.

This guards the full pipeline from "agent exits" to "rows queryable in
the memory DB".
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import struct
import subprocess
import sys
from pathlib import Path


from heuresis import Workspace
from heuresis.experiment import record_run
from heuresis.memory import MemoryStore
from heuresis.memory.embeddings import DIM
from heuresis.models import RunResult
from heuresis.store import ResultStore
from heuresis.tools.defaults import MEMORY


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "src" / "heuresis" / "tools" / "memory.py"


class _FakeEmbedder:
    model = "fake-memory-v1"
    dim = DIM

    def embed(self, texts):
        return [self.embed_one(t) for t in texts]

    def embed_one(self, text):
        h = hashlib.shake_256(text.encode()).digest(DIM // 8)
        bits: list[float] = []
        for byte in h:
            for i in range(8):
                bits.append(1.0 if (byte >> i) & 1 else 0.0)
        return bits


def _make_empty_venv(path: Path) -> None:
    """Minimal fake venv so Workspace._link_venv is happy."""
    (path / "bin").mkdir(parents=True, exist_ok=True)
    (path / "bin" / "python").write_text("#!/bin/sh\nexec /usr/bin/env python3 \"$@\"\n")
    (path / "bin" / "python").chmod(0o755)


def _memory_rpc(sock_path: Path, req: dict) -> dict:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(str(sock_path))
    try:
        data = json.dumps(req).encode()
        sock.sendall(struct.pack("!I", len(data)) + data)
        header = b""
        while len(header) < 4:
            header += sock.recv(4 - len(header))
        (length,) = struct.unpack("!I", header)
        payload = b""
        while len(payload) < length:
            payload += sock.recv(length - len(payload))
        return json.loads(payload)
    finally:
        sock.close()


def test_end_to_end_real_experiment_real_memory(tmp_path: Path):
    """Simulate one iteration of the bbob_linear loop with all real classes.

    What we verify:

    1. Workspace.setup() writes real .workspace_id / .workspace_role /
       .memory_socket_path markers.
    2. ResultStore records the run in store.db (normal framework path).
    3. record_run() passes ``memory`` through, MemoryStore writes the row
       into memory.db.
    4. The resulting experiments row carries the *real* workspace UUIDs
       we read from disk — not dummy strings.
    5. A subsequent semantic search from a third-party workspace (the
       ideator, a distinct .workspace_id) returns that row, exactly as
       an ideator would see it on its next turn.
    """
    # --- Set up real directories the way a real campaign does ------------
    exp_root = tmp_path / "runs"
    store = ResultStore(db_path=exp_root / "store.db")
    exp = store.experiment("memory-integration", root=exp_root, task="test")

    ideator_dir = exp.dir / "ideator_0"
    exec_dir = exp.dir / "exec_001"

    # Fake venv so Workspace._link_venv doesn't warn.
    fake_venv = tmp_path / "venv"
    _make_empty_venv(fake_venv)

    ideator_ws = Workspace(
        tools=[MEMORY], files={}, prompt="", venv=fake_venv,
        role="ideator",
    )
    executor_ws = Workspace(
        tools=[MEMORY], files={}, prompt="", venv=fake_venv,
        role="executor",
    )

    # --- Start a real MemoryStore with a fake embedder -------------------
    with MemoryStore(exp.dir / "memory.db", embedder=_FakeEmbedder()) as memory:
        ideator_ws.memory_socket = memory.socket_path
        executor_ws.memory_socket = memory.socket_path

        # --- Real Workspace.setup (writes markers, links venv, etc.) ----
        ideator_ws.setup(ideator_dir)
        executor_ws.setup(exec_dir)

        # Sanity: markers are real and on disk.
        ideator_id = (ideator_dir / ".workspace_id").read_text().strip()
        executor_id = (exec_dir / ".workspace_id").read_text().strip()
        assert len(ideator_id) == 12
        assert len(executor_id) == 12
        assert ideator_id != executor_id
        assert (ideator_dir / ".workspace_role").read_text() == "ideator"
        assert (exec_dir / ".workspace_role").read_text() == "executor"
        assert (ideator_dir / ".memory_socket_path").read_text() == str(memory.socket_path)

        # Simulate the executor writing a retrospective note —
        # record_run() picks this up and folds it into the embedding.
        (exec_dir / "notes.md").write_text(
            "Halved the trust region on failure; mean_log_gap improved by 0.3."
        )

        # --- Drive record_run the same way bbob_linear body() does ------
        result = RunResult(workspace=exec_dir, exit_code=0, stats={"duration": 42.0})
        idea = "Trust-region method with adaptive shrinkage on Rastrigin."
        record_run(
            exp, "exec_001",
            result=result,
            info={"valid": True, "best_score": 1.23},
            strategy_meta={"generation": 0, "features": None},
            iteration=0, run_type="executor",
            idea=idea, parent_ids=None,
            memory=memory, ideator_workspace=ideator_dir,
        )

        # --- 1. Real ResultStore persisted the run ----------------------
        runs = exp.runs(run_type="executor")
        assert len(runs) == 1
        assert runs[0].run_id == "exec_001"
        assert runs[0].score == 1.23
        assert runs[0].valid is True

        # --- 2. Real memory.db has the experiment row with REAL UUIDs ---
        conn = memory._connect(readonly=True)
        try:
            row = conn.execute(
                "SELECT ideator_id, executor_id, score, idea_md "
                "FROM experiments"
            ).fetchone()
            vec_count = conn.execute(
                "SELECT COUNT(*) FROM experiments_vec WHERE executor_id = ?",
                (executor_id,),
            ).fetchone()[0]
        finally:
            conn.close()

        assert row["ideator_id"] == ideator_id, (
            "memory.experiments.ideator_id should match the real .workspace_id "
            "on the ideator dir, not a dummy string"
        )
        assert row["executor_id"] == executor_id
        assert row["score"] == 1.23
        assert row["idea_md"] == idea
        assert vec_count == 1

        # --- 3. From the ideator's sandbox perspective, the row is found
        #     via semantic search. Uses the real socket path the CLI would
        #     resolve from .memory_socket_path.
        env = os.environ.copy()
        env["MEMORY_SOCKET"] = str(
            Path((ideator_dir / ".memory_socket_path").read_text())
        )
        env["WORKSPACE_ID"] = ideator_id
        env["WORKSPACE_ROLE"] = "ideator"

        # Exact-idea query should find our just-ingested row first.
        cli = subprocess.run(
            [sys.executable, str(CLI), "search", idea,
             "--table", "experiments", "--k", "3"],
            env=env, capture_output=True, text=True, timeout=30,
        )
        assert cli.returncode == 0, cli.stderr
        body = json.loads(cli.stdout)
        assert body["ok"] is True
        assert body["rows"][0]["executor_id"] == executor_id
        assert body["rows"][0]["ideator_id"] == ideator_id

        # --- 4. From the executor perspective, append a learning; verify
        #     it's attributed to the REAL executor workspace id.
        env["WORKSPACE_ID"] = executor_id
        env["WORKSPACE_ROLE"] = "executor"
        cli = subprocess.run(
            [sys.executable, str(CLI),
             "append", "Rastrigin is multimodal; single-point optimizers stall.",
             "--tags", "rastrigin,multimodal"],
            env=env, capture_output=True, text=True, timeout=30,
        )
        assert cli.returncode == 0, cli.stderr
        learning_id = json.loads(cli.stdout)["learning_id"]

        conn = memory._connect(readonly=True)
        try:
            learning = conn.execute(
                "SELECT author_id, author_role FROM learnings "
                "WHERE learning_id = ?", (learning_id,),
            ).fetchone()
        finally:
            conn.close()
        assert learning["author_id"] == executor_id, (
            "CLI must stamp author_id from the real on-disk workspace uuid"
        )
        assert learning["author_role"] == "executor"

    # --- 5. On `with` exit, the real socket file is cleaned up ---------
    assert not memory.socket_path.exists()

    # --- 6. memory.db survives (it's persistent, not socket-lifetime) ---
    db = exp.dir / "memory.db"
    assert db.exists()
    # And the row is still there after shutdown (WAL checkpoint on close).
    import sqlite3
    conn = sqlite3.connect(str(db))
    try:
        conn.row_factory = sqlite3.Row
        final = conn.execute(
            "SELECT ideator_id, executor_id FROM experiments"
        ).fetchone()
    finally:
        conn.close()
    assert final["ideator_id"] == ideator_id
    assert final["executor_id"] == executor_id
