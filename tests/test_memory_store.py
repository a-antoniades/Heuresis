"""Tests for MemoryStore — schema, ingest_experiment, and socket ops.

The store is exercised in-process via its public API (no subprocess, no
real Gemini calls). A fake deterministic embedder produces stable
DIM-dim vectors (whatever DIM currently is) so search semantics are
checkable.
"""
from __future__ import annotations

import hashlib
import json
import socket
import struct
import tempfile
from pathlib import Path


from heuresis.memory.embeddings import DIM
from heuresis.memory.store import MemoryStore, short_socket_path


# -- helpers ----------------------------------------------------------------


class FakeEmbedder:
    """Deterministic embedder at the production DIM. Same text -> same vector.

    Uses shake_256 (a SHA-3 extendable-output hash) to produce exactly
    DIM/8 bytes of deterministic "entropy" per text, then interprets
    the bits as a 0/1 vector. Coarse by design — same text will
    vec-match exactly, different texts get measurable distance.
    """

    model = "fake-memory-v1"
    dim = DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        h = hashlib.shake_256(text.encode("utf-8")).digest(DIM // 8)
        bits: list[float] = []
        for byte in h:
            for i in range(8):
                bits.append(1.0 if (byte >> i) & 1 else 0.0)
        assert len(bits) == DIM
        return bits


def _store(tmp_path: Path) -> MemoryStore:
    """Fresh MemoryStore wired to a fake embedder (no network)."""
    db = tmp_path / "memory.db"
    return MemoryStore(db, embedder=FakeEmbedder())


def _rpc(sock_path: Path, req: dict) -> dict:
    """Call the running MemoryStore over its Unix socket."""
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


# -- socket path -------------------------------------------------------------


def test_short_socket_path_is_deterministic_and_short(tmp_path: Path):
    p = short_socket_path(tmp_path)
    # Lives in the system temp dir (usually /tmp; respects $TMPDIR), named
    # qd-memory-<hash>.sock. Don't hardcode /tmp — CI/sandboxes set $TMPDIR.
    assert p.parent == Path(tempfile.gettempdir())
    assert p.name.startswith("qd-memory-") and p.suffix == ".sock"
    assert len(str(p)) < 108  # Unix socket path hard limit
    # Same anchor -> same socket. Different anchor -> different socket.
    assert short_socket_path(tmp_path) == p
    assert short_socket_path(tmp_path / "other") != p


# -- schema + ingest --------------------------------------------------------


def test_schema_is_created_on_init(tmp_path: Path):
    """All tables and views should exist right after MemoryStore() returns."""
    store = _store(tmp_path)
    conn = store._connect(readonly=True)
    try:
        names = {
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type IN ('table', 'view')"
            ).fetchall()
        }
    finally:
        conn.close()
    # Base tables + views + vec shadow tables
    assert "experiments" in names
    assert "learnings" in names
    assert "memory_experiments_v" in names
    assert "memory_learnings_v" in names
    # vec0 creates a top-level table with the given name
    assert "experiments_vec" in names
    assert "learnings_vec" in names


def test_wal_mode_enabled(tmp_path: Path):
    """WAL is what makes N-readers / 1-writer viable under load."""
    store = _store(tmp_path)
    conn = store._connect()
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()
    assert mode.lower() == "wal"


def test_ingest_experiment_writes_row_and_vec(tmp_path: Path):
    store = _store(tmp_path)
    store.ingest_experiment(
        ideator_id="abc111", executor_id="xyz222",
        valid=True, score=1.5,
        features={"axis_a": 0.42}, parent_ids=["p1"],
        generation=3, idea_md="trust region with restart",
        notes_md=None,
    )
    conn = store._connect(readonly=True)
    try:
        row = conn.execute(
            "SELECT ideator_id, executor_id, valid, score, features_json,"
            " parent_ids_json, generation, idea_md FROM experiments"
        ).fetchone()
        vec_count = conn.execute(
            "SELECT COUNT(*) FROM experiments_vec "
            "WHERE executor_id = ?", ("xyz222",)
        ).fetchone()[0]
    finally:
        conn.close()

    assert row["ideator_id"] == "abc111"
    assert row["executor_id"] == "xyz222"
    assert row["valid"] == 1
    assert row["score"] == 1.5
    assert json.loads(row["features_json"]) == {"axis_a": 0.42}
    assert json.loads(row["parent_ids_json"]) == ["p1"]
    assert row["generation"] == 3
    assert row["idea_md"] == "trust region with restart"
    assert vec_count == 1


def test_ingest_experiment_notes_change_embedding(tmp_path: Path):
    """idea+notes should embed differently than idea alone (same executor)."""
    store = _store(tmp_path)

    store.ingest_experiment(
        ideator_id="i1", executor_id="e1",
        valid=True, score=1.0, features=None, parent_ids=None,
        generation=0, idea_md="idea text", notes_md=None,
    )
    store.ingest_experiment(
        ideator_id="i1", executor_id="e2",
        valid=True, score=1.0, features=None, parent_ids=None,
        generation=0, idea_md="idea text", notes_md="NOTES: x was surprising",
    )
    # A fake-embedder search for the bare "idea text" should prefer e1.
    query_vec = FakeEmbedder().embed_one("idea text")
    from heuresis.memory.store import _vec_blob
    conn = store._connect(readonly=True)
    try:
        row = conn.execute(
            """
            SELECT executor_id FROM (
                SELECT executor_id, distance
                FROM experiments_vec
                WHERE embedding MATCH ? AND k = 1
            )
            """,
            (_vec_blob(query_vec),),
        ).fetchone()
    finally:
        conn.close()
    assert row["executor_id"] == "e1"


def test_ingest_experiment_reingest_is_idempotent_on_executor_id(tmp_path: Path):
    """Re-ingesting the same executor should overwrite, not duplicate."""
    store = _store(tmp_path)
    for score in (1.0, 2.0):
        store.ingest_experiment(
            ideator_id="i1", executor_id="e1",
            valid=True, score=score, features=None, parent_ids=None,
            generation=0, idea_md="idea", notes_md=None,
        )
    conn = store._connect(readonly=True)
    try:
        row = conn.execute("SELECT COUNT(*) c, MAX(score) s FROM experiments").fetchone()
        vec = conn.execute(
            "SELECT COUNT(*) FROM experiments_vec WHERE executor_id = 'e1'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert row["c"] == 1
    assert row["s"] == 2.0
    assert vec == 1


# -- socket: append ---------------------------------------------------------


def test_append_writes_learning_and_vec(tmp_path: Path):
    store = _store(tmp_path)
    with store:
        resp = _rpc(store.socket_path, {
            "op": "append",
            "content": "Nelder-Mead collapses on Rastrigin.",
            "tags": ["nelder-mead", "rastrigin"],
            "related": ["exec-x"],
            "author_id": "abc111",
            "author_role": "executor",
        })
    assert resp.get("ok") is True
    learning_id = resp["learning_id"]

    conn = store._connect(readonly=True)
    try:
        row = conn.execute(
            "SELECT author_id, author_role, tags_json, related_executor_ids_json, content"
            " FROM learnings WHERE learning_id = ?", (learning_id,),
        ).fetchone()
        vec_count = conn.execute(
            "SELECT COUNT(*) FROM learnings_vec WHERE learning_id = ?",
            (learning_id,),
        ).fetchone()[0]
    finally:
        conn.close()

    assert row["author_id"] == "abc111"
    assert row["author_role"] == "executor"
    assert json.loads(row["tags_json"]) == ["nelder-mead", "rastrigin"]
    assert json.loads(row["related_executor_ids_json"]) == ["exec-x"]
    assert "Nelder-Mead" in row["content"]
    assert vec_count == 1


def test_append_rejects_empty_content(tmp_path: Path):
    store = _store(tmp_path)
    with store:
        resp = _rpc(store.socket_path, {
            "op": "append", "content": "  ",
            "author_id": "a", "author_role": "ideator",
        })
    assert resp.get("ok") is False
    assert "content" in resp.get("error", "").lower()


def test_append_rejects_bad_role(tmp_path: Path):
    store = _store(tmp_path)
    with store:
        resp = _rpc(store.socket_path, {
            "op": "append", "content": "x",
            "author_id": "a", "author_role": "admin",
        })
    assert resp.get("ok") is False
    assert "role" in resp.get("error", "").lower()


def test_append_validates_tags_type(tmp_path: Path):
    store = _store(tmp_path)
    with store:
        resp = _rpc(store.socket_path, {
            "op": "append", "content": "x",
            "tags": "not-a-list",
            "author_id": "a", "author_role": "ideator",
        })
    assert resp.get("ok") is False


# -- socket: search ---------------------------------------------------------


def test_search_experiments_returns_nearest(tmp_path: Path):
    store = _store(tmp_path)
    store.ingest_experiment(
        ideator_id="i1", executor_id="closer",
        valid=True, score=1.0, features=None, parent_ids=None,
        generation=0, idea_md="apple banana cherry", notes_md=None,
    )
    store.ingest_experiment(
        ideator_id="i1", executor_id="farther",
        valid=True, score=1.0, features=None, parent_ids=None,
        generation=0, idea_md="totally unrelated text", notes_md=None,
    )
    with store:
        resp = _rpc(store.socket_path, {
            "op": "search", "query": "apple banana cherry",
            "table": "experiments", "k": 2,
        })
    assert resp.get("ok") is True
    rows = resp["rows"]
    assert len(rows) == 2
    assert rows[0]["executor_id"] == "closer"
    # First result should include ideator_id from the join + score
    assert rows[0]["ideator_id"] == "i1"
    assert rows[0]["score"] == 1.0


def test_search_learnings_returns_nearest(tmp_path: Path):
    store = _store(tmp_path)
    with store:
        a = _rpc(store.socket_path, {
            "op": "append", "content": "simplex collapses at 200 FEvals",
            "author_id": "a1", "author_role": "executor",
        })["learning_id"]
        _rpc(store.socket_path, {
            "op": "append", "content": "unrelated text",
            "author_id": "a1", "author_role": "executor",
        })
        resp = _rpc(store.socket_path, {
            "op": "search", "query": "simplex collapses at 200 FEvals",
            "table": "learnings", "k": 1,
        })
    assert resp.get("ok") is True
    rows = resp["rows"]
    assert len(rows) == 1
    assert rows[0]["learning_id"] == a
    assert rows[0]["author_role"] == "executor"


def test_search_rejects_bad_table(tmp_path: Path):
    store = _store(tmp_path)
    with store:
        resp = _rpc(store.socket_path, {
            "op": "search", "query": "x", "table": "SECRETS",
        })
    assert resp.get("ok") is False


def test_search_requires_query(tmp_path: Path):
    store = _store(tmp_path)
    with store:
        resp = _rpc(store.socket_path, {
            "op": "search", "query": "  ", "table": "experiments",
        })
    assert resp.get("ok") is False


def test_search_k_is_clamped(tmp_path: Path):
    """Very large k should not crash; store clamps to a reasonable max."""
    store = _store(tmp_path)
    store.ingest_experiment(
        ideator_id="i1", executor_id="e1",
        valid=True, score=1.0, features=None, parent_ids=None,
        generation=0, idea_md="thing", notes_md=None,
    )
    with store:
        resp = _rpc(store.socket_path, {
            "op": "search", "query": "thing",
            "table": "experiments", "k": 9999,
        })
    assert resp.get("ok") is True


# -- socket: read -----------------------------------------------------------


def test_read_allows_select_over_views(tmp_path: Path):
    store = _store(tmp_path)
    store.ingest_experiment(
        ideator_id="i1", executor_id="e1",
        valid=True, score=3.14, features=None, parent_ids=None,
        generation=0, idea_md="idea", notes_md=None,
    )
    with store:
        resp = _rpc(store.socket_path, {
            "op": "read",
            "sql": "SELECT executor_id, score FROM memory_experiments_v",
        })
    assert resp.get("ok") is True
    assert resp["rows"] == [{"executor_id": "e1", "score": 3.14}]


def test_read_rejects_non_select(tmp_path: Path):
    store = _store(tmp_path)
    with store:
        for bad in ("DELETE FROM experiments",
                    "UPDATE experiments SET score=0",
                    "DROP TABLE experiments",
                    "INSERT INTO learnings (ts, author_id, author_role, content) VALUES (0, 'x', 'ideator', 'y')"):
            resp = _rpc(store.socket_path, {"op": "read", "sql": bad})
            assert resp.get("ok") is False, f"should reject: {bad}"


def test_read_rejects_underlying_table_access(tmp_path: Path):
    """Agents must only name the views, never the underlying tables."""
    store = _store(tmp_path)
    with store:
        for bad in (
            "SELECT * FROM experiments",
            "SELECT * FROM learnings",
            "SELECT * FROM experiments_vec",
        ):
            resp = _rpc(store.socket_path, {"op": "read", "sql": bad})
            assert resp.get("ok") is False, f"should reject: {bad}"


def test_read_rejects_multiple_statements(tmp_path: Path):
    store = _store(tmp_path)
    with store:
        resp = _rpc(store.socket_path, {
            "op": "read",
            "sql": "SELECT 1 FROM memory_experiments_v; DROP TABLE experiments",
        })
    assert resp.get("ok") is False


def test_read_allows_with_cte(tmp_path: Path):
    """CTEs are read-only and useful for aggregations."""
    store = _store(tmp_path)
    store.ingest_experiment(
        ideator_id="i1", executor_id="e1",
        valid=True, score=1.0, features=None, parent_ids=None,
        generation=0, idea_md="a", notes_md=None,
    )
    with store:
        resp = _rpc(store.socket_path, {
            "op": "read",
            "sql": (
                "WITH scored AS (SELECT executor_id, score "
                "FROM memory_experiments_v WHERE valid = 1) "
                "SELECT COUNT(*) AS n FROM scored"
            ),
        })
    assert resp.get("ok") is True
    assert resp["rows"] == [{"n": 1}]


# -- socket: error surfaces -------------------------------------------------


def test_bad_op_returns_error(tmp_path: Path):
    store = _store(tmp_path)
    with store:
        resp = _rpc(store.socket_path, {"op": "NONSENSE"})
    assert resp.get("ok") is False
    assert "unknown op" in resp.get("error", "").lower()


def test_malformed_json_returns_error(tmp_path: Path):
    store = _store(tmp_path)
    with store:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(str(store.socket_path))
        bad = b"not-json-at-all"
        sock.sendall(struct.pack("!I", len(bad)) + bad)
        header = b""
        while len(header) < 4:
            header += sock.recv(4 - len(header))
        (length,) = struct.unpack("!I", header)
        payload = b""
        while len(payload) < length:
            payload += sock.recv(length - len(payload))
        sock.close()
    resp = json.loads(payload)
    assert resp.get("ok") is False
    assert "json" in resp.get("error", "").lower()


# -- lifecycle --------------------------------------------------------------


def test_context_manager_cleans_up_socket(tmp_path: Path):
    store = _store(tmp_path)
    with store:
        assert store.socket_path.exists()
    # After exit, the socket file is removed.
    assert not store.socket_path.exists()
