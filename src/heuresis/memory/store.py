"""MemoryStore: host-side Unix-socket server backed by SQLite + sqlite-vec.

One instance per campaign. Started once at the top of ``run.py``
(typically as a ``with`` block wrapping ``parallel_ideators``), stopped
once at the end. Owns:

- the per-campaign SQLite database (``exp.dir/memory.db``)
- the Gemini embedding client
- a Unix-socket server used by the in-sandbox ``memory`` CLI

The server speaks the same length-prefixed JSON protocol as
:class:`heuresis.grading.GradingServer`. Three operations are
exposed:

- ``append``   — write a row to ``learnings``, embed its content
- ``search``   — semantic vector search over ``experiments`` or ``learnings``
- ``read``     — execute a read-only ``SELECT`` against the two views

Framework-side writes (the experiments table) go through
:meth:`ingest_experiment` directly, satisfying
:class:`heuresis.memory.protocol.MemoryIngest`. Agent-side writes
go through the socket.
"""

from __future__ import annotations

import hashlib
import json
import logging
import socket
import sqlite3
import struct
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from heuresis.memory.embeddings import GeminiMemoryEmbedder

logger = logging.getLogger(__name__)

_HEADER_FMT = "!I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_SCHEMA_SQL = (Path(__file__).parent / "schema.sql").read_text()

# Read-only view names — the only tables the `read` op is allowed to name.
_READ_VIEWS = {"memory_experiments_v", "memory_learnings_v"}


def short_socket_path(anchor: Path) -> Path:
    """Short socket path under /tmp to avoid the 108-char Unix-socket limit.

    Uses the same sha16-of-resolved-path scheme as the grading server so
    the two primitives are easy to debug side-by-side.
    """
    h = hashlib.sha256(str(anchor.resolve()).encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"qd-memory-{h}.sock"


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = conn.recv(min(remaining, 65536))
        if not chunk:
            raise ConnectionError("Connection closed before all data received")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_msg(conn: socket.socket, data: bytes) -> None:
    conn.sendall(struct.pack(_HEADER_FMT, len(data)) + data)


def _recv_msg(conn: socket.socket) -> bytes:
    header = _recv_exact(conn, _HEADER_SIZE)
    (length,) = struct.unpack(_HEADER_FMT, header)
    return _recv_exact(conn, length)


def _vec_blob(values: list[float]) -> bytes:
    """Pack a float vector for sqlite-vec (little-endian float32)."""
    import struct as _s
    return _s.pack(f"<{len(values)}f", *values)


class MemoryStore:
    """Unix-socket memory server.

    Lifecycle::

        with MemoryStore(exp.dir / "memory.db") as memory:
            ideator_ws.memory_socket = memory.socket_path
            executor_ws.memory_socket = memory.socket_path
            parallel_ideators(harnesses, body)

    Thread-safety:
      - One server thread accepts connections one at a time.
      - A single ``_write_lock`` serializes all writes (experiments_vec,
        learnings, learnings_vec). SQLite WAL mode still permits
        concurrent readers.
      - ``ingest_experiment`` (called from host-side ``record_run``) also
        takes ``_write_lock``, so it's safe to interleave with agent
        appends arriving over the socket.
    """

    def __init__(
        self,
        db_path: Path,
        *,
        embedder: GeminiMemoryEmbedder | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path = short_socket_path(self.db_path.parent)

        self._embedder = embedder  # lazy: built on first use if None
        self._write_lock = threading.Lock()
        self._server_socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._init_schema()

    # -- schema + connection helpers --------------------------------------

    def _connect(self, *, readonly: bool = False) -> sqlite3.Connection:
        if readonly:
            uri = f"file:{self.db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=10.0)
        else:
            conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.enable_load_extension(True)
        import sqlite_vec  # type: ignore[import-not-found]
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(_SCHEMA_SQL)

    @property
    def embedder(self) -> GeminiMemoryEmbedder:
        if self._embedder is None:
            self._embedder = GeminiMemoryEmbedder()
        return self._embedder

    # -- framework-facing: ingest_experiment -------------------------------

    def ingest_experiment(
        self,
        *,
        ideator_id: str,
        executor_id: str,
        valid: bool,
        score: float | None,
        features: dict[str, Any] | None,
        parent_ids: list[str] | None,
        generation: int,
        idea_md: str,
        notes_md: str | None,
    ) -> None:
        """Insert one ``experiments`` row + its embedding.

        Called from :func:`heuresis.experiment.record_run` when both
        ``memory`` and ``ideator_workspace`` are passed. Embeddings are
        computed synchronously — the run is "done" only after memory has
        accepted the row. Failures raise and get logged by the caller.
        """
        text_for_embed = idea_md
        if notes_md:
            text_for_embed = f"{idea_md}\n\nNOTES:\n{notes_md}"
        vec = self.embedder.embed_one(text_for_embed)

        with self._write_lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO experiments
                    (ideator_id, executor_id, ts, valid, score,
                     features_json, parent_ids_json, generation, idea_md)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ideator_id, executor_id, time.time(),
                        1 if valid else 0,
                        score,
                        json.dumps(features) if features else None,
                        json.dumps(parent_ids) if parent_ids else None,
                        generation,
                        idea_md,
                    ),
                )
                # experiments_vec has executor_id as PK; re-insert overwrites
                # cleanly on retry / ingest-after-resume.
                conn.execute(
                    "DELETE FROM experiments_vec WHERE executor_id = ?",
                    (executor_id,),
                )
                conn.execute(
                    "INSERT INTO experiments_vec (executor_id, ideator_id, embedding) "
                    "VALUES (?, ?, ?)",
                    (executor_id, ideator_id, _vec_blob(vec)),
                )

    # -- server lifecycle --------------------------------------------------

    def start(self) -> None:
        if self.socket_path.exists():
            self.socket_path.unlink()
        self._server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_socket.bind(str(self.socket_path))
        self._server_socket.listen(8)
        self._server_socket.settimeout(1.0)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        logger.info("MemoryStore started at %s (db=%s)", self.socket_path, self.db_path)

    def stop(self) -> None:
        self._stop_event.set()
        if self._server_socket:
            try:
                self._server_socket.close()
            except OSError:
                pass
        if self._thread:
            self._thread.join(timeout=5)
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError:
                pass
        logger.info("MemoryStore stopped")

    def __enter__(self) -> "MemoryStore":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.stop()
        return False

    # -- server internals --------------------------------------------------

    def _serve(self) -> None:
        assert self._server_socket is not None
        while not self._stop_event.is_set():
            try:
                conn, _ = self._server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                self._handle(conn)
            except Exception:
                logger.exception("MemoryStore: error handling request")
            finally:
                conn.close()

    def _handle(self, conn: socket.socket) -> None:
        raw = _recv_msg(conn)
        try:
            req = json.loads(raw)
        except json.JSONDecodeError as exc:
            _send_msg(conn, json.dumps({"ok": False, "error": f"bad json: {exc}"}).encode())
            return

        op = req.get("op")
        try:
            if op == "append":
                resp = self._op_append(req)
            elif op == "search":
                resp = self._op_search(req)
            elif op == "read":
                resp = self._op_read(req)
            else:
                resp = {"ok": False, "error": f"unknown op: {op!r}"}
        except Exception as exc:
            logger.exception("MemoryStore op %s failed", op)
            resp = {"ok": False, "error": str(exc)}

        _send_msg(conn, json.dumps(resp).encode())

    # -- op handlers -------------------------------------------------------

    def _op_append(self, req: dict[str, Any]) -> dict[str, Any]:
        content = req.get("content")
        if not isinstance(content, str) or not content.strip():
            return {"ok": False, "error": "append requires non-empty 'content'"}
        author_id = req.get("author_id") or ""
        author_role = req.get("author_role") or "executor"
        if author_role not in {"ideator", "executor"}:
            return {"ok": False, "error": f"bad author_role: {author_role!r}"}
        tags = req.get("tags") or []
        related = req.get("related") or []
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            return {"ok": False, "error": "tags must be list[str]"}
        if not isinstance(related, list) or not all(isinstance(r, str) for r in related):
            return {"ok": False, "error": "related must be list[str]"}

        # Embed content + tags together (tags add signal to the embedding).
        to_embed = content
        if tags:
            to_embed = f"{content}\ntags: {', '.join(tags)}"
        vec = self.embedder.embed_one(to_embed)

        with self._write_lock:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO learnings
                    (ts, author_id, author_role, tags_json,
                     related_executor_ids_json, content)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        time.time(), author_id, author_role,
                        json.dumps(tags) if tags else None,
                        json.dumps(related) if related else None,
                        content,
                    ),
                )
                learning_id = cur.lastrowid
                conn.execute(
                    "INSERT INTO learnings_vec (learning_id, embedding) VALUES (?, ?)",
                    (learning_id, _vec_blob(vec)),
                )

        return {"ok": True, "learning_id": learning_id}

    def _op_search(self, req: dict[str, Any]) -> dict[str, Any]:
        query = req.get("query")
        if not isinstance(query, str) or not query.strip():
            return {"ok": False, "error": "search requires 'query'"}
        table = req.get("table", "experiments")
        k = int(req.get("k", 5))
        k = max(1, min(k, 50))

        vec = self.embedder.embed_one(query)
        vec_blob = _vec_blob(vec)

        with self._connect(readonly=True) as conn:
            if table == "experiments":
                # sqlite-vec: the KNN query must live in its own (sub)query,
                # with `MATCH ?` and `k = ?` as the only filters on vec0
                # rows. We then join to the base table on the PK outside.
                sql = """
                    SELECT vq.executor_id, vq.distance,
                           e.ideator_id, e.ts, e.valid, e.score,
                           e.features_json, e.parent_ids_json, e.generation,
                           e.idea_md
                    FROM (
                        SELECT executor_id, distance
                        FROM experiments_vec
                        WHERE embedding MATCH ? AND k = ?
                    ) AS vq
                    JOIN experiments e ON e.executor_id = vq.executor_id
                    ORDER BY vq.distance
                """
                rows = conn.execute(sql, (vec_blob, k)).fetchall()
            elif table == "learnings":
                sql = """
                    SELECT vq.learning_id, vq.distance,
                           l.ts, l.author_id, l.author_role,
                           l.tags_json, l.related_executor_ids_json, l.content
                    FROM (
                        SELECT learning_id, distance
                        FROM learnings_vec
                        WHERE embedding MATCH ? AND k = ?
                    ) AS vq
                    JOIN learnings l ON l.learning_id = vq.learning_id
                    ORDER BY vq.distance
                """
                rows = conn.execute(sql, (vec_blob, k)).fetchall()
            else:
                return {"ok": False, "error": f"table must be 'experiments' or 'learnings' (got {table!r})"}

        return {"ok": True, "rows": [dict(r) for r in rows]}

    def _op_read(self, req: dict[str, Any]) -> dict[str, Any]:
        sql = req.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            return {"ok": False, "error": "read requires 'sql'"}
        stripped = sql.strip().rstrip(";").lstrip()
        lowered = stripped.lower()
        if not (lowered.startswith("select") or lowered.startswith("with")):
            return {"ok": False, "error": "read only accepts SELECT / WITH statements"}
        if ";" in stripped:
            return {"ok": False, "error": "only a single statement is allowed"}

        # Cheap allow-list: the string must name at least one of the views.
        # Combined with the read-only connection below, this blocks agents
        # from reading underlying tables or calling SQL functions that
        # leak schema metadata.
        if not any(v in lowered for v in _READ_VIEWS):
            return {"ok": False, "error": f"query must reference one of {_READ_VIEWS}"}

        with self._connect(readonly=True) as conn:
            try:
                rows = conn.execute(stripped).fetchall()
            except sqlite3.Error as exc:
                return {"ok": False, "error": f"sql error: {exc}"}

        return {"ok": True, "rows": [dict(r) for r in rows]}
