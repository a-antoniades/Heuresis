"""Host-side grading server for secure in-sandbox scoring.

Runs outside the bwrap sandbox and communicates with the in-sandbox
``grade`` tool via a Unix socket.  The test answers never enter the
sandbox — only the JSON score is returned.

Protocol (length-prefixed binary over Unix socket):
    Request:  [4-byte big-endian length][JSON payload]
              payload = {"files": {"filename": "<base64 bytes>", ...}}
    Response: [4-byte big-endian length][JSON result]
              result = {"score": ..., "valid": ..., "details": {...}}

Task-specific grading is implemented by subclassing ``GradingServer``
and overriding the ``grade()`` method.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import socket
import struct
import tempfile
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_HEADER_FMT = "!I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


def short_socket_path(workspace: Path) -> Path:
    """Return a short socket path under /tmp to avoid the 108-char Unix socket limit."""
    h = hashlib.sha256(str(workspace.resolve()).encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"qd-grade-{h}.sock"


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


class GradingServer(ABC):
    """Unix-socket grading server.

    Listens for submissions from the in-sandbox ``grade`` tool.
    Each request is a JSON payload containing base64-encoded files.
    The server decodes them and calls ``grade(files)`` which
    task-specific subclasses implement.

    Subclasses declare ``input_files`` to enable host-side fallback
    scoring when the agent dies without calling the grade tool. The
    orchestration layer (``heuresis.experiment.execute``) reads
    those files from the workspace and re-invokes ``grade`` directly.

    Usage::

        with MyGradingServer(socket_path) as server:
            harness.run(...)
    """

    #: Files the grader consumes. When non-empty, ``execute()`` will
    #: read them from the workspace as a fallback if the agent never
    #: invoked the in-sandbox grade tool.
    input_files: list[str] = []

    def __init__(self, socket_path: Path) -> None:
        self.socket_path = Path(socket_path)
        self._server_socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @abstractmethod
    def grade(self, files: dict[str, bytes]) -> dict[str, Any]:
        """Grade submitted files. Return {"score": ..., "valid": ..., "details": {...}}.

        ``files`` is a dict mapping filename to raw bytes.
        Task-specific subclasses implement this with their own logic.
        """
        ...

    def start(self) -> None:
        # Always use /tmp socket — survives agent deleting workspace contents,
        # avoids 108-char Unix socket path limit.
        actual_path = short_socket_path(self.socket_path.parent)
        self._actual_socket_path = actual_path

        if actual_path.exists():
            actual_path.unlink()

        self._server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_socket.bind(str(actual_path))
        self._server_socket.listen(4)
        self._server_socket.settimeout(1.0)

        marker = self.socket_path.parent / ".grade_socket_path"
        try:
            marker.write_text(str(actual_path))
        except OSError:
            pass

        # NOTE: We intentionally do NOT set os.environ["GRADE_SOCKET"] here.
        # With multiple parallel grading servers (one per executor thread),
        # concurrent start() calls would race on the global env, and each
        # harness.run() could snapshot the wrong socket path. The in-sandbox
        # `grade` tool resolves via the per-workspace .grade_socket_path
        # marker (and hash-of-workspace-path as a final fallback).

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        logger.info("GradingServer started on %s", actual_path)

    def stop(self) -> None:
        self._stop_event.set()
        if self._server_socket:
            try:
                self._server_socket.close()
            except OSError:
                pass
        if self._thread:
            self._thread.join(timeout=5)
        actual = getattr(self, "_actual_socket_path", self.socket_path)
        for p in (actual, self.socket_path):
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass
        marker = self.socket_path.parent / ".grade_socket_path"
        if marker.exists():
            try:
                marker.unlink()
            except OSError:
                pass
        # No env var to unset; see start() note on the race we're avoiding.
        logger.info("GradingServer stopped")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    def _serve(self) -> None:
        while not self._stop_event.is_set():
            try:
                conn, _ = self._server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                self._handle_connection(conn)
            except Exception:
                logger.exception("Error handling grading request")
            finally:
                conn.close()

    def _handle_connection(self, conn: socket.socket) -> None:
        raw = _recv_msg(conn)

        try:
            payload = json.loads(raw)
            encoded_files = payload.get("files", {})
            files = {
                name: base64.b64decode(data)
                for name, data in encoded_files.items()
            }
        except (json.JSONDecodeError, KeyError, Exception) as e:
            result = {"score": None, "valid": False,
                      "details": {"error": f"Invalid payload: {e}"}}
            _send_msg(conn, json.dumps(result).encode())
            return

        try:
            result = self.grade(files)
        except Exception as e:
            logger.exception("Grading failed")
            result = {"score": None, "valid": False,
                      "details": {"error": str(e)}}

        _send_msg(conn, json.dumps(result).encode())
