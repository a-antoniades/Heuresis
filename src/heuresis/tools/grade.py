#!/usr/bin/env python3
"""Universal grading tool.

Sends submission files to the host-side GradingServer over a Unix socket
and prints the JSON result. The agent runs ``grade <files>`` and gets back
a score — it never needs to know how grading works internally.

Each call creates a timestamped attempt directory under ``attempts/`` with
the submitted files and the returned score.

Protocol: sends a JSON payload with all files (base64-encoded, keyed by
filename) in a single request. The server decides which files matter
and how to grade them.
"""

import argparse
import base64
import hashlib
import json
import os
import shutil
import socket
import struct
import sys
import tempfile
import time
from pathlib import Path

_ATTEMPTS_DIR = "attempts"
_SANDBOX_SOCKET = Path("/workspace/.grade.sock")
_MARKER = Path("/workspace/.grade_socket_path")


def _resolve_socket_path() -> Path:
    """Resolve the grading socket in priority order.

    Inside a bwrap sandbox, ``/tmp`` is shadowed by ``--tmpfs /tmp`` so the
    host path recorded in ``.grade_socket_path`` is unreachable. The
    harness bind-mounts the real socket at ``/workspace/.grade.sock``
    via ``_mount_grade_socket`` — check that first.

    Priority:
      1. ``/workspace/.grade.sock`` (sandbox bind mount — primary path)
      2. ``$GRADE_SOCKET`` env var
      3. ``/workspace/.grade_socket_path`` marker (host path, works only
         when running on the host, not inside the sandbox)
      4. ``/tmp/qd-grade-<sha16-of-WORKSPACE_PATH>.sock`` fallback
    """
    if _SANDBOX_SOCKET.exists():
        return _SANDBOX_SOCKET
    env = os.environ.get("GRADE_SOCKET")
    if env:
        return Path(env)
    if _MARKER.exists():
        try:
            return Path(_MARKER.read_text().strip())
        except OSError:
            pass
    ws = os.environ.get("WORKSPACE_PATH", "/workspace")
    h = hashlib.sha256(ws.encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"qd-grade-{h}.sock"


def _next_attempt_dir() -> Path:
    base = Path(_ATTEMPTS_DIR)
    base.mkdir(exist_ok=True)
    from datetime import datetime
    name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    attempt_dir = base / name
    attempt_dir.mkdir()
    return attempt_dir


def _send_msg(sock: socket.socket, data: bytes) -> None:
    sock.sendall(struct.pack("!I", len(data)) + data)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks: list[bytes] = []
    while n > 0:
        chunk = sock.recv(min(n, 65536))
        if not chunk:
            raise ConnectionError("Server closed connection")
        chunks.append(chunk)
        n -= len(chunk)
    return b"".join(chunks)


def _recv_msg(sock: socket.socket) -> bytes:
    header = _recv_exact(sock, 4)
    (length,) = struct.unpack("!I", header)
    return _recv_exact(sock, length)


def main(argv: list[str] | None = None) -> int:
    """Submit files for grading and print the JSON result."""
    parser = argparse.ArgumentParser(
        prog="grade",
        description="Submit files to the grading server and print the JSON result.",
    )
    parser.add_argument("files", nargs="*",
                        help="Files to submit (default: submission.csv).")
    args = parser.parse_args(argv)

    files = tuple(args.files) or ("submission.csv",)
    file_paths = [Path(f) for f in files]
    missing = [str(f) for f in file_paths if not f.exists()]
    if missing:
        result = {"score": None, "valid": False,
                  "details": {"error": f"Files not found: {missing}"}}
        print(json.dumps(result))
        return 1

    payload = {
        "files": {
            fp.name: base64.b64encode(fp.read_bytes()).decode()
            for fp in file_paths
        }
    }

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(str(_resolve_socket_path()))
        _send_msg(sock, json.dumps(payload).encode())
        response = _recv_msg(sock)
        sock.close()
        result = json.loads(response)
    except Exception as e:
        result = {"score": None, "valid": False,
                  "details": {"error": str(e)}}

    print(json.dumps(result))

    try:
        attempt_dir = _next_attempt_dir()
        for fp in file_paths:
            shutil.copy2(fp, attempt_dir / fp.name)
        result_with_meta = {
            **result,
            "timestamp": time.time(),
            "files": [str(p) for p in file_paths],
        }
        (attempt_dir / "result.json").write_text(json.dumps(result_with_meta, indent=2))
    except Exception:
        pass

    return 0 if result.get("valid") else 1


if __name__ == "__main__":
    sys.exit(main())
