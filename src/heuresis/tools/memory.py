#!/usr/bin/env python3
"""Universal memory CLI — append / search / read.

Stdlib-only. Speaks the length-prefixed JSON protocol to the host-side
:class:`heuresis.memory.MemoryStore` over a Unix socket.

Identity markers (written by :meth:`heuresis.workspace.Workspace.setup`):

- ``/workspace/.workspace_id``   -> author_id (required for append)
- ``/workspace/.workspace_role`` -> author_role (default "executor")

Agents never type their own ID.

Socket resolution priority:

1. ``/workspace/.memory.sock``   (sandbox bind mount — primary path)
2. ``$MEMORY_SOCKET`` env var
3. ``/workspace/.memory_socket_path`` marker (host path — CLI from host)
4. ``/tmp/qd-memory-<sha16-of-WORKSPACE_PATH>.sock`` final fallback
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import struct
import sys
import tempfile
from pathlib import Path

_SANDBOX_SOCKET = Path("/workspace/.memory.sock")
_SOCKET_MARKER = Path("/workspace/.memory_socket_path")
_ID_MARKER = Path("/workspace/.workspace_id")
_ROLE_MARKER = Path("/workspace/.workspace_role")


def _resolve_socket_path() -> Path:
    if _SANDBOX_SOCKET.exists():
        return _SANDBOX_SOCKET
    env = os.environ.get("MEMORY_SOCKET")
    if env:
        return Path(env)
    if _SOCKET_MARKER.exists():
        try:
            return Path(_SOCKET_MARKER.read_text().strip())
        except OSError:
            pass
    ws = os.environ.get("WORKSPACE_PATH", "/workspace")
    h = hashlib.sha256(ws.encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"qd-memory-{h}.sock"


def _resolve_author() -> tuple[str, str]:
    author_id = os.environ.get("WORKSPACE_ID", "")
    if not author_id and _ID_MARKER.exists():
        try:
            author_id = _ID_MARKER.read_text().strip()
        except OSError:
            pass
    role = os.environ.get("WORKSPACE_ROLE", "")
    if not role and _ROLE_MARKER.exists():
        try:
            role = _ROLE_MARKER.read_text().strip()
        except OSError:
            pass
    role = role or "executor"
    return author_id, role


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


def _rpc(request: dict) -> dict:
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(str(_resolve_socket_path()))
        _send_msg(sock, json.dumps(request).encode())
        resp = _recv_msg(sock)
        sock.close()
        return json.loads(resp)
    except Exception as exc:
        return {"ok": False, "error": f"memory rpc failed: {exc}"}


def _split_csv(raw: str) -> list[str]:
    return [t.strip() for t in raw.split(",") if t.strip()]


def _cmd_append(args: argparse.Namespace) -> int:
    author_id, role = _resolve_author()
    if not author_id:
        print(json.dumps({"ok": False, "error": "no .workspace_id marker found"}))
        return 1
    result = _rpc({
        "op": "append",
        "content": args.content,
        "tags": _split_csv(args.tags),
        "related": _split_csv(args.related),
        "author_id": author_id,
        "author_role": role,
    })
    print(json.dumps(result))
    return 0 if result.get("ok") else 1


def _cmd_search(args: argparse.Namespace) -> int:
    result = _rpc({
        "op": "search",
        "query": args.query,
        "table": args.table,
        "k": args.k,
    })
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


def _cmd_read(args: argparse.Namespace) -> int:
    result = _rpc({"op": "read", "sql": args.sql})
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memory",
        description="Shared campaign memory primitive.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    ap = sub.add_parser("append", help="Record a learning (id/role auto-stamped).")
    ap.add_argument("content")
    ap.add_argument("--tags", default="", help="Comma-separated tags.")
    ap.add_argument("--related", default="", help="Comma-separated executor ids.")
    ap.set_defaults(func=_cmd_append)

    sp = sub.add_parser("search", help="Semantic search over a table.")
    sp.add_argument("query")
    sp.add_argument("--table", choices=["experiments", "learnings"], default="experiments")
    sp.add_argument("--k", type=int, default=5)
    sp.set_defaults(func=_cmd_search)

    rp = sub.add_parser("read", help="SQL SELECT/WITH over memory_*_v views.")
    rp.add_argument("sql")
    rp.set_defaults(func=_cmd_read)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
