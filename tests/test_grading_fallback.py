"""Tests for grade-tool socket resolution and GradingServer /tmp socket usage.

Task 11 of foundation-refactor adoption:
- GradingServer always uses /tmp/qd-grade-<hash>.sock (rm-safe)
- tools/grade.py _resolve_socket_path() uses 3-level fallback:
    1. $GRADE_SOCKET env var
    2. /workspace/.grade_socket_path marker file
    3. hash-from-$WORKSPACE_PATH
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from unittest.mock import patch



class TestGradingServerAlwaysUsesTmpSocket:
    """GradingServer.start() must always use /tmp regardless of workspace path length."""

    def _make_concrete_server(self, socket_path: Path):
        from heuresis.grading import GradingServer

        class _TestServer(GradingServer):
            def grade(self, files):
                return {"score": 1.0, "valid": True, "details": {}}

        return _TestServer(socket_path=socket_path)

    def test_grading_server_always_uses_tmp_socket_short_path(self, tmp_path):
        """Even with a short workspace path, socket must land in /tmp."""
        sock_path = tmp_path / ".grade.sock"
        server = self._make_concrete_server(sock_path)
        server.start()
        try:
            actual = server._actual_socket_path
            assert str(actual).startswith("/tmp/"), (
                f"Expected /tmp socket, got {actual}"
            )
            assert actual.name.startswith("qd-grade-")
        finally:
            server.stop()

    def test_grading_server_always_uses_tmp_socket_long_path(self, tmp_path):
        """Long workspace paths must also use /tmp socket (same as short path)."""
        # Construct a deeply nested path that would exceed 100 chars
        long_dir = tmp_path / ("a" * 80) / "workspace"
        long_dir.mkdir(parents=True)
        sock_path = long_dir / ".grade.sock"
        server = self._make_concrete_server(sock_path)
        server.start()
        try:
            actual = server._actual_socket_path
            assert str(actual).startswith("/tmp/"), (
                f"Expected /tmp socket, got {actual}"
            )
            assert actual.name.startswith("qd-grade-")
        finally:
            server.stop()

    def test_grading_server_does_not_touch_grade_socket_env(self, tmp_path):
        """start()/stop() must NOT touch the GRADE_SOCKET env var.

        With parallel grading servers (one per executor thread), concurrent
        start() calls would race on the global env. Resolution happens via the
        per-workspace .grade_socket_path marker + hash-of-workspace fallback
        (see grading.py:~124 for rationale).
        """
        sock_path = tmp_path / ".grade.sock"
        server = self._make_concrete_server(sock_path)

        os.environ.pop("GRADE_SOCKET", None)

        server.start()
        try:
            assert "GRADE_SOCKET" not in os.environ, \
                "start() must NOT set GRADE_SOCKET (thread-safety requirement)"
        finally:
            server.stop()

        assert "GRADE_SOCKET" not in os.environ


class TestGradeToolResolveSocketPath:
    """_resolve_socket_path() 3-level fallback: env → marker → hash."""

    def _resolve(self):
        from heuresis.tools.grade import _resolve_socket_path
        return _resolve_socket_path()

    def test_grade_tool_resolves_socket_from_env(self, tmp_path, monkeypatch):
        """Level 1: $GRADE_SOCKET env var wins when set."""
        expected = tmp_path / "my-custom.sock"
        monkeypatch.setenv("GRADE_SOCKET", str(expected))
        monkeypatch.delenv("WORKSPACE_PATH", raising=False)
        result = self._resolve()
        assert result == expected

    def test_grade_tool_resolves_socket_from_marker_file(self, tmp_path, monkeypatch):
        """Level 2: marker file /workspace/.grade_socket_path used when env absent."""
        monkeypatch.delenv("GRADE_SOCKET", raising=False)
        monkeypatch.delenv("WORKSPACE_PATH", raising=False)

        marker_sock = tmp_path / "qd-grade-abc123.sock"
        marker = tmp_path / ".grade_socket_path"
        marker.write_text(str(marker_sock))

        from heuresis.tools import grade as grade_mod
        with patch.object(grade_mod, "_MARKER", marker):
            result = self._resolve()

        assert result == marker_sock

    def test_grade_tool_resolves_socket_from_workspace_hash(self, tmp_path, monkeypatch):
        """Level 3: hash from $WORKSPACE_PATH when env and marker both absent."""
        monkeypatch.delenv("GRADE_SOCKET", raising=False)
        monkeypatch.delenv("WORKSPACE_PATH", raising=False)

        from heuresis.tools import grade as grade_mod
        # Ensure marker doesn't exist
        with patch.object(grade_mod, "_MARKER", tmp_path / "nonexistent_marker"):
            result = self._resolve()

        # Default ws = "/workspace"
        expected_hash = hashlib.sha256(b"/workspace").hexdigest()[:16]
        expected = Path(tempfile.gettempdir()) / f"qd-grade-{expected_hash}.sock"
        assert result == expected

    def test_grade_tool_hash_changes_with_workspace_path(self, tmp_path, monkeypatch):
        """Level 3 hash is determined by WORKSPACE_PATH value."""
        monkeypatch.delenv("GRADE_SOCKET", raising=False)
        custom_ws = "/custom/workspace/path"
        monkeypatch.setenv("WORKSPACE_PATH", custom_ws)

        from heuresis.tools import grade as grade_mod
        with patch.object(grade_mod, "_MARKER", tmp_path / "nonexistent_marker"):
            result = self._resolve()

        expected_hash = hashlib.sha256(custom_ws.encode()).hexdigest()[:16]
        expected = Path(tempfile.gettempdir()) / f"qd-grade-{expected_hash}.sock"
        assert result == expected

    def test_grade_tool_env_takes_precedence_over_marker(self, tmp_path, monkeypatch):
        """Level 1 wins even when level 2 marker also exists."""
        env_sock = tmp_path / "env-level.sock"
        marker_sock = tmp_path / "marker-level.sock"
        monkeypatch.setenv("GRADE_SOCKET", str(env_sock))

        marker = tmp_path / ".grade_socket_path"
        marker.write_text(str(marker_sock))

        from heuresis.tools import grade as grade_mod
        with patch.object(grade_mod, "_MARKER", marker):
            result = self._resolve()

        assert result == env_sock
