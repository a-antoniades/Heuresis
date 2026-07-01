"""Tests for Claude Code session ID UUID format requirements."""

from __future__ import annotations

import re


def test_claude_session_id_is_dashed_uuid():
    """Claude Code requires dashed UUID format for --session-id."""
    from heuresis.agent import CLAUDE_CODE, generate_session_id
    sid = generate_session_id(CLAUDE_CODE, "/usr/local/bin/claude")
    assert sid is not None
    assert re.match(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", sid
    ), sid


def test_opencode_session_id_returns_none():
    """OpenCode has no session_id_flag, so generate_session_id returns None."""
    from heuresis.agent import OPENCODE, generate_session_id
    sid = generate_session_id(OPENCODE, "/usr/local/bin/opencode")
    assert sid is None
