"""Tests for Harness strip_env behavior."""
from __future__ import annotations

from heuresis.harness import Harness


def test_claude_strips_anthropic_api_key_by_default():
    """Claude agent auto-strips ANTHROPIC_API_KEY so subscription auth is used."""
    h = Harness("claude")
    assert "ANTHROPIC_API_KEY" in h._strip_env


def test_opencode_does_not_strip():
    h = Harness("opencode")
    assert "ANTHROPIC_API_KEY" not in h._strip_env


def test_explicit_strip_env_override():
    h = Harness("opencode", strip_env=["MY_SECRET"])
    assert "MY_SECRET" in h._strip_env


def test_explicit_override_replaces_default_for_claude():
    h = Harness("claude", strip_env=["DIFFERENT"])
    # Explicit list replaces the auto-detected list
    assert h._strip_env == ["DIFFERENT"]
    assert "ANTHROPIC_API_KEY" not in h._strip_env
