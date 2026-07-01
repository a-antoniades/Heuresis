"""Tests for heuresis.api_keys."""

from __future__ import annotations

import pytest

from heuresis.api_keys import load_api_key, load_api_keys


def test_gemini_api_keys_comma_separated(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENERATIVE_AI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEYS", "alpha,beta,gamma")
    assert load_api_keys("gemini") == ["alpha", "beta", "gamma"]


def test_gemini_api_keys_colon_separated(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENERATIVE_AI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEYS", "alpha:beta")
    assert load_api_keys("gemini") == ["alpha", "beta"]


def test_gemini_api_keys_newline_separated(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENERATIVE_AI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEYS", "alpha\nbeta\n")
    assert load_api_keys("gemini") == ["alpha", "beta"]


def test_gemini_single_key_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GOOGLE_GENERATIVE_AI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "solo")
    assert load_api_keys("gemini") == ["solo"]


def test_gemini_google_env_fallback(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_GENERATIVE_AI_API_KEY", "google-solo")
    assert load_api_keys("gemini") == ["google-solo"]


def test_gemini_keys_file_env_is_ignored(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENERATIVE_AI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_KEYS_FILE", "/tmp/gemini-keys.txt")
    assert load_api_keys("gemini") == []


def test_gemini_precedence_multi_over_single(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEYS", "multi")
    monkeypatch.setenv("GEMINI_API_KEY", "single")
    monkeypatch.setenv("GOOGLE_GENERATIVE_AI_API_KEY", "google")
    assert load_api_keys("gemini") == ["multi"]


def test_gemini_empty_when_unset(monkeypatch):
    for var in (
        "GEMINI_API_KEYS",
        "GEMINI_API_KEY",
        "GOOGLE_GENERATIVE_AI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    assert load_api_keys("gemini") == []


def test_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    assert load_api_key("openai") == "sk-openai"


def test_anthropic_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")
    assert load_api_key("anthropic") == "anthropic-test-key"


def test_openrouter_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key")
    assert load_api_key("openrouter") == "openrouter-test-key"


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="unknown provider"):
        load_api_keys("not-a-provider")
