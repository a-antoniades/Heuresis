"""Unit tests for GeminiMemoryEmbedder.

These tests never hit the network: they inject a ``client_factory`` that
returns a MagicMock. The mock shapes its response to look like the real
google-genai ``embed_content`` return.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from heuresis.memory.embeddings import DIM, MODEL, GeminiMemoryEmbedder


def _mock_resp(vectors: list[list[float]]) -> MagicMock:
    resp = MagicMock()
    resp.embeddings = [MagicMock(values=v) for v in vectors]
    return resp


def _ok_dim_vec(fill: float = 0.1) -> list[float]:
    """A correctly-dimensioned vector for the production DIM."""
    return [fill] * DIM


# -- construction / key sourcing --------------------------------------------


def _clear_gemini_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENERATIVE_AI_API_KEY", raising=False)


def test_model_and_dim_are_locked():
    assert MODEL == "gemini-embedding-001"
    assert DIM == 3072  # native dim; see embeddings.py for why not MRL-truncated


def test_raises_when_no_keys(monkeypatch):
    _clear_gemini_env(monkeypatch)
    with pytest.raises(RuntimeError, match="requires an API key"):
        GeminiMemoryEmbedder()


def test_loads_key_from_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    monkeypatch.delenv("GOOGLE_GENERATIVE_AI_API_KEY", raising=False)
    emb = GeminiMemoryEmbedder(client_factory=lambda k: MagicMock())
    assert emb._keys == ["env-key"]


def test_env_precedence_gemini_over_google(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "primary")
    monkeypatch.setenv("GOOGLE_GENERATIVE_AI_API_KEY", "secondary")
    emb = GeminiMemoryEmbedder(client_factory=lambda k: MagicMock())
    assert emb._keys == ["primary"]


def test_falls_back_to_google_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_GENERATIVE_AI_API_KEY", "backup")
    emb = GeminiMemoryEmbedder(client_factory=lambda k: MagicMock())
    assert emb._keys == ["backup"]


def test_loads_keys_from_file(tmp_path: Path, monkeypatch):
    _clear_gemini_env(monkeypatch)
    keyfile = tmp_path / "keys.txt"
    keyfile.write_text("alpha\n# comment\nbravo\n\ncharlie\n")
    emb = GeminiMemoryEmbedder(
        api_keys_file=keyfile, client_factory=lambda k: MagicMock(),
    )
    assert emb._keys == ["alpha", "bravo", "charlie"]


def test_explicit_keys_win_over_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-only")
    emb = GeminiMemoryEmbedder(
        api_keys=["explicit"], client_factory=lambda k: MagicMock(),
    )
    assert emb._keys == ["explicit"]


# -- embed path --------------------------------------------------------------


def test_embed_returns_one_vector_per_input():
    client = MagicMock()
    client.models.embed_content.return_value = _mock_resp([
        _ok_dim_vec(0.1), _ok_dim_vec(0.2),
    ])
    emb = GeminiMemoryEmbedder(
        api_keys=["k1"], client_factory=lambda k: client,
    )
    out = emb.embed(["a", "b"])
    assert len(out) == 2
    assert all(len(v) == DIM for v in out)
    assert out[0][0] == 0.1
    assert out[1][0] == 0.2


def test_embed_empty_batch_skips_api():
    factory = MagicMock()
    emb = GeminiMemoryEmbedder(api_keys=["k1"], client_factory=factory)
    assert emb.embed([]) == []
    factory.assert_not_called()


def test_embed_one_returns_single_vector():
    client = MagicMock()
    client.models.embed_content.return_value = _mock_resp([_ok_dim_vec(0.5)])
    emb = GeminiMemoryEmbedder(
        api_keys=["k1"], client_factory=lambda k: client,
    )
    v = emb.embed_one("hello")
    assert len(v) == DIM
    assert v[0] == 0.5


def test_passes_output_dimensionality_config():
    """Regression: output_dimensionality must match DIM in schema.sql."""
    client = MagicMock()
    client.models.embed_content.return_value = _mock_resp([_ok_dim_vec()])
    emb = GeminiMemoryEmbedder(
        api_keys=["k1"], client_factory=lambda k: client,
    )
    emb.embed(["a"])
    _, kwargs = client.models.embed_content.call_args
    cfg = kwargs.get("config")
    assert cfg is not None, "config kwarg must be passed"
    assert getattr(cfg, "output_dimensionality", None) == DIM


def test_uses_gemini_embedding_001_model():
    """Regression: model name must match the schema.sql vec0 declaration."""
    client = MagicMock()
    client.models.embed_content.return_value = _mock_resp([_ok_dim_vec()])
    emb = GeminiMemoryEmbedder(
        api_keys=["k1"], client_factory=lambda k: client,
    )
    emb.embed(["a"])
    _, kwargs = client.models.embed_content.call_args
    assert kwargs.get("model") == MODEL


# -- retry / rotation --------------------------------------------------------


def test_retries_on_transient_failure_then_succeeds():
    client = MagicMock()
    client.models.embed_content.side_effect = [
        RuntimeError("429 quota"),
        _mock_resp([_ok_dim_vec()]),
    ]
    emb = GeminiMemoryEmbedder(
        api_keys=["k1"],
        max_retries=2, retry_backoff_s=0.0,
        client_factory=lambda k: client,
    )
    out = emb.embed(["x"])
    assert len(out) == 1
    assert client.models.embed_content.call_count == 2


def test_rotates_to_next_key_after_retries_exhausted():
    client_a = MagicMock()
    client_a.models.embed_content.side_effect = RuntimeError("boom")
    client_b = MagicMock()
    client_b.models.embed_content.return_value = _mock_resp([_ok_dim_vec()])

    factory = MagicMock(side_effect=[client_a, client_b])
    emb = GeminiMemoryEmbedder(
        api_keys=["k1", "k2"],
        max_retries=2, retry_backoff_s=0.0,
        client_factory=factory,
    )
    out = emb.embed(["x"])
    assert len(out) == 1
    # First key tried (and exhausted), then second key succeeded.
    assert factory.call_count == 2


def test_raises_when_all_keys_exhausted():
    def failing(_k):
        c = MagicMock()
        c.models.embed_content.side_effect = RuntimeError("always")
        return c
    emb = GeminiMemoryEmbedder(
        api_keys=["k1", "k2"],
        max_retries=1, retry_backoff_s=0.0,
        client_factory=failing,
    )
    with pytest.raises(RuntimeError, match="keys exhausted"):
        emb.embed(["x"])


def test_dim_mismatch_raises_immediately():
    """Shape errors are bugs, not transient failures — do NOT rotate."""
    client = MagicMock()
    # Wrong dim: should raise before rotating.
    client.models.embed_content.return_value = _mock_resp([[0.1, 0.2]])
    factory = MagicMock(return_value=client)
    emb = GeminiMemoryEmbedder(
        api_keys=["k1", "k2"],
        max_retries=1, retry_backoff_s=0.0,
        client_factory=factory,
    )
    with pytest.raises(RuntimeError, match="embedding dim"):
        emb.embed(["x"])
    assert factory.call_count == 1


def test_response_length_mismatch_raises_immediately():
    client = MagicMock()
    # Two inputs, one vector back.
    client.models.embed_content.return_value = _mock_resp([_ok_dim_vec()])
    factory = MagicMock(return_value=client)
    emb = GeminiMemoryEmbedder(
        api_keys=["k1"], max_retries=1, retry_backoff_s=0.0,
        client_factory=factory,
    )
    with pytest.raises(RuntimeError, match="response length"):
        emb.embed(["x", "y"])


def test_client_is_cached_per_key():
    client = MagicMock()
    client.models.embed_content.return_value = _mock_resp([_ok_dim_vec()])
    factory = MagicMock(return_value=client)
    emb = GeminiMemoryEmbedder(
        api_keys=["k1"], client_factory=factory,
    )
    emb.embed(["a"])
    emb.embed(["b"])
    # Factory called once total, even for two distinct batches on the same key.
    assert factory.call_count == 1
