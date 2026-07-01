"""Tests for Embedder protocol and FakeEmbedder."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from heuresis.qd.core.embedding import Embedder, FakeEmbedder, GeminiEmbedder, canonicalize_text, text_hash


class TestCanonicalize:
    def test_normalizes_line_endings(self):
        assert canonicalize_text("a\r\nb\r\nc") == "a\nb\nc"

    def test_strips_trailing_whitespace_per_line(self):
        assert canonicalize_text("a   \nb\t\nc") == "a\nb\nc"

    def test_strips_outer_whitespace(self):
        assert canonicalize_text("  \n  hello  \n  ") == "hello"


class TestTextHash:
    def test_stable(self):
        a = text_hash("hello world")
        b = text_hash("hello world")
        assert a == b
        assert len(a) == 64  # sha256 hex

    def test_canonicalization_before_hash(self):
        # Differing whitespace should hash the same
        assert text_hash("hello  \n") == text_hash("hello\r\n")

    def test_distinct_for_different_content(self):
        assert text_hash("a") != text_hash("b")


class TestFakeEmbedder:
    def test_returns_batch_shape(self):
        emb = FakeEmbedder(dim=8)
        out = emb.embed(["x", "y", "z"])
        assert out.shape == (3, 8)
        assert out.dtype == np.float32

    def test_deterministic(self):
        emb = FakeEmbedder(dim=8)
        a = emb.embed(["hello"])
        b = emb.embed(["hello"])
        np.testing.assert_allclose(a, b)

    def test_distinct_texts_distinct_vectors(self):
        emb = FakeEmbedder(dim=8)
        out = emb.embed(["foo", "bar"])
        assert not np.allclose(out[0], out[1])

    def test_embed_one_returns_1d(self):
        emb = FakeEmbedder(dim=8)
        v = emb.embed_one("hello")
        assert v.shape == (8,)

    def test_protocol_compat(self):
        emb = FakeEmbedder(dim=8)
        assert isinstance(emb, Embedder)
        assert emb.dim == 8
        assert emb.model.startswith("fake")

    def test_empty_batch(self):
        emb = FakeEmbedder(dim=8)
        out = emb.embed([])
        assert out.shape == (0, 8)


# ---------------------------------------------------------------------------
# GeminiEmbedder tests — use a mock genai.Client injected via a factory.
# ---------------------------------------------------------------------------


def _mock_embed_response(vectors: list[list[float]]) -> MagicMock:
    """Shape a mock response to match google.genai's embed_content return."""
    resp = MagicMock()
    resp.embeddings = [MagicMock(values=v) for v in vectors]
    return resp


class TestGeminiEmbedder:
    def test_basic_embed(self):
        client = MagicMock()
        client.models.embed_content.return_value = _mock_embed_response(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        factory = MagicMock(return_value=client)
        emb = GeminiEmbedder(
            api_keys=["k1"],
            dim=3,
            batch_size=20,
            client_factory=factory,
        )
        out = emb.embed(["a", "b"])
        assert out.shape == (2, 3)
        np.testing.assert_allclose(out[0], [1.0, 0.0, 0.0])
        factory.assert_called_once_with("k1")

    def test_batching(self):
        client = MagicMock()
        # First batch returns 2 vectors, second returns 1
        client.models.embed_content.side_effect = [
            _mock_embed_response([[1, 0], [0, 1]]),
            _mock_embed_response([[0.5, 0.5]]),
        ]
        emb = GeminiEmbedder(
            api_keys=["k1"], dim=2, batch_size=2,
            client_factory=lambda k: client,
        )
        out = emb.embed(["a", "b", "c"])
        assert out.shape == (3, 2)
        assert client.models.embed_content.call_count == 2

    def test_key_rotation_on_failure(self):
        client1 = MagicMock()
        client1.models.embed_content.side_effect = RuntimeError("429 rate limit")
        client2 = MagicMock()
        client2.models.embed_content.return_value = _mock_embed_response([[1.0, 0.0]])

        factory = MagicMock(side_effect=[client1, client2])
        emb = GeminiEmbedder(
            api_keys=["k1", "k2"], dim=2, batch_size=20,
            client_factory=factory,
            retry_backoff_s=0.0,  # no sleep in tests
        )
        out = emb.embed(["a"])
        np.testing.assert_allclose(out[0], [1.0, 0.0])
        assert factory.call_count == 2  # k1 tried, then k2

    def test_raise_when_all_keys_exhausted(self):
        def failing_client(key):
            c = MagicMock()
            c.models.embed_content.side_effect = RuntimeError(f"boom {key}")
            return c

        emb = GeminiEmbedder(
            api_keys=["k1", "k2", "k3"], dim=2, batch_size=20,
            client_factory=failing_client,
            retry_backoff_s=0.0,
        )
        with pytest.raises(RuntimeError, match="all .* keys exhausted"):
            emb.embed(["a"])

    def test_empty_batch_no_api_call(self):
        factory = MagicMock()
        emb = GeminiEmbedder(
            api_keys=["k1"], dim=3, batch_size=20, client_factory=factory,
        )
        out = emb.embed([])
        assert out.shape == (0, 3)
        factory.assert_not_called()

    def test_embed_one(self):
        client = MagicMock()
        client.models.embed_content.return_value = _mock_embed_response([[0.1, 0.2]])
        emb = GeminiEmbedder(
            api_keys=["k1"], dim=2, batch_size=20,
            client_factory=lambda k: client,
        )
        v = emb.embed_one("hi")
        assert v.shape == (2,)
        np.testing.assert_allclose(v, [0.1, 0.2])

    def test_preserves_input_order_across_batches(self):
        client = MagicMock()
        client.models.embed_content.side_effect = [
            _mock_embed_response([[1, 0, 0]]),  # batch 1 for "a"
            _mock_embed_response([[0, 1, 0]]),  # batch 2 for "b"
            _mock_embed_response([[0, 0, 1]]),  # batch 3 for "c"
        ]
        emb = GeminiEmbedder(
            api_keys=["k1"], dim=3, batch_size=1,
            client_factory=lambda k: client,
        )
        out = emb.embed(["a", "b", "c"])
        np.testing.assert_allclose(out, np.eye(3, dtype=np.float32))

    def test_keys_from_file(self, tmp_path):
        key_file = tmp_path / "keys.txt"
        key_file.write_text("key_a\n# comment\nkey_b\n\nkey_c\n")
        client = MagicMock()
        client.models.embed_content.return_value = _mock_embed_response([[1.0]])
        factory = MagicMock(return_value=client)
        emb = GeminiEmbedder(
            api_keys_file=key_file, dim=1, batch_size=20,
            client_factory=factory,
        )
        emb.embed(["x"])
        # First call uses the first non-comment, non-blank key
        factory.assert_called_once_with("key_a")

    def test_client_cached_across_batches(self):
        """Factory should be called once per key, regardless of batch count."""
        client = MagicMock()
        client.models.embed_content.side_effect = [
            _mock_embed_response([[1, 0, 0]]),
            _mock_embed_response([[0, 1, 0]]),
            _mock_embed_response([[0, 0, 1]]),
        ]
        factory = MagicMock(return_value=client)
        emb = GeminiEmbedder(
            api_keys=["k1"], dim=3, batch_size=1,
            client_factory=factory,
        )
        emb.embed(["a", "b", "c"])   # 3 batches, same key
        assert factory.call_count == 1, "client factory should cache per key"

    def test_dim_mismatch_raises_immediately(self):
        """Our own shape-validation errors should NOT trigger key rotation."""
        client = MagicMock()
        # Return an embedding of wrong dim
        client.models.embed_content.return_value = _mock_embed_response([[1.0, 0.0]])  # dim=2
        factory = MagicMock(return_value=client)
        emb = GeminiEmbedder(
            api_keys=["k1", "k2", "k3"], dim=4,  # expected dim=4 but got 2
            batch_size=20, client_factory=factory,
            retry_backoff_s=0.0,
        )
        with pytest.raises(RuntimeError, match="embedding dim"):
            emb.embed(["a"])
        # Key rotation should NOT have been triggered:
        assert factory.call_count == 1
