"""Embedder protocol, canonicalization, and test fake.

Gemini implementation lives alongside but imports google.genai lazily.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


def canonicalize_text(text: str) -> str:
    """Return a canonical form of text suitable for hashing and embedding.

    Rules:
    - CRLF and CR line endings normalize to LF.
    - Trailing whitespace on each line is stripped.
    - Leading/trailing whitespace on the whole text is stripped.

    These rules exist so that whitespace-only edits don't cause spurious
    cache misses against text_hash.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip()


def text_hash(text: str) -> str:
    """Return sha256 hex of the canonicalized text."""
    return hashlib.sha256(canonicalize_text(text).encode("utf-8")).hexdigest()


@runtime_checkable
class Embedder(Protocol):
    """Protocol for text embedders."""

    model: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch. Returns (N, D) float32. Order matches inputs."""
        ...

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text. Returns (D,) float32."""
        ...


class FakeEmbedder:
    """Deterministic fake embedder for tests.

    Uses sha256 of each text as a seed to produce stable, distinct vectors.
    Callers are responsible for canonicalizing text before embedding; this class
    does not call ``canonicalize_text`` itself.  This matches the production
    ``Embedder`` contract.
    """

    def __init__(self, dim: int = 16, model: str = "fake-v1") -> None:
        self.dim = dim
        self.model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int.from_bytes(
                hashlib.sha256(t.encode("utf-8")).digest()[:8], "little"
            )
            rng = np.random.default_rng(seed)
            out[i] = rng.standard_normal(self.dim).astype(np.float32)
        return out

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class GeminiEmbedder:
    """Gemini embedding backend with 3-key rotation, batching, retries.

    Policy: try each key in order. On failure, rotate to the next key.
    If all keys fail for a batch, raise RuntimeError. No TF-IDF fallback
    — callers upstream decide whether to degrade.

    `client_factory` is injected for testability. In production, the default
    factory creates a `google.genai.Client` per key.
    """

    model = "gemini-embedding-001"

    def __init__(
        self,
        *,
        api_keys: list[str] | None = None,
        api_keys_file: Path | None = None,
        dim: int = 3072,
        batch_size: int = 20,
        retry_backoff_s: float = 0.5,
        client_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.dim = dim
        self.batch_size = batch_size
        self._retry_backoff_s = retry_backoff_s

        if api_keys is not None:
            self._api_keys = list(api_keys)
        elif api_keys_file is not None:
            from heuresis.api_keys import read_keys_file

            self._api_keys = read_keys_file(api_keys_file)
        else:
            from heuresis.api_keys import load_api_keys

            self._api_keys = load_api_keys("gemini")

        if not self._api_keys:
            raise ValueError(
                "GeminiEmbedder requires at least one API key "
                "(set GEMINI_API_KEYS / GEMINI_API_KEY / GOOGLE_GENERATIVE_AI_API_KEY, "
                "or pass api_keys=[...] / api_keys_file=Path(...))"
            )

        self._client_factory = client_factory or self._default_factory
        self._clients: dict[str, Any] = {}

    @staticmethod
    def _default_factory(api_key: str) -> Any:
        from google import genai  # type: ignore[import-not-found]
        return genai.Client(api_key=api_key)

    def _get_client(self, api_key: str) -> Any:
        if api_key not in self._clients:
            self._clients[api_key] = self._client_factory(api_key)
        return self._clients[api_key]

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors = self._embed_batch_with_rotation(batch)
            out[start : start + len(batch)] = vectors
        return out

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def _embed_batch_with_rotation(self, batch: list[str]) -> np.ndarray:
        last_exc: Exception | None = None
        for attempt, api_key in enumerate(self._api_keys):
            try:
                client = self._get_client(api_key)
                response = client.models.embed_content(
                    model=self.model,
                    contents=batch,
                )
            except Exception as exc:
                logger.info(
                    "GeminiEmbedder: key #%d failed (%s); rotating",
                    attempt, exc,
                )
                last_exc = exc
                if self._retry_backoff_s > 0:
                    time.sleep(self._retry_backoff_s)
                continue

            # Validation runs outside the except clause so our own
            # RuntimeErrors are never mistaken for transient API failures.
            vectors = [emb.values for emb in response.embeddings]
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"embedding response length {len(vectors)} "
                    f"!= batch length {len(batch)}"
                )
            arr = np.asarray(vectors, dtype=np.float32)
            if arr.shape[1] != self.dim:
                raise RuntimeError(
                    f"embedding dim {arr.shape[1]} != expected {self.dim}"
                )
            return arr

        raise RuntimeError(
            f"GeminiEmbedder: all {len(self._api_keys)} keys exhausted "
            f"(last error: {last_exc})"
        ) from last_exc
