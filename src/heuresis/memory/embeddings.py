"""Gemini embedding client for the memory primitive.

Thin wrapper around ``google-genai``'s ``embed_content`` call, locked to
``gemini-embedding-001`` at the native 3072-dim (no MRL truncation).
Embeddings are written into sqlite-vec virtual tables declared
``float[3072]`` in :file:`schema.sql` — the two must stay in sync. We
use the native dim to avoid the MRL-truncated-vectors-aren't-normalized
gotcha; disk/index size is irrelevant at campaign scale (~hundreds of
rows per DB).

Key sourcing (in priority order):

1. ``api_keys`` argument
2. ``api_keys_file`` argument (one key per line, ``#`` comments allowed)
3. Central env vars via :func:`heuresis.api_keys.load_api_keys`:
   ``GEMINI_API_KEYS`` → ``GEMINI_API_KEY`` → ``GOOGLE_GENERATIVE_AI_API_KEY``

Fail-closed: on total exhaustion of keys/retries, ``embed_one`` raises
``RuntimeError``. The caller (MemoryStore) propagates the error so the
in-sandbox ``memory`` CLI returns non-zero and the agent sees the
failure.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

MODEL = "gemini-embedding-001"
# Native dim. Must match vec0 column width in schema.sql.
# We deliberately avoid MRL truncation (e.g. 768) because truncated
# Gemini vectors require explicit re-normalization to be usable with
# L2 distance — messy, and disk size is a non-issue at our scale.
DIM = 3072


class GeminiMemoryEmbedder:
    """Single-purpose embedder for memory rows.

    Batched call (``embed``) is used for ingest_experiment; ``embed_one`` is
    used by search queries. Retries on any exception (primarily 429 / 5xx)
    with exponential backoff, then rotates to the next key. If all keys
    fail for a batch, raises ``RuntimeError``.
    """

    model = MODEL
    dim = DIM

    def __init__(
        self,
        *,
        api_keys: list[str] | None = None,
        api_keys_file: Path | None = None,
        max_retries: int = 3,
        retry_backoff_s: float = 0.5,
        client_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._retries = max_retries
        self._backoff = retry_backoff_s

        if api_keys is None:
            if api_keys_file is not None and api_keys_file.exists():
                from heuresis.api_keys import read_keys_file

                api_keys = read_keys_file(api_keys_file)
            else:
                from heuresis.api_keys import load_api_keys

                api_keys = load_api_keys("gemini")
        if not api_keys:
            raise RuntimeError(
                "GeminiMemoryEmbedder requires an API key. Set GEMINI_API_KEYS, "
                "GEMINI_API_KEYS, GEMINI_API_KEY, GOOGLE_GENERATIVE_AI_API_KEY, "
                "or pass api_keys / api_keys_file."
            )
        self._keys = list(api_keys)
        self._client_factory = client_factory or self._default_factory
        self._clients: dict[str, Any] = {}

    @staticmethod
    def _default_factory(api_key: str) -> Any:
        from google import genai  # type: ignore[import-not-found]
        return genai.Client(api_key=api_key)

    def _client(self, key: str) -> Any:
        if key not in self._clients:
            self._clients[key] = self._client_factory(key)
        return self._clients[key]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one DIM-dim vector per input, order preserved."""
        if not texts:
            return []
        # google-genai supports batched embed_content calls; no reason to chunk
        # unless batches get huge. Memory writes are small (O(1) per op), so
        # we take the request as-is and rely on the SDK to chunk internally.
        return self._embed_with_rotation(texts)

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def _embed_with_rotation(self, batch: list[str]) -> list[list[float]]:
        from google.genai.types import EmbedContentConfig  # type: ignore[import-not-found]
        cfg = EmbedContentConfig(output_dimensionality=DIM)
        last_exc: Exception | None = None

        for key_idx, key in enumerate(self._keys):
            for attempt in range(self._retries):
                try:
                    client = self._client(key)
                    resp = client.models.embed_content(
                        model=MODEL, contents=batch, config=cfg,
                    )
                except Exception as exc:
                    last_exc = exc
                    logger.info(
                        "GeminiMemoryEmbedder: key #%d attempt %d failed (%s)",
                        key_idx, attempt, exc,
                    )
                    if self._backoff > 0:
                        time.sleep(self._backoff * (2 ** attempt))
                    continue

                vectors = [list(emb.values) for emb in resp.embeddings]
                if len(vectors) != len(batch):
                    raise RuntimeError(
                        f"embedding response length {len(vectors)} "
                        f"!= batch length {len(batch)}"
                    )
                for v in vectors:
                    if len(v) != DIM:
                        raise RuntimeError(
                            f"embedding dim {len(v)} != expected {DIM}"
                        )
                return vectors

        raise RuntimeError(
            f"GeminiMemoryEmbedder: all {len(self._keys)} keys exhausted "
            f"(last error: {last_exc})"
        ) from last_exc
