"""Inline tag extraction for curiosity-plus Change B.

Returns ``{components, technique_tags}`` for an idea via a single Gemini
call. Used by ``CuriosityPlusSearch`` to compute novelty / repetition penalty
in component-tag space rather than text-embedding space — text-embedding
distance rewards vocabulary diversity, while tag-Jaccard distance rewards
architectural diversity.

The classifier degrades gracefully: any error or missing dependency
returns empty lists, which the caller treats as "no tag info" and falls
back to text-embedding cosine.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class TagClassifier(Protocol):
    """A small two-method interface so callers can swap a fake in tests."""

    def classify(self, idea_text: str) -> dict[str, list[str]]:
        """Return ``{"components": [...], "technique_tags": [...]}``.

        Both lists may be empty on classifier failure — callers must
        handle that case.
        """
        ...


_PROMPT_TMPL = """Classify the following NanoGPT speedrun research idea.

Return ONE JSON object, no markdown fences, no commentary:
{{"components": [...], "technique_tags": [...]}}

- components: which architectural / training pieces does the idea modify?
  Use lowercase snake_case nouns. Examples: value_embeddings, attention,
  mlp, optimizer, lr_schedule, embedding, loss, regularization,
  initialization, normalization, depth_routing.
- technique_tags: specific named techniques referenced or applied.
  Examples: soft_moe, cayley_orthogonal, householder_flow, rope, z_loss,
  swiglu, neftune, ema, gqa, mixup.

Idea:
{idea}
"""


class GeminiTagClassifier:
    """Gemini-based tag extractor with key rotation.

    Mirrors the rotation pattern used by ``GeminiEmbedder``. Keys are loaded
    from ``api_keys``, ``api_keys_file``, or the standard Gemini env vars.
    Lazy-imports ``google.genai`` so test environments without it can still
    load the module.
    """

    def __init__(
        self,
        *,
        api_keys: list[str] | None = None,
        api_keys_file: str | Path | None = None,
        model: str = "gemini-3-flash-preview",
        max_retries: int = 3,
        timeout_s: float = 30.0,
    ) -> None:
        if api_keys is not None:
            self._keys = list(api_keys)
        elif api_keys_file is not None:
            from heuresis.api_keys import read_keys_file

            self._keys = read_keys_file(Path(api_keys_file))
        else:
            from heuresis.api_keys import load_api_keys

            self._keys = load_api_keys("gemini")
        if not self._keys:
            raise ValueError(
                "GeminiTagClassifier requires at least one API key "
                "(set GEMINI_API_KEYS / GEMINI_API_KEY / GOOGLE_GENERATIVE_AI_API_KEY, "
                "or pass api_keys / api_keys_file)"
            )
        self._key_idx = 0
        self._model = model
        self._max_retries = max_retries
        self._timeout_s = timeout_s

    def _client(self) -> Any:
        # Lazy import so module loads without google-genai installed.
        from google import genai
        return genai.Client(api_key=self._keys[self._key_idx])

    def _rotate(self) -> None:
        self._key_idx = (self._key_idx + 1) % len(self._keys)

    def classify(self, idea_text: str) -> dict[str, list[str]]:
        prompt = _PROMPT_TMPL.format(idea=idea_text[:6000])
        for attempt in range(self._max_retries):
            try:
                client = self._client()
                resp = client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                )
                text = (resp.text or "").strip()
                # Strip ```json fences just in case the model adds them.
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
                obj = json.loads(text)
                components = [str(c) for c in obj.get("components", []) if c]
                tags = [str(t) for t in obj.get("technique_tags", []) if t]
                return {"components": components, "technique_tags": tags}
            except Exception as exc:
                logger.warning(
                    "GeminiTagClassifier attempt %d/%d failed: %s",
                    attempt + 1, self._max_retries, exc,
                )
                self._rotate()
                time.sleep(0.5 * (attempt + 1))
        # All retries exhausted — return empty so callers fall back to cosine.
        return {"components": [], "technique_tags": []}


class FakeTagClassifier:
    """Deterministic test tagger: pulls capitalized words / lowercase nouns
    from idea text. Useful in tests where Gemini is not available."""

    _COMPONENT_HINTS = (
        "value_embeddings", "attention", "mlp", "optimizer", "lr_schedule",
        "embedding", "loss", "regularization", "initialization",
        "normalization", "depth", "head",
    )

    def classify(self, idea_text: str) -> dict[str, list[str]]:
        low = idea_text.lower()
        comps = [c for c in self._COMPONENT_HINTS if c in low]
        # Crude tag extraction: snake_case-able tokens 4–30 chars
        tags = sorted({
            m.group(0).lower()
            for m in re.finditer(r"\b[A-Za-z][A-Za-z0-9_]{3,29}\b", idea_text)
        })[:8]
        return {"components": comps[:4], "technique_tags": tags}


def jaccard_distance(a: set[str], b: set[str]) -> float:
    """Symmetric set distance in [0, 1]. Returns 1.0 if both sets empty."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return 1.0 - (len(a & b) / len(union))
