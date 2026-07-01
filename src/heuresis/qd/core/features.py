"""Feature classifier hierarchy for MAP-Elites feature extraction.

The classifier is orthogonal to the search strategy — MAP-Elites consumes
a features: dict[str, float]. Strategy doesn't care whether it came from
keywords or an LLM.

Three classes:
  FeatureClassifier (ABC)
  KeywordClassifier — deterministic, no API
  LLMClassifier — Gemini with 3-key rotation, falls back to KeywordClassifier
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from heuresis.qd.core.archive import Feature

logger = logging.getLogger(__name__)


class FeatureClassifier(ABC):
    """Abstract base class for feature extraction."""

    features: list[Feature]

    @abstractmethod
    def classify(self, idea: str, workspace: Path | None = None) -> dict[str, float]: ...


class KeywordClassifier(FeatureClassifier):
    """Deterministic keyword-based classification.

    keywords[axis][bin_idx] = [kw1, kw2, ...]. For each axis, the bin with
    the highest keyword-hit count wins; ties broken by first bin index.
    Falls back to bin 0 if no hits anywhere.
    """

    def __init__(
        self,
        features: list[Feature],
        keywords: dict[str, dict[int, list[str]]],
    ) -> None:
        self.features = features
        self._keywords = keywords

    def classify(self, idea: str, workspace: Path | None = None) -> dict[str, float]:
        text = idea or ""
        if workspace is not None:
            train_py = workspace / "train.py"
            if train_py.exists():
                try:
                    text += "\n" + train_py.read_text(errors="replace")
                except OSError:
                    pass

        text_lower = text.lower()
        out: dict[str, float] = {}
        for axis in self.features:
            axis_kw = self._keywords.get(axis.name, {})
            best_bin = 0
            best_hits = 0
            for bin_idx, kws in axis_kw.items():
                hits = sum(text_lower.count(kw.lower()) for kw in kws)
                if hits > best_hits:
                    best_hits = hits
                    best_bin = bin_idx
            out[axis.name] = float(best_bin)
        return out


class LLMClassifier(FeatureClassifier):
    """LLM-based classification with key rotation + keyword fallback.

    On any failure (429, 500, malformed JSON, missing key), rotates to
    the next API key. If all keys fail, falls back to the KeywordClassifier.
    """

    def __init__(
        self,
        features: list[Feature],
        *,
        fallback: FeatureClassifier,
        api_keys: list[str] | None = None,
        api_keys_file: Path | None = None,
        model: str = "gemini-3-flash-preview",
        classification_prompt: str = "",
        temperature: float = 1.0,
    ) -> None:
        self.features = features
        self.fallback = fallback
        self.model = model
        self.classification_prompt = classification_prompt
        self.temperature = temperature

        if api_keys is not None:
            self._api_keys = list(api_keys)
        elif api_keys_file is not None:
            try:
                self._api_keys = [
                    line.strip() for line in api_keys_file.read_text().splitlines()
                    if line.strip() and not line.startswith("#")
                ]
            except OSError:
                logger.warning(
                    "LLMClassifier: api_keys_file %s not readable — no keys loaded",
                    api_keys_file,
                )
                self._api_keys = []
        else:
            from heuresis.api_keys import load_api_keys

            self._api_keys = load_api_keys("gemini")
        if not self._api_keys:
            logger.warning("LLMClassifier has no API keys — will always fall back")

    def classify(self, idea: str, workspace: Path | None = None) -> dict[str, float]:
        last_error: Exception | None = None
        for api_key in self._api_keys:
            try:
                return self._try_llm(idea, workspace, api_key)
            except Exception as e:
                logger.info("LLMClassifier key rotation: %s", e)
                last_error = e
                time.sleep(0.5)  # brief backoff
        if last_error:
            logger.warning("LLMClassifier: all keys failed (%s), using keyword fallback", last_error)
        return self.fallback.classify(idea, workspace)

    def _try_llm(
        self, idea: str, workspace: Path | None, api_key: str
    ) -> dict[str, float]:
        """One LLM call with a specific API key. Raises on any failure."""
        from google import genai  # type: ignore[import-not-found]

        text = idea
        if workspace is not None:
            train_py = workspace / "train.py"
            if train_py.exists():
                try:
                    text += "\n\n--- train.py ---\n" + train_py.read_text(errors="replace")[:30_000]
                except OSError:
                    pass

        client = genai.Client(api_key=api_key)
        schema = self._build_response_schema()
        response = client.models.generate_content(
            model=self.model,
            contents=[
                {"role": "user", "parts": [{"text": f"{self.classification_prompt}\n\n{text}"}]},
            ],
            config={
                "temperature": self.temperature,
                "response_mime_type": "application/json",
                "response_schema": schema,
            },
        )
        raw = getattr(response, "text", None)
        if raw is None:
            raise RuntimeError("LLM returned no text")
        parsed = self._parse(raw)
        if parsed is None:
            raise RuntimeError(f"Failed to parse LLM response: {raw[:200]}")
        return parsed

    def _build_response_schema(self) -> dict[str, Any]:
        props: dict[str, Any] = {}
        for f in self.features:
            props[f.name] = {"type": "INTEGER", "minimum": int(f.min_val),
                             "maximum": int(f.max_val)}
        return {
            "type": "OBJECT",
            "properties": props,
            "required": [f.name for f in self.features],
        }

    def _parse(self, raw: str) -> dict[str, float] | None:
        text = raw.strip()
        # Strip markdown fences
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        out: dict[str, float] = {}
        for f in self.features:
            if f.name not in data:
                return None
            try:
                val = float(data[f.name])
            except (TypeError, ValueError):
                return None
            if not (f.min_val <= val <= f.max_val):
                return None
            out[f.name] = val
        return out
