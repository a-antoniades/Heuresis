"""Tests for FeatureClassifier hierarchy."""
from __future__ import annotations

from unittest.mock import patch


from heuresis.qd import Feature
from heuresis.qd.core.features import (
    KeywordClassifier,
    LLMClassifier,
)


FEATURES = [
    Feature("component", 0, 2, num_bins=3),
    Feature("approach", 0, 1, num_bins=2),
]

KEYWORDS = {
    "component": {
        0: ["attention", "qkv"],
        1: ["mlp", "ffn"],
        2: ["norm", "rmsnorm"],
    },
    "approach": {
        0: ["tune", "scale"],
        1: ["replace", "rewrite"],
    },
}


def test_keyword_classifier_picks_highest_hits():
    c = KeywordClassifier(FEATURES, KEYWORDS)
    features = c.classify("Rewrite the attention mechanism with new QKV projections", None)
    assert features["component"] == 0  # attention + qkv
    assert features["approach"] == 1  # rewrite


def test_keyword_classifier_default_on_no_hits():
    c = KeywordClassifier(FEATURES, KEYWORDS)
    features = c.classify("totally unrelated text zzz", None)
    assert features["component"] == 0
    assert features["approach"] == 0


def test_llm_classifier_falls_back_on_api_error():
    fallback = KeywordClassifier(FEATURES, KEYWORDS)
    llm = LLMClassifier(
        FEATURES,
        fallback=fallback,
        api_keys=["bogus"],
        model="gemini-3-flash-preview",
        classification_prompt="classify",
    )
    with patch.object(llm, "_try_llm", side_effect=RuntimeError("api down")):
        features = llm.classify("Rewrite the attention", None)
    assert features["component"] == 0  # fell back to keyword


def test_llm_classifier_success_returns_llm_result():
    fallback = KeywordClassifier(FEATURES, KEYWORDS)
    llm = LLMClassifier(
        FEATURES,
        fallback=fallback,
        api_keys=["k1"],
        model="gemini-3-flash-preview",
        classification_prompt="classify",
    )
    with patch.object(llm, "_try_llm", return_value={"component": 2.0, "approach": 1.0}):
        features = llm.classify("irrelevant text", None)
    assert features == {"component": 2.0, "approach": 1.0}


def test_llm_classifier_rotates_keys_on_429():
    fallback = KeywordClassifier(FEATURES, KEYWORDS)
    llm = LLMClassifier(
        FEATURES,
        fallback=fallback,
        api_keys=["k1", "k2", "k3"],
        model="gemini-3-flash-preview",
        classification_prompt="classify",
    )
    # First call with k1 raises 429; second with k2 succeeds
    calls = []
    def _fake_llm(idea, workspace, api_key):
        calls.append(api_key)
        if api_key == "k1":
            raise RuntimeError("429 rate limit")
        return {"component": 1.0, "approach": 0.0}
    with patch.object(llm, "_try_llm", side_effect=_fake_llm):
        features = llm.classify("irrelevant", None)
    assert features == {"component": 1.0, "approach": 0.0}
    assert calls == ["k1", "k2"]


def test_llm_classifier_missing_api_keys_file_falls_back(tmp_path):
    """Missing api_keys_file should not crash — falls back to empty list + keyword fallback."""
    fallback = KeywordClassifier(FEATURES, KEYWORDS)
    missing = tmp_path / "does_not_exist.txt"
    llm = LLMClassifier(
        FEATURES,
        fallback=fallback,
        api_keys_file=missing,
        model="gemini-3-flash-preview",
        classification_prompt="classify",
    )
    # No keys loaded, so classify() immediately delegates to fallback
    features = llm.classify("Rewrite the attention", None)
    assert features["component"] == 0  # keyword classifier hit 'attention'
