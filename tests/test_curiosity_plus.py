"""Tests for explicit curiosity-plus primitives."""

import numpy as np
import pytest

from heuresis.qd.core.embedding import FakeEmbedder
from heuresis.qd.curiosity_plus import CuriosityPlusSearch
from heuresis.qd.curiosity_plus.embedding_store import EmbeddingStore
from heuresis.qd.curiosity_plus.selection import select_anchor
from heuresis.qd.curiosity_plus.surprise import Prediction, apply_memory_discount
from heuresis.qd.curiosity_plus.tag_classifier import (
    FakeTagClassifier,
    jaccard_distance,
)


@pytest.fixture
def embedder():
    return FakeEmbedder(dim=16)


def test_curiosity_plus_accepts_plus_knobs(embedder):
    strategy = CuriosityPlusSearch(
        embedder,
        score_weight=0.5,
        tag_classifier=FakeTagClassifier(),
        memory_strength=0.3,
    )

    assert strategy.score_weight == 0.5


def test_score_weight_one_picks_best_lower_is_better(embedder):
    store = EmbeddingStore(embedder)
    store.add("good", "idea good", iteration=0, surprise=0.1, score=0.95, valid=True)
    store.add("ok", "idea ok", iteration=1, surprise=0.1, score=1.10, valid=True)
    store.add("bad", "idea bad", iteration=2, surprise=0.1, score=1.50, valid=True)

    rng = np.random.default_rng(0)
    counts: dict[str, int] = {}
    for _ in range(400):
        run_id = select_anchor(
            store,
            candidate_window=20,
            tau=1.0,
            rng=rng,
            score_weight=1.0,
            lower_is_better=True,
        )
        counts[run_id] = counts.get(run_id, 0) + 1

    assert counts.get("good", 0) > counts.get("ok", 0) > counts.get("bad", 0)
    assert counts.get("bad", 0) == 0


def test_tag_repetition_uses_tag_distance(embedder):
    store = EmbeddingStore(embedder)
    store.add("a", "idea a", iteration=0, surprise=0.5, valid=True)
    store.add("b", "idea b", iteration=1, surprise=0.5, valid=True)
    store.add("c", "idea c", iteration=2, surprise=0.5, valid=True)
    store.update_tags("a", components=["value_embeddings"], technique_tags=["soft_moe"])
    store.update_tags("b", components=["value_embeddings"], technique_tags=["soft_moe"])
    store.update_tags("c", components=["attention"], technique_tags=["gqa"])

    rng = np.random.default_rng(0)
    counts: dict[str, int] = {}
    for _ in range(400):
        run_id = select_anchor(
            store,
            candidate_window=20,
            tau=1.0,
            rng=rng,
            anchor_history=["a"],
            M=5,
            tag_repetition=True,
        )
        counts[run_id] = counts.get(run_id, 0) + 1

    assert counts.get("c", 0) > counts.get("b", 0)


def test_apply_memory_discount_full_when_memory_perfect():
    prediction = Prediction(predicted_valid=True, predicted_fitness=0.5)
    surprise, explanation = apply_memory_discount(
        raw_surprise=2.0,
        prediction=prediction,
        mem_pred_score=1.0,
        mem_pred_valid=True,
        actual_fitness=1.0,
        actual_valid=True,
        sigma=1.0,
        alpha=1.0,
    )

    assert explanation == pytest.approx(1.0)
    assert surprise == pytest.approx(0.0)


def test_curiosity_plus_tags_ideas_inline(embedder):
    strategy = CuriosityPlusSearch(embedder, tag_classifier=FakeTagClassifier())
    meta = strategy.on_result(
        "r1",
        score=0.9,
        idea="Replace value_embeddings with a Soft MoE variant",
        parent_ids=[],
        valid=True,
    )

    assert "curiosity_components" in meta
    assert "value_embeddings" in meta["curiosity_components"]
    assert "value_embeddings" in strategy.store.get_tag_set("r1")


def test_jaccard_distance_basic():
    assert jaccard_distance({"a", "b"}, {"a", "b"}) == 0.0
    assert jaccard_distance({"a"}, {"b"}) == 1.0
    assert abs(jaccard_distance({"a", "b"}, {"a", "c"}) - 2 / 3) < 1e-9
