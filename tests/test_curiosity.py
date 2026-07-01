"""Tests for curiosity search primitives (Milestones 1–4)."""

import numpy as np
import pytest

from heuresis.qd.core.embedding import FakeEmbedder
from heuresis.qd.curiosity.curiosity_signals import (
    build_curiosity_context,
    build_prediction_context,
)
from heuresis.qd.curiosity.embedding_store import EmbeddingStore
from heuresis.qd.curiosity.learning_progress import (
    compute_all_lp,
    compute_lp,
)
from heuresis.qd.curiosity.prediction import (
    _extract_text_from_agent_log,
    parse_prediction,
    predict_outcome,
)
from heuresis.qd.curiosity.search import CuriositySearch
from heuresis.qd.curiosity.seeding import parse_candidates, select_seed
from heuresis.qd.curiosity.selection import select_anchor
from heuresis.qd.curiosity.surprise import (
    Prediction,
    SigmaTracker,
    surprise,
)


# ---------------------------------------------------------------------------
# Embedding Store
# ---------------------------------------------------------------------------

@pytest.fixture
def embedder():
    return FakeEmbedder(dim=16)


@pytest.fixture
def store(embedder):
    return EmbeddingStore(embedder)


class TestEmbeddingStore:
    def test_add_and_size(self, store):
        store.add("r1", "idea one", iteration=0)
        store.add("r2", "idea two", iteration=1)
        assert store.size == 2

    def test_get(self, store):
        store.add("r1", "idea one", iteration=0, surprise=0.5)
        entry = store.get("r1")
        assert entry.run_id == "r1"
        assert entry.surprise == 0.5
        assert entry.iteration == 0

    def test_update_surprise(self, store):
        store.add("r1", "idea one", iteration=0)
        assert store.get("r1").surprise is None
        store.update_surprise("r1", 0.75, score=0.95, valid=True)
        entry = store.get("r1")
        assert entry.surprise == 0.75
        assert entry.score == 0.95
        assert entry.valid is True

    def test_knn_returns_nearest(self, store):
        store.add("r1", "alpha", iteration=0)
        store.add("r2", "beta", iteration=1)
        store.add("r3", "gamma", iteration=2)
        # Query by r1's vector — r1 should be nearest to itself
        neighbors = store.knn_by_run_id("r1", k=2, include_self=True)
        assert len(neighbors) == 2
        assert neighbors[0][0].run_id == "r1"
        # Similarities should be descending
        assert neighbors[0][1] >= neighbors[1][1]

    def test_knn_exclude_self(self, store):
        store.add("r1", "alpha", iteration=0)
        store.add("r2", "beta", iteration=1)
        neighbors = store.knn_by_run_id("r1", k=5, include_self=False)
        run_ids = {e.run_id for e, _ in neighbors}
        assert "r1" not in run_ids

    def test_farthest_point_empty_store(self, store, embedder):
        candidates = embedder.embed(["a", "b", "c"])
        idx = store.farthest_point(candidates)
        assert idx == 0  # default when store is empty

    def test_farthest_point_selects_distant(self, store, embedder):
        # Add one idea, then pick the candidate farthest from it
        store.add("r1", "close idea", iteration=0)
        candidates = embedder.embed(["close idea", "very different thing", "another variant"])
        idx = store.farthest_point(candidates)
        # "close idea" should NOT be selected (it's identical to r1)
        assert idx != 0

    def test_novelty_check_empty(self, store, embedder):
        vec = embedder.embed_one("anything")
        assert store.novelty_check(vec, threshold=0.9) is True

    def test_novelty_check_duplicate(self, store, embedder):
        store.add("r1", "exact text", iteration=0)
        vec = embedder.embed_one("exact text")
        # Same text → same embedding → similarity ~1.0 → not novel
        assert store.novelty_check(vec, threshold=0.9) is False

    def test_recent_entries(self, store):
        for i in range(10):
            store.add(f"r{i}", f"idea {i}", iteration=i)
        recent = store.recent_entries(3)
        assert len(recent) == 3
        assert recent[0].run_id == "r7"
        assert recent[2].run_id == "r9"

    def test_overwrite_existing_run_id(self, store):
        store.add("r1", "original", iteration=0, surprise=0.5)
        store.add("r1", "updated", iteration=1, surprise=0.8)
        assert store.size == 1
        assert store.get("r1").idea == "updated"
        assert store.get("r1").surprise == 0.8

    def test_save_load(self, store, tmp_path, embedder):
        store.add("r1", "idea one", iteration=0, surprise=0.5, score=0.95, valid=True)
        store.add("r2", "idea two", iteration=1)
        path = tmp_path / "store.npz"
        store.save(path)

        loaded = EmbeddingStore(embedder)
        loaded.load(path)
        assert loaded.size == 2
        e1 = loaded.get("r1")
        assert e1.surprise == 0.5
        assert e1.score == 0.95
        assert e1.valid is True
        e2 = loaded.get("r2")
        assert e2.surprise is None
        assert e2.score is None
        assert e2.valid is None


# ---------------------------------------------------------------------------
# Surprise
# ---------------------------------------------------------------------------


class TestSurprise:
    def test_none_prediction(self):
        assert surprise(None, True, 0.95) is None

    def test_both_invalid(self):
        pred = Prediction(predicted_valid=False, predicted_fitness=None)
        assert surprise(pred, False, None) == 0.0

    def test_validity_mismatch_predicted_valid(self):
        pred = Prediction(predicted_valid=True, predicted_fitness=0.95)
        s = surprise(pred, False, None, lambda_mismatch=1.5)
        assert s == 1.5

    def test_validity_mismatch_predicted_invalid(self):
        pred = Prediction(predicted_valid=False, predicted_fitness=None)
        s = surprise(pred, True, 0.95)
        assert s == 1.0  # default lambda

    def test_both_valid_no_sigma(self):
        pred = Prediction(predicted_valid=True, predicted_fitness=0.95)
        s = surprise(pred, True, 0.90)
        assert s == pytest.approx(0.05)

    def test_both_valid_with_sigma(self):
        tracker = SigmaTracker(window=50)
        for v in [0.90, 0.92, 0.94, 0.96, 0.98]:
            tracker.observe(v)
        pred = Prediction(predicted_valid=True, predicted_fitness=0.95)
        s = surprise(pred, True, 0.90, sigma_tracker=tracker)
        assert s is not None
        assert s > 0


class TestSigmaTracker:
    def test_default_sigma(self):
        tracker = SigmaTracker()
        assert tracker.sigma == 1.0  # default before data

    def test_single_value(self):
        tracker = SigmaTracker()
        tracker.observe(0.5)
        assert tracker.sigma == 1.0  # need ≥2

    def test_known_values(self):
        tracker = SigmaTracker()
        tracker.observe(2.0)
        tracker.observe(4.0)
        # std of [2, 4] with ddof=1 = sqrt(2) ≈ 1.414
        assert tracker.sigma == pytest.approx(np.std([2.0, 4.0], ddof=1), rel=1e-6)

    def test_window_limit(self):
        tracker = SigmaTracker(window=3)
        for v in [1.0, 2.0, 3.0, 100.0]:
            tracker.observe(v)
        assert tracker.count == 3
        # Only [2.0, 3.0, 100.0] in window


# ---------------------------------------------------------------------------
# Learning Progress
# ---------------------------------------------------------------------------


class TestLearningProgress:
    def _build_store_with_trajectory(self, embedder, surprises):
        """Build a store where all ideas are similar (same neighborhood)."""
        store = EmbeddingStore(embedder)
        for i, s in enumerate(surprises):
            # Use slightly varied ideas so embeddings differ but remain close
            store.add(f"r{i}", f"similar idea variant {i}", iteration=i, surprise=s)
        return store

    def test_decreasing_surprise_positive_lp(self, embedder):
        # Surprise decreasing over time → LP should be positive
        surprises = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        store = self._build_store_with_trajectory(embedder, surprises)
        lp, confident = compute_lp("r5", store, k=10)
        assert lp > 0

    def test_increasing_surprise_negative_lp(self, embedder):
        # Surprise increasing → LP should be negative
        surprises = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        store = self._build_store_with_trajectory(embedder, surprises)
        lp, confident = compute_lp("r5", store, k=10)
        assert lp < 0

    def test_flat_surprise_zero_lp(self, embedder):
        # Constant surprise → LP ≈ 0
        surprises = [0.5] * 10
        store = self._build_store_with_trajectory(embedder, surprises)
        lp, confident = compute_lp("r5", store, k=10)
        assert lp == pytest.approx(0.0)

    def test_too_few_neighbors(self, embedder):
        store = EmbeddingStore(embedder)
        store.add("r0", "only idea", iteration=0, surprise=0.5)
        lp, confident = compute_lp("r0", store, k=10)
        assert lp == 0.0
        assert confident is False

    def test_compute_all_lp(self, embedder):
        surprises = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        store = self._build_store_with_trajectory(embedder, surprises)
        results = compute_all_lp(
            [f"r{i}" for i in range(10)], store, k=10,
        )
        assert len(results) == 10
        for rid, lp, confident in results:
            assert isinstance(lp, float)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


class TestSelection:
    def test_empty_store(self, store):
        result = select_anchor(store, candidate_window=20)
        assert result is None

    def test_single_entry(self, store):
        store.add("r1", "idea one", iteration=0, surprise=0.5, valid=True)
        result = select_anchor(store, candidate_window=20)
        assert result == "r1"

    def test_returns_valid_run_id(self, embedder):
        store = EmbeddingStore(embedder)
        for i in range(10):
            store.add(
                f"r{i}", f"idea {i}", iteration=i,
                surprise=float(i) / 10, valid=True,
            )
        rng = np.random.default_rng(42)
        result = select_anchor(store, candidate_window=20, tau=1.0, rng=rng)
        assert result is not None
        assert result.startswith("r")

    def test_high_surprise_preferred(self, embedder):
        # Small store → LP not confident → per-entry fallback to raw surprise,
        # so the high-surprise entry should still be preferred.
        store = EmbeddingStore(embedder)
        store.add("low1", "boring idea a", iteration=0, surprise=0.01, valid=True)
        store.add("low2", "boring idea b", iteration=1, surprise=0.01, valid=True)
        store.add("high", "surprising idea", iteration=2, surprise=10.0, valid=True)
        store.add("low3", "boring idea c", iteration=3, surprise=0.01, valid=True)

        counts: dict[str, int] = {}
        rng = np.random.default_rng(42)
        for _ in range(200):
            r = select_anchor(store, candidate_window=20, tau=0.5, rng=rng)
            counts[r] = counts.get(r, 0) + 1
        assert counts.get("high", 0) > counts.get("low1", 0)

    def test_repetition_penalty(self, embedder):
        store = EmbeddingStore(embedder)
        for i in range(5):
            store.add(f"r{i}", f"idea {i}", iteration=i, surprise=0.5, valid=True)

        rng = np.random.default_rng(42)
        history = ["r0", "r0", "r0", "r0", "r0"]
        counts_with: dict[str, int] = {}
        for _ in range(200):
            r = select_anchor(
                store, candidate_window=20, tau=1.0,
                anchor_history=history, M=5, rng=rng,
            )
            counts_with[r] = counts_with.get(r, 0) + 1

        counts_without: dict[str, int] = {}
        rng2 = np.random.default_rng(42)
        for _ in range(200):
            r = select_anchor(
                store, candidate_window=20, tau=1.0,
                anchor_history=[], M=5, rng=rng2,
            )
            counts_without[r] = counts_without.get(r, 0) + 1

        assert counts_with.get("r0", 0) <= counts_without.get("r0", 0)

    def test_skips_invalid_and_inflight(self, embedder):
        # Realistic store: some valid scored runs, an invalid run, and an
        # in-flight seed reservation (valid=None). select_anchor should
        # only return one of the valid entries.
        store = EmbeddingStore(embedder)
        store.add("ok1", "good idea a", iteration=0, surprise=0.3, valid=True)
        store.add("ok2", "good idea b", iteration=1, surprise=0.4, valid=True)
        store.add("failed", "broken idea", iteration=2, surprise=0.9, valid=False)
        store.add("inflight", "reserved seed", iteration=3)  # valid=None
        rng = np.random.default_rng(0)
        picks = {select_anchor(store, candidate_window=20, rng=rng) for _ in range(50)}
        assert picks.issubset({"ok1", "ok2"})
        assert "failed" not in picks
        assert "inflight" not in picks

    def test_returns_none_when_nothing_valid(self, embedder):
        store = EmbeddingStore(embedder)
        store.add("failed", "broken idea", iteration=0, surprise=0.9, valid=False)
        store.add("inflight", "reserved", iteration=1)  # valid=None
        assert select_anchor(store, candidate_window=20) is None


# ---------------------------------------------------------------------------
# Seeding (M5)
# ---------------------------------------------------------------------------


class TestSeeding:
    def test_select_seed_empty_store(self, embedder):
        store = EmbeddingStore(embedder)
        idx, idea = select_seed(store, ["a", "b", "c"])
        assert idx == 0  # first when store is empty
        assert idea == "a"

    def test_select_seed_picks_distant(self, embedder):
        store = EmbeddingStore(embedder)
        store.add("r1", "anchor idea", iteration=0)
        # Same text → identical embedding; expect a different candidate
        idx, idea = select_seed(store, ["anchor idea", "very different concept", "another option"])
        assert idx != 0

    def test_select_seed_no_candidates_raises(self, embedder):
        store = EmbeddingStore(embedder)
        with pytest.raises(ValueError):
            select_seed(store, [])

    def test_parse_candidates_dashes(self):
        text = """## Idea 1
First idea body
---
## Idea 2
Second idea body
---
## Idea 3
Third idea body"""
        candidates = parse_candidates(text, expected=3)
        assert len(candidates) == 3
        assert "First idea body" in candidates[0]
        assert "Third idea body" in candidates[2]

    def test_parse_candidates_numbered(self):
        text = """## Idea 1
A
## Idea 2
B"""
        candidates = parse_candidates(text, expected=2)
        assert len(candidates) >= 2

    def test_parse_candidates_fallback_single(self):
        text = "Just one idea, no separators."
        candidates = parse_candidates(text, expected=5)
        assert len(candidates) == 1
        assert "Just one idea" in candidates[0]


# ---------------------------------------------------------------------------
# Prediction parsing (M2 / M7)
# ---------------------------------------------------------------------------


class TestParsePrediction:
    def test_clean_json(self):
        raw = '{"valid": true, "fitness": 0.95, "confidence": 0.7, "reasoning": "test"}'
        p = parse_prediction(raw)
        assert p is not None
        assert p.predicted_valid is True
        assert p.predicted_fitness == 0.95
        assert p.confidence == 0.7
        assert p.reasoning == "test"

    def test_predicted_invalid_with_null_fitness(self):
        raw = '{"valid": false, "fitness": null, "reasoning": "will fail"}'
        p = parse_prediction(raw)
        assert p is not None
        assert p.predicted_valid is False
        assert p.predicted_fitness is None

    def test_string_bool(self):
        raw = '{"valid": "true", "fitness": 0.9, "reasoning": ""}'
        p = parse_prediction(raw)
        assert p is not None
        assert p.predicted_valid is True

    def test_extract_from_noisy(self):
        raw = 'Here is my prediction:\n```json\n{"valid": true, "fitness": 0.88, "reasoning": "good"}\n```\nDone.'
        p = parse_prediction(raw)
        assert p is not None
        assert p.predicted_fitness == 0.88

    def test_invalid_json_returns_none(self):
        assert parse_prediction("not json at all") is None
        assert parse_prediction("") is None
        assert parse_prediction("   ") is None

    def test_missing_valid_field_returns_none(self):
        raw = '{"fitness": 0.95, "reasoning": "no valid field"}'
        assert parse_prediction(raw) is None


class TestExtractTextFromAgentLog:
    def test_missing_file_returns_none(self, tmp_path):
        assert _extract_text_from_agent_log(tmp_path / "nope.log") is None

    def test_last_text_event_wins(self, tmp_path):
        # JSONL with two text events — last one should be picked up. The
        # agent wrote the JSON as chat text inside a ```json fence (the
        # failure mode we hit on tid=1/tid=2 in the bbob_curiosity smoke).
        log = tmp_path / "agent.log"
        log.write_text(
            '{"type": "step_start", "part": {}}\n'
            '{"type": "text", "part": {"text": "interim thought"}}\n'
            '{"type": "text", "part": {"text": "```json\\n'
            '{\\"valid\\": true, \\"fitness\\": -1.2, \\"reasoning\\": \\"x\\"}\\n```"}}\n'
            '{"type": "step_finish", "part": {}}\n'
        )
        text = _extract_text_from_agent_log(log)
        assert text is not None
        pred = parse_prediction(text)
        assert pred is not None
        assert pred.predicted_valid is True
        assert pred.predicted_fitness == -1.2

    def test_no_text_events_returns_none(self, tmp_path):
        log = tmp_path / "agent.log"
        log.write_text('{"type": "step_start", "part": {}}\n')
        assert _extract_text_from_agent_log(log) is None

    def test_malformed_lines_are_skipped(self, tmp_path):
        log = tmp_path / "agent.log"
        log.write_text(
            'not json\n'
            '{"type": "text", "part": {"text": "ok"}}\n'
            '   \n'
        )
        assert _extract_text_from_agent_log(log) == "ok"


class TestPredictOutcomeRetry:
    """predict_outcome must retry once on parse failure, then give up."""

    def test_retries_then_succeeds(self, tmp_path):
        """First attempt emits garbage; second attempt emits valid JSON."""
        calls = {"n": 0}
        valid_json = (
            '{"valid": true, "fitness": -1.5, "reasoning": "ok", "confidence": 0.8}'
        )

        class FakeHandle:
            def result(self_inner):
                calls["n"] += 1
                # First attempt: write garbage. Second: write valid JSON.
                if calls["n"] == 1:
                    (tmp_path / "prediction.json").write_text("not json")
                else:
                    (tmp_path / "prediction.json").write_text(valid_json)

        class FakeHarness:
            def run(self, *a, **kw):
                return FakeHandle()

        pred = predict_outcome(
            FakeHarness(), None, tmp_path,
            prompt_vars={}, timeout=1, max_attempts=2,
        )
        assert calls["n"] == 2
        assert pred is not None
        assert pred.predicted_valid is True
        assert pred.predicted_fitness == -1.5

    def test_returns_none_after_all_attempts_fail(self, tmp_path):
        """Every attempt emits garbage → max_attempts exhausted → None."""
        calls = {"n": 0}

        class FakeHandle:
            def result(self_inner):
                calls["n"] += 1
                (tmp_path / "prediction.json").write_text("not json")

        class FakeHarness:
            def run(self, *a, **kw):
                return FakeHandle()

        pred = predict_outcome(
            FakeHarness(), None, tmp_path,
            prompt_vars={}, timeout=1, max_attempts=2,
        )
        assert calls["n"] == 2
        assert pred is None

    def test_no_retry_when_first_attempt_succeeds(self, tmp_path):
        """First attempt emits valid JSON → single harness call."""
        calls = {"n": 0}
        valid_json = '{"valid": true, "fitness": 0.5, "reasoning": "x"}'

        class FakeHandle:
            def result(self_inner):
                calls["n"] += 1
                (tmp_path / "prediction.json").write_text(valid_json)

        class FakeHarness:
            def run(self, *a, **kw):
                return FakeHandle()

        pred = predict_outcome(
            FakeHarness(), None, tmp_path,
            prompt_vars={}, timeout=1, max_attempts=2,
        )
        assert calls["n"] == 1
        assert pred is not None
        assert pred.predicted_fitness == 0.5


# ---------------------------------------------------------------------------
# Curiosity context (M6)
# ---------------------------------------------------------------------------


class TestCuriosityContext:
    def test_anchor_context_with_neighbors(self, embedder):
        store = EmbeddingStore(embedder)
        store.add("r1", "anchor idea here", iteration=0, surprise=0.3, score=0.95, valid=True)
        store.add("r2", "neighbor idea two", iteration=1, surprise=0.4, score=0.94, valid=True)
        store.add("r3", "neighbor idea three", iteration=2, surprise=0.2, score=0.93, valid=True)
        ctx = build_curiosity_context("r1", store, k=5)
        assert "r1" in ctx
        assert "Curiosity Anchor" in ctx
        assert "Why This Region" in ctx

    def test_anchor_context_unknown_id_raises(self, embedder):
        store = EmbeddingStore(embedder)
        with pytest.raises(KeyError):
            build_curiosity_context("nonexistent", store)

    def test_prediction_context_empty_store(self, embedder):
        store = EmbeddingStore(embedder)
        ctx = build_prediction_context(store)
        assert "No prior" in ctx

    def test_prediction_context_skips_entries_without_prediction(self, embedder):
        # Entries without a prediction (e.g. Phase 1 seeds) shouldn't appear
        store = EmbeddingStore(embedder)
        for i in range(5):
            store.add(
                f"r{i}", f"idea {i}", iteration=i,
                surprise=0.5, score=0.9 + i * 0.01, valid=True,
            )
        ctx = build_prediction_context(store, max_history=3)
        assert "No prior" in ctx

    def test_prediction_context_with_predictions(self, embedder):
        store = EmbeddingStore(embedder)
        for i in range(5):
            pred = Prediction(
                predicted_valid=True,
                predicted_fitness=0.95,
                reasoning=f"Thought this would work because reason {i}",
                confidence=0.6,
            )
            store.add(
                f"r{i}", f"idea {i}", iteration=i,
                surprise=0.5, score=0.9 + i * 0.01, valid=True,
                prediction=pred,
            )
        ctx = build_prediction_context(store, max_history=3)
        assert "Past Predictions" in ctx
        # Shows only last 3
        assert "r4" in ctx
        assert "r3" in ctx
        assert "r2" in ctx
        assert "r1" not in ctx
        # Shows predicted vs actual
        assert "Predicted:" in ctx
        assert "Actual:" in ctx
        assert "val_bpb=0.95" in ctx  # predicted
        # Shows the reasoning
        assert "reason 4" in ctx
        # Shows a verdict label
        assert any(
            v in ctx for v in ("on target", "somewhat off", "badly off", "VALIDITY MISS")
        )

    def test_prediction_context_validity_miss(self, embedder):
        store = EmbeddingStore(embedder)
        # Predicted valid but actually invalid
        pred = Prediction(
            predicted_valid=True,
            predicted_fitness=0.9,
            reasoning="should work",
        )
        store.add(
            "r1", "optimistic idea", iteration=0,
            surprise=1.0, score=None, valid=False,
            prediction=pred,
        )
        ctx = build_prediction_context(store)
        assert "VALIDITY MISS" in ctx


# ---------------------------------------------------------------------------
# CuriositySearch (M7)
# ---------------------------------------------------------------------------


class TestCuriositySearch:
    def test_base_strategy_rejects_plus_knobs(self, embedder):
        """Base curiosity must not expose curiosity-plus behavior switches."""
        with pytest.raises(TypeError):
            CuriositySearch(embedder, score_weight=0.5)
        with pytest.raises(TypeError):
            CuriositySearch(embedder, tag_classifier=object())
        with pytest.raises(TypeError):
            CuriositySearch(embedder, memory_strength=0.3)

    def test_base_stored_idea_has_no_plus_tag_fields(self, store):
        """Base curiosity stores text/embedding/outcome only, not plus tags."""
        store.add("r1", "plain base idea", iteration=0, score=0.9, valid=True)
        entry = store.get("r1")
        assert not hasattr(entry, "components")
        assert not hasattr(entry, "technique_tags")

    def test_seeding_transitions_to_steady(self, embedder):
        s = CuriositySearch(embedder, n_seed=3)
        assert s.is_seeding() is True
        for i in range(3):
            s.on_result(
                f"r{i}", score=0.9, idea=f"seed {i}",
                parent_ids=[], valid=True,
            )
        assert s.is_seeding() is False

    def test_select_parents_during_seeding_returns_empty(self, embedder):
        s = CuriositySearch(embedder, n_seed=10)
        assert s.select_parents(ideator_id=0) == []

    def test_select_parents_steady_state_returns_anchor(self, embedder):
        s = CuriositySearch(embedder, n_seed=2)
        for i in range(3):
            pred = Prediction(predicted_valid=True, predicted_fitness=0.9)
            s.on_result(
                f"r{i}", score=0.9 + i * 0.01, idea=f"idea {i}",
                parent_ids=[], prediction=pred, valid=True,
            )
        parents = s.select_parents(ideator_id=0)
        assert len(parents) == 1
        assert parents[0].startswith("r")

    def test_on_result_records_surprise(self, embedder):
        s = CuriositySearch(embedder)
        pred = Prediction(predicted_valid=True, predicted_fitness=0.95)
        meta = s.on_result(
            "r1", score=0.90, idea="test idea", parent_ids=[],
            prediction=pred, valid=True,
        )
        assert "curiosity_surprise" in meta
        assert "curiosity_phase" in meta
        assert meta["curiosity_predicted_valid"] is True
        assert meta["curiosity_predicted_fitness"] == 0.95

    def test_on_result_no_prediction_no_surprise(self, embedder):
        s = CuriositySearch(embedder)
        meta = s.on_result(
            "r1", score=0.90, idea="seed idea",
            parent_ids=[], prediction=None, valid=True,
        )
        assert "curiosity_surprise" not in meta
        assert "curiosity_predicted_valid" not in meta

    def test_select_seed_candidate(self, embedder):
        s = CuriositySearch(embedder)
        idx, idea = s.select_seed_candidate(["a", "b", "c"])
        assert idx == 0
        assert idea == "a"

    def test_summary(self, embedder):
        s = CuriositySearch(embedder)
        for i in range(3):
            s.on_result(f"r{i}", score=0.9, idea=f"i {i}", parent_ids=[], valid=True)
        summary = s.summary()
        assert "seeding" in summary or "steady" in summary
        assert "store=3" in summary

    def test_anchor_history_repetition_penalty(self, embedder):
        s = CuriositySearch(embedder, n_seed=2, anchor_history=2)
        for i in range(3):
            pred = Prediction(predicted_valid=True, predicted_fitness=0.9)
            s.on_result(f"r{i}", score=0.9, idea=f"i {i}", parent_ids=[], prediction=pred, valid=True)
        s.select_parents(ideator_id=0)
        s.select_parents(ideator_id=0)
        # History should be capped at anchor_history (M=2)
        assert len(s._anchor_history.get(0, [])) <= 2

    def test_rebuild_restores_generation(self, embedder):
        s = CuriositySearch(embedder)
        records = [
            ("r1", 0.9, {"generation": 0, "idea": "first"}),
            ("r2", 0.85, {"generation": 1, "idea": "second"}),
        ]
        s.rebuild(records)
        assert s._generation_map["r1"] == 0
        assert s._generation_map["r2"] == 1
        # σ tracker should have observed both scores
        assert s._sigma_tracker.count == 2
