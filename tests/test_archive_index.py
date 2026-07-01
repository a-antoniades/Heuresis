"""Tests for ArchiveIndex — in-memory buckets + KNN + anchor sampling."""
from __future__ import annotations

import threading

import numpy as np
import pytest

from heuresis.qd.core.archive_index import ArchiveIndex, Neighbor
from heuresis.qd.core.embedding import FakeEmbedder


@pytest.fixture
def index():
    return ArchiveIndex(embedder=FakeEmbedder(dim=16))


class TestAcceptedBucketAddAndQuery:
    def test_add_returns_none_and_grows_bucket(self, index):
        assert index.accepted_size == 0
        index.add_accepted("r1", "plan for r1", score=0.95)
        assert index.accepted_size == 1
        index.add_accepted("r2", "plan for r2", score=0.80)
        assert index.accepted_size == 2

    def test_top_k_from_text_returns_neighbors(self, index):
        index.add_accepted("r1", "plan one", score=0.1)
        index.add_accepted("r2", "plan two", score=0.2)
        index.add_accepted("r3", "plan three", score=0.3)
        neighbors = index.top_k_from_text("plan one", k=2, bucket="accepted")
        assert len(neighbors) == 2
        assert all(isinstance(n, Neighbor) for n in neighbors)
        # The closest to "plan one" should be r1 itself (same text → same hash → same seed)
        assert neighbors[0].run_id == "r1"
        # Similarity must be descending
        assert neighbors[0].similarity >= neighbors[1].similarity

    def test_top_k_from_run_id_excludes_self_by_default(self, index):
        index.add_accepted("r1", "plan", score=0.1)
        index.add_accepted("r2", "plan b", score=0.2)
        neighbors = index.top_k_from_run_id("r1", k=5, bucket="accepted")
        ids = {n.run_id for n in neighbors}
        assert "r1" not in ids
        assert "r2" in ids

    def test_top_k_from_run_id_include_self(self, index):
        index.add_accepted("r1", "plan", score=0.1)
        index.add_accepted("r2", "plan b", score=0.2)
        neighbors = index.top_k_from_run_id("r1", k=5, bucket="accepted", include_self=True)
        ids = {n.run_id for n in neighbors}
        assert "r1" in ids

    def test_top_k_empty_bucket(self, index):
        assert index.top_k_from_text("anything", k=5, bucket="accepted") == []

    def test_top_k_k_larger_than_bucket(self, index):
        index.add_accepted("r1", "plan", score=0.1)
        neighbors = index.top_k_from_text("other", k=10, bucket="accepted")
        assert len(neighbors) == 1

    def test_vectors_normalized_at_insert(self, index):
        index.add_accepted("r1", "plan", score=0.1)
        # Internal state check — normalized vectors have unit length
        vec = index._buckets["accepted"].matrix[0]
        np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-6)

    def test_metadata_preserved(self, index):
        index.add_accepted("r1", "plan one", score=0.95)
        entry = index.get_entry("r1")
        assert entry.run_id == "r1"
        assert entry.plan == "plan one"
        assert entry.bucket == "accepted"
        assert entry.score == 0.95

    def test_duplicate_run_id_overwrites(self, index):
        index.add_accepted("r1", "first plan", score=0.5)
        index.add_accepted("r1", "second plan", score=0.9)
        assert index.accepted_size == 1
        assert index.get_entry("r1").plan == "second plan"
        assert index.get_entry("r1").score == 0.9


class TestNeighborShape:
    def test_neighbor_fields(self, index):
        index.add_accepted("r1", "plan", score=0.5)
        n = index.top_k_from_text("plan", k=1, bucket="accepted")[0]
        assert n.run_id == "r1"
        assert 0.0 <= n.similarity <= 1.0 + 1e-6
        assert n.bucket == "accepted"
        assert n.plan == "plan"


class TestRejectedBuckets:
    def test_add_rejected_moi(self, index):
        index.add_rejected_moi("r1", "plan", reasoning="too similar to r5")
        assert index.size("failed_moi") == 1
        assert index.size("accepted") == 0
        entry = index.get_entry("r1")
        assert entry.bucket == "failed_moi"
        assert entry.reasoning == "too similar to r5"

    def test_add_rejected_train(self, index):
        index.add_rejected_train("r1", "plan", failure_mode="timeout")
        assert index.size("failed_train") == 1
        entry = index.get_entry("r1")
        assert entry.bucket == "failed_train"
        assert entry.failure_mode == "timeout"

    def test_top_k_filters_by_bucket(self, index):
        index.add_accepted("a1", "plan a", score=0.5)
        index.add_rejected_moi("m1", "plan a", reasoning="x")
        index.add_rejected_train("t1", "plan a", failure_mode="x")

        accepted_hits = index.top_k_from_text("plan a", k=5, bucket="accepted")
        moi_hits = index.top_k_from_text("plan a", k=5, bucket="failed_moi")
        train_hits = index.top_k_from_text("plan a", k=5, bucket="failed_train")

        assert {n.run_id for n in accepted_hits} == {"a1"}
        assert {n.run_id for n in moi_hits} == {"m1"}
        assert {n.run_id for n in train_hits} == {"t1"}

    def test_bucket_reassignment_on_re_add(self, index):
        # A run initially failed_train can be re-added as accepted
        # (e.g., resume path hydrates from scratch)
        index.add_rejected_train("r1", "plan", failure_mode="crash")
        assert index.size("failed_train") == 1
        index.add_accepted("r1", "plan", score=0.9)
        assert index.size("failed_train") == 0
        assert index.size("accepted") == 1
        assert index.get_entry("r1").bucket == "accepted"


class TestSampleAnchor:
    def test_empty_archive_returns_none(self, index):
        assert index.sample_anchor() is None

    def test_single_entry_always_returned(self, index):
        index.add_accepted("r1", "plan", score=0.5)
        for _ in range(10):
            assert index.sample_anchor() == "r1"

    def test_all_start_equiprobable(self, index):
        # With seed=42 and equal probs, we should see every run_id over many draws
        for i in range(5):
            index.add_accepted(f"r{i}", f"plan {i}", score=0.5)
        # Reset probs to pristine state before sampling
        picks = [index.sample_anchor(mark_used=False) for _ in range(500)]
        counts = {rid: picks.count(rid) for rid in picks}
        # Each should be sampled roughly 100 times, loose bound
        assert all(80 <= c <= 120 for c in counts.values()), counts

    def test_mark_used_decays_probability(self, index):
        for i in range(3):
            index.add_accepted(f"r{i}", f"plan {i}", score=0.5)
        # If r0 is marked used, subsequent sampling should avoid r0 until
        # other probabilities regrow.
        first = index.sample_anchor()
        # After mark_used the picked id has prob 0; others have had +=1
        assert first in {"r0", "r1", "r2"}
        # Next draw should NOT be `first` unless randomness lines up; run many draws
        picks = [index.sample_anchor(mark_used=False) for _ in range(100)]
        # first should appear meaningfully less often than the other two
        c_first = picks.count(first)
        c_others = sum(picks.count(rid) for rid in {"r0", "r1", "r2"} - {first})
        assert c_first < c_others

    def test_probs_regrow_over_time(self, index):
        for i in range(3):
            index.add_accepted(f"r{i}", f"plan {i}", score=0.5)
        index.sample_anchor()  # one "used" entry
        # After many increments (non-mark draws), all probs should be > 0 again
        for _ in range(20):
            index.sample_anchor(mark_used=False)
        probs = index.anchor_probs()
        assert all(p > 0 for p in probs)

    def test_new_accepted_gets_zero_prob(self, index):
        # New entries start at 0 probability (they've just been seen as
        # the chosen candidate); they regrow over time.
        index.add_accepted("r1", "plan", score=0.5)
        index.add_accepted("r2", "plan2", score=0.5)
        probs = index.anchor_probs()
        # Both start at 1.0, not 0, but OMNI-EPIC initializes `taskgen_choose_probs`
        # to 1 and zeros only on use. We follow that: new entries start at 1.0.
        assert probs[0] == 1.0
        assert probs[1] == 1.0


class TestPersistence:
    def test_save_embedding_called_on_add(self, tmp_path):
        import numpy as np

        from heuresis.models import RunResult
        from heuresis.store import ResultStore

        store = ResultStore(db_path=tmp_path / "store.db")
        exp = store.experiment("t", root=tmp_path / "runs")
        exp.save("r1", result=RunResult(workspace=tmp_path, exit_code=0, stats={}),
                 run_type="executor", iteration=0)
        index = ArchiveIndex(embedder=FakeEmbedder(dim=16), experiment=exp)
        index.add_accepted("r1", "plan one", score=0.95)

        stored = exp.get_embeddings(embedder="fake-v1", text_kind="plan")
        assert "r1" in stored
        np.testing.assert_allclose(
            np.linalg.norm(stored["r1"]), 1.0, atol=1e-6
        )

    def test_rebuild_from_experiment_restores_state(self, tmp_path):
        from heuresis.models import RunResult
        from heuresis.store import ResultStore

        store = ResultStore(db_path=tmp_path / "store.db")
        exp = store.experiment("t", root=tmp_path / "runs")
        for i, (rid, plan, score) in enumerate([
            ("r1", "plan one", 0.9),
            ("r2", "plan two", 0.8),
            ("r3", "plan three", 0.7),
        ]):
            exp.save(rid, result=RunResult(workspace=tmp_path, exit_code=0, stats={}),
                     run_type="executor", iteration=i, valid=True,
                     idea=plan, metadata={"best_score": score, "omniepic_bucket": "accepted"})

        # Populate index #1
        idx1 = ArchiveIndex(embedder=FakeEmbedder(dim=16), experiment=exp)
        idx1.add_accepted("r1", "plan one", score=0.9)
        idx1.add_accepted("r2", "plan two", score=0.8)
        idx1.add_accepted("r3", "plan three", score=0.7)

        # Fresh index, rebuild from DB
        idx2 = ArchiveIndex(embedder=FakeEmbedder(dim=16), experiment=exp)
        idx2.rebuild_from_experiment(exp)

        assert idx2.accepted_size == 3
        # Top-k from idx2 must match idx1
        hits1 = idx1.top_k_from_text("plan one", k=3, bucket="accepted")
        hits2 = idx2.top_k_from_text("plan one", k=3, bucket="accepted")
        assert [h.run_id for h in hits1] == [h.run_id for h in hits2]

    def test_rebuild_assigns_buckets_from_metadata(self, tmp_path):
        from heuresis.models import RunResult
        from heuresis.store import ResultStore

        store = ResultStore(db_path=tmp_path / "store.db")
        exp = store.experiment("t", root=tmp_path / "runs")
        # One run per bucket
        exp.save("a1", result=RunResult(workspace=tmp_path, exit_code=0, stats={}),
                 run_type="executor", valid=True, iteration=0,
                 idea="plan a", metadata={"best_score": 0.9, "omniepic_bucket": "accepted"})
        exp.save("m1", result=RunResult(workspace=tmp_path, exit_code=0, stats={}),
                 run_type="executor", valid=False, iteration=1,
                 idea="plan m", metadata={"omniepic_bucket": "failed_moi",
                                          "omniepic_reasoning": "too similar"})
        exp.save("t1", result=RunResult(workspace=tmp_path, exit_code=0, stats={}),
                 run_type="executor", valid=False, iteration=2,
                 idea="plan t", metadata={"omniepic_bucket": "failed_train",
                                          "omniepic_failure_mode": "timeout"})

        idx = ArchiveIndex(embedder=FakeEmbedder(dim=16), experiment=exp)
        idx.add_accepted("a1", "plan a", score=0.9)
        idx.add_rejected_moi("m1", "plan m", reasoning="too similar")
        idx.add_rejected_train("t1", "plan t", failure_mode="timeout")

        idx2 = ArchiveIndex(embedder=FakeEmbedder(dim=16), experiment=exp)
        idx2.rebuild_from_experiment(exp)

        assert idx2.size("accepted") == 1
        assert idx2.size("failed_moi") == 1
        assert idx2.size("failed_train") == 1


class TestConcurrency:
    def test_concurrent_reads_during_writes(self):
        """N writer threads + M reader threads, no crashes, no data corruption."""
        index = ArchiveIndex(embedder=FakeEmbedder(dim=16))
        errors: list[Exception] = []

        def writer(thread_id: int) -> None:
            try:
                for i in range(50):
                    index.add_accepted(f"w{thread_id}_r{i}", f"plan {thread_id} {i}",
                                       score=float(i))
            except Exception as e:
                errors.append(e)

        def reader(thread_id: int) -> None:
            try:
                for _ in range(100):
                    # Mix of read operations
                    index.top_k_from_text("plan", k=3, bucket="accepted")
                    index.sample_anchor(mark_used=False)
                    index.accepted_size
            except Exception as e:
                errors.append(e)

        writers = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        readers = [threading.Thread(target=reader, args=(i,)) for i in range(4)]
        for t in writers + readers:
            t.start()
        for t in writers + readers:
            t.join()

        assert not errors, f"concurrency errors: {errors}"
        assert index.accepted_size == 4 * 50, f"expected 200 rows, got {index.accepted_size}"


class TestCrossBucketAnchor:
    def test_anchor_in_failed_moi_query_accepted(self, index):
        """Anchor lives in one bucket, retrieve from a different bucket.

        This is the path LocalMoIReviewer uses on retry: the ideator's
        rejected candidate becomes the anchor, and we retrieve its closest
        accepted neighbors to show the reviewer what it was "too close to."
        """
        index.add_accepted("a1", "muon optimizer on matrix params", score=0.9)
        index.add_accepted("a2", "depth 8 to 16 with FFN halved", score=0.85)
        index.add_rejected_moi(
            "m1", "muon on all 2D weights with AdamW elsewhere",
            reasoning="too close to a1",
        )

        hits = index.top_k_from_run_id("m1", k=3, bucket="accepted")
        assert len(hits) == 2
        assert hits[0].run_id == "a1"
        assert hits[0].similarity >= hits[1].similarity


def test_add_accepted_logs_warning_when_save_embedding_fails(
    caplog: "pytest.LogCaptureFixture",
) -> None:
    """A failing save_embedding must NOT crash the loop; it logs a warning."""
    from unittest.mock import MagicMock

    from heuresis.qd.core.archive_index import ArchiveIndex
    from heuresis.qd.core.embedding import FakeEmbedder

    bad_exp = MagicMock()
    bad_exp.save_embedding.side_effect = RuntimeError("disk full")
    idx = ArchiveIndex(embedder=FakeEmbedder(dim=16), experiment=bad_exp)
    with caplog.at_level("WARNING"):
        idx.add_accepted(run_id="r1", plan="some plan", score=0.9)
    # Entry was still added in-memory.
    assert idx.accepted_size == 1
    # Warning was logged.
    assert any("save_embedding" in rec.message for rec in caplog.records)
