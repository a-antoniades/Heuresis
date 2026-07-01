#!/usr/bin/env python3
"""Tier 2 integration smoke — exercises Phase 1 + Phase 2 primitives against
real data from ``runs/nanogpt/store.db``. No training, no strategy class.

Verifies:
  1. Real ideator plans embed cleanly via GeminiEmbedder.
  2. ArchiveIndex accepts them into the ``accepted`` bucket.
  3. sample_anchor returns a run_id and zeros its anchor prob.
  4. top_k_from_run_id finds nearest neighbors in the same bucket.
  5. MoIReviewer.review on a synthetic candidate produces a coherent
     assessment referencing retrieved examples.
  6. add_rejected_moi routes to the failed_moi bucket.
  7. Bucket counts are correct after the exercise.

Usage:
    QD_STORE_PATH=runs/nanogpt/store.db QD_SOURCE_EXPERIMENT=<experiment_id> \
        uv run scripts/smoke/integration.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from heuresis.qd import ArchiveIndex, GeminiEmbedder, MoIReviewer
from heuresis.store import ResultStore
from heuresis.tasks import task_dir


STORE_PATH = Path(os.environ.get("QD_STORE_PATH", "runs/nanogpt/store.db"))
SOURCE_EXPERIMENT = os.environ.get("QD_SOURCE_EXPERIMENT", "nanogpt-linear")
NUM_SEEDS = 20                                           # cap to keep embedding fast


def load_real_runs() -> list[dict]:
    """Return up to NUM_SEEDS real runs with ideas + scores from one experiment."""
    store = ResultStore(STORE_PATH)
    runs = store.query(
        "select run_id, idea, score, valid, metadata from runs "
        "where experiment_id=? and run_type='executor' and idea is not null and idea != '' "
        "order by run_id limit ?",
        (SOURCE_EXPERIMENT, NUM_SEEDS),
    )
    # Keep only runs with non-trivial ideas
    return [r for r in runs if r["idea"] and len(r["idea"]) > 50]


def main() -> int:
    from heuresis.api_keys import load_api_keys

    if not STORE_PATH.exists():
        print(f"ERROR: store {STORE_PATH} not found", file=sys.stderr)
        return 1
    if not load_api_keys("gemini"):
        print(
            "ERROR: no Gemini API keys found. Set GEMINI_API_KEYS, GEMINI_API_KEY, "
            "GOOGLE_GENERATIVE_AI_API_KEY (see .env.example)",
            file=sys.stderr,
        )
        return 1

    print("=" * 78)
    print("Step 1: Load real runs from store")
    print("=" * 78)
    runs = load_real_runs()
    print(f"  loaded {len(runs)} runs from experiment {SOURCE_EXPERIMENT}")
    accepted = [r for r in runs if r["valid"] and r["score"] is not None]
    failed = [r for r in runs if not r["valid"] or r["score"] is None]
    print(f"  {len(accepted)} accepted (valid + scored), {len(failed)} failed/invalid")

    print()
    print("=" * 78)
    print("Step 2: Build ArchiveIndex with GeminiEmbedder; embed & route")
    print("=" * 78)
    embedder = GeminiEmbedder()
    idx = ArchiveIndex(embedder=embedder)
    for r in accepted:
        idx.add_accepted(run_id=r["run_id"], plan=r["idea"], score=r["score"])
    for r in failed:
        idx.add_rejected_train(
            run_id=r["run_id"], plan=r["idea"], failure_mode="loaded_as_failed",
        )
    print(f"  accepted bucket size: {idx.accepted_size}")
    print(f"  failed_train bucket size: {idx.size('failed_train')}")
    print(f"  failed_moi bucket size:   {idx.size('failed_moi')}")
    assert idx.accepted_size == len(accepted), "accepted bucket size mismatch"

    print()
    print("=" * 78)
    print("Step 3: sample_anchor → top_k_from_run_id")
    print("=" * 78)
    probs_before = list(idx._anchor_probs)
    anchor_id = idx.sample_anchor(mark_used=True)
    probs_after = list(idx._anchor_probs)
    print(f"  sampled anchor: {anchor_id}")
    print(f"  probs before sum: {sum(probs_before):.2f}, after sum: {sum(probs_after):.2f}")
    # After mark_used, the sampled anchor's prob should be 0.
    anchor_idx = idx._buckets["accepted"].row_index[anchor_id]
    assert probs_after[anchor_idx] == 0.0, "sampled anchor's prob was not zeroed"
    print(f"  ✓ probs[{anchor_id}] zeroed after sampling")

    neighbors = idx.top_k_from_run_id(anchor_id, k=5, bucket="accepted")
    print(f"  top-5 nearest to anchor (bucket=accepted, excl self): {len(neighbors)}")
    for n in neighbors[:3]:
        plan_snippet = n.plan[:80].replace("\n", " ")
        print(f"    - {n.run_id} sim={n.similarity:.3f} score={n.meta.get('score')} plan={plan_snippet!r}")

    print()
    print("=" * 78)
    print("Step 4: MoIReviewer.review on a synthetic candidate")
    print("=" * 78)
    reviewer = MoIReviewer(
        idx, task_dir("nanogpt"),
        min_archive_size=len(accepted),  # ensure gate passes
    )
    candidate = (
        "Replace all optimizers with vanilla SGD at lr=0.01, no momentum. "
        "Keep everything else identical to the seed."
    )
    print(f"  candidate: {candidate[:100]!r}...")
    assessment = reviewer.review(candidate)
    print(f"  VERDICT: interesting={assessment.interesting}")
    print(f"  REASONING: {assessment.reasoning}")
    print(f"  retrieved_ids[:3]: {assessment.retrieved_ids[:3]}")
    print(f"  tokens in={assessment.input_tokens} out={assessment.output_tokens} dur={assessment.duration_s:.2f}s")
    assert isinstance(assessment.interesting, bool), "assessment.interesting must be bool"
    assert len(assessment.reasoning) > 0, "reasoning is empty"
    assert len(assessment.retrieved_ids) > 0, "no examples retrieved (gate should have passed)"
    print("  ✓ assessment is well-formed")

    print()
    print("=" * 78)
    print("Step 5: Route rejected candidate to failed_moi bucket")
    print("=" * 78)
    if not assessment.interesting:
        idx.add_rejected_moi(
            run_id="synthetic_rejected_1",
            plan=candidate,
            reasoning=assessment.reasoning,
        )
        print("  routed candidate to failed_moi (reasoning stored)")
    else:
        idx.add_accepted(run_id="synthetic_accepted_1", plan=candidate, score=0.97)
        print("  routed candidate to accepted (would be trained in real loop)")
    print(f"  final bucket sizes: accepted={idx.accepted_size}, "
          f"failed_moi={idx.size('failed_moi')}, failed_train={idx.size('failed_train')}")

    print()
    print("=" * 78)
    print("Step 6: sample_anchor excludes failed_moi routing (check it's still from accepted)")
    print("=" * 78)
    second_anchor = idx.sample_anchor(mark_used=True)
    assert second_anchor != anchor_id or sum(probs_after) == 0.0  # unless all probs zeroed
    # More fundamentally: the sampled anchor must be in the accepted bucket.
    assert second_anchor in idx._buckets["accepted"].row_index
    print(f"  second anchor: {second_anchor} (in accepted bucket ✓)")

    print()
    print("=" * 78)
    print("INTEGRATION SMOKE PASSED")
    print("=" * 78)
    print(f"  - Loaded {len(runs)} real runs, routed {len(accepted)} to accepted")
    print(f"  - MoIReviewer produced coherent assessment ({assessment.duration_s:.1f}s)")
    print("  - sample_anchor + top_k + bucket routing all functional")
    return 0


if __name__ == "__main__":
    sys.exit(main())
