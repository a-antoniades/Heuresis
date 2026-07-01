#!/usr/bin/env python3
"""OmniEpicSearch strategy smoke — exercises on_result / on_moi_rejected
without real training.

Complements scripts/smoke/integration.py (primitives) and the live
nanogpt_omni_epic_smoke experiment (pipeline + training). Run this to
verify the strategy-level bucket routing end-to-end with the MoI gate
hot, in seconds rather than minutes.

Usage:
    QD_STORE_PATH=runs/nanogpt/store.db QD_SOURCE_EXPERIMENT=<experiment_id> \
        uv run scripts/smoke/strategy.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from heuresis.qd import ArchiveIndex, GeminiEmbedder, MoIReviewer
from heuresis.qd.omni_epic.search import OmniEpicSearch
from heuresis.store import ResultStore
from heuresis.tasks import task_dir


STORE_PATH = Path(os.environ.get("QD_STORE_PATH", "runs/nanogpt/store.db"))
SOURCE_EXPERIMENT = os.environ.get("QD_SOURCE_EXPERIMENT", "nanogpt-linear")
NUM_SEEDS = 15


def load_real_accepted(store: ResultStore, n: int) -> list[dict]:
    runs = store.query(
        "select run_id, idea, score, metadata from runs "
        "where experiment_id=? and run_type='executor' "
        "and valid=1 and score is not null and idea is not null "
        "order by run_id limit ?",
        (SOURCE_EXPERIMENT, n),
    )
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

    store = ResultStore(STORE_PATH)
    real = load_real_accepted(store, NUM_SEEDS)
    print(f"Loaded {len(real)} real accepted runs for seeding")

    print()
    print("Building ArchiveIndex + MoIReviewer + OmniEpicSearch...")
    embedder = GeminiEmbedder()
    idx = ArchiveIndex(embedder=embedder)
    reviewer = MoIReviewer(
        idx, task_dir("nanogpt"), min_archive_size=5,
    )
    strategy = OmniEpicSearch(idx, reviewer, lower_is_better=True)

    # --- Seed the archive via on_result (simulating many completed iterations) ---
    print()
    print("=" * 78)
    print(f"Seeding: call strategy.on_result() for {len(real)} real runs")
    print("=" * 78)
    for r in real:
        meta = strategy.on_result(
            run_id=f"seed_{r['run_id']}",
            score=r["score"],
            idea=r["idea"],
            parent_ids=[],
            ideator_id=0,
        )
        bucket = meta["omniepic_bucket"]
        assert bucket == "accepted", f"expected accepted, got {bucket} for {r['run_id']}"
    print(f"  ✓ all {len(real)} routed to accepted")
    print(f"  {strategy.summary()}")

    # --- Simulate a training failure via on_result with score=None ---
    print()
    print("=" * 78)
    print("Test: strategy.on_result(score=None) routes to failed_train")
    print("=" * 78)
    meta = strategy.on_result(
        run_id="exec_failed_train_1",
        score=None,
        idea="A synthetic idea that would have failed training.",
        parent_ids=["seed_" + real[0]["run_id"]],
        ideator_id=0,
    )
    assert meta["omniepic_bucket"] == "failed_train", f"unexpected bucket {meta['omniepic_bucket']}"
    assert meta["omniepic_failure_mode"] == "training failed or invalid"
    print(f"  ✓ bucket={meta['omniepic_bucket']}, failure_mode={meta['omniepic_failure_mode']}")
    print(f"  generation computed: {meta['generation']}")
    print(f"  {strategy.summary()}")

    # --- Select parents + verify post-seed MoI gate ---
    print()
    print("=" * 78)
    print("Test: select_parents + MoI gate (post-seed, archive > min)")
    print("=" * 78)
    parents = strategy.select_parents(ideator_id=0)
    print(f"  select_parents returned: {parents}")
    assert len(parents) == 1, "expected exactly one anchor"

    # A candidate that's very similar to existing AdamW entries → expect reject
    trivial = "Use AdamW with learning rate 1e-3 and cosine schedule over the 20 minute budget."
    print(f"  candidate (trivial): {trivial[:80]!r}...")
    assessment = strategy.review_idea(trivial)
    print(f"  MoI verdict: interesting={assessment.interesting}")
    print(f"  reasoning: {assessment.reasoning[:150]}...")
    print(f"  retrieved_ids ({len(assessment.retrieved_ids)}): {assessment.retrieved_ids[:3]}...")
    assert len(assessment.retrieved_ids) > 0, "post-seed gate should retrieve examples"

    # --- Route the MoI result ---
    print()
    print("=" * 78)
    print("Test: on_moi_rejected vs on_result based on MoI verdict")
    print("=" * 78)
    if assessment.interesting:
        meta = strategy.on_result(
            run_id="exec_moi_accepted_1", score=0.97, idea=trivial,
            parent_ids=parents,
        )
        print(f"  MoI accepted → on_result → bucket={meta['omniepic_bucket']}")
        assert meta["omniepic_bucket"] == "accepted"
    else:
        meta = strategy.on_moi_rejected(
            run_id="exec_moi_rejected_1", idea=trivial, assessment=assessment,
            parent_ids=parents,
        )
        print(f"  MoI rejected → on_moi_rejected → bucket={meta['omniepic_bucket']}")
        assert meta["omniepic_bucket"] == "failed_moi"
        assert meta["omniepic_reasoning"] == assessment.reasoning
        assert meta["omniepic_retrieved_ids"] == assessment.retrieved_ids
    print(f"  {strategy.summary()}")

    # --- Verify anchor probabilities evolve across multiple samplings ---
    print()
    print("=" * 78)
    print("Test: repeated select_parents diversifies anchors")
    print("=" * 78)
    seen = set()
    for _ in range(5):
        p = strategy.select_parents(ideator_id=0)
        if p:
            seen.add(p[0])
    print(f"  sampled {len(seen)} distinct anchors across 5 calls (of {idx.accepted_size} accepted)")
    assert len(seen) >= 2, "expected at least 2 distinct anchors"

    print()
    print("=" * 78)
    print("STRATEGY SMOKE PASSED")
    print("=" * 78)
    print(f"  {strategy.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
