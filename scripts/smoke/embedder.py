#!/usr/bin/env python3
"""Live smoke test for GeminiEmbedder + ArchiveIndex.

Runs real API calls. Verifies:
  1. Embedding of 3 plans returns (3, 3072) float32.
  2. Cosine similarity between similar plans > dissimilar plans.
  3. ArchiveIndex top_k returns the expected neighbor ordering.

Usage:
    uv run scripts/smoke/embedder.py
"""

from __future__ import annotations

import sys

import numpy as np

from heuresis.qd.core.archive_index import ArchiveIndex
from heuresis.qd.core.embedding import GeminiEmbedder


PLANS = {
    "muon":       "Replace AdamW with Muon optimizer on matrix parameters; keep AdamW on norms and biases.",
    "muon_tweak": "Use Muon on all 2D weights and AdamW elsewhere; standard learning rate.",
    "depth":      "Increase model depth from 8 to 16 layers; halve the FFN width to keep parameter count constant.",
}


def main() -> int:
    from heuresis.api_keys import load_api_keys

    keys = load_api_keys("gemini")
    if not keys:
        print(
            "ERROR: no Gemini API keys found. Set GEMINI_API_KEYS, GEMINI_API_KEY, "
            "GOOGLE_GENERATIVE_AI_API_KEY (see .env.example)",
            file=sys.stderr,
        )
        return 1

    emb = GeminiEmbedder()
    print(f"Embedder: {emb.model}, dim={emb.dim}, keys={len(emb._api_keys)}")

    # --- Raw embedding check ---
    texts = list(PLANS.values())
    vecs = emb.embed(texts)
    assert vecs.shape == (3, 3072), f"unexpected shape {vecs.shape}"
    assert vecs.dtype == np.float32, f"unexpected dtype {vecs.dtype}"

    # Normalize and compare
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    nvecs = vecs / norms
    sim_muon_pair = float(nvecs[0] @ nvecs[1])       # muon vs muon_tweak
    sim_muon_depth = float(nvecs[0] @ nvecs[2])      # muon vs depth

    print(f"cos(muon, muon_tweak) = {sim_muon_pair:.4f}")
    print(f"cos(muon, depth)      = {sim_muon_depth:.4f}")
    assert sim_muon_pair > sim_muon_depth, (
        "Similar plans should have higher cosine similarity than dissimilar plans "
        f"({sim_muon_pair} !> {sim_muon_depth})"
    )

    # --- ArchiveIndex check ---
    index = ArchiveIndex(embedder=emb)
    for rid, plan in PLANS.items():
        index.add_accepted(rid, plan, score=0.5)

    hits = index.top_k_from_text(PLANS["muon"], k=3, bucket="accepted")
    print("\nTop-3 neighbors of muon plan:")
    for n in hits:
        print(f"  {n.run_id}: sim={n.similarity:.4f}")

    # muon itself should be first; muon_tweak should be closer than depth
    assert hits[0].run_id == "muon"
    assert hits[1].run_id == "muon_tweak"
    assert hits[2].run_id == "depth"

    print("\nSMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
