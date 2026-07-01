#!/usr/bin/env python3
"""Live smoke test for MoIReviewer.

Builds an ArchiveIndex of 12 hand-written nanogpt-style ideas, then runs
two MoI reviews and prints the verdicts. Pass criterion: human spot-check
that the reviewer rejects trivial variants and accepts genuinely novel
ideas, with reasoning that references the retrieved examples.

Usage:
    uv run scripts/smoke/moi_reviewer.py
"""
from __future__ import annotations

import sys

from heuresis.qd import ArchiveIndex, GeminiEmbedder, MoIReviewer
from heuresis.tasks import task_dir


SEED_IDEAS = [
    ("seed_adamw_lr1e3",     "Train with AdamW, learning rate 1e-3, no warmup, weight decay 0.1.", 0.972),
    ("seed_adamw_lr3e4",     "Use AdamW with learning rate 3e-4 and cosine schedule for 30 minutes.", 0.969),
    ("seed_muon_2d",         "Replace AdamW with Muon optimizer on all 2D matrix parameters; AdamW elsewhere.", 0.951),
    ("seed_muon_alt",        "Use Muon for matrix params with a small momentum buffer; tune learning rate.", 0.948),
    ("seed_depth_16",        "Increase depth from 8 to 16 layers; halve FFN width to keep param count constant.", 0.962),
    ("seed_depth_12_wider",  "Use 12 layers with hidden dim 640 instead of 8 layers with hidden dim 512.", 0.965),
    ("seed_attn_flash",      "Switch to FlashAttention v2 for the self-attention forward pass.", 0.971),
    ("seed_norm_rms",        "Replace LayerNorm with RMSNorm everywhere in the transformer block.", 0.970),
    ("seed_init_scale",      "Scale initialization for residual projections by 1/sqrt(2*depth) per GPT-NeoX.", 0.968),
    ("seed_pos_rope",        "Use rotary position embeddings instead of learned absolute positions.", 0.966),
    ("seed_warmup_50",       "Linear warmup over the first 50 steps then constant LR for remainder.", 0.973),
    ("seed_dropout_0p1",     "Add dropout=0.1 to attention and residual streams, keep optimizer otherwise.", 0.974),
]

CANDIDATES = [
    ("Use SGD with Nesterov momentum=0.9 instead of any adaptive optimizer; lr 0.1 with cosine decay.",
     "novel — no SGD in seed, different optimizer family"),
    ("Use AdamW with lr 5e-4 and warmup of 100 steps.",
     "trivial — minor variant of multiple seed AdamW entries"),
]


def main() -> int:
    from heuresis.api_keys import load_api_keys

    if not load_api_keys("gemini"):
        print(
            "ERROR: no Gemini API keys found. Set GEMINI_API_KEYS, GEMINI_API_KEY, "
            "GOOGLE_GENERATIVE_AI_API_KEY (see .env.example)",
            file=sys.stderr,
        )
        return 1

    print("Building ArchiveIndex...")
    emb = GeminiEmbedder()
    idx = ArchiveIndex(embedder=emb)
    for run_id, plan, score in SEED_IDEAS:
        idx.add_accepted(run_id=run_id, plan=plan, score=score)
    print(f"  archive has {idx.accepted_size} accepted entries")

    print("Constructing MoIReviewer...")
    reviewer = MoIReviewer(idx, task_dir("nanogpt"))

    for candidate, expected in CANDIDATES:
        print()
        print("=" * 78)
        print(f"CANDIDATE: {candidate}")
        print(f"EXPECTED:  {expected}")
        out = reviewer.review(candidate)
        print(f"VERDICT:   interesting={out.interesting}")
        print(f"REASONING: {out.reasoning}")
        print(f"RETRIEVED: {out.retrieved_ids[:5]}{'...' if len(out.retrieved_ids) > 5 else ''}")
        print(f"TOKENS:    in={out.input_tokens}, out={out.output_tokens}, dur={out.duration_s:.2f}s")

    print()
    print("Smoke complete. Verify by inspection:")
    print("  - Candidate 1 (SGD) should be interesting=True")
    print("  - Candidate 2 (AdamW lr=5e-4) should be interesting=False")
    print("  - Reasoning should reference the retrieved examples specifically")
    return 0


if __name__ == "__main__":
    sys.exit(main())
