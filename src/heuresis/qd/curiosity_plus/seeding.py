"""Phase 1: Diversity-enforced seeding via farthest-point greedy (§6.2).

Goal: maximize coverage of the embedding space so k-NN neighborhoods
are well-populated when LP turns on. No curiosity signal — just coverage.

Each iteration generates B candidate ideas, embeds all, and selects
the one farthest from everything already in the store.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from heuresis.qd.curiosity_plus.embedding_store import EmbeddingStore


def select_seed(
    store: EmbeddingStore,
    candidates: list[str],
    *,
    reserve_run_id: str | None = None,
    reserve_iteration: int | None = None,
) -> tuple[int, str]:
    """Pick the candidate that maximizes minimum distance to the store.

    Returns (index, candidate_text). If the store is empty, returns the
    first candidate.

    When ``reserve_run_id`` and ``reserve_iteration`` are provided, the
    chosen candidate's embedding is inserted into the store immediately so
    concurrent seed picks see it — preserving farthest-point semantics
    across parallel ideators. ``on_result`` later upserts score/surprise.
    """
    if not candidates:
        raise ValueError("No candidates provided for seeding")

    vectors = store.embed_texts(candidates)
    idx = store.farthest_point(vectors)
    chosen = candidates[idx]
    if reserve_run_id is not None and reserve_iteration is not None:
        store.add(
            reserve_run_id,
            chosen,
            iteration=reserve_iteration,
            vector=vectors[idx],
        )
    return idx, chosen


def parse_candidates(raw_text: str, expected: int = 5) -> list[str]:
    """Parse B candidate ideas from LLM output.

    Expects ideas separated by a delimiter like "---" or numbered headers.
    Falls back to treating the entire text as a single candidate.
    """
    # Try splitting on markdown-style delimiters
    parts = re.split(r'\n---+\n', raw_text)
    if len(parts) >= 2:
        return [p.strip() for p in parts if p.strip()]

    # Try splitting on numbered headers like "## Idea 1", "## Idea 2"
    parts = re.split(r'\n##\s+(?:Idea|Candidate)\s+\d+', raw_text, flags=re.IGNORECASE)
    if len(parts) >= 2:
        return [p.strip() for p in parts if p.strip()]

    # Try splitting on "1.", "2.", etc. at start of line
    parts = re.split(r'\n(?=\d+\.\s)', raw_text)
    if len(parts) >= 2:
        return [p.strip() for p in parts if p.strip()]

    return [raw_text.strip()] if raw_text.strip() else []
