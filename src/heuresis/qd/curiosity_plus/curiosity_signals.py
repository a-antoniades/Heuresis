"""Build natural-language curiosity context for the ideator prompt (§6, M6).

Given an anchor idea and its kNN neighborhood, produces text describing:
- The anchor idea the ideator should build on
- The prediction track record in that region
- Whether predictions are improving or deteriorating (LP signal)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from heuresis.qd.curiosity_plus.embedding_store import EmbeddingStore, StoredIdea

from heuresis.qd.curiosity_plus.learning_progress import compute_lp
from heuresis.qd.curiosity_plus.surprise import Prediction


def build_curiosity_context(
    anchor_run_id: str,
    store: EmbeddingStore,
    *,
    k: int = 10,
    include_predictions: bool = True,
) -> str:
    """Build natural-language curiosity context for the ideator.

    Returns a multi-section text block that can be injected into the
    ideator prompt.
    """
    anchor = store.get(anchor_run_id)
    neighbors = store.knn_by_run_id(anchor_run_id, k=k, include_self=False)
    lp, confident = compute_lp(anchor_run_id, store, k=k)

    lines: list[str] = []

    # --- Anchor description ---
    lines.append("## Curiosity Anchor")
    lines.append(f"Build on this direction (run `{anchor.run_id}`):")
    score_str = f"val_bpb={anchor.score}" if anchor.score is not None else "no score"
    lines.append(f"- Score: {score_str}")
    lines.append(f"- Idea: \"{_truncate(anchor.idea, 200)}\"")
    lines.append("")

    # --- Why this region ---
    lines.append("## Why This Region")
    if confident:
        if lp > 0.1:
            lines.append(
                f"Learning progress is POSITIVE (LP={lp:.3f}) — predictions about "
                "ideas in this neighborhood have been improving. This means there's "
                "learnable structure here worth exploring further."
            )
        elif lp < -0.1:
            lines.append(
                f"Learning progress is NEGATIVE (LP={lp:.3f}) — predictions are "
                "getting worse here, suggesting unexpected complexity. Fresh "
                "approaches in this direction might reveal something."
            )
        else:
            lines.append(
                f"Learning progress is near zero (LP={lp:.3f}) — this region is "
                "fairly understood. Try a meaningful variation rather than a tweak."
            )
    else:
        lines.append(
            "This region has limited data — your results here will help map "
            "unexplored territory."
        )
    lines.append("")

    # --- Neighborhood results ---
    scored = [(e, sim) for e, sim in neighbors if e.score is not None]
    if scored:
        lines.append(f"## Nearby Results ({len(scored)} scored neighbors)")
        for entry, sim in scored[:5]:
            surprise_str = f"surprise={entry.surprise:.2f}" if entry.surprise is not None else ""
            parts = [
                f"- {entry.run_id}: val_bpb={entry.score}",
                f"sim={sim:.3f}",
            ]
            if surprise_str:
                parts.append(surprise_str)
            parts.append(f'"{_truncate(entry.idea, 100)}"')
            lines.append(" | ".join(parts))
        lines.append("")

    # --- Prediction track record ---
    if include_predictions:
        with_surprise = [(e, sim) for e, sim in neighbors if e.surprise is not None]
        if with_surprise:
            lines.append("## Prediction Track Record")
            high_surprise = [e for e, _ in with_surprise if e.surprise is not None and e.surprise > 0.5]
            low_surprise = [e for e, _ in with_surprise if e.surprise is not None and e.surprise < 0.3]
            if high_surprise:
                lines.append(
                    f"  {len(high_surprise)} ideas in this region had surprising "
                    "outcomes (predictions were wrong) — the landscape here is "
                    "not what we expected."
                )
            if low_surprise:
                lines.append(
                    f"  {len(low_surprise)} ideas had predictable outcomes — "
                    "the landscape here is well-understood."
                )
            lines.append("")

    return "\n".join(lines)


def build_prediction_context(
    store: EmbeddingStore,
    *,
    max_history: int = 10,
) -> str:
    """Build past-predictions context for the prediction prompt (§7.3 self-reflective loop).

    Shows each prior entry as:
      - what the LLM predicted (valid, fitness, reasoning)
      - what actually happened (valid, fitness)
      - the resulting surprise

    The LLM can then recognize where its past reasoning was right/wrong
    and adjust the current prediction accordingly.
    """
    entries = store.all_entries()
    # Only include steady-state entries (those that had a prediction)
    completed = [
        e for e in entries
        if e.prediction is not None and e.surprise is not None
    ]
    if not completed:
        return "(No prior predictions to reference yet.)"

    recent = completed[-max_history:]
    lines = ["## Past Predictions and Outcomes"]
    lines.append("")
    for e in recent:
        pred = e.prediction
        actual_valid = "valid" if e.valid else "invalid"
        predicted_valid = "valid" if pred.predicted_valid else "invalid"
        predicted_f = (
            f"{pred.predicted_fitness:.4f}" if pred.predicted_fitness is not None else "N/A"
        )
        actual_f = f"{e.score:.4f}" if e.score is not None else "N/A"
        right_or_wrong = _verdict(pred, e)

        lines.append(f"### {e.run_id} — {right_or_wrong} (surprise={e.surprise:.2f})")
        lines.append(f"- Idea: \"{_truncate(e.idea, 120)}\"")
        lines.append(
            f"- Predicted: {predicted_valid}, val_bpb={predicted_f}"
            + (f" (confidence={pred.confidence:.2f})" if pred.confidence is not None else "")
        )
        lines.append(f"- Actual:    {actual_valid}, val_bpb={actual_f}")
        if pred.reasoning:
            lines.append(f"- Predicted reasoning: \"{_truncate(pred.reasoning, 220)}\"")
        lines.append("")
    return "\n".join(lines).rstrip()


def _verdict(pred: Prediction, entry: "StoredIdea") -> str:
    """One-word label for how well the prediction matched reality."""
    if pred.predicted_valid != bool(entry.valid):
        return "VALIDITY MISS"
    if entry.surprise is None:
        return "n/a"
    if entry.surprise < 0.25:
        return "on target"
    if entry.surprise < 0.75:
        return "somewhat off"
    return "badly off"


def _truncate(text: str, max_chars: int) -> str:
    flat = text.replace("\n", " ").strip()
    if len(flat) <= max_chars:
        return flat
    return flat[:max_chars] + "..."
