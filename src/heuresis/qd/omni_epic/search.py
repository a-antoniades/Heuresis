"""OmniEpicSearch — MAP-Elites-style + MoI-gated ideation.

Composes Phase 1's ``ArchiveIndex`` (3 buckets: accepted / failed_moi /
failed_train) with Phase 2's ``MoIReviewer``. Each iteration:

1. ``select_parents`` → ``archive_index.sample_anchor`` returns one anchor
   from the accepted bucket (diversity-weighted via ``_anchor_probs``).
2. ``context`` formats the top-K accepted + top-K failed_train neighbors
   of that anchor into text the ideator can read.
3. Caller's run.py hands the anchor's context to the ideator, ideator
   produces ``idea.md``.
4. Caller calls ``review_idea`` → ``MoIReviewer.review``. Verdict routes:
      - ``interesting=True`` → execute → ``on_result(score=...)`` →
        either ``accepted`` (valid score) or ``failed_train`` (no score).
      - ``interesting=False`` → ``on_moi_rejected`` → ``failed_moi``.
5. OMNI-EPIC paper behavior: **no retry on MoI rejection** (the idea is
   simply dropped into ``failed_moi``). The retry decision is a loop
   concern outside this class.

Persistence: every call returns a metadata dict to be merged into
``runs.metadata`` by ``record_run``. MoI calls carry tokens / duration /
raw_response so debugging the reviewer doesn't require rerunning it.
"""
from __future__ import annotations

from typing import Any

from heuresis.qd.core.archive_index import ArchiveIndex, Neighbor
from heuresis.qd.omni_epic.reviewer import MoIAssessment, MoIReviewer
from heuresis.qd.core.base import SearchStrategy


class OmniEpicSearch(SearchStrategy):
    """OMNI-EPIC-style strategy: anchor-sampled ideation + MoI gate.

    Ideator-side context (``context()``) is optional; a caller can also
    receive the anchor from ``select_parents`` and build its own context.
    The stub Phase 2 smoke used the latter; Phase 3 run.py should use
    ``context()`` directly.
    """

    def __init__(
        self,
        archive_index: ArchiveIndex,
        reviewer: MoIReviewer,
        *,
        lower_is_better: bool = True,
        success_context_k: int = 5,
        failed_context_k: int = 5,
        memory: bool = False,
    ) -> None:
        self.archive_index = archive_index
        self.reviewer = reviewer
        self.lower_is_better = lower_is_better
        self.success_context_k = success_context_k
        self.failed_context_k = failed_context_k
        # Campaign-level flag for the experiment loop — strategy itself
        # doesn't use memory. See LinearSearch for the pattern.
        self.memory = memory
        self._generation_map: dict[str, int] = {}
        # Per-ideator anchor pinned when select_parents runs, so context()
        # reads from the same anchor that will be reported in on_result.
        self._current_anchor: dict[int, str | None] = {}

    # --- SearchStrategy interface -----------------------------------------

    def select_parents(self, *, ideator_id: int = 0) -> list[str]:
        """Sample a single anchor from the accepted bucket.

        Returns empty list if the archive has no accepted entries (pre-seed
        phase). The anchor's prob is zeroed so subsequent calls prefer
        different anchors (OMNI-EPIC's ``taskgen_choose_probs``).
        """
        anchor = self.archive_index.sample_anchor(mark_used=True)
        self._current_anchor[ideator_id] = anchor
        return [anchor] if anchor else []

    def context(self, *, ideator_id: int = 0) -> str:
        """Return formatted top-K accepted + top-K failed_train neighbors.

        Mirrors OMNI-EPIC §2's ``RAG(anchor, codepaths, k=5) + RAG(anchor,
        failedtrain, k=5)``. Returns empty string when there is no anchor
        (e.g., archive empty or ``select_parents`` not yet called).
        """
        anchor = self._current_anchor.get(ideator_id)
        if anchor is None:
            return ""

        accepted: list[Neighbor] = []
        failed: list[Neighbor] = []
        if self.archive_index.accepted_size > 0:
            accepted = self.archive_index.top_k_from_run_id(
                anchor, k=self.success_context_k, bucket="accepted",
                include_self=False,
            )
        if self.archive_index.size("failed_train") > 0:
            failed = self.archive_index.top_k_from_run_id(
                anchor, k=self.failed_context_k, bucket="failed_train",
                include_self=False,
            )

        lines: list[str] = []
        if accepted:
            lines.append(
                f"## Similar accepted ideas ({len(accepted)} nearest to anchor `{anchor}`)"
            )
            for n in accepted:
                score = n.meta.get("score")
                lines.append(
                    f"- {n.run_id}: val_bpb={score} | similarity={n.similarity:.3f} | "
                    f'plan: "{n.plan[:140].replace(chr(10), " ")}"'
                )
            lines.append("")
        if failed:
            lines.append(
                f"## Similar failed-training ideas ({len(failed)} nearest to anchor `{anchor}`)"
            )
            for n in failed:
                lines.append(
                    f"- {n.run_id}: FAILED | similarity={n.similarity:.3f} | "
                    f'plan: "{n.plan[:140].replace(chr(10), " ")}"'
                )
            lines.append("")
        return "\n".join(lines)

    def on_result(
        self,
        run_id: str,
        score: float | None,
        features: dict[str, float] | None = None,
        *,
        idea: str | None = None,
        parent_ids: list[str] | None = None,
        ideator_id: int = 0,
        moi_assessment: MoIAssessment | None = None,
    ) -> dict[str, Any]:
        """Route a post-execution run into accepted or failed_train.

        Always called AFTER the MoI gate has passed (otherwise use
        ``on_moi_rejected``). If ``moi_assessment`` is supplied, persist
        its bookkeeping into metadata so the reviewer's work is traceable.
        """
        plan = idea or ""
        generation = self._next_generation(parent_ids)
        self._generation_map[run_id] = generation

        if score is not None:
            self.archive_index.add_accepted(run_id, plan, score=score)
            bucket = "accepted"
            extra: dict[str, Any] = {}
        else:
            self.archive_index.add_rejected_train(
                run_id, plan, failure_mode="training failed or invalid",
            )
            bucket = "failed_train"
            extra = {"omniepic_failure_mode": "training failed or invalid"}

        meta: dict[str, Any] = {
            "parent_ids": parent_ids or [],
            "generation": generation,
            "omniepic_bucket": bucket,
            **extra,
        }
        if idea is not None:
            meta["idea"] = idea
        if moi_assessment is not None:
            meta.update(_assessment_metadata(moi_assessment))
        return meta

    def on_moi_rejected(
        self,
        run_id: str,
        idea: str,
        assessment: MoIAssessment,
        *,
        parent_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Route a pre-execution MoI rejection into failed_moi.

        Persists the full reviewer bookkeeping: reasoning, retrieved_ids,
        raw_response, tokens, duration — so the decision can be audited
        without rerunning the reviewer.
        """
        generation = self._next_generation(parent_ids)
        self._generation_map[run_id] = generation
        self.archive_index.add_rejected_moi(
            run_id, idea, reasoning=assessment.reasoning,
        )
        meta: dict[str, Any] = {
            "parent_ids": parent_ids or [],
            "generation": generation,
            "omniepic_bucket": "failed_moi",
            "idea": idea,
        }
        meta.update(_assessment_metadata(assessment))
        return meta

    def review_idea(self, idea: str) -> MoIAssessment:
        """Convenience pass-through to the MoI reviewer."""
        return self.reviewer.review(idea)

    def rebuild(
        self, records: list[tuple[str, float | None, dict[str, Any]]]
    ) -> None:
        """Rebuild the generation map. The archive is hydrated separately
        via ``ArchiveIndex.rebuild_from_experiment`` by the caller.
        """
        for run_id, _score, metadata in records:
            self._generation_map[run_id] = metadata.get("generation", 0)

    def summary(self) -> str:
        return (
            f"OmniEpic: accepted={self.archive_index.accepted_size}, "
            f"failed_moi={self.archive_index.size('failed_moi')}, "
            f"failed_train={self.archive_index.size('failed_train')}"
        )

    # --- helpers ----------------------------------------------------------

    def _next_generation(self, parent_ids: list[str] | None) -> int:
        if not parent_ids:
            return 0
        parent_gens = [self._generation_map.get(pid, 0) for pid in parent_ids]
        return max(parent_gens, default=0) + 1


def _assessment_metadata(assessment: MoIAssessment) -> dict[str, Any]:
    """Serialize a ``MoIAssessment`` to metadata-safe fields.

    ``raw_response`` can be large; callers can pop it if the store grows.
    """
    return {
        "omniepic_reasoning": assessment.reasoning,
        "omniepic_retrieved_ids": assessment.retrieved_ids,
        "moi_interesting": assessment.interesting,
        "moi_raw_response": assessment.raw_response,
        "moi_input_tokens": assessment.input_tokens,
        "moi_output_tokens": assessment.output_tokens,
        "moi_duration_s": assessment.duration_s,
        "moi_total_cost": assessment.total_cost,
    }
