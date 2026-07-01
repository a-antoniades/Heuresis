"""NoveltyMapElitesSearch: MAP-Elites with a novelty-review gate on ideas.

Wraps ``MapElitesSearch`` for archive management and ``NoveltyReviewer``
for idea filtering. Adds ``review_idea()`` which runs the novelty gate
between ideation and execution.

This was previously named ``OmniEpicSearch`` (pre-Phase 3). The Phase 3
canonical OMNI-EPIC implementation lives in ``qd.omni_epic.search`` and
uses ``ArchiveIndex`` + ``MoIReviewer`` instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from heuresis.qd.core.archive import Feature
from heuresis.qd.core.base import SearchStrategy
from heuresis.qd.map_elites.search import MapElitesSearch

if TYPE_CHECKING:
    from heuresis.novelty import NoveltyAssessment, NoveltyReviewer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IdeaReview:
    """Result of reviewing an idea for novelty."""

    accepted: bool
    assessment: "NoveltyAssessment"
    attempt: int


class NoveltyMapElitesSearch(SearchStrategy):
    """MAP-Elites with novelty-gated ideation.

    The 5 core SearchStrategy methods delegate to the inner
    MapElitesSearch. The novelty gate is an additional step the loop
    calls explicitly between ideation and execution.
    """

    def __init__(
        self,
        features: list[Feature],
        reviewer: "NoveltyReviewer",
        *,
        novelty_threshold: int = 2,
        maximize: bool = True,
        parent_k_range: tuple[int, int] = (1, 2),
        selection_policy: str = "tournament",
        feature_name_fn: Callable[[dict[str, float]], dict[str, str]] | None = None,
        seed: int = 42,
    ) -> None:
        self._me = MapElitesSearch(
            features,
            maximize=maximize,
            parent_k_range=parent_k_range,
            selection_policy=selection_policy,
            feature_name_fn=feature_name_fn,
            seed=seed,
        )
        self._reviewer = reviewer
        self.novelty_threshold = novelty_threshold
        self._total_reviews = 0
        self._accepted_reviews = 0
        self._rejected_reviews = 0

    @property
    def archive(self):
        """Access the underlying GridArchive."""
        return self._me.archive

    def select_parents(self, *, ideator_id: int = 0) -> list[str]:
        return self._me.select_parents(ideator_id=ideator_id)

    def context(self, *, ideator_id: int = 0) -> str:
        ctx = self._me.context(ideator_id=ideator_id)
        if ctx:
            ctx += "\n\n"
        ctx += (
            f"NOVELTY REQUIREMENT: Your idea will be reviewed for novelty.\n"
            f"Score >= {self.novelty_threshold}/3 required for execution.\n"
            f"Search for prior work before proposing. Aim for genuinely new mechanisms."
        )
        return ctx

    def review_idea(
        self,
        idea: str,
        *,
        workspace_path: Path | None = None,
        idea_id: str | None = None,
    ) -> IdeaReview:
        """Run novelty review on an idea. Call between ideation and execution."""

        self._total_reviews += 1
        assessment = self._reviewer.assess(
            idea, workspace_path=workspace_path, idea_id=idea_id,
        )
        accepted = assessment.novelty >= self.novelty_threshold
        if accepted:
            self._accepted_reviews += 1
        else:
            self._rejected_reviews += 1
        logger.info(
            "Novelty review: %s (score=%d, threshold=%d)",
            "ACCEPTED" if accepted else "REJECTED",
            assessment.novelty, self.novelty_threshold,
        )
        return IdeaReview(
            accepted=accepted,
            assessment=assessment,
            attempt=self._total_reviews,
        )

    def on_result(
        self,
        run_id: str,
        score: float | None,
        features: dict[str, float] | None = None,
        *,
        idea: str | None = None,
        parent_ids: list[str] | None = None,
        ideator_id: int = 0,
    ) -> dict[str, Any]:
        return self._me.on_result(
            run_id, score, features,
            idea=idea, parent_ids=parent_ids, ideator_id=ideator_id,
        )

    def rebuild(self, records: list[tuple[str, float | None, dict[str, Any]]]) -> None:
        self._me.rebuild(records)

    def summary(self) -> str:
        me_summary = self._me.summary()
        novelty_line = (
            f"Novelty gate: {self._accepted_reviews}/{self._total_reviews} accepted "
            f"(threshold >= {self.novelty_threshold}), "
            f"{self._rejected_reviews} rejected"
        )
        return f"{me_summary}\n{novelty_line}"
