"""LinearSearch: top-K parents by score, no archive."""

from __future__ import annotations

from typing import Any

from heuresis.qd.core.base import (
    SearchStrategy,
    compute_generation,
    extract_summary,
)


class LinearSearch(SearchStrategy):
    """Linear/greedy: no archive, select top-K previous runs by score.

    The ideator sees a stateful session across iterations (managed by the
    experiment's run.py via Harness(stateful=True)). should_reset_session()
    exposes when to force a fresh session to prevent context bloat.
    """

    def __init__(
        self,
        *,
        max_parents: int = 5,
        maximize: bool = True,
        session_reset_every: int | None = 10,
        memory: bool = False,
    ) -> None:
        self.max_parents = max_parents
        self.maximize = maximize
        self.session_reset_every = session_reset_every
        # Flag for the experiment script to decide whether to spin up
        # MemoryStore + add the MEMORY tool + include memory blocks in
        # prompts. The strategy itself doesn't use memory directly; this
        # is the single campaign-level source of truth the loop reads.
        self.memory = memory
        self._scored: dict[str, float] = {}
        self._generation_map: dict[str, int] = {}
        self._idea_summaries: dict[str, str] = {}

    def select_parents(self, *, ideator_id: int = 0) -> list[str]:
        scored = [(rid, s) for rid, s in self._scored.items() if s is not None]
        scored.sort(key=lambda t: t[1], reverse=self.maximize)
        return [rid for rid, _ in scored[: self.max_parents]]

    def context(self, *, ideator_id: int = 0) -> str:
        # Past parents are injected into the prompt via select_parents; no extra context.
        return ""

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
        generation = compute_generation(parent_ids, self._generation_map)
        self._generation_map[run_id] = generation

        if idea:
            self._idea_summaries[run_id] = extract_summary(idea)
        if score is not None:
            self._scored[run_id] = score

        return {
            "parent_ids": parent_ids or [],
            "generation": generation,
            **({"idea": idea} if idea is not None else {}),
        }

    def rebuild(
        self, records: list[tuple[str, float | None, dict[str, Any]]]
    ) -> None:
        for run_id, score, metadata in records:
            if score is not None:
                self._scored[run_id] = score
            gen = metadata.get("generation", 0)
            self._generation_map[run_id] = gen
            idea = metadata.get("idea")
            if idea:
                self._idea_summaries[run_id] = extract_summary(idea)

    def summary(self) -> str:
        n = len(self._scored)
        if not self._scored:
            return "Linear: 0 scored runs"
        best = min(self._scored.values()) if not self.maximize else max(self._scored.values())
        return f"Linear: {n} scored runs, best={best:.4f}"

    def should_reset_session(self, iteration: int) -> bool:
        """Return True if the ideator should reset its stateful session."""
        if self.session_reset_every is None or self.session_reset_every <= 0:
            return False
        return iteration > 0 and (iteration % self.session_reset_every == 0)
