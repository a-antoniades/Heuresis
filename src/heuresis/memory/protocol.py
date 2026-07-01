"""Protocols for the memory primitive.

Defined in their own module so ``heuresis.experiment`` can depend
on the interface without importing the concrete :class:`MemoryStore`
(which pulls in google-genai / sqlite-vec).
"""

from __future__ import annotations

from typing import Any, Protocol


class MemoryIngest(Protocol):
    """Framework-side writer for the ``experiments`` table.

    The only entry point called from the experiment loop (via
    :func:`heuresis.experiment.record_run`). Agent-written rows go
    through the in-sandbox ``memory append`` CLI directly against the
    socket, never through this Protocol.
    """

    def ingest_experiment(
        self,
        *,
        ideator_id: str,
        executor_id: str,
        valid: bool,
        score: float | None,
        features: dict[str, Any] | None,
        parent_ids: list[str] | None,
        generation: int,
        idea_md: str,
        notes_md: str | None,
    ) -> None:
        ...
