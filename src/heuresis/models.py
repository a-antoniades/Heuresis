"""Core data types for the heuresis library."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TokenUsage:
    """Token/cost accounting for older tests and agent integrations."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0


@dataclass
class RunResult:
    """Returned when a run completes (via RunFuture.result()).

    ``workspace`` and ``exit_code`` are always present. ``stats`` is an
    open dict populated by parsing the agent log — duration, tokens,
    cost, and any agent-specific metrics.  Adding new fields to stats
    doesn't break existing code.
    """

    workspace: Path
    exit_code: int
    stats: dict[str, Any] = field(default_factory=dict)
    duration: float | None = None
    log_path: Path | None = None
    tag: str | None = None
    tokens: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class RunRecord:
    """A persisted run as stored in ResultStore and returned by queries."""
    experiment_id: str
    run_id: str
    score: float | None
    workspace: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    # First-class columns (restored from pre-refactor schema)
    iteration: int | None = None
    run_type: str | None = None
    valid: bool | None = None
    started_at: str | None = None
    duration_s: float | None = None
    exit_code: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_cost: float | None = None
    parent_ids: list[str] = field(default_factory=list)
    generation: int = 0
    idea: str | None = None
