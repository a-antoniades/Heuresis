"""Island migration: topology functions and migration logic.

Pure functions — no mutable state. The IslandSearch strategy owns the
populations and calls these functions when migration triggers.
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ScoredSolution:
    """A solution with its fitness score."""

    id: str
    score: float


@dataclass
class MigrationEvent:
    """Record of one island-to-island migration."""

    from_island: int
    to_islands: list[int] = field(default_factory=list)
    solution_ids: list[str] = field(default_factory=list)


def ring_neighbors(island_id: int, num_islands: int) -> list[int]:
    """Return the next island in a unidirectional ring."""
    if num_islands <= 1:
        return []
    return [(island_id + 1) % num_islands]


def fully_connected_neighbors(island_id: int, num_islands: int) -> list[int]:
    """Return all other islands."""
    return [i for i in range(num_islands) if i != island_id]


def migrate(
    populations: list[list[ScoredSolution]],
    *,
    topology_fn: Callable[[int, int], list[int]],
    k: int,
    maximize: bool,
    rng: _random.Random | None = None,
    donor_pool_size: int = 5,
) -> list[MigrationEvent]:
    """Copy solutions from each island to its topology neighbors.

    Donors are sampled randomly from the top-``donor_pool_size`` solutions
    on each island (or the full population if smaller).  This stochastic
    selection ensures rare migration events (typical for expensive-eval
    campaigns) transfer diverse genetic material rather than repeatedly
    sending the same top-1 solution.

    Mutates *populations* in place. Returns a log of migration events.
    """
    _rng = rng or _random.Random()
    num_islands = len(populations)
    # Snapshot migrants before mutating any population.
    # Populations are assumed to be sorted best-first (maintained by
    # IslandSearch._eliminate), so we skip re-sorting here.
    migrants: list[tuple[int, list[ScoredSolution]]] = []
    for island_id in range(num_islands):
        pop = populations[island_id]
        if not pop:
            migrants.append((island_id, []))
            continue
        pool = pop[:min(donor_pool_size, len(pop))]
        selected = _rng.sample(pool, k=min(k, len(pool)))
        migrants.append((island_id, selected))

    events: list[MigrationEvent] = []
    for island_id, to_migrate in migrants:
        neighbors = topology_fn(island_id, num_islands)
        migrated_ids = [s.id for s in to_migrate]
        for neighbor in neighbors:
            existing_ids = {s.id for s in populations[neighbor]}
            for sol in to_migrate:
                if sol.id not in existing_ids:
                    populations[neighbor].append(ScoredSolution(sol.id, sol.score))
                    existing_ids.add(sol.id)
        events.append(
            MigrationEvent(
                from_island=island_id,
                to_islands=neighbors,
                solution_ids=migrated_ids,
            )
        )
    return events
