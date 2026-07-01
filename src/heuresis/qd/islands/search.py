"""IslandSearch: multi-population evolutionary search with migration."""

from __future__ import annotations

import bisect
import random as _random
from typing import Any, Literal

from heuresis.qd.core.base import (
    SearchStrategy,
    compute_generation,
    extract_summary,
)
from heuresis.qd.core.migration import (
    ScoredSolution,
    fully_connected_neighbors,
    migrate,
    ring_neighbors,
)
from heuresis.qd.core.selection import Selector

_TOPOLOGY_FNS = {
    "ring": ring_neighbors,
    "fully_connected": fully_connected_neighbors,
}


class IslandSearch(SearchStrategy):
    """Island-model evolutionary search with mutation, crossover, and migration.

    Maintains multiple isolated populations. Each ideator_id maps to an
    island via round-robin. Periodically migrates top solutions between
    topology-adjacent islands.
    """

    def __init__(
        self,
        *,
        num_islands: int = 8,
        topology: Literal["ring", "fully_connected"] = "ring",
        max_population: int = 30,
        maximize: bool = True,
        crossover_rate: float = 0.4,
        parent_k: int = 2,
        tournament_size: int = 2,
        migration_interval: int = 24,
        migration_k: int = 1,
        seed: int = 42,
        memory: bool = False,
    ) -> None:
        if topology not in _TOPOLOGY_FNS:
            raise ValueError(f"topology must be one of {list(_TOPOLOGY_FNS)}, got {topology!r}")
        self.num_islands = num_islands
        self.topology = topology
        self.max_population = max_population
        self.maximize = maximize
        self.crossover_rate = crossover_rate
        self.parent_k = parent_k
        self.tournament_size = tournament_size
        self.migration_interval = migration_interval
        self.migration_k = migration_k
        # Campaign-level flag for the experiment loop — strategy itself
        # doesn't use memory. See LinearSearch for the pattern.
        self.memory = memory
        self.selector = Selector(seed=seed)
        self._rng = _random.Random(seed)

        self._islands: list[list[ScoredSolution]] = [[] for _ in range(num_islands)]
        self._eval_count = 0
        self._migration_count = 0
        self._last_operator: dict[int, str | None] = {}
        self._generation_map: dict[str, int] = {}
        self._idea_summaries: dict[str, str] = {}

    def _island_for(self, ideator_id: int) -> int:
        return ideator_id % self.num_islands

    def select_parents(self, *, ideator_id: int = 0) -> list[str]:
        island_idx = self._island_for(ideator_id)
        pop = self._islands[island_idx]
        if not pop:
            self._last_operator[ideator_id] = None
            return []

        want_crossover = self._rng.random() < self.crossover_rate
        if want_crossover and len(pop) < self.parent_k:
            want_crossover = False

        self.selector.update(scores={s.id: s.score for s in pop})
        if want_crossover:
            self._last_operator[ideator_id] = "crossover"
            return self.selector.tournament(k=self.parent_k, tournament_size=self.tournament_size)
        else:
            self._last_operator[ideator_id] = "mutation"
            return self.selector.tournament(k=1, tournament_size=self.tournament_size)

    def context(self, *, ideator_id: int = 0) -> str:
        island_idx = self._island_for(ideator_id)
        pop = self._islands[island_idx]
        n = len(pop)
        best = f"{pop[0].score:.4f}" if pop else "n/a"
        lines = [
            f"You are exploring on island {island_idx} "
            f"({n} solution{'s' if n != 1 else ''}, best={best}).",
        ]
        op = self._last_operator.get(ideator_id)
        if op == "mutation":
            lines.append(
                "You are MUTATING the solution above. "
                "Propose a modification that improves it or explores a new direction."
            )
        elif op == "crossover":
            lines.append(
                "You are COMBINING the solutions above. "
                "Synthesize the best elements into something better than any individual."
            )
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
    ) -> dict[str, Any]:
        island_idx = self._island_for(ideator_id)
        generation = compute_generation(parent_ids, self._generation_map)
        self._generation_map[run_id] = generation

        if idea:
            self._idea_summaries[run_id] = extract_summary(idea)

        metadata: dict[str, Any] = {
            "island_id": island_idx,
            "operator": self._last_operator.get(ideator_id),
            "parent_ids": parent_ids or [],
            "generation": generation,
        }
        if idea is not None:
            metadata["idea"] = idea

        if score is None:
            metadata["rank"] = None
            return metadata

        self._eval_count += 1
        rank = self._eliminate(island_idx, ScoredSolution(id=run_id, score=score))
        metadata["rank"] = rank

        if self._eval_count % self.migration_interval == 0:
            self._migrate()
            metadata["migrated"] = True
        else:
            metadata["migrated"] = False

        return metadata

    def _eliminate(self, island_idx: int, sol: ScoredSolution) -> int:
        """Insert solution into island, drop worst if over capacity. Returns rank."""
        pop = self._islands[island_idx]
        if self.maximize:
            keys = [-s.score for s in pop]
            pos = bisect.bisect_left(keys, -sol.score)
        else:
            keys = [s.score for s in pop]
            pos = bisect.bisect_left(keys, sol.score)
        pop.insert(pos, sol)
        if len(pop) > self.max_population:
            pop.pop()
        return pos

    def _migrate(self) -> None:
        migrate(
            self._islands,
            topology_fn=_TOPOLOGY_FNS[self.topology],
            k=self.migration_k,
            maximize=self.maximize,
            rng=self._rng,
        )
        self._migration_count += 1

    def rebuild(self, records: list[tuple[str, float | None, dict[str, Any]]]) -> None:
        for run_id, score, metadata in records:
            island_idx = metadata.get("island_id")
            if island_idx is None or score is None:
                continue
            self._eliminate(int(island_idx), ScoredSolution(id=run_id, score=score))
            self._eval_count += 1
            gen = metadata.get("generation", 0)
            self._generation_map[run_id] = gen
            idea = metadata.get("idea")
            if idea:
                self._idea_summaries[run_id] = extract_summary(idea)

    def summary(self) -> str:
        lines: list[str] = []
        total = 0
        for idx, pop in enumerate(self._islands):
            if pop:
                lines.append(f"  Island {idx}: {len(pop)} solutions, best={pop[0].score:.4f}")
            else:
                lines.append(f"  Island {idx}: empty")
            total += len(pop)
        lines.insert(0, f"Islands: {self.num_islands}, total solutions: {total}, "
                        f"migrations: {self._migration_count}")
        return "\n".join(lines)
