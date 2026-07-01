"""Tests for IslandSearch v2 parameters and stochastic migration."""

from __future__ import annotations

import random

import pytest

from heuresis.qd.core.migration import ScoredSolution, migrate, ring_neighbors
from heuresis.qd.islands.search import IslandSearch


class TestDefaults:
    """Constructor defaults match the v2 spec."""

    def test_v2_defaults(self) -> None:
        s = IslandSearch()
        assert s.num_islands == 8
        assert s.max_population == 30
        assert s.crossover_rate == 0.4
        assert s.parent_k == 2
        assert s.tournament_size == 2
        assert s.migration_interval == 24
        assert s.migration_k == 1

    def test_preserve_best_removed(self) -> None:
        with pytest.raises(TypeError):
            IslandSearch(preserve_best=True)  # type: ignore[call-arg]


class TestStochasticMigration:
    """Migration selects randomly from top-5 (donor_pool_size)."""

    @staticmethod
    def _make_pop(n: int, maximize: bool = False) -> list[ScoredSolution]:
        """Sorted population: best first. IDs are 'sol_0' (best) .. 'sol_{n-1}' (worst)."""
        scores = list(range(n, 0, -1)) if maximize else list(range(n))
        return [ScoredSolution(id=f"sol_{i}", score=float(s)) for i, s in enumerate(scores)]

    def test_migrant_always_from_top5(self) -> None:
        """Every migrant must come from the top-5 of the source population."""
        pop = self._make_pop(10, maximize=False)
        top5_ids = {s.id for s in pop[:5]}

        for seed in range(50):
            pops = [list(pop), []]
            migrate(
                pops,
                topology_fn=ring_neighbors,
                k=1,
                maximize=False,
                rng=random.Random(seed),
            )
            new_on_island1 = [s for s in pops[1] if s.id in {s.id for s in pop}]
            for s in new_on_island1:
                assert s.id in top5_ids, f"seed={seed}: migrant {s.id} not in top-5"

    def test_stochastic_variety(self) -> None:
        """Across different seeds, migration selects different solutions."""
        pop = self._make_pop(10, maximize=False)
        migrant_ids: set[str] = set()

        for seed in range(50):
            pops = [list(pop), []]
            migrate(
                pops,
                topology_fn=ring_neighbors,
                k=1,
                maximize=False,
                rng=random.Random(seed),
            )
            for s in pops[1]:
                migrant_ids.add(s.id)

        assert len(migrant_ids) > 1, "Migration always sent the same solution"

    def test_small_population(self) -> None:
        """Migration works when population is smaller than donor_pool_size."""
        pop = self._make_pop(3, maximize=False)
        pops = [list(pop), []]
        migrate(
            pops,
            topology_fn=ring_neighbors,
            k=1,
            maximize=False,
            rng=random.Random(0),
        )
        assert len(pops[1]) == 1
        assert pops[1][0].id in {s.id for s in pop}

    def test_deterministic_with_seed(self) -> None:
        """Same RNG seed produces identical migration results."""
        pop = self._make_pop(10, maximize=False)

        def run_migration(seed: int) -> list[str]:
            pops = [list(pop), []]
            migrate(
                pops,
                topology_fn=ring_neighbors,
                k=1,
                maximize=False,
                rng=random.Random(seed),
            )
            return [s.id for s in pops[1]]

        for seed in range(10):
            assert run_migration(seed) == run_migration(seed)

    def test_custom_donor_pool_size(self) -> None:
        """donor_pool_size parameter limits the eligible pool."""
        pop = self._make_pop(10, maximize=False)
        top3_ids = {s.id for s in pop[:3]}

        for seed in range(50):
            pops = [list(pop), []]
            migrate(
                pops,
                topology_fn=ring_neighbors,
                k=1,
                maximize=False,
                rng=random.Random(seed),
                donor_pool_size=3,
            )
            for s in pops[1]:
                assert s.id in top3_ids


class TestMigrationInterval:
    """Migration fires at the correct global eval count."""

    def test_fires_at_interval(self) -> None:
        s = IslandSearch(num_islands=2, migration_interval=24, maximize=True)
        for i in range(1, 25):
            s.on_result(f"r{i}", score=float(i), ideator_id=i % 2)
        assert s._migration_count == 1

    def test_does_not_fire_before_interval(self) -> None:
        s = IslandSearch(num_islands=2, migration_interval=24, maximize=True)
        for i in range(1, 24):
            s.on_result(f"r{i}", score=float(i), ideator_id=i % 2)
        assert s._migration_count == 0


class TestEndToEnd:
    """Smoke test: full island cycle with migration."""

    def test_cross_pollination(self) -> None:
        s = IslandSearch(
            num_islands=2,
            max_population=30,
            migration_interval=4,
            maximize=True,
        )
        # Add solutions to island 0 only (ideator_id=0)
        for i in range(4):
            s.on_result(f"r{i}", score=float(i + 1) * 10, ideator_id=0)

        # Migration should have fired (4 evals = interval)
        assert s._migration_count == 1
        # Island 1 should have received a solution from island 0
        assert len(s._islands[1]) > 0
        migrant_ids = {sol.id for sol in s._islands[1]}
        source_ids = {sol.id for sol in s._islands[0]}
        assert migrant_ids & source_ids, "Island 1 should contain a solution from island 0"
