"""MapElitesSearch and CellTargetedMapElitesSearch."""

from __future__ import annotations

import itertools
import random as _random
from typing import Any, Callable

from heuresis.qd.core.archive import Feature, GridArchive
from heuresis.qd.core.base import (
    SearchStrategy,
    compute_generation,
    default_feature_names,
    extract_summary,
    format_archive_context,
)
from heuresis.qd.core.metrics import best_fitness, coverage, qd_score
from heuresis.qd.core.selection import Selector


class MapElitesSearch(SearchStrategy):
    """MAP-Elites: maintain a grid archive of diverse elites.

    Wraps GridArchive + Selector. Provides archive context formatting,
    diverse parent selection, and lineage tracking.
    """

    def __init__(
        self,
        features: list[Feature],
        *,
        maximize: bool = True,
        parent_k_range: tuple[int, int] = (1, 2),
        selection_policy: str = "tournament",
        feature_name_fn: Callable[[dict[str, float]], dict[str, str]] | None = None,
        seed: int = 42,
        memory: bool = False,
    ) -> None:
        lo, hi = parent_k_range
        if lo < 1 or hi < lo:
            raise ValueError(f"parent_k_range must satisfy 1 <= lo <= hi, got {parent_k_range}")
        self.archive = GridArchive(features, maximize=maximize)
        self.selector = Selector(seed=seed)
        self.parent_k_range = parent_k_range
        self.selection_policy = selection_policy
        self.feature_name_fn = feature_name_fn or default_feature_names
        # Campaign-level flag for the experiment loop — strategy itself
        # doesn't use memory. See LinearSearch for the pattern.
        self.memory = memory
        self._n_elites = 0
        self._generation_map: dict[str, int] = {}
        self._idea_summaries: dict[str, str] = {}
        self._k_rng = _random.Random(seed)

    def select_parents(self, *, ideator_id: int = 0) -> list[str]:
        if self.archive.size == 0:
            return []
        self.selector.update(archive=self.archive)
        lo, hi = self.parent_k_range
        k = self._k_rng.randint(lo, hi)
        return self.selector.sample(self.selection_policy, k=k)

    def context(self, *, ideator_id: int = 0) -> str:
        if self.archive.size == 0:
            return ""
        return format_archive_context(
            self.archive,
            summaries=self._idea_summaries,
            feature_name_fn=self.feature_name_fn,
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
        generation = compute_generation(parent_ids, self._generation_map)
        self._generation_map[run_id] = generation

        if idea:
            self._idea_summaries[run_id] = extract_summary(idea)

        metadata: dict[str, Any] = {
            "parent_ids": parent_ids or [],
            "generation": generation,
        }
        if idea is not None:
            metadata["idea"] = idea
        if features is not None:
            metadata["qd_features"] = features
            metadata["feature_names"] = self.feature_name_fn(features)

        if score is None or features is None:
            metadata["archive_status"] = "invalid"
            return metadata

        added = self.archive.add(run_id, score, features)
        if added:
            self._n_elites += 1
            metadata["archive_status"] = "elite"
            metadata["cell_key"] = self.archive.cell_key_for(features)
            displaced = self.archive.last_displaced
            if displaced is not None:
                metadata["displaced_id"] = displaced.id
                metadata["displaced_fitness"] = displaced.fitness
        else:
            metadata["archive_status"] = "dominated"

        return metadata

    def rebuild(self, records: list[tuple[str, float | None, dict[str, Any]]]) -> None:
        for run_id, score, metadata in records:
            feats = metadata.get("qd_features")
            if feats and score is not None:
                self.archive.add(run_id, score, feats)
            gen = metadata.get("generation", 0)
            self._generation_map[run_id] = gen
            idea = metadata.get("idea")
            if idea:
                self._idea_summaries[run_id] = extract_summary(idea)

    def summary(self) -> str:
        return (
            f"Archive: {self.archive.size}/{self.archive.cell_count()} cells\n"
            f"Coverage: {coverage(self.archive):.0%}\n"
            f"QD Score: {qd_score(self.archive):.4f}\n"
            f"Best Fitness: {best_fitness(self.archive):.4f}\n"
            f"New elites: {self._n_elites}"
        )


class CellTargetedMapElitesSearch(MapElitesSearch):
    """MAP-Elites variant: random cell selection (weighted toward empty) with LLM focus.

    Instead of showing the full archive and letting the LLM choose where
    to explore, we pick a target cell (weighted toward empty cells) and
    describe it explicitly. Empty target cells get no parents. Occupied
    target cells either mutate the target elite or cross it over with one
    additional archive elite.
    """

    def __init__(
        self,
        features: list[Feature],
        *,
        maximize: bool = True,
        empty_weight: float = 3.0,
        crossover_rate: float = 0.5,
        feature_name_fn: Callable[[dict[str, float]], dict[str, str]] | None = None,
        seed: int = 42,
        memory: bool = False,
    ) -> None:
        if empty_weight < 0:
            raise ValueError(f"empty_weight must be >= 0, got {empty_weight}")
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError(f"crossover_rate must be in [0, 1], got {crossover_rate}")
        super().__init__(
            features, maximize=maximize, parent_k_range=(1, 1),
            feature_name_fn=feature_name_fn, seed=seed, memory=memory,
        )
        self.empty_weight = empty_weight
        self.crossover_rate = crossover_rate
        self._rng = _random.Random(seed)
        self._last_target: dict[int, tuple[int, ...]] = {}
        self._last_operator: dict[int, str] = {}

    def _all_cells(self) -> list[tuple[int, ...]]:
        dims = [f.num_bins for f in self.archive.features]
        return list(itertools.product(*(range(d) for d in dims)))

    def _cell_weights(self, cells: list[tuple[int, ...]]) -> list[float]:
        occupied = {c for c, _ in self.archive.occupied_cells()}
        return [(self.empty_weight if c not in occupied else 1.0) for c in cells]

    def _sample_target_cells(self, k: int) -> list[tuple[int, ...]]:
        cells = self._all_cells()
        weights = self._cell_weights(cells)
        selected: list[tuple[int, ...]] = []
        for _ in range(min(k, len(cells))):
            if sum(weights) <= 0:
                weights = [1.0] * len(cells)
            idx = self._rng.choices(range(len(cells)), weights=weights, k=1)[0]
            selected.append(cells.pop(idx))
            weights.pop(idx)
        return selected

    def _sample_target_cell(self) -> tuple[int, ...]:
        return self._sample_target_cells(1)[0]

    def select_parents(self, *, ideator_id: int = 0) -> list[str]:
        target_cells = self._sample_target_cells(2)
        target = target_cells[0]
        self._last_target[ideator_id] = target
        occupied = dict(self.archive.occupied_cells())
        elite = occupied.get(target)
        if elite is None:
            self._last_operator[ideator_id] = "explore_empty"
            return []

        parents = [elite.id]
        want_crossover = self._rng.random() < self.crossover_rate
        donor = occupied.get(target_cells[1]) if len(target_cells) > 1 else None
        if want_crossover and donor is not None:
            parents.append(donor.id)
            self._last_operator[ideator_id] = "crossover"
        else:
            self._last_operator[ideator_id] = "mutate"
        return parents

    def context(self, *, ideator_id: int = 0) -> str:
        if ideator_id not in self._last_target:
            self.select_parents(ideator_id=ideator_id)
        target = self._last_target[ideator_id]
        feat_dict = {
            f.name: float(target[i])
            for i, f in enumerate(self.archive.features)
        }
        names = self.feature_name_fn(feat_dict)
        label = " x ".join(names.values())
        occupied = dict(self.archive.occupied_cells())
        elite = occupied.get(target)
        total = self.archive.cell_count()
        pct = f"{self.archive.size / total:.0%}" if total else "0%"
        lines = [
            f"Target cell: {label}",
            f"Archive: {self.archive.size}/{total} cells filled ({pct}).",
        ]
        if elite is not None:
            lines.append(
                f"This cell is occupied by {elite.id} (fitness={elite.fitness:.4f})."
            )
            op = self._last_operator.get(ideator_id, "mutate")
            if op == "crossover":
                lines.extend([
                    "",
                    "Operator: CROSSOVER",
                    "Combine the two selected parents. Parent A is the target-cell elite; "
                    "Parent B is another archive elite. Identify the load-bearing mechanism "
                    "from each, then design a coherent synthesis. Avoid simply stacking "
                    "incompatible changes.",
                ])
            else:
                lines.extend([
                    "",
                    "Operator: MUTATE",
                    "Continue from the selected parent. Identify its load-bearing mechanism, "
                    "then propose a material change that improves it or branches into a "
                    "stronger direction. The result does not need to stay in the same "
                    "archive cell.",
                ])
        else:
            lines.append(
                "This cell is EMPTY. Propose a new approach aimed at this unexplored region."
            )
        return "\n".join(lines)
