"""Go-Explore-inspired cell-targeted MAP-Elites strategy."""

from __future__ import annotations

import math
from typing import Callable

from heuresis.qd.core.archive import Feature
from heuresis.qd.core.metrics import best_fitness, coverage, qd_score
from heuresis.qd.map_elites.search import CellTargetedMapElitesSearch


class GoExploreSearch(CellTargetedMapElitesSearch):
    """Cell-targeted MAP-Elites with score/visit-weighted cell sampling.

    Parent/operator semantics are inherited from
    :class:`CellTargetedMapElitesSearch`. The only algorithmic difference is
    target-cell selection:

        weight = (quality + alpha) / sqrt(visits + 1)

    where quality is improvement over the task baseline in the configured
    optimization direction. Empty cells have quality 0 and remain selectable
    through the alpha floor.
    """

    def __init__(
        self,
        features: list[Feature],
        *,
        maximize: bool = False,
        baseline_score: float = 0.0,
        alpha: float = 0.01,
        crossover_rate: float = 0.5,
        feature_name_fn: Callable[[dict[str, float]], dict[str, str]] | None = None,
        seed: int = 42,
        memory: bool = False,
    ) -> None:
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        super().__init__(
            features,
            maximize=maximize,
            empty_weight=1.0,
            crossover_rate=crossover_rate,
            feature_name_fn=feature_name_fn,
            seed=seed,
            memory=memory,
        )
        self.baseline_score = baseline_score
        self.alpha = alpha

    def _cell_weights(self, cells: list[tuple[int, ...]]) -> list[float]:
        visits = self.archive.cell_visits()
        occupied = dict(self.archive.occupied_cells())
        weights: list[float] = []
        for cell in cells:
            elite = occupied.get(cell)
            if elite is None:
                quality = 0.0
            elif self.archive.maximize:
                quality = max(0.0, elite.fitness - self.baseline_score)
            else:
                quality = max(0.0, self.baseline_score - elite.fitness)
            weights.append((quality + self.alpha) / math.sqrt(visits.get(cell, 0) + 1))
        return weights

    def summary(self) -> str:
        visits = self.archive.cell_visits()
        occupied_keys = {cell for cell, _ in self.archive.occupied_cells()}
        occupied_visits = [v for cell, v in visits.items() if cell in occupied_keys]
        if occupied_visits:
            visit_str = (
                "Visits (occupied min/max/mean): "
                f"{min(occupied_visits)}/"
                f"{max(occupied_visits)}/"
                f"{sum(occupied_visits) / len(occupied_visits):.1f}"
            )
        else:
            visit_str = "Visits (occupied min/max/mean): 0/0/0.0"
        return (
            f"Archive: {self.archive.size}/{self.archive.cell_count()} cells\n"
            f"Coverage: {coverage(self.archive):.0%}\n"
            f"QD Score: {qd_score(self.archive):.4f}\n"
            f"Best Fitness: {best_fitness(self.archive):.4f}\n"
            f"{visit_str}"
        )
