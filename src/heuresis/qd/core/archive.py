"""MAP-Elites archives: GridArchive and CVTArchive.

Ported from science-codeevolve/src/codeevolve/database.py (Apache 2.0),
decoupled from the Program dataclass.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from heuresis.qd.core.cvt import closest_centroid_idx, cvt


@dataclass(frozen=True)
class Feature:
    """Defines one dimension of the MAP-Elites feature space.

    Attributes:
        name: Feature name (must match keys in the ``features`` dict passed to
              :meth:`Archive.add`).
        min_val: Lower bound of the feature range.
        max_val: Upper bound of the feature range.
        num_bins: Number of bins for grid discretisation.  Required for
                  :class:`GridArchive`; ignored by :class:`CVTArchive`.
        bin_names: Optional human-readable label per bin (aligned to bin index
                  0..num_bins-1). When present, names resolve generically (see
                  :func:`heuresis.qd.feature_namer`) — tasks need not ship a
                  per-task naming function.
    """

    name: str
    min_val: float
    max_val: float
    num_bins: int | None = None
    bin_names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class EliteEntry:
    """A single elite stored in an archive cell.

    Attributes:
        id: Caller-provided identifier for the solution.
        fitness: Fitness value of the solution.
        features: Feature values that placed this solution in its cell.
    """

    id: str
    fitness: float
    features: dict[str, float] = field(default_factory=dict)


class Archive(ABC):
    """Abstract base class for a MAP-Elites archive."""

    def __init__(self, features: list[Feature], *, maximize: bool = True) -> None:
        self.features: list[Feature] = features
        self.maximize: bool = maximize

    def _is_better(self, new: float, old: float) -> bool:
        """Return True if *new* fitness is better than *old*."""
        return new > old if self.maximize else new < old

    @abstractmethod
    def add(self, id: str, fitness: float, features: dict[str, float]) -> bool:
        """Try to insert a solution into the archive.

        Args:
            id: Unique identifier for the solution.
            fitness: Fitness value.
            features: Dict mapping feature name → value.

        Returns:
            ``True`` if the solution was inserted (new cell or better fitness),
            ``False`` otherwise.
        """

    def add_batch(
        self,
        ids: list[str],
        fitnesses: list[float],
        features_list: list[dict[str, float]],
    ) -> list[bool]:
        """Add multiple solutions. Returns list of booleans (inserted or not)."""
        return [
            self.add(id, fit, feat)
            for id, fit, feat in zip(ids, fitnesses, features_list)
        ]

    @abstractmethod
    def elites(self) -> list[EliteEntry]:
        """Return all elite entries currently in the archive."""

    @property
    def elite_ids(self) -> list[str]:
        """Return the IDs of all elites."""
        return [e.id for e in self.elites()]

    @property
    def size(self) -> int:
        """Number of occupied cells."""
        return len(self.elites())

    @abstractmethod
    def cell_count(self) -> int:
        """Total number of cells (occupied + empty)."""

    @abstractmethod
    def cell_visits(self) -> dict:
        """Return mapping of cell → total visit count."""

    @abstractmethod
    def cell_improvements(self) -> dict:
        """Return mapping of cell → improvement count."""

    @property
    def total_visits(self) -> int:
        """Total number of add() calls across all cells."""
        return sum(self.cell_visits().values())


class GridArchive(Archive):
    """Fixed-grid MAP-Elites archive.

    Each feature dimension is discretised into ``num_bins`` bins, producing a
    regular grid.  Each cell stores at most one elite (the highest-fitness
    solution mapped to that cell).
    """

    def __init__(self, features: list[Feature], *, maximize: bool = True) -> None:
        super().__init__(features, maximize=maximize)
        for f in self.features:
            if f.num_bins is None:
                raise ValueError(
                    f"Feature '{f.name}' must have num_bins set for GridArchive."
                )
        self._num_cells: int = math.prod(f.num_bins for f in self.features)
        self._map: dict[tuple[int, ...], EliteEntry] = {}
        self._visits: dict[tuple[int, ...], int] = {}
        self._improvements: dict[tuple[int, ...], int] = {}
        self._last_displaced: EliteEntry | None = None

    def cell_count(self) -> int:
        return self._num_cells

    # ------------------------------------------------------------------

    def _cell_index(self, features: dict[str, float]) -> tuple[int, ...] | None:
        indices: list[int] = []
        for feat in self.features:
            value = features.get(feat.name)
            if value is None:
                return None
            value = max(feat.min_val, min(value, feat.max_val))
            span = feat.max_val - feat.min_val
            proportion = (value - feat.min_val) / span if span > 0 else 0.0
            idx = min(int(proportion * feat.num_bins), feat.num_bins - 1)
            indices.append(idx)
        return tuple(indices)

    @property
    def last_displaced(self) -> EliteEntry | None:
        """The elite displaced by the most recent successful add(), or None."""
        return self._last_displaced

    def cell_key_for(self, features: dict[str, float]) -> str | None:
        """Return the string cell key for a feature vector, or None if invalid."""
        cell = self._cell_index(features)
        return self._key_to_str(cell) if cell is not None else None

    def add(self, id: str, fitness: float, features: dict[str, float]) -> bool:
        cell = self._cell_index(features)
        if cell is None:
            self._last_displaced = None
            return False
        self._visits[cell] = self._visits.get(cell, 0) + 1
        existing = self._map.get(cell)
        if existing is None or self._is_better(fitness, existing.fitness):
            self._last_displaced = existing  # None if cell was empty
            self._map[cell] = EliteEntry(id=id, fitness=fitness, features=features)
            self._improvements[cell] = self._improvements.get(cell, 0) + 1
            return True
        self._last_displaced = None
        return False

    def elites(self) -> list[EliteEntry]:
        return list(self._map.values())

    def occupied_cells(self) -> list[tuple[tuple[int, ...], EliteEntry]]:
        """Return all occupied cells as (cell_index, elite) pairs."""
        return list(self._map.items())

    def cell_visits(self) -> dict[tuple[int, ...], int]:
        return dict(self._visits)

    def cell_improvements(self) -> dict[tuple[int, ...], int]:
        return dict(self._improvements)

    # -- Serialization --------------------------------------------------

    @staticmethod
    def _key_to_str(key: tuple[int, ...]) -> str:
        return ",".join(str(k) for k in key)

    @staticmethod
    def _key_from_str(s: str) -> tuple[int, ...]:
        return tuple(int(x) for x in s.split(","))

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict of the full archive state."""
        return {
            "features": [
                {"name": f.name, "min_val": f.min_val,
                 "max_val": f.max_val, "num_bins": f.num_bins,
                 "bin_names": list(f.bin_names) if f.bin_names else None}
                for f in self.features
            ],
            "maximize": self.maximize,
            "map": {
                self._key_to_str(k): {
                    "id": e.id, "fitness": e.fitness, "features": e.features,
                }
                for k, e in self._map.items()
            },
            "visits": {self._key_to_str(k): v for k, v in self._visits.items()},
            "improvements": {self._key_to_str(k): v for k, v in self._improvements.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> GridArchive:
        """Reconstruct a GridArchive from a dict produced by :meth:`to_dict`."""
        features = [
            Feature(name=f["name"], min_val=f["min_val"],
                    max_val=f["max_val"], num_bins=f["num_bins"],
                    bin_names=tuple(f["bin_names"]) if f.get("bin_names") else None)
            for f in data["features"]
        ]
        archive = cls(features, maximize=data.get("maximize", True))
        archive._map = {
            cls._key_from_str(k): EliteEntry(
                id=e["id"], fitness=e["fitness"], features=e["features"],
            )
            for k, e in data["map"].items()
        }
        archive._visits = {
            cls._key_from_str(k): v for k, v in data["visits"].items()
        }
        archive._improvements = {
            cls._key_from_str(k): v for k, v in data["improvements"].items()
        }
        return archive

    def __repr__(self) -> str:
        return f"GridArchive(cells={self._num_cells}, occupied={self.size})"


class CVTArchive(Archive):
    """Centroidal Voronoi Tessellation MAP-Elites archive.

    The feature space is partitioned into Voronoi regions around centroids
    computed via Lloyd's algorithm.
    """

    def __init__(
        self,
        features: list[Feature],
        num_centroids: int,
        num_init_samples: int = 10_000,
        max_iter: int = 300,
        tolerance: float = 1e-6,
        *,
        maximize: bool = True,
    ) -> None:
        super().__init__(features, maximize=maximize)
        feature_bounds = [(f.min_val, f.max_val) for f in features]
        self.centroids: np.ndarray = cvt(
            num_centroids, num_init_samples, feature_bounds, max_iter, tolerance
        )
        self._map: dict[int, EliteEntry] = {}
        self._visits: dict[int, int] = {}
        self._improvements: dict[int, int] = {}
        self._last_displaced: EliteEntry | None = None

    def cell_count(self) -> int:
        return len(self.centroids)

    @property
    def last_displaced(self) -> EliteEntry | None:
        """The elite displaced by the most recent successful add(), or None."""
        return self._last_displaced

    def add(self, id: str, fitness: float, features: dict[str, float]) -> bool:
        point = np.zeros(len(self.features))
        for i, feat in enumerate(self.features):
            value = features.get(feat.name)
            if value is None:
                self._last_displaced = None
                return False
            point[i] = value

        idx = closest_centroid_idx(point, self.centroids)
        self._visits[idx] = self._visits.get(idx, 0) + 1
        existing = self._map.get(idx)
        if existing is None or self._is_better(fitness, existing.fitness):
            self._last_displaced = existing  # None if cell was empty
            self._map[idx] = EliteEntry(id=id, fitness=fitness, features=features)
            self._improvements[idx] = self._improvements.get(idx, 0) + 1
            return True
        self._last_displaced = None
        return False

    def elites(self) -> list[EliteEntry]:
        return list(self._map.values())

    def cell_visits(self) -> dict[int, int]:
        return dict(self._visits)

    def cell_improvements(self) -> dict[int, int]:
        return dict(self._improvements)

    def __repr__(self) -> str:
        return f"CVTArchive(centroids={len(self.centroids)}, occupied={self.size})"
