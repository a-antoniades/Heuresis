"""SearchStrategy ABC and shared helpers for QD algorithms.

Every strategy (linear, map-elites, islands, omni-epic) implements the
same five methods so experiment loops can swap algorithms without
changing structure.

Helpers in this module are shared across strategies:
  - ``compute_generation``: lineage depth from parent generations
  - ``extract_summary``: one-line summary from an idea document
  - ``default_feature_names``: fallback feature-name formatter
  - ``format_archive_context``: textual archive snapshot for ideator prompts
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Callable

from heuresis.qd.core.archive import Feature, GridArchive
from heuresis.qd.core.metrics import best_fitness, coverage, qd_score


class SearchStrategy(ABC):
    """Common interface for QD search algorithms.

    Every strategy exposes the same five methods so experiment loops
    can swap algorithms without changing structure.
    """

    @abstractmethod
    def select_parents(self, *, ideator_id: int = 0) -> list[str]:
        """Return run IDs the ideator should see as parent context."""
        ...

    @abstractmethod
    def context(self, *, ideator_id: int = 0) -> str:
        """Extra text for the ideator prompt (archive state, island info)."""
        ...

    @abstractmethod
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
        """Update internal state with a completed run.

        Returns metadata dict to persist (lineage, archive status, etc.).
        """
        ...

    @abstractmethod
    def rebuild(self, records: list[tuple[str, float | None, dict[str, Any]]]) -> None:
        """Restore state from (run_id, score, metadata) triples."""
        ...

    @abstractmethod
    def summary(self) -> str:
        """Human-readable stats about the search."""
        ...


def compute_generation(parent_ids: list[str] | None, gen_map: dict[str, int]) -> int:
    """Compute generation number from parent generations."""
    if not parent_ids:
        return 0
    parent_gens = [gen_map.get(pid, 0) for pid in parent_ids]
    return max(parent_gens) + 1


def extract_summary(idea: str, max_chars: int = 120) -> str:
    """Extract a short summary from an idea document."""
    for line in idea.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("## strategy"):
            after = stripped[len("## strategy"):].strip().lstrip(":").strip()
            if after and len(after) > 10:
                return after[:max_chars]
            continue
        if stripped and not stripped.startswith("#") and len(stripped) > 10:
            return stripped[:max_chars]
    return idea[:max_chars].replace("\n", " ").strip()


def default_feature_names(features: dict[str, float]) -> dict[str, str]:
    """Fallback: use raw feature values as names."""
    return {k: str(v) for k, v in features.items()}


def feature_namer(
    features: Sequence[Feature],
) -> Callable[[dict[str, float]], dict[str, str]]:
    """Build a name function from labeled :class:`Feature` axes.

    Maps each feature value to ``bin_names[int(value)]``, falling back to the
    raw value for axes that carry no labels. This replaces per-task naming
    functions: labels live on the ``Feature`` list, the lookup is generic.
    """
    by_name = {f.name: f for f in features}

    def _namer(values: dict[str, float]) -> dict[str, str]:
        out: dict[str, str] = {}
        for name, value in values.items():
            f = by_name.get(name)
            idx = int(value)
            if f is not None and f.bin_names and 0 <= idx < len(f.bin_names):
                out[name] = f.bin_names[idx]
            else:
                out[name] = str(value)
        return out

    return _namer


def format_archive_context(
    archive: GridArchive,
    *,
    summaries: dict[str, str] | None = None,
    feature_name_fn: Callable[[dict[str, float]], dict[str, str]] | None = None,
) -> str:
    """Format archive state as text for the ideator prompt."""
    if feature_name_fn is None:
        feature_name_fn = default_feature_names
    summaries = summaries or {}

    total = archive.cell_count()
    occupied = archive.size
    lines = [
        "=== Quality-Diversity Archive ===",
        f"Coverage: {occupied}/{total} cells ({coverage(archive):.0%})",
        f"QD Score: {qd_score(archive):.4f}",
        f"Best Fitness: {best_fitness(archive):.4f}",
    ]

    if occupied > 0:
        lines.append("")
        lines.append("Elite solutions (one per explored region):")
        for elite in sorted(archive.elites(), key=lambda e: e.fitness, reverse=True):
            feat_str = ""
            if elite.features:
                names = feature_name_fn(elite.features)
                feat_str = " | ".join(f"{k}: {v}" for k, v in names.items())
            strategy = summaries.get(elite.id, "")
            parts = [f"- {elite.id}: score={elite.fitness:.4f}"]
            if feat_str:
                parts.append(f"[{feat_str}]")
            if strategy:
                parts.append(f'"{strategy}"')
            lines.append(" ".join(parts))

    if isinstance(archive, GridArchive) and occupied < total:
        lines.append("")
        lines.append("Unexplored regions to target:")
        occupied_indices = {cell for cell, _ in archive.occupied_cells()}
        dims = [f.num_bins for f in archive.features]
        all_cells = list(itertools.product(*(range(d) for d in dims)))
        empty_cells = [c for c in all_cells if c not in occupied_indices]
        for cell_idx in empty_cells[:10]:
            feat_dict = {f.name: float(cell_idx[i]) for i, f in enumerate(archive.features)}
            names = feature_name_fn(feat_dict)
            label = " x ".join(names.values())
            lines.append(f"  - {label}")
        if len(empty_cells) > 10:
            lines.append(f"  ... and {len(empty_cells) - 10} more")

    return "\n".join(lines)
