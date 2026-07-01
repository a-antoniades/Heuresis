"""Standard MAP-Elites quality-diversity metrics."""

from __future__ import annotations

from heuresis.qd.core.archive import Archive


def coverage(archive: Archive) -> float:
    """Fraction of cells occupied: ``size / cell_count``."""
    total = archive.cell_count()
    if total == 0:
        return 0.0
    return archive.size / total


def qd_score(archive: Archive) -> float:
    """Sum of fitness across all occupied cells."""
    return sum(e.fitness for e in archive.elites())


def best_fitness(archive: Archive) -> float:
    """Maximum fitness across all elites.  Returns ``-inf`` if empty."""
    elites = archive.elites()
    if not elites:
        return float("-inf")
    return max(e.fitness for e in elites)
