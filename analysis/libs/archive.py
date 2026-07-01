"""Archive reconstruction and QD metric utilities.

Provides incremental archive reconstruction from run DataFrames,
QD score computation, and grid conversion for heatmap visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from heuresis.qd.core.archive import Feature, GridArchive

if TYPE_CHECKING:
    import pandas as pd

# Ensure qd library is importable (kept for backward compat with scripts that
# import this module without the heuresis package on sys.path)
_QD_SRC = Path(__file__).resolve().parents[2] / "src"

BASELINE_BPB = 0.998


@dataclass
class ArchiveSnapshot:
    """State of the archive after processing one iteration."""

    iteration: int
    coverage: float  # fraction of cells occupied (0-1)
    best_fitness: float
    qd_score: float  # improvement-based QD score
    n_elites: int


def reconstruct_archive_incremental(
    df: "pd.DataFrame",
    features: list[Feature],
    *,
    maximize: bool = False,
    baseline_bpb: float = BASELINE_BPB,
) -> tuple[list[ArchiveSnapshot], GridArchive]:
    """Rebuild an archive one iteration at a time, recording snapshots.

    Args:
        df: DataFrame with columns ``iteration``, ``score``,
            ``valid_filtered``, ``mod_target``, ``int_style``.
        features: Feature definitions for the GridArchive.
        maximize: Whether higher fitness is better.
        baseline_bpb: Reference score for QD score computation.

    Returns:
        (snapshots, final_archive) where snapshots is a list of
        ArchiveSnapshot objects (one per iteration that had a valid run).
    """
    import pandas as pd

    archive = GridArchive(features, maximize=maximize)
    snapshots: list[ArchiveSnapshot] = []
    total_cells = archive.cell_count()

    for _, row in df.sort_values("iteration").iterrows():
        if not row.get("valid_filtered", False):
            continue
        if pd.isna(row["mod_target"]) or pd.isna(row["int_style"]):
            continue

        feat_dict = {
            features[0].name: float(row["mod_target"]),
            features[1].name: float(row["int_style"]),
        }
        archive.add(row["run_id"], row["score"], feat_dict)

        elites = archive.elites()
        best = min(e.fitness for e in elites) if not maximize else max(e.fitness for e in elites)
        qd = qd_score_improvement(archive, baseline_bpb)

        snapshots.append(ArchiveSnapshot(
            iteration=int(row["iteration"]),
            coverage=archive.size / total_cells,
            best_fitness=best,
            qd_score=qd,
            n_elites=archive.size,
        ))

    return snapshots, archive


def qd_score_improvement(
    archive: GridArchive,
    baseline_bpb: float = BASELINE_BPB,
) -> float:
    """Compute improvement-based QD score (higher is better).

    For a minimization archive, QD score = sum of improvements over baseline
    across all occupied cells: sum(max(0, baseline - elite_bpb)).
    """
    return sum(max(0.0, baseline_bpb - e.fitness) for e in archive.elites())


def archive_to_grid(
    archive: GridArchive,
    n_rows: int,
    n_cols: int,
) -> np.ma.MaskedArray:
    """Convert archive to a 2D grid for heatmap visualization.

    Returns a masked array where empty cells are masked.
    Rows = feature 0 (modification_target), Cols = feature 1 (intervention_style).
    """
    grid = np.full((n_rows, n_cols), np.nan)
    for cell_idx, elite in archive.occupied_cells():
        row, col = cell_idx
        grid[row, col] = elite.fitness
    return np.ma.masked_invalid(grid)
