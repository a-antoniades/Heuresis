"""Archive visualization for MAP-Elites.

Renders a GridArchive as a heatmap showing fitness per cell,
with human-readable axis labels and summary statistics.

Requires matplotlib (optional dependency).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from heuresis.qd.core.archive import GridArchive


def plot_archive(
    archive: "GridArchive",
    path: str | Path,
    *,
    feature_name_fn: Callable[[dict[str, float]], dict[str, str]] | None = None,
    title: str = "MAP-Elites Archive",
    cmap: str = "viridis_r",
    empty_color: str = "#e0e0e0",
    figsize: tuple[float, float] | None = None,
    annotate: bool = True,
    total_runs: int | None = None,
    successful_runs: int | None = None,
) -> Path:
    """Save a heatmap of the archive to *path*.

    Args:
        archive: A :class:`GridArchive` (must be 2D).
        path: Output file path (.png, .pdf, .svg).
        feature_name_fn: Maps ``{feature_name: float_val}`` to
            ``{feature_name: str_label}``. Used for axis tick labels.
        title: Figure title.
        cmap: Matplotlib colormap. Use ``"viridis_r"`` for lower-is-better,
            ``"viridis"`` for higher-is-better.
        empty_color: Fill color for unoccupied cells.
        figsize: ``(width, height)`` in inches. Auto-sized if ``None``.
        annotate: If ``True``, show fitness values and elite IDs in cells.
        total_runs: Total number of runs attempted (for subtitle).
        successful_runs: Number of valid/successful runs (for subtitle).

    Returns:
        The resolved output path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from heuresis.qd.core.metrics import coverage, qd_score

    features = archive.features
    if len(features) != 2:
        raise ValueError(f"plot_archive requires a 2D archive, got {len(features)}D")

    nrows = features[0].num_bins
    ncols = features[1].num_bins

    grid = np.full((nrows, ncols), np.nan)
    id_grid: list[list[str | None]] = [[None] * ncols for _ in range(nrows)]
    for cell_idx, elite in archive.occupied_cells():
        r, c = cell_idx
        grid[r, c] = elite.fitness
        id_grid[r][c] = elite.id

    if feature_name_fn:
        row_labels = [
            feature_name_fn({features[0].name: float(i)})[features[0].name]
            for i in range(nrows)
        ]
        col_labels = [
            feature_name_fn({features[1].name: float(i)})[features[1].name]
            for i in range(ncols)
        ]
    else:
        row_labels = [str(i) for i in range(nrows)]
        col_labels = [str(i) for i in range(ncols)]

    if figsize is None:
        figsize = (max(8, ncols * 1.6), max(6, nrows * 0.8))

    fig, ax = plt.subplots(figsize=figsize)

    valid_vals = grid[~np.isnan(grid)]
    outlier_mask = np.full_like(grid, False, dtype=bool)
    outlier_color = "#aaaaaa"
    if len(valid_vals) >= 3:
        median = np.median(valid_vals)
        mad = np.median(np.abs(valid_vals - median))
        if mad > 0:
            threshold = 10.0 * mad
            for r in range(nrows):
                for c in range(ncols):
                    if not np.isnan(grid[r, c]) and abs(grid[r, c] - median) > threshold:
                        outlier_mask[r, c] = True

    display = grid.copy()
    display[outlier_mask] = np.nan
    masked = np.ma.masked_invalid(display)
    im = ax.imshow(masked, cmap=cmap, aspect="auto")

    for r in range(nrows):
        for c in range(ncols):
            if np.isnan(grid[r, c]):
                ax.add_patch(plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    fill=True, color=empty_color, ec="white", lw=0.5,
                ))
            elif outlier_mask[r, c]:
                ax.add_patch(plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    fill=True, color=outlier_color, ec="white", lw=0.5,
                ))
                if annotate:
                    ax.text(c, r - 0.12, f"{grid[r, c]:.4f}",
                            ha="center", va="center",
                            fontsize=7, fontweight="bold", color="white")
                    if id_grid[r][c]:
                        ax.text(c, r + 0.22, id_grid[r][c],
                                ha="center", va="center",
                                fontsize=5, color="white", alpha=0.8)
            elif annotate:
                ax.text(c, r - 0.12, f"{grid[r, c]:.4f}",
                        ha="center", va="center",
                        fontsize=7, fontweight="bold", color="white")
                if id_grid[r][c]:
                    ax.text(c, r + 0.22, id_grid[r][c],
                            ha="center", va="center",
                            fontsize=5, color="white", alpha=0.8)

    ax.set_xticks(range(ncols))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel(features[1].name.replace("_", " ").title())
    ax.set_ylabel(features[0].name.replace("_", " ").title())

    if masked.count() > 0:
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Fitness")

    parts = [
        f"Coverage: {archive.size}/{archive.cell_count()} ({coverage(archive):.0%})",
        f"QD Score: {qd_score(archive):.4f}",
    ]
    if total_runs is not None:
        if successful_runs is not None:
            parts.append(f"Runs: {successful_runs}/{total_runs}")
        else:
            parts.append(f"Attempts: {total_runs}")
    subtitle = "  |  ".join(parts)
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
