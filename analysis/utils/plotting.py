"""Shared plotting utilities for MAP-Elites analysis.

Provides consistent styling, color scheme, and reusable plot components.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Consistent color scheme across all analyses
COLORS = {
    "LLM-guided": "#2196F3",
    "Baseline (cell-targeted)": "#FF9800",
    "Greedy (no MAP-Elites)": "#4CAF50",
}

# Shortened feature labels for heatmaps
TARGET_LABELS = [
    "Attention", "MLP/FFN", "Norm/Init", "Pos. Enc.",
    "Embed/Vocab", "Layer Struct.", "Window/Ctx",
    "Aux. Module", "Optim/Sched", "Data/Reg",
]

STYLE_LABELS = [
    "Swap/Replace", "Add/Augment", "Tune/Scale",
    "Sched/Curric", "Sparsify", "Stabilize", "Adaptive",
]


def setup_style() -> None:
    """Apply consistent matplotlib style for publication figures."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def plot_heatmap(
    ax: plt.Axes,
    grid: np.ma.MaskedArray,
    row_labels: list[str],
    col_labels: list[str],
    *,
    cmap: str = "viridis_r",
    vmin: float | None = None,
    vmax: float | None = None,
    annotate: bool = True,
    fmt: str = ".3f",
    empty_color: str = "#e0e0e0",
) -> plt.cm.ScalarMappable:
    """Draw an annotated heatmap on the given axes.

    Returns the ScalarMappable for creating a shared colorbar.
    """
    n_rows, n_cols = grid.shape

    # Draw empty cell background
    for i in range(n_rows):
        for j in range(n_cols):
            if grid.mask[i, j] if np.ma.is_masked(grid) else np.isnan(grid[i, j]):
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor=empty_color, edgecolor="white", linewidth=0.5,
                ))

    im = ax.imshow(
        grid, cmap=cmap, vmin=vmin, vmax=vmax,
        aspect="auto", interpolation="nearest",
    )

    if annotate:
        for i in range(n_rows):
            for j in range(n_cols):
                val = grid[i, j]
                if np.ma.is_masked(grid) and grid.mask[i, j]:
                    continue
                if np.isnan(val):
                    continue
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                        fontsize=7, color="white" if val < (vmin or 0) + 0.5 * ((vmax or 1) - (vmin or 0)) else "black")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)

    return im


def savefig(fig: plt.Figure, path: str | Path) -> None:
    """Save figure to path, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
