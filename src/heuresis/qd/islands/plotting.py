"""Island-search visualization.

Renders a multi-panel figure showing:
  1. Per-island fitness distributions (violin + strip plot)
  2. Fitness trajectories (best-so-far over evaluations)
  3. Top-k idea tables per island (max 4 columns, wraps to rows)

Requires matplotlib (optional dependency).

Works with our RunRecord (idea/island_id in metadata dict).
"""

from __future__ import annotations

import math
import re
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from heuresis.models import RunRecord


def _extract_idea_summary(text: str, max_chars: int) -> str:
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or re.match(r"^[-*_]{3,}$", line):
            continue
        line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        line = re.sub(r"`(.+?)`", r"\1", line)
        return line[:max_chars - 1] + "…" if len(line) > max_chars else line
    first = text.split("\n", 1)[0].strip().lstrip("#").strip()
    return first[:max_chars - 1] + "…" if len(first) > max_chars else first


def _island_colors(n: int) -> list[str]:
    palette = [
        "#2196F3", "#E91E63", "#4CAF50", "#FF9800",
        "#9C27B0", "#00BCD4", "#F44336", "#8BC34A",
    ]
    if n <= len(palette):
        return palette[:n]
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10")
    return [
        "#{:02x}{:02x}{:02x}".format(*(int(c * 255) for c in cmap(i / max(n - 1, 1))[:3]))
        for i in range(n)
    ]


def _build_island_populations(
    runs: list[RunRecord],
    num_islands: int,
    lower_is_better: bool = True,
) -> list[list[tuple[str, float]]]:
    """Reconstruct sorted island populations from RunRecords.

    Returns a list of (run_id, score) pairs per island, sorted best-first.
    """
    islands: list[list[tuple[str, float]]] = [[] for _ in range(num_islands)]
    for run in runs:
        island_id = run.metadata.get("island_id")
        if island_id is None or run.score is None:
            continue
        island_id = int(island_id)
        if 0 <= island_id < num_islands:
            islands[island_id].append((run.run_id, run.score))

    for pop in islands:
        pop.sort(key=lambda x: x[1], reverse=not lower_is_better)
    return islands


def _build_fitness_timeline(
    runs: list[RunRecord],
    num_islands: int,
    lower_is_better: bool = True,
) -> tuple[list[list[int]], list[list[float]], list[list[float]], list[list[float]]]:
    eval_steps: list[list[int]] = [[] for _ in range(num_islands)]
    best_so_far: list[list[float]] = [[] for _ in range(num_islands)]
    pop_best: list[list[float]] = [[] for _ in range(num_islands)]
    pop_worst: list[list[float]] = [[] for _ in range(num_islands)]

    running_best: list[float | None] = [None] * num_islands
    island_scores: list[list[float]] = [[] for _ in range(num_islands)]
    global_step = 0

    for run in runs:
        island_id = run.metadata.get("island_id")
        if island_id is None or run.score is None:
            continue
        island_id = int(island_id)
        if island_id < 0 or island_id >= num_islands:
            continue

        global_step += 1
        island_scores[island_id].append(run.score)

        prev = running_best[island_id]
        if prev is None:
            running_best[island_id] = run.score
        elif lower_is_better and run.score < prev:
            running_best[island_id] = run.score
        elif not lower_is_better and run.score > prev:
            running_best[island_id] = run.score

        eval_steps[island_id].append(global_step)
        best_so_far[island_id].append(running_best[island_id])
        pop_best[island_id].append(
            min(island_scores[island_id]) if lower_is_better
            else max(island_scores[island_id])
        )
        pop_worst[island_id].append(
            max(island_scores[island_id]) if lower_is_better
            else min(island_scores[island_id])
        )

    return eval_steps, best_so_far, pop_best, pop_worst


def plot_islands(
    runs: list[RunRecord],
    num_islands: int,
    path: str | Path,
    *,
    title: str = "Island Search",
    top_k: int = 5,
    idea_max_chars: int = 70,
    figsize: tuple[float, float] | None = None,
    lower_is_better: bool = True,
    max_table_cols: int = 4,
) -> Path:
    """Save a multi-panel island-search visualization to *path*.

    Args:
        runs: All completed RunRecords (island_id/idea in metadata).
        num_islands: Number of islands in the search.
        path: Output file path (.png, .pdf, .svg).
        title: Figure title.
        top_k: Number of top ideas to show per island.
        idea_max_chars: Max characters for truncated idea text.
        figsize: Figure size in inches. Auto-sized if ``None``.
        lower_is_better: If True, lower scores are better.
        max_table_cols: Max island tables per row (wraps to extra rows).

    Returns:
        The resolved output path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    islands = _build_island_populations(runs, num_islands, lower_is_better)
    colors = _island_colors(num_islands)
    run_map: dict[str, RunRecord] = {r.run_id: r for r in runs}

    # Table grid dimensions
    table_cols = min(num_islands, max_table_cols)
    table_rows = math.ceil(num_islands / table_cols)

    if figsize is None:
        w = max(16, table_cols * 4)
        h = 5.5 + (0.45 * top_k + 1.5) * table_rows
        figsize = (w, h)

    table_height_ratio = 1.0 * table_rows
    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.0, table_height_ratio],
        width_ratios=[1.2, 1.0],
        hspace=0.25, wspace=0.18,
        left=0.03, right=0.97, top=0.92, bottom=0.02,
    )

    ax_dist = fig.add_subplot(gs[0, 0])
    ax_traj = fig.add_subplot(gs[0, 1])
    gs_tables = gs[1, :].subgridspec(table_rows, table_cols, wspace=0.08, hspace=0.3)

    # --- Panel 1: Fitness distributions ---

    all_scores = [score for pop in islands for _, score in pop]
    island_data = [
        np.array([score for _, score in pop]) if pop else np.array([])
        for pop in islands
    ]

    # Y-axis: tight min/max of in-fence scores so the grid matches the data.
    # We still compute an IQR fence to decide what counts as an outlier
    # (rendered as ▼ markers, excluded from violin KDE) but the axis limits
    # themselves come from min/max of non-outlier values.
    if all_scores:
        q1, q3 = np.percentile(all_scores, [25, 75])
        iqr = q3 - q1
        fence_lo = q1 - 1.5 * iqr
        fence_hi = q3 + 1.5 * iqr
        in_fence = [s for s in all_scores if fence_lo <= s <= fence_hi]
        if in_fence:
            y_lo = min(in_fence)
            y_hi = max(in_fence)
        else:
            y_lo = min(all_scores)
            y_hi = max(all_scores)
        margin = (y_hi - y_lo) * 0.01 or 0.002
        y_lo -= margin
        y_hi += margin
    else:
        y_lo, y_hi = 0, 1

    # Violin kernel density: use only fence-clipped values so outliers don't
    # distort the visible shape. Outliers remain visible as ▼ markers.
    island_data_for_violin = [
        d[(d >= y_lo) & (d <= y_hi)] for d in island_data
    ]
    non_empty = [(i, d) for i, d in enumerate(island_data_for_violin) if len(d) >= 2]
    if non_empty:
        vp = ax_dist.violinplot(
            [d for _, d in non_empty],
            positions=[i for i, _ in non_empty],
            showmedians=False, showextrema=False, widths=0.6,
        )
        for i, body in enumerate(vp["bodies"]):
            idx = non_empty[i][0]
            body.set_facecolor(colors[idx])
            body.set_alpha(0.25)
            body.set_edgecolor(colors[idx])

    n_outliers = 0
    for idx, scores in enumerate(island_data):
        if len(scores) == 0:
            continue
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(scores))
        # Dim outliers that fall outside the fence
        in_range = (scores >= y_lo) & (scores <= y_hi)
        if in_range.any():
            ax_dist.scatter(idx + jitter[in_range], scores[in_range],
                            color=colors[idx], s=25, alpha=0.7,
                            zorder=3, edgecolors="white", linewidths=0.5)
        if (~in_range).any():
            n_outliers += int((~in_range).sum())
            # Place outlier markers at the axis edge so they're visible
            edge_y = y_hi if lower_is_better else y_lo
            ax_dist.scatter(idx + jitter[~in_range],
                            [edge_y] * int((~in_range).sum()),
                            color=colors[idx], s=30, alpha=0.5, marker="v",
                            zorder=3, edgecolors="white", linewidths=0.5)
        best_val = scores.min() if lower_is_better else scores.max()
        ax_dist.scatter([idx], [best_val], color=colors[idx], s=120, marker="*",
                        zorder=4, edgecolors="white", linewidths=0.8)
        med = float(np.median(scores))
        ax_dist.hlines(med, idx - 0.2, idx + 0.2, colors=colors[idx], linewidths=2, zorder=4)

    ax_dist.set_ylim(y_lo, y_hi)
    ax_dist.set_xticks(range(num_islands))
    ax_dist.set_xticklabels([f"Island {i}" for i in range(num_islands)],
                            fontsize=12, fontweight="bold")
    for label, color in zip(ax_dist.get_xticklabels(), colors):
        label.set_color(color)
    ax_dist.set_ylabel("Score", fontsize=12)
    if lower_is_better:
        ax_dist.invert_yaxis()
    ax_dist.grid(True, axis="y", alpha=0.2)
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)

    total_sols = sum(len(p) for p in islands)
    global_best = (min(all_scores) if lower_is_better else max(all_scores)) if all_scores else None
    parts = [f"{total_sols} solutions"]
    if global_best is not None:
        parts.append(f"best: {global_best:.4f}")
    if n_outliers:
        parts.append(f"{n_outliers} outliers clipped")
    ax_dist.set_title("Fitness Distributions", fontsize=13, fontweight="bold", pad=18)
    # ax_dist.text(0.5, 1.005, "  |  ".join(parts), ha="center", va="bottom",
    #              fontsize=10, color="#666666", transform=ax_dist.transAxes)

    # --- Panel 2: Trajectories ---

    eval_steps, best_so_far, pop_best, pop_worst = _build_fitness_timeline(
        runs, num_islands, lower_is_better,
    )

    for idx in range(num_islands):
        if not eval_steps[idx]:
            continue
        ax_traj.plot(eval_steps[idx], best_so_far[idx], color=colors[idx], lw=2.5,
                     label=f"Island {idx}", marker="o", markersize=4, alpha=0.9)

    mig_steps = [
        i + 1 for i, r in enumerate(runs)
        if r.metadata.get("migrated")
    ]
    if mig_steps:
        ax_traj.axvline(mig_steps[0], color="#888888", ls="--", lw=1.4, alpha=0.7,
                        zorder=0, label="Migration")
        for step in mig_steps[1:]:
            ax_traj.axvline(step, color="#888888", ls="--", lw=1.4, alpha=0.7, zorder=0)

    if global_best is not None:
        ax_traj.axhline(global_best, color="#333333", ls=":", lw=1.0, alpha=0.4)

    # Y-axis: tight to best_so_far range. Keep the full low end (we want to see
    # the best scores) but clip the high end via IQR fence so a single
    # unlucky initial score on one island doesn't inflate the panel with
    # empty space.
    traj_values = [v for series in best_so_far for v in series]
    if global_best is not None:
        traj_values.append(global_best)
    if traj_values:
        arr = np.asarray(traj_values, dtype=float)
        t_lo = float(arr.min())
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        fence_hi = q3 + 1.5 * iqr if iqr > 0 else float(arr.max())
        t_hi = float(min(arr.max(), fence_hi))
        if t_hi <= t_lo:
            t_hi = float(arr.max())
        # ~3% padding so lines don't butt against axes (distributions stay tight)
        tpad = (t_hi - t_lo) * 0.03 or 0.005
        ax_traj.set_ylim(t_lo - tpad, t_hi + tpad)
    else:
        ax_traj.set_ylim(y_lo, y_hi)
    all_steps = [s for island_steps in eval_steps for s in island_steps]
    if all_steps:
        x_lo, x_hi = min(all_steps), max(all_steps)
        xpad = max(1, (x_hi - x_lo) * 0.02)
        ax_traj.set_xlim(x_lo - xpad, x_hi + xpad)
    ax_traj.set_xlabel("Evaluation #", fontsize=12)
    ax_traj.set_ylabel("Best Score" + (" (lower=better)" if lower_is_better else ""), fontsize=12)
    ax_traj.set_title("Fitness Trajectories", fontsize=13, fontweight="bold")
    ax_traj.legend(fontsize=10, loc="lower right" if lower_is_better else "upper right",
                   framealpha=0.9)
    if lower_is_better:
        ax_traj.invert_yaxis()
    ax_traj.grid(True, alpha=0.2)
    ax_traj.spines["top"].set_visible(False)
    ax_traj.spines["right"].set_visible(False)

    # --- Panel 3: Top-k tables (max_table_cols per row) ---

    trunc_width = max(30, idea_max_chars // max(1, table_cols - 1))

    for idx in range(num_islands):
        row = idx // table_cols
        col = idx % table_cols
        ax_tbl = fig.add_subplot(gs_tables[row, col])
        ax_tbl.axis("off")

        pop = islands[idx]
        best_score = pop[0][1] if pop else None
        header = f"Island {idx}  —  best: {best_score:.4f}" if best_score is not None else f"Island {idx}"
        ax_tbl.set_title(header, fontsize=11, fontweight="bold", color=colors[idx], loc="left", pad=6)

        if not pop:
            ax_tbl.text(0.5, 0.5, "empty", fontsize=8, color="#999999",
                        ha="center", va="center", transform=ax_tbl.transAxes)
            continue

        cell_text: list[list[str]] = []
        cell_colors: list[list[str]] = []
        for rank, (run_id, score) in enumerate(pop[:top_k]):
            run = run_map.get(run_id)
            idea_raw = "—"
            if run:
                idea_raw = _extract_idea_summary(
                    run.metadata.get("idea", "—"), trunc_width * 3,
                )
            lines = textwrap.wrap(idea_raw, width=trunc_width)
            wrapped = "\n".join(lines[:3])
            if len(lines) > 3:
                wrapped = wrapped.rstrip("…") + "…"
            cell_text.append([f"{score:.4f}", wrapped])
            bg = "#e8f5e9" if rank == 0 else ("#fafafa" if rank % 2 == 0 else "white")
            cell_colors.append([bg, bg])

        table = ax_tbl.table(
            cellText=cell_text, colLabels=["Score", "Idea"],
            colWidths=[0.15, 0.85], cellLoc="left",
            bbox=[0.0, 0.0, 1.0, 1.0], cellColours=cell_colors,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        table.scale(1.0, 1.8)

        for col_idx in range(2):
            cell = table[0, col_idx]
            cell.set_facecolor(colors[idx])
            cell.set_text_props(color="white", fontweight="bold", fontsize=9.5)
            cell.set_edgecolor("white")

    # Hide any leftover grid cells
    remainder = num_islands % table_cols
    if remainder:
        for col in range(remainder, table_cols):
            ax_empty = fig.add_subplot(gs_tables[table_rows - 1, col])
            ax_empty.axis("off")

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.0)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
