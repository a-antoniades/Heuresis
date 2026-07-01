"""Lineage-tree loading and plotting for search runs.

The heuresis store already records enough ancestry to reconstruct a
parent-to-child DAG for executor runs: ``parent_ids``, ``generation``, and
strategy metadata such as ``operator`` and ``island_id``.  This module keeps
the lineage reconstruction separate from Matplotlib so the inference logic is
easy to test.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

GroupBy = Literal["auto", "none", "island", "cell", "archive_status", "operator"]


@dataclass
class LineageNode:
    run_id: str
    score: float | None
    iteration: int | None
    generation: int
    group: str = "search"
    parent_ids: list[str] = field(default_factory=list)
    operator: str | None = None
    valid: bool | None = None
    archive_status: str | None = None
    cell_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    is_ghost: bool = False


@dataclass(frozen=True)
class LineageEdge:
    parent_id: str
    child_id: str
    kind: str


@dataclass
class LineageGraph:
    nodes: dict[str, LineageNode]
    edges: list[LineageEdge]


@dataclass
class LayeredLayout:
    positions: dict[str, tuple[float, float]]
    groups: list[str]
    group_nodes: dict[str, list[str]]


def load_lineage_nodes(
    db_path: str | Path,
    experiment_id: str,
    *,
    run_type: str = "executor",
    scored_only: bool = False,
) -> list[LineageNode]:
    """Load persisted runs as lineage nodes.

    ``scored_only`` mirrors older analysis scripts that plotted only scored
    variants.  Leaving it false is usually better for current runs because
    invalid executor attempts can still be parents or useful failed branches.
    """
    where = ["experiment_id = ?", "run_type = ?"]
    params: list[Any] = [experiment_id, run_type]
    if scored_only:
        where.append("score IS NOT NULL")

    query = (
        "SELECT run_id, iteration, score, valid, parent_ids, generation, metadata "
        "FROM runs WHERE "
        + " AND ".join(where)
        + " ORDER BY generation, iteration, started_at, run_id"
    )

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(query, tuple(params)).fetchall()
    finally:
        conn.close()

    nodes: list[LineageNode] = []
    for row in rows:
        metadata = _parse_json_dict(row["metadata"])
        parent_ids = [p for p in (row["parent_ids"] or "").split(",") if p]
        group = _metadata_group(metadata, group_by="auto")
        nodes.append(
            LineageNode(
                run_id=row["run_id"],
                score=row["score"],
                iteration=row["iteration"],
                generation=row["generation"] or int(metadata.get("generation", 0) or 0),
                group=group,
                parent_ids=parent_ids,
                operator=metadata.get("operator"),
                valid=bool(row["valid"]) if row["valid"] is not None else None,
                archive_status=metadata.get("archive_status"),
                cell_key=metadata.get("cell_key"),
                metadata=metadata,
            )
        )
    return nodes


def build_lineage_graph(nodes: Iterable[LineageNode]) -> LineageGraph:
    """Build a parent-to-child DAG and ghost missing parents.

    Missing parents can happen after filtering or when importing partial
    histories.  Ghost nodes keep the rendered tree connected enough to explain
    ancestry without pretending there is a scored run for that parent.
    """
    graph_nodes = {node.run_id: node for node in nodes}
    edges: list[LineageEdge] = []

    for child in sorted(graph_nodes.values(), key=_node_sort_key):
        for idx, parent_id in enumerate(child.parent_ids):
            if parent_id not in graph_nodes:
                graph_nodes[parent_id] = LineageNode(
                    run_id=parent_id,
                    score=None,
                    iteration=None,
                    generation=max(child.generation - 1, 0),
                    group=child.group,
                    is_ghost=True,
                )
            parent = graph_nodes[parent_id]
            edges.append(
                LineageEdge(
                    parent_id=parent_id,
                    child_id=child.run_id,
                    kind=_infer_edge_kind(
                        parent,
                        child,
                        parent_index=idx,
                        parent_count=len(child.parent_ids),
                    ),
                )
            )

    return LineageGraph(nodes=graph_nodes, edges=edges)


def compute_layered_layout(
    graph: LineageGraph,
    *,
    group_by: GroupBy = "auto",
    x_spacing: float = 1.15,
    y_spacing: float = 1.0,
) -> LayeredLayout:
    """Compute a deterministic generation-layered layout.

    Within each group, y is ``-generation`` and x is creation order inside that
    generation.  This simple layout is more stable than force-directed layouts
    for repeated experiment reports.
    """
    group_nodes: dict[str, list[str]] = {}
    for node_id, node in graph.nodes.items():
        group = group_for_node(node, group_by=group_by, all_nodes=graph.nodes.values())
        group_nodes.setdefault(group, []).append(node_id)

    groups = sorted(group_nodes, key=_natural_key)
    positions: dict[str, tuple[float, float]] = {}
    for group in groups:
        by_generation: dict[int, list[str]] = {}
        for node_id in group_nodes[group]:
            gen = graph.nodes[node_id].generation
            by_generation.setdefault(gen, []).append(node_id)

        for generation, node_ids in by_generation.items():
            ordered = sorted(node_ids, key=lambda n: _node_sort_key(graph.nodes[n]))
            midpoint = (len(ordered) - 1) / 2.0
            for idx, node_id in enumerate(ordered):
                positions[node_id] = ((idx - midpoint) * x_spacing, -generation * y_spacing)

    return LayeredLayout(positions=positions, groups=groups, group_nodes=group_nodes)


def group_for_node(
    node: LineageNode,
    *,
    group_by: GroupBy = "auto",
    all_nodes: Iterable[LineageNode] | None = None,
) -> str:
    if group_by == "none":
        return "search"
    if group_by == "island":
        return _metadata_group(node.metadata, group_by="island")
    if group_by == "cell":
        return node.cell_key or "no_cell"
    if group_by == "archive_status":
        return node.archive_status or "unknown"
    if group_by == "operator":
        return node.operator or "seed"

    all_nodes = list(all_nodes or [node])
    if any(n.metadata.get("island_id") is not None for n in all_nodes):
        return _metadata_group(node.metadata, group_by="island")
    if any(n.group != "search" for n in all_nodes):
        return node.group
    return "search"


def plot_lineage_tree(
    graph: LineageGraph,
    path: str | Path,
    *,
    title: str = "Search Evolution Tree",
    group_by: GroupBy = "auto",
    lower_is_better: bool = True,
    max_labels_per_group: int = 3,
    figsize: tuple[float, float] | None = None,
) -> Path:
    """Render a generation-layered hereditary tree to ``path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    layout = compute_layered_layout(graph, group_by=group_by)
    n_groups = max(len(layout.groups), 1)
    ncols = min(n_groups, 4)
    nrows = math.ceil(n_groups / ncols)
    if figsize is None:
        max_generation = max((node.generation for node in graph.nodes.values()), default=0)
        width = max(7.0 * ncols, 10.0)
        height = max(4.5 * nrows, 3.0 + 0.45 * max_generation) * nrows
        figsize = (width, height)

    fig, axes_grid = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = [ax for row in axes_grid for ax in row]
    scores = [node.score for node in graph.nodes.values() if node.score is not None]
    if scores:
        norm = Normalize(vmin=min(scores), vmax=max(scores))
    else:
        norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.cm.RdYlGn_r if lower_is_better else plt.cm.RdYlGn

    for ax, group in zip(axes, layout.groups):
        node_ids = layout.group_nodes[group]
        node_set = set(node_ids)
        _draw_edges(ax, graph, layout.positions, node_set)
        _draw_nodes(ax, graph, layout.positions, node_ids, norm, cmap, lower_is_better)
        _label_best_nodes(
            ax,
            graph,
            layout.positions,
            node_ids,
            lower_is_better=lower_is_better,
            max_labels=max_labels_per_group,
        )

        scored = [graph.nodes[n].score for n in node_ids if graph.nodes[n].score is not None]
        best = None
        if scored:
            best = min(scored) if lower_is_better else max(scored)
        suffix = f"best={best:.4g}" if best is not None else "no scored runs"
        ax.set_title(
            f"{_format_group_label(group, group_by)} ({suffix})",
            fontweight="bold",
            pad=16,
        )
        ax.set_xlabel("creation order within generation")
        ax.set_ylabel("generation")
        y_values = [layout.positions[n][1] for n in node_ids]
        x_values = [layout.positions[n][0] for n in node_ids]
        ax.set_ylim(min(y_values) - 0.25, max(y_values) + 0.35)
        ax.set_xlim(min(x_values) - 0.75, max(x_values) + 0.75)
        ax.set_yticks(sorted(set(y_values), reverse=True))
        ax.set_yticklabels([
            str(int(abs(y))) for y in sorted(set(y_values), reverse=True)
        ])
        ax.grid(axis="y", alpha=0.2)
        ax.set_xticks([])
        for spine in ("top", "right", "bottom"):
            ax.spines[spine].set_visible(False)

    for ax in axes[len(layout.groups):]:
        ax.axis("off")

    handles = [
        Line2D([0], [0], color="#8a8a8a", lw=1.0, label="mutation / parent"),
        Line2D([0], [0], color="#e08a00", lw=1.2, linestyle=":", label="crossover"),
        Line2D([0], [0], color="#7b3fb3", lw=1.2, linestyle="--", label="migration"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor="white",
               lw=0, label="unscored / ghost"),
        mpatches.Patch(color=cmap(norm(min(scores))) if scores else cmap(0.0),
                       label="best score color"),
        mpatches.Patch(color=cmap(norm(max(scores))) if scores else cmap(1.0),
                       label="worst score color"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9, frameon=False)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _draw_edges(ax, graph: LineageGraph, positions: dict[str, tuple[float, float]], node_set: set[str]) -> None:
    styles = {
        "mutation": ("#8a8a8a", "-"),
        "parent": ("#8a8a8a", "-"),
        "crossover_a": ("#e08a00", ":"),
        "crossover_b": ("#e08a00", ":"),
        "migration": ("#7b3fb3", "--"),
    }
    for edge in graph.edges:
        if edge.parent_id not in node_set and edge.child_id in node_set and edge.kind == "migration":
            child_x, child_y = positions[edge.child_id]
            parent = graph.nodes.get(edge.parent_id)
            parent_y = -float(parent.generation) if parent is not None else child_y + 0.5
            ax.annotate(
                "",
                xy=(child_x, child_y),
                xytext=(child_x - 0.65, parent_y),
                arrowprops={
                    "arrowstyle": "-",
                    "lw": 0.85,
                    "color": "#7b3fb3",
                    "linestyle": "--",
                    "alpha": 0.75,
                },
                zorder=1,
            )
            continue
        if edge.parent_id not in node_set or edge.child_id not in node_set:
            continue
        if edge.parent_id not in positions or edge.child_id not in positions:
            continue
        color, linestyle = styles.get(edge.kind, ("#8a8a8a", "-"))
        ax.annotate(
            "",
            xy=positions[edge.child_id],
            xytext=positions[edge.parent_id],
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.65,
                "color": color,
                "linestyle": linestyle,
                "alpha": 0.65,
            },
            zorder=1,
        )


def _draw_nodes(
    ax,
    graph: LineageGraph,
    positions: dict[str, tuple[float, float]],
    node_ids: list[str],
    norm,
    cmap,
    lower_is_better: bool,
) -> None:
    scored = [n for n in node_ids if graph.nodes[n].score is not None and not graph.nodes[n].is_ghost]
    unscored = [n for n in node_ids if n not in scored]

    if scored:
        xs = [positions[n][0] for n in scored]
        ys = [positions[n][1] for n in scored]
        colors = [cmap(norm(graph.nodes[n].score)) for n in scored]
        sizes = [
            45.0 + 115.0 * _quality_fraction(graph.nodes[n].score, norm, lower_is_better)
            for n in scored
        ]
        ax.scatter(xs, ys, c=colors, s=sizes, edgecolor="black", linewidth=0.5, zorder=3)

    if unscored:
        ax.scatter(
            [positions[n][0] for n in unscored],
            [positions[n][1] for n in unscored],
            facecolors="white",
            edgecolors="#777777",
            s=55,
            linewidth=0.6,
            zorder=2,
        )


def _label_best_nodes(
    ax,
    graph: LineageGraph,
    positions: dict[str, tuple[float, float]],
    node_ids: list[str],
    *,
    lower_is_better: bool,
    max_labels: int,
) -> None:
    scored = [graph.nodes[n] for n in node_ids if graph.nodes[n].score is not None]
    scored.sort(key=lambda n: n.score if lower_is_better else -n.score)
    top_y = max((positions[n][1] for n in node_ids), default=0.0)
    for node in scored[:max_labels]:
        _, y = positions[node.run_id]
        y_offset = -20 if y == top_y else 6
        ax.annotate(
            f"{node.run_id}\n{node.score:.4g}",
            positions[node.run_id],
            textcoords="offset points",
            xytext=(6, y_offset),
            fontsize=8,
            fontweight="bold",
            va="top" if y_offset < 0 else "bottom",
            zorder=4,
        )


def _infer_edge_kind(
    parent: LineageNode,
    child: LineageNode,
    *,
    parent_index: int,
    parent_count: int,
) -> str:
    if parent.group != child.group and parent.group != "search" and child.group != "search":
        return "migration"

    operator = (child.operator or "").lower()
    if parent_count > 1 or "cross" in operator or "combine" in operator:
        return "crossover_a" if parent_index == 0 else "crossover_b"
    if "migrat" in operator:
        return "migration"
    if "mutat" in operator:
        return "mutation"
    return "parent"


def _metadata_group(metadata: dict[str, Any], *, group_by: Literal["auto", "island"]) -> str:
    island_id = metadata.get("island_id")
    if island_id is not None:
        return str(island_id)
    if group_by == "island":
        return "no_island"
    return "search"


def _parse_json_dict(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _node_sort_key(node: LineageNode) -> tuple[int, int, str]:
    iteration = node.iteration if node.iteration is not None else 10**12
    return (node.generation, iteration, node.run_id)


def _natural_key(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):08d}")
    except ValueError:
        return (1, value)


def _quality_fraction(score: float | None, norm, lower_is_better: bool) -> float:
    if score is None:
        return 0.0
    scaled = float(norm(score))
    if math.isnan(scaled):
        return 0.5
    return 1.0 - scaled if lower_is_better else scaled


def _format_group_label(group: str, group_by: GroupBy) -> str:
    if group_by in {"auto", "island"} and group not in {"search", "no_island"}:
        return f"Island {group}"
    return group.replace("_", " ").title()
