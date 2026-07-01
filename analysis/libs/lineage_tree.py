"""Lineage / hereditary tree plot for QD search runs.

Builds a parent→child DAG from the ``runs`` table (``parent_ids`` column)
and lays nodes out by ``generation``, colored by ``score``. For
``IslandSearch`` runs, splits the figure into one panel per island.

Usage (CLI):
    python -m libs.lineage_tree <experiment_id> [--out path.png]

Usage (library):
    from libs.lineage_tree import build_tree, plot_tree
    G = build_tree(experiment_id)
    plot_tree(G, "tree.png", lower_is_better=True)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

RESEARCH_AGENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = RESEARCH_AGENT_ROOT / "runs" / "nanogpt" / "store.db"

EDGE_STYLE = {
    # (color, linestyle, linewidth, alpha, connectionstyle)
    "mutation":  ("#6B7280", "-",  1.0, 0.55, "arc3,rad=0.0"),    # gray-500 straight
    "crossover": ("#F97316", "-",  1.4, 0.80, "arc3,rad=0.18"),   # orange-500 curved
    "migration": ("#A855F7", "--", 1.4, 0.85, "arc3,rad=0.25"),   # purple-500 dashed-curved
}

# Highlight colors for special nodes
ROOT_RING_COLOR  = "#1F2937"   # near-black ring around founder nodes
BEST_RING_COLOR  = "#FACC15"   # yellow-400 ring around the best run per panel


# ---------- Tree construction ------------------------------------------------

def _split_parents(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def _classify_edge(parent_meta: dict, child_meta: dict) -> str:
    """Pick edge kind from metadata. ``crossover`` and ``migration`` are
    asserted explicitly via the strategy's ``operator``/``migrated`` keys —
    we don't infer crossover from parent count, because Linear records the
    full top-K context as ``parent_ids`` even though no recombination occurs.
    """
    op = (child_meta.get("operator") or "").lower()
    if "crossover" in op:
        return "crossover"
    p_isl = parent_meta.get("island_id")
    c_isl = child_meta.get("island_id")
    if p_isl is not None and c_isl is not None and p_isl != c_isl:
        return "migration"
    if child_meta.get("migrated"):
        return "migration"
    return "mutation"


def build_tree(
    experiment_id: str,
    *,
    db_path: Path = DEFAULT_DB,
    run_type: str = "executor",
) -> nx.DiGraph:
    """Build a parent→child DAG for one experiment.

    Each node carries ``score``, ``valid``, ``generation``, ``iteration``,
    ``island_id`` (when present), and the parsed ``metadata`` dict.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = list(conn.execute(
        "SELECT run_id, iteration, score, valid, parent_ids, generation, metadata "
        "FROM runs WHERE experiment_id = ? AND run_type = ? "
        "ORDER BY iteration",
        (experiment_id, run_type),
    ))
    conn.close()

    G = nx.DiGraph()
    nodes_by_iter: dict[str, dict] = {}
    for r in rows:
        meta = json.loads(r["metadata"]) if r["metadata"] else {}
        attrs = {
            "run_id": r["run_id"],
            "iteration": r["iteration"],
            "score": r["score"],
            "valid": bool(r["valid"]) if r["valid"] is not None else False,
            "generation": r["generation"] or 0,
            "island_id": meta.get("island_id"),
            "operator": meta.get("operator"),
            "archive_status": meta.get("archive_status"),
            "cell_key": meta.get("cell_key"),
            "migrated": meta.get("migrated"),
            "parent_ids_raw": r["parent_ids"],
        }
        # Use (run_id, iteration) as node key so re-used run_ids across
        # threads stay distinct (executor workspaces are per-thread).
        key = f"{r['run_id']}@{r['iteration']}"
        attrs["key"] = key
        G.add_node(key, **attrs)
        nodes_by_iter[key] = attrs

    # Build edges: resolve parent_ids by latest preceding occurrence of run_id.
    # For non-crossover children with multiple parents (Linear's top-K context),
    # only the first listed parent is drawn — the rest are context, not lineage.
    last_seen: dict[str, str] = {}
    for r in rows:
        key = f"{r['run_id']}@{r['iteration']}"
        parents = _split_parents(r["parent_ids"])
        child_meta = G.nodes[key]
        is_crossover = "crossover" in (child_meta.get("operator") or "").lower()
        if not is_crossover:
            parents = parents[:1]
        for pid in parents:
            pkey = last_seen.get(pid)
            if pkey is None or pkey not in G:
                continue
            parent_meta = G.nodes[pkey]
            kind = _classify_edge(parent_meta, child_meta)
            G.add_edge(pkey, key, kind=kind)
        last_seen[r["run_id"]] = key

    return G


# ---------- Layout ----------------------------------------------------------

def _layout_by_generation(nodes: list[str], G: nx.DiGraph) -> dict[str, tuple[float, float]]:
    """Lay nodes out with generation on the y-axis (root at top), spread on x.

    Within each generation, nodes are ordered by iteration so the time
    axis runs left → right.
    """
    by_gen: dict[int, list[str]] = defaultdict(list)
    for n in nodes:
        by_gen[G.nodes[n]["generation"]].append(n)

    pos: dict[str, tuple[float, float]] = {}
    for gen, ns in by_gen.items():
        ns.sort(key=lambda n: G.nodes[n]["iteration"])
        # Center horizontally; spacing = 1.0 between siblings.
        offset = -(len(ns) - 1) / 2.0
        for i, n in enumerate(ns):
            pos[n] = (offset + i, -gen)
    return pos


# ---------- Plotting --------------------------------------------------------

def _draw_panel(
    ax,
    G: nx.DiGraph,
    nodes: list[str],
    *,
    lower_is_better: bool,
    title: str,
    score_range: tuple[float, float] | None,
) -> None:
    if not nodes:
        ax.set_title(f"{title} (empty)")
        ax.set_xticks([]); ax.set_yticks([])
        return

    sub = G.subgraph(nodes)
    pos = _layout_by_generation(nodes, G)

    # --- edges
    for u, v, d in sub.edges(data=True):
        if u not in pos or v not in pos:
            continue
        color, ls, lw, alpha, conn = EDGE_STYLE.get(
            d.get("kind", "mutation"), EDGE_STYLE["mutation"])
        ax.annotate(
            "", xy=pos[v], xytext=pos[u],
            arrowprops=dict(arrowstyle="-", lw=lw, color=color,
                            linestyle=ls, alpha=alpha,
                            connectionstyle=conn),
            zorder=1,
        )

    # --- nodes
    scored = [n for n in nodes if G.nodes[n]["score"] is not None]
    failed = [n for n in nodes if G.nodes[n]["score"] is None]

    if scored:
        s_vals = [G.nodes[n]["score"] for n in scored]
        if score_range is not None:
            vmin, vmax = score_range
        else:
            vmin, vmax = min(s_vals), max(s_vals)
        # Lower is better → invert (so good = green, bad = red).
        cmap = plt.cm.RdYlGn_r if lower_is_better else plt.cm.RdYlGn
        norm = Normalize(vmin=vmin, vmax=vmax)
        ax.scatter(
            [pos[n][0] for n in scored],
            [pos[n][1] for n in scored],
            c=[cmap(norm(s)) for s in s_vals],
            s=70, edgecolor="black", linewidth=0.5, zorder=3,
        )

    if failed:
        ax.scatter(
            [pos[n][0] for n in failed],
            [pos[n][1] for n in failed],
            facecolors="white", edgecolors="#9CA3AF",
            s=40, linewidth=0.6, zorder=2,
        )

    # --- label top-3 by score
    if scored:
        ranked = sorted(scored, key=lambda n: G.nodes[n]["score"],
                        reverse=not lower_is_better)
        for n in ranked[:3]:
            s = G.nodes[n]["score"]
            ax.annotate(
                f"{G.nodes[n]['run_id']}\n{s:.4f}",
                pos[n], textcoords="offset points",
                xytext=(0, 12), ha="center", va="bottom",
                fontsize=7, color="#1F2937",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.85),
            )

        best = G.nodes[ranked[0]]["score"]
        title = f"{title}  (best={best:.4f}, n={len(scored)})"

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#D1D5DB")
    ax.set_ylabel("← generation")


def _hierarchy_layout(
    G: nx.DiGraph,
    root: str,
    *,
    nodes_subset: set[str] | None = None,
    x_gap: float = 1.8,
    y_gap: float = 1.6,
    return_depths: bool = False,
) -> dict[str, tuple[float, float]] | tuple[dict[str, tuple[float, float]], dict[str, int]]:
    """Top-down tree layout: root at y=0, descendants below, siblings spread
    horizontally in proportion to subtree leaf count. Multi-parent children
    are placed under their first-seen parent.
    """
    in_subset = (lambda n: n in nodes_subset) if nodes_subset else (lambda n: True)

    children: dict[str, list[str]] = {}
    seen: set[str] = set()

    def collect(n: str) -> None:
        if n in seen or not in_subset(n):
            return
        seen.add(n)
        kids = [c for c in G.successors(n) if in_subset(c) and c not in seen]
        kids.sort(key=lambda c: G.nodes[c]["iteration"])
        children[n] = kids
        for c in kids:
            collect(c)

    collect(root)

    leaves: dict[str, int] = {}

    def count_leaves(n: str) -> int:
        kids = children.get(n, [])
        if not kids:
            leaves[n] = 1
            return 1
        total = sum(count_leaves(c) for c in kids)
        leaves[n] = total
        return total

    count_leaves(root)

    pos: dict[str, tuple[float, float]] = {}
    depths: dict[str, int] = {}

    def place(n: str, x_left: float, depth: int) -> None:
        width = leaves[n] * x_gap
        pos[n] = (x_left + width / 2.0, -depth * y_gap)
        depths[n] = depth
        cursor = x_left
        for c in children.get(n, []):
            cw = leaves[c] * x_gap
            place(c, cursor, depth + 1)
            cursor += cw

    place(root, 0.0, 0)
    if return_depths:
        return pos, depths
    return pos


def _draw_subtree_panel(
    ax,
    G: nx.DiGraph,
    root: str,
    *,
    lower_is_better: bool,
    title: str,
    score_range: tuple[float, float] | None,
    cmap=None,
    norm=None,
    prune_failed_leaves: bool = False,
) -> None:
    """Render a single descendant subtree on ``ax``.

    Conventions: root has a dark ring, best-scoring node has a gold ring,
    failed runs are small open circles, edges use ``EDGE_STYLE``.

    ``prune_failed_leaves=True`` drops dead-end failed runs — those carry no
    lineage info (they have no children) and, when there are thousands of
    them (e.g. OMNI-EPIC), they dominate the figure visually.
    """
    same_island_only = G.nodes[root]["island_id"]
    if same_island_only is not None:
        subset = {
            n for n in nx.descendants(G, root) | {root}
            if G.nodes[n]["island_id"] == same_island_only
        }
    else:
        subset = nx.descendants(G, root) | {root}

    if prune_failed_leaves:
        subset = {
            n for n in subset
            if (G.nodes[n]["score"] is not None) or (G.out_degree(n) > 0)
            or n == root
        }

    pos, depths = _hierarchy_layout(G, root, nodes_subset=subset, return_depths=True)
    if not pos:
        ax.set_title(f"{title} (empty)")
        return

    sub = G.subgraph(pos.keys())

    # --- edges (drawn first, behind nodes)
    for u, v, d in sub.edges(data=True):
        if u not in pos or v not in pos:
            continue
        color, ls, lw, alpha, conn = EDGE_STYLE.get(
            d.get("kind", "mutation"), EDGE_STYLE["mutation"])
        ax.annotate(
            "", xy=pos[v], xytext=pos[u],
            arrowprops=dict(
                arrowstyle="-", lw=lw, color=color,
                linestyle=ls, alpha=alpha,
                connectionstyle=conn,
            ),
            zorder=1,
        )

    # --- nodes
    scored = [n for n in pos if G.nodes[n]["score"] is not None]
    failed = [n for n in pos if G.nodes[n]["score"] is None]

    if cmap is None:
        cmap = plt.cm.RdYlGn_r if lower_is_better else plt.cm.RdYlGn
    if norm is None:
        if scored:
            s_vals = [G.nodes[n]["score"] for n in scored]
            if score_range is None:
                score_range = (min(s_vals), max(s_vals))
            norm = Normalize(*score_range)

    if scored and norm is not None:
        s_vals = [G.nodes[n]["score"] for n in scored]
        ax.scatter(
            [pos[n][0] for n in scored],
            [pos[n][1] for n in scored],
            c=[cmap(norm(s)) for s in s_vals],
            s=420, edgecolor="#1F2937", linewidth=0.9, zorder=3,
        )
    if failed:
        ax.scatter(
            [pos[n][0] for n in failed],
            [pos[n][1] for n in failed],
            facecolors="#F3F4F6", edgecolors="#D1D5DB",
            s=140, linewidth=0.7, zorder=2,
        )

    # --- highlight root and best
    rx, ry = pos[root]
    ax.scatter([rx], [ry], facecolors="none", edgecolors=ROOT_RING_COLOR,
               s=820, linewidth=2.0, zorder=4)

    best_node = None
    if scored:
        best_node = min(scored, key=lambda n: G.nodes[n]["score"]) if lower_is_better \
                    else max(scored, key=lambda n: G.nodes[n]["score"])
        bx, by = pos[best_node]
        ax.scatter([bx], [by], facecolors="none", edgecolors=BEST_RING_COLOR,
                   s=820, linewidth=2.4, zorder=5)

    # --- labels: root, best, and top-3 by score (deduped)
    label_set: list[str] = []
    seen: set[str] = set()
    if best_node is not None and best_node not in seen:
        label_set.append(best_node); seen.add(best_node)
    if root not in seen:
        label_set.append(root); seen.add(root)
    if scored:
        ranked = sorted(scored, key=lambda n: G.nodes[n]["score"],
                        reverse=not lower_is_better)
        for n in ranked:
            if n in seen:
                continue
            label_set.append(n); seen.add(n)
            if len(label_set) >= 4:
                break

    for n in label_set:
        s = G.nodes[n]["score"]
        rid = G.nodes[n]["run_id"]
        if s is None:
            txt = f"{rid}\n(root, no score)" if n == root else f"{rid}\n(failed)"
        elif n == root and n == best_node:
            txt = f"{rid}\n{s:.4f}  (root, best)"
        elif n == root:
            txt = f"{rid}\n{s:.4f}  (root)"
        elif n == best_node:
            txt = f"{rid}\n{s:.4f}  (best)"
        else:
            txt = f"{rid}\n{s:.4f}"
        ax.annotate(
            txt, pos[n], textcoords="offset points",
            xytext=(0, 18), ha="center", va="bottom",
            fontsize=8.5, color="#1F2937",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#E5E7EB", linewidth=0.5, alpha=0.92),
            zorder=6,
        )

    # --- depth gridlines / y-tick labels (layout depth = primary-parent
    # chain length, NOT the strategy's stored ``generation`` which uses
    # max-parent-gen + 1 across all top-K parents).
    depth_to_y = {d: pos[n][1] for n, d in depths.items()}
    unique_depths = sorted(depth_to_y)
    if unique_depths:
        for d in unique_depths:
            ax.axhline(depth_to_y[d], color="#F3F4F6", lw=0.8, zorder=0)
        # avoid label crowding for very deep trees
        max_labels = 25
        if len(unique_depths) > max_labels:
            stride = (len(unique_depths) + max_labels - 1) // max_labels
            label_depths = unique_depths[::stride]
            if label_depths[-1] != unique_depths[-1]:
                label_depths.append(unique_depths[-1])
        else:
            label_depths = unique_depths
        ax.set_yticks([depth_to_y[d] for d in label_depths])
        ax.set_yticklabels([f"depth {d}" for d in label_depths],
                           fontsize=8, color="#6B7280")

    # --- frame: clean axes, hide ticks marks but keep tick labels
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0, pad=2)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#E5E7EB")
    ax.spines["left"].set_linewidth(0.8)

    ax.set_title(title, fontsize=11, color="#1F2937", pad=8, loc="left")


def _summary_subtitle(G: nx.DiGraph, root: str, ndesc: int,
                      *, pruned: bool = False) -> str:
    """One-line stats blurb for a panel title."""
    same_isl = G.nodes[root]["island_id"]
    if same_isl is not None:
        nodes = [n for n in nx.descendants(G, root) | {root}
                 if G.nodes[n]["island_id"] == same_isl]
    else:
        nodes = list(nx.descendants(G, root) | {root})
    scored = [n for n in nodes if G.nodes[n]["score"] is not None]
    n_total = len(nodes)
    n_scored = len(scored)
    n_failed = n_total - n_scored
    base = f"{n_total} runs  •  {n_scored} valid  •  {n_failed} failed"
    if pruned and n_failed > 0:
        base += "  (failed leaves hidden)"
    return base


def plot_descendant_trees(
    G: nx.DiGraph,
    out_path: str | Path,
    *,
    lower_is_better: bool = True,
    title: str | None = None,
    min_descendants: int = 5,
    prune_failed_leaves: bool | None = None,
) -> Path:
    """For each island (or the whole graph if islands aren't used), pick the
    root with the most descendants and render its subtree as a top-down
    genealogy. Useful for showing how one founder idea branched and
    drifted across generations.
    """
    nodes = list(G.nodes())
    islands = sorted({
        G.nodes[n]["island_id"] for n in nodes
        if G.nodes[n]["island_id"] is not None
    })

    # Auto-prune failed leaves when they dominate the visual (>5x scored
    # nodes). Failed leaves carry no lineage info — dropping them is safe.
    if prune_failed_leaves is None:
        n_failed_leaves = sum(
            1 for n in nodes
            if G.nodes[n]["score"] is None and G.out_degree(n) == 0
        )
        n_scored = sum(1 for n in nodes if G.nodes[n]["score"] is not None)
        prune_failed_leaves = (n_scored > 0 and n_failed_leaves > 5 * n_scored)

    # Shared color scale across all panels — clip to 5th-95th percentile so a
    # single diverged run doesn't crush the colormap.
    s_vals = [G.nodes[n]["score"] for n in nodes if G.nodes[n]["score"] is not None]
    if s_vals:
        s_sorted = sorted(s_vals)
        lo = s_sorted[max(0, int(0.05 * (len(s_sorted) - 1)))]
        hi = s_sorted[min(len(s_sorted) - 1, int(0.95 * (len(s_sorted) - 1)))]
        score_range = (lo, hi) if lo < hi else (min(s_vals), max(s_vals))
    else:
        score_range = None
    cmap = plt.cm.RdYlGn_r if lower_is_better else plt.cm.RdYlGn
    norm = Normalize(*score_range) if score_range else None

    def _score_root(n: str, isl_filter: int | None) -> tuple[int, int]:
        """Rank candidate roots by (max depth, descendant count) — deeper
        subtrees beat wider-but-shallow ones, since depth is what makes a
        lineage tree visually interesting."""
        if isl_filter is not None:
            desc_set = {d for d in nx.descendants(G, n)
                        if G.nodes[d]["island_id"] == isl_filter}
        else:
            desc_set = nx.descendants(G, n)
        if not desc_set:
            return (0, 0)
        root_gen = G.nodes[n]["generation"]
        max_depth = max(G.nodes[d]["generation"] for d in desc_set) - root_gen
        return (max_depth, len(desc_set))

    if islands:
        roots: list[tuple[str, int, str]] = []
        for isl in islands:
            isl_nodes = [n for n in nodes if G.nodes[n]["island_id"] == isl]
            cand = []
            for n in isl_nodes:
                preds_in_isl = [
                    p for p in G.predecessors(n)
                    if G.nodes[p]["island_id"] == isl
                ]
                if preds_in_isl:
                    continue
                rank = _score_root(n, isl)
                cand.append((n, rank))
            cand.sort(key=lambda x: (-x[1][0], -x[1][1]))
            if cand and cand[0][1][1] >= min_descendants:
                roots.append((cand[0][0], cand[0][1][1], f"Island {isl}"))
            else:
                roots.append((cand[0][0] if cand else "", 0, f"Island {isl} (empty)"))

        # widest panel sets the figure width — count leaves per dominant tree
        max_leaves = 1
        for root, _, _ in roots:
            if not root:
                continue
            isl = G.nodes[root]["island_id"]
            if isl is not None:
                desc = {n for n in nx.descendants(G, root) | {root}
                        if G.nodes[n]["island_id"] == isl}
            else:
                desc = nx.descendants(G, root) | {root}
            if prune_failed_leaves:
                desc = {n for n in desc
                        if G.nodes[n]["score"] is not None
                        or G.out_degree(n) > 0
                        or n == root}
            leaves = sum(1 for n in desc if all(c not in desc for c in G.successors(n)))
            max_leaves = max(max_leaves, leaves)

        n = len(roots)
        cols = min(n, 2)
        rows = (n + cols - 1) // cols
        # 0.32 inches per leaf — keeps marker overlap below ~30%
        per_panel_w = max(12.5, min(28.0, 0.32 * max_leaves))
        fig, axes = plt.subplots(
            rows, cols, figsize=(per_panel_w * cols, 7.5 * rows),
            squeeze=False,
        )
        for ax, (root, ndesc, label) in zip(axes.flat, roots):
            if not root:
                ax.set_title(label, fontsize=11, loc="left")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values(): s.set_visible(False)
                continue
            head_iter = G.nodes[root]["iteration"]
            head_run = G.nodes[root]["run_id"]
            stats = _summary_subtitle(G, root, ndesc, pruned=prune_failed_leaves)
            panel_title = (
                f"{label}   ·   founder {head_run} (iter {head_iter})\n"
                f"{stats}"
            )
            _draw_subtree_panel(
                ax, G, root,
                lower_is_better=lower_is_better,
                title=panel_title, score_range=score_range,
                cmap=cmap, norm=norm,
                prune_failed_leaves=prune_failed_leaves,
            )
        for ax in axes.flat[len(roots):]:
            ax.set_visible(False)
    else:
        # rank roots by (max-depth, descendant count) — deeper trees win
        candidates = []
        for n in nodes:
            if G.in_degree(n) != 0:
                continue
            rank = _score_root(n, None)
            candidates.append((n, rank))
        candidates.sort(key=lambda x: (-x[1][0], -x[1][1]))
        if not candidates:
            raise SystemExit("no roots found")
        root, (depth, ndesc) = candidates[0]
        # scale figure width with effective leaf count
        desc = nx.descendants(G, root) | {root}
        if prune_failed_leaves:
            desc = {n for n in desc
                    if G.nodes[n]["score"] is not None
                    or G.out_degree(n) > 0
                    or n == root}
        leaves = sum(1 for n in desc if all(c not in desc for c in G.successors(n)))
        fig_w = max(15.0, min(36.0, 0.32 * leaves))
        fig, ax = plt.subplots(figsize=(fig_w, 9))
        stats = _summary_subtitle(G, root, ndesc, pruned=prune_failed_leaves)
        head_iter = G.nodes[root]["iteration"]
        head_run = G.nodes[root]["run_id"]
        _draw_subtree_panel(
            ax, G, root,
            lower_is_better=lower_is_better,
            title=f"founder {head_run} (iter {head_iter})\n{stats}",
            score_range=score_range,
            cmap=cmap, norm=norm,
            prune_failed_leaves=prune_failed_leaves,
        )

    # --- compact figure title and legend block above the panels
    if title:
        fig.suptitle(title, fontsize=16, color="#1F2937", y=1.005,
                     fontweight="normal")

    # --- shared colorbar + legend, both at the bottom
    if norm is not None:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.32, 0.045, 0.36, 0.010])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        direction = "(lower is better)" if lower_is_better else "(higher is better)"
        cbar.set_label(f"val_bpb  {direction}", fontsize=11, color="#1F2937",
                       labelpad=6)
        cbar.ax.tick_params(labelsize=10, colors="#374151")
        cbar.outline.set_edgecolor("#E5E7EB")
        cbar.outline.set_linewidth(0.6)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor=ROOT_RING_COLOR, markersize=14, markeredgewidth=2.2,
               label="founder (root)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor=BEST_RING_COLOR, markersize=14, markeredgewidth=2.6,
               label="best score in panel"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#F3F4F6",
               markeredgecolor="#D1D5DB", markersize=10, markeredgewidth=0.8,
               label="failed run"),
        Line2D([0], [0], color=EDGE_STYLE["mutation"][0],  lw=2.0, ls="-",
               label="mutation"),
        Line2D([0], [0], color=EDGE_STYLE["crossover"][0], lw=2.2, ls="-",
               label="crossover (curved)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=12,
               frameon=False, bbox_to_anchor=(0.5, 0.085))

    fig.subplots_adjust(left=0.05, right=0.98, top=0.94, bottom=0.13,
                         hspace=0.40, wspace=0.08)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def plot_tree(
    G: nx.DiGraph,
    out_path: str | Path,
    *,
    lower_is_better: bool = True,
    title: str | None = None,
    split_by_island: bool | None = None,
) -> Path:
    """Render the lineage tree.

    If the runs carry ``island_id`` and ``split_by_island`` is None or True,
    one panel is drawn per island. Otherwise a single panel is drawn.
    """
    nodes = list(G.nodes())
    islands = sorted({
        G.nodes[n]["island_id"] for n in nodes
        if G.nodes[n]["island_id"] is not None
    })
    do_split = (split_by_island is not False) and len(islands) >= 2

    # Shared score range for cross-panel comparability. Clip to 5th-95th
    # percentile so a single diverged run doesn't crush the colormap.
    s_vals = [G.nodes[n]["score"] for n in nodes if G.nodes[n]["score"] is not None]
    if s_vals:
        s_sorted = sorted(s_vals)
        lo = s_sorted[max(0, int(0.05 * (len(s_sorted) - 1)))]
        hi = s_sorted[min(len(s_sorted) - 1, int(0.95 * (len(s_sorted) - 1)))]
        if lo == hi:
            lo, hi = min(s_vals), max(s_vals)
        score_range = (lo, hi)
    else:
        score_range = None

    if do_split:
        n = len(islands)
        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.5 * rows),
                                  squeeze=False)
        for ax, isl in zip(axes.flat, islands):
            isl_nodes = [k for k in nodes if G.nodes[k]["island_id"] == isl]
            _draw_panel(
                ax, G, isl_nodes,
                lower_is_better=lower_is_better,
                title=f"island {isl}",
                score_range=score_range,
            )
        for ax in axes.flat[len(islands):]:
            ax.set_visible(False)
    else:
        fig, ax = plt.subplots(figsize=(11, 7))
        _draw_panel(
            ax, G, nodes,
            lower_is_better=lower_is_better,
            title=title or "lineage",
            score_range=score_range,
        )

    # --- legend
    direction = "lower=better" if lower_is_better else "higher=better"
    handles = [
        mpatches.Patch(color=plt.cm.RdYlGn_r(0.0) if lower_is_better else plt.cm.RdYlGn(1.0),
                       label=f"best score ({direction})"),
        mpatches.Patch(color=plt.cm.RdYlGn_r(1.0) if lower_is_better else plt.cm.RdYlGn(0.0),
                       label="worst score"),
        mpatches.Patch(facecolor="white", edgecolor="#9CA3AF", label="failed run"),
        Line2D([0], [0], color=EDGE_STYLE["mutation"][0],  lw=1.2, ls="-",  label="mutation"),
        Line2D([0], [0], color=EDGE_STYLE["crossover"][0], lw=1.2, ls=":",  label="crossover"),
        Line2D([0], [0], color=EDGE_STYLE["migration"][0], lw=1.2, ls="--", label="migration"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))

    if title:
        fig.suptitle(title, fontsize=13, y=0.995)

    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


# ---------- CLI -------------------------------------------------------------

def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment_id")
    p.add_argument("--out", default=None,
                   help="Output path (default: analysis/experiments/<exp>/figures/lineage_tree.png)")
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--higher-is-better", action="store_true",
                   help="Default assumes lower-is-better (val_bpb)")
    p.add_argument("--no-split-island", action="store_true",
                   help="Force single-panel even if island_id is present")
    p.add_argument("--mode", choices=("flat", "descendants"), default="flat",
                   help="'flat' = generation-layered DAG; 'descendants' = "
                        "top-down genealogy of the dominant root per island")
    p.add_argument("--prune-failed", choices=("auto", "yes", "no"), default="auto",
                   help="Hide failed runs that have no children. 'auto' "
                        "kicks in when failed >> scored (default).")
    p.add_argument("--title", default=None)
    args = p.parse_args(argv)

    G = build_tree(args.experiment_id, db_path=Path(args.db))
    if G.number_of_nodes() == 0:
        print(f"no executor runs found for {args.experiment_id}", file=sys.stderr)
        return 1

    default_name = "lineage_descendants.png" if args.mode == "descendants" else "lineage_tree.png"
    out = Path(args.out) if args.out else (
        RESEARCH_AGENT_ROOT / "analysis" / "experiments" / args.experiment_id
        / "figures" / default_name
    )
    prune_map = {"auto": None, "yes": True, "no": False}
    if args.mode == "descendants":
        plot_descendant_trees(
            G, out,
            lower_is_better=not args.higher_is_better,
            title=args.title or args.experiment_id,
            prune_failed_leaves=prune_map[args.prune_failed],
        )
    else:
        plot_tree(
            G, out,
            lower_is_better=not args.higher_is_better,
            title=args.title or args.experiment_id,
            split_by_island=False if args.no_split_island else None,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
