#!/usr/bin/env python3
"""Per-island UMAP of idea embeddings for any IslandSearch run.

Mirrors the embedding / UMAP setup used by the
``heuresis:analyzing-search-runs`` skill
(``maintainer/claude/skills/heuresis:analyzing-search-runs/scripts/figures.py``,
``plot_diversity_map``):

  - Gemini embedding model: ``gemini-embedding-001`` (3072-dim)
  - Batch size: 20
  - UMAP: n_neighbors=min(15, n-1), min_dist=0.2, metric="cosine",
    random_state=42
  - TF-IDF fallback if the Gemini API is unavailable

The only difference vs the skill's figure: grouping/coloring by ``island_id``
pulled from the store's run metadata instead of by strategy.

Outputs (in ``--output-dir``, default ``<run_dir>/figures/``):
  island_embedding_map.png      — combined UMAP + per-island small multiples
                                  + within-island pairwise distance violin
  island_embedding_cache.json   — cached embeddings (keyed by executor_id list)

Usage
-----
  python scripts/island_embedding_map.py --run runs/nanogpt/<run_dir> \\
      [--exp <experiment_id>] [--store-db <path>] \\
      [--output-dir <path>] [--force]

Run from the heuresis/ repo root. If ``--exp`` is omitted, the script
reads the ``experiments`` table of ``--store-db`` and matches the run dir by
its ``dir`` column (falling back to the directory's basename).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
# Candidate store.db locations, in priority order. The first non-empty file
# that contains the requested experiment is used.
CANDIDATE_STORES = [
    REPO_ROOT / "runs/nanogpt/store.db",   # current (post-refactor) default
    REPO_ROOT / "store.db",                # older default
    REPO_ROOT / "runs/_legacy/store.db",   # pre-refactor runs
]


def autodetect_store(run_dir: Path, experiment_id: str | None) -> Path:
    """Pick the first candidate store that contains the run or experiment."""
    run_dir_abs = str(run_dir.resolve())
    basename = run_dir.name
    for path in CANDIDATE_STORES:
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            conn = sqlite3.connect(str(path))
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='experiments'"
            )
            if not cur.fetchone():
                conn.close()
                continue
            cur = conn.execute(
                "SELECT experiment_id FROM experiments WHERE dir=? OR experiment_id=? "
                "OR experiment_id=? LIMIT 1",
                (run_dir_abs, experiment_id or basename, basename),
            )
            row = cur.fetchone()
            conn.close()
            if row:
                return path
        except sqlite3.DatabaseError:
            continue
    raise SystemExit(
        f"Could not locate a store with run {run_dir.name}. "
        f"Tried: {', '.join(str(p) for p in CANDIDATE_STORES)}. "
        f"Pass --store-db explicitly."
    )
EMBED_MODEL = "gemini-embedding-001"
EMBED_BATCH_SIZE = 20

ISLAND_COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 12,
    "font.family": "sans-serif",
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Experiment resolution
# ---------------------------------------------------------------------------

def resolve_experiment_id(store_db: Path, run_dir: Path) -> str:
    """Look up the experiment_id for ``run_dir`` from the store.

    Tries (1) matching the store's ``experiments.dir`` column against the
    absolute ``run_dir`` path, then (2) falling back to the directory's
    basename as the experiment_id (matches the project convention where
    ``<run_dir_name> == experiment_id``).
    """
    run_dir_abs = str(run_dir.resolve())
    conn = sqlite3.connect(str(store_db))
    try:
        cur = conn.execute(
            "SELECT experiment_id FROM experiments WHERE dir=? LIMIT 1",
            (run_dir_abs,),
        )
        row = cur.fetchone()
        if row:
            return row[0]
        # Fallback: directory basename == experiment_id
        basename = run_dir.name
        cur = conn.execute(
            "SELECT experiment_id FROM experiments WHERE experiment_id=? LIMIT 1",
            (basename,),
        )
        row = cur.fetchone()
        if row:
            return row[0]
        raise SystemExit(
            f"Could not resolve experiment_id for run dir {run_dir}. "
            f"Pass --exp explicitly."
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_island_ideas(
    store_db: Path, experiment_id: str, run_dir: Path,
) -> list[dict]:
    """Pull (executor_id, score, island_id, idea_text) rows from the store."""
    conn = sqlite3.connect(str(store_db))
    try:
        cur = conn.execute(
            "SELECT run_id, score, metadata FROM runs "
            "WHERE experiment_id=? AND score IS NOT NULL ORDER BY run_id",
            (experiment_id,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    out: list[dict] = []
    missing_text = 0
    missing_island = 0
    for executor_id, score, metadata in rows:
        meta = json.loads(metadata) if metadata else {}
        island_id = meta.get("island_id")
        if island_id is None:
            missing_island += 1
            continue

        try:
            iteration = int(executor_id.rsplit("_", 1)[-1])
        except ValueError:
            missing_text += 1
            continue

        idea_text = _read_idea_md(run_dir, iteration)
        if not idea_text:
            missing_text += 1
            continue

        out.append({
            "executor_id": executor_id,
            "iteration": iteration,
            "score": float(score),
            "island_id": int(island_id),
            "operator": meta.get("operator"),
            "idea_text": idea_text[:2500],  # same truncation as the skill
        })

    print(
        f"Loaded {len(out)} ideas "
        f"(skipped: {missing_island} missing island_id, "
        f"{missing_text} missing idea.md)"
    )
    if missing_island and not out:
        print(
            "No runs carry island_id metadata — is this really an "
            "IslandSearch run?", file=sys.stderr,
        )
    return out


def _read_idea_md(run_dir: Path, iteration: int) -> str | None:
    """Locate the accepted idea.md for a given iteration's ideator dir."""
    ideator_dir = run_dir / f"ideator_{iteration:03d}"
    if not ideator_dir.exists():
        return None
    attempts = sorted(ideator_dir.glob("attempt_*/idea.md"))
    if not attempts:
        return None
    return attempts[-1].read_text()


# ---------------------------------------------------------------------------
# Embeddings (same helpers as the skill's figures.py)
# ---------------------------------------------------------------------------

def get_gemini_embeddings(texts: list[str]) -> np.ndarray | None:
    try:
        from heuresis.qd.core.embedding import GeminiEmbedder
    except ImportError:
        return None
    try:
        emb = GeminiEmbedder(batch_size=EMBED_BATCH_SIZE)
        return emb.embed(texts)
    except Exception as exc:
        print(f"  embedding error: {exc}")
        return None


def get_tfidf_embeddings(texts: list[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(
        max_features=1000, stop_words="english", sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b",
    ).fit_transform(texts).toarray()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_island_embedding_map(
    ideas: list[dict],
    embeddings: np.ndarray,
    output: Path,
    *,
    title_suffix: str = "",
) -> None:
    import umap
    from sklearn.metrics.pairwise import cosine_distances

    n_neighbors = min(15, len(ideas) - 1)
    projected = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.2,
        metric="cosine",
        random_state=42,
    ).fit_transform(embeddings)

    island_ids = np.array([i["island_id"] for i in ideas])
    scores = np.array([i["score"] for i in ideas])
    iterations = np.array([i["iteration"] for i in ideas])

    unique_islands = sorted(set(island_ids.tolist()))

    n_isl = len(unique_islands)
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(
        2, n_isl,
        height_ratios=[1.35, 1.0],
        hspace=0.28, wspace=0.22,
    )
    ax_map = fig.add_subplot(gs[0, : max(1, n_isl - 1)])
    ax_dist = fig.add_subplot(gs[0, max(1, n_isl - 1):])
    ax_small = [fig.add_subplot(gs[1, i]) for i in range(n_isl)]

    # --- Combined UMAP (shuffled plot order so no color is buried) ---
    rng = np.random.default_rng(0)
    order = rng.permutation(len(projected))

    color_of: dict[int, str] = {
        isl: ISLAND_COLORS[idx % len(ISLAND_COLORS)]
        for idx, isl in enumerate(unique_islands)
    }
    point_colors = np.array([color_of[int(i)] for i in island_ids])

    ax_map.scatter(
        projected[order, 0], projected[order, 1],
        c=point_colors[order], s=55, alpha=0.75,
        edgecolors="white", linewidths=0.4,
    )

    for isl in unique_islands:
        mask = island_ids == isl
        ax_map.scatter([], [], c=color_of[isl], s=55,
                       label=f"Island {isl} (n={mask.sum()})",
                       edgecolors="white", linewidths=0.4)

    # First (earliest iteration) = diamond, best (lowest score) = star
    first_points: dict[int, tuple[int, int]] = {}
    best_points: dict[int, tuple[int, int, float]] = {}
    for isl in unique_islands:
        mask = island_ids == isl
        if not mask.any():
            continue
        idx_in_mask = np.where(mask)[0]

        first_local = int(np.argmin(iterations[mask]))
        first_global = int(idx_in_mask[first_local])
        first_points[isl] = (first_global, int(iterations[first_global]))

        best_local = int(np.argmin(scores[mask]))
        best_global = int(idx_in_mask[best_local])
        best_points[isl] = (
            best_global, int(iterations[best_global]), float(scores[best_global]),
        )

        ax_map.scatter(
            projected[first_global, 0], projected[first_global, 1],
            marker="D", s=160, facecolor="white",
            edgecolor=color_of[isl], linewidths=2.2, zorder=6,
        )
        ax_map.annotate(
            f"i{iterations[first_global]}",
            xy=(projected[first_global, 0], projected[first_global, 1]),
            xytext=(9, 7), textcoords="offset points",
            fontsize=8, fontweight="bold", color="black",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white",
                  "edgecolor": color_of[isl], "alpha": 0.9},
        )

        ax_map.scatter(
            projected[best_global, 0], projected[best_global, 1],
            marker="*", s=420, color=color_of[isl],
            edgecolors="black", linewidths=1.1, zorder=7,
        )
        ax_map.annotate(
            f"i{iterations[best_global]}  {scores[best_global]:.3f}",
            xy=(projected[best_global, 0], projected[best_global, 1]),
            xytext=(10, -12), textcoords="offset points",
            fontsize=8, fontweight="bold", color="black",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white",
                  "edgecolor": color_of[isl], "alpha": 0.95},
        )

    ax_map.scatter([], [], marker="D", s=120, facecolor="white",
                   edgecolor="gray", linewidths=2.2,
                   label="First scored idea")
    ax_map.scatter([], [], marker="*", s=260, color="gray",
                   edgecolors="black", linewidths=1.1,
                   label="Best idea")
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    ax_map.set_xlabel("UMAP 1", fontsize=10)
    ax_map.set_ylabel("UMAP 2", fontsize=10)
    ax_map.set_title(
        "Idea Space — all islands (UMAP, Gemini embeddings)",
        fontsize=13, fontweight="bold", pad=8,
    )
    ax_map.legend(fontsize=9, loc="best", framealpha=0.9)

    # --- Per-island small multiples ---
    xlim = ax_map.get_xlim()
    ylim = ax_map.get_ylim()
    for idx, isl in enumerate(unique_islands):
        ax = ax_small[idx]
        mask = island_ids == isl
        color = color_of[isl]

        ax.scatter(
            projected[~mask, 0], projected[~mask, 1],
            c="#d0d0d0", s=18, alpha=0.55, edgecolors="none",
        )
        ax.scatter(
            projected[mask, 0], projected[mask, 1],
            c=color, s=42, alpha=0.85,
            edgecolors="white", linewidths=0.4,
        )
        if isl in first_points:
            first_global, _ = first_points[isl]
            ax.scatter(
                projected[first_global, 0], projected[first_global, 1],
                marker="D", s=130, facecolor="white",
                edgecolor=color, linewidths=2.0, zorder=6,
            )
        if isl in best_points:
            best_global, best_iter, best_score = best_points[isl]
            ax.scatter(
                projected[best_global, 0], projected[best_global, 1],
                marker="*", s=340, color=color,
                edgecolors="black", linewidths=1.1, zorder=7,
            )
            ax.set_title(
                f"Island {isl}  (n={mask.sum()})   best={best_score:.3f} @ i{best_iter}",
                fontsize=11, fontweight="bold", color=color, pad=4,
            )
        else:
            ax.set_title(
                f"Island {isl}  (n={mask.sum()})",
                fontsize=12, fontweight="bold", color=color, pad=4,
            )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])

    # --- Within-island pairwise cosine distance violin ---
    dist_data: list[np.ndarray] = []
    dist_labels: list[str] = []
    for isl in unique_islands:
        mask = island_ids == isl
        if mask.sum() < 2:
            continue
        dmat = cosine_distances(embeddings[mask])
        triu = np.triu_indices(mask.sum(), k=1)
        dist_data.append(dmat[triu])
        dist_labels.append(f"Island {isl}\n(n={mask.sum()})")

    if dist_data:
        parts = ax_dist.violinplot(
            dist_data, showmeans=True, showmedians=True, showextrema=False,
        )
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(ISLAND_COLORS[i % len(ISLAND_COLORS)])
            pc.set_alpha(0.5)
            pc.set_edgecolor(ISLAND_COLORS[i % len(ISLAND_COLORS)])
        parts["cmeans"].set_color("black")
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linestyle("--")

        ax_dist.boxplot(
            dist_data, widths=0.1, patch_artist=True, showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
            boxprops={"facecolor": "white", "edgecolor": "gray", "alpha": 0.8},
            whiskerprops={"color": "gray"}, capprops={"color": "gray"},
        )
        ax_dist.set_xticks(range(1, len(dist_labels) + 1))
        ax_dist.set_xticklabels(dist_labels, fontsize=11, fontweight="bold")
        ax_dist.set_ylabel("Pairwise Cosine Distance", fontsize=11)
        ax_dist.set_title(
            "Within-island Semantic Spread",
            fontsize=13, fontweight="bold", pad=8,
        )
        y_lo = ax_dist.get_ylim()[0]
        for i, dists in enumerate(dist_data):
            ax_dist.text(
                i + 1, y_lo + 0.005,
                f"mean={np.mean(dists):.3f}\nmed={np.median(dists):.3f}",
                ha="center", va="bottom", fontsize=8,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "#f0f0f0",
                      "edgecolor": "#cccccc"},
            )

    title = "Island Search — Per-Island Idea Embeddings"
    if title_suffix:
        title += f"\n{title_suffix}"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {output}")

    # Print per-island stats so they can be cited in text.
    print("\nPer-island mean pairwise cosine distance:")
    for lbl, dists in zip(dist_labels, dist_data):
        print(f"  {lbl.replace(chr(10), ' ')}: "
              f"mean={np.mean(dists):.4f}  median={np.median(dists):.4f}  "
              f"n_pairs={len(dists)}")

    full_dmat = cosine_distances(embeddings)
    triu = np.triu_indices(len(embeddings), k=1)
    all_pairs = full_dmat[triu]
    same_island_mask = island_ids[triu[0]] == island_ids[triu[1]]
    print(f"All pairs:          mean={all_pairs.mean():.4f}")
    print(f"Same-island pairs:  mean={all_pairs[same_island_mask].mean():.4f}"
          f"  (n={same_island_mask.sum()})")
    print(f"Cross-island pairs: mean={all_pairs[~same_island_mask].mean():.4f}"
          f"  (n={(~same_island_mask).sum()})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="Path to the island run directory")
    parser.add_argument("--exp", type=str, default=None,
                        help="experiment_id in the store (auto-detected if omitted)")
    parser.add_argument("--store-db", type=Path, default=None,
                        help="Path to store.db (auto-detected if omitted)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: <run>/figures/)")
    parser.add_argument("--force", action="store_true",
                        help="Re-embed even if cache exists")
    args = parser.parse_args()

    run_dir: Path = args.run.resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run dir does not exist: {run_dir}")

    store_db: Path = args.store_db or autodetect_store(run_dir, args.exp)
    if not store_db.exists():
        raise SystemExit(f"Store DB does not exist: {store_db}")

    exp_id = args.exp or resolve_experiment_id(store_db, run_dir)
    output_dir: Path = (args.output_dir or (run_dir / "figures")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / "island_embedding_cache.json"
    figure_path = output_dir / "island_embedding_map.png"

    print(f"Run:        {run_dir}")
    print(f"Experiment: {exp_id}")
    print(f"Store DB:   {store_db}")
    print(f"Output:     {output_dir}")

    ideas = load_island_ideas(store_db, exp_id, run_dir)
    if len(ideas) < 10 and cache_path.exists() and not args.force:
        # Run filesystem may have been pruned. Fall back to reconstructing
        # ideas from the cached embeddings' metadata so we can still plot.
        cached_obj = json.loads(cache_path.read_text())
        cached_ids = cached_obj.get("executor_ids", [])
        cached_islands = cached_obj.get("island_ids", [])
        cached_scores = cached_obj.get("scores", [])
        if cached_ids and len(cached_ids) == len(cached_islands) == len(cached_scores):
            print(
                f"Filesystem pruned ({len(ideas)} ideas loadable); "
                f"reconstructing from cache ({len(cached_ids)} ideas)."
            )
            ideas = [
                {
                    "executor_id": eid,
                    "iteration": int(eid.rsplit("_", 1)[-1]),
                    "score": float(s),
                    "island_id": int(isl),
                    "operator": None,
                    "idea_text": "",  # unused when cache is authoritative
                }
                for eid, isl, s in zip(cached_ids, cached_islands, cached_scores)
            ]

    if len(ideas) < 10:
        raise SystemExit(f"Too few ideas ({len(ideas)}); nothing to plot.")

    cached: dict | None = None
    if cache_path.exists() and not args.force:
        cached = json.loads(cache_path.read_text())
        cached_ids = cached.get("executor_ids", [])
        if cached_ids == [i["executor_id"] for i in ideas]:
            print(f"Using cached embeddings ({cache_path.name})")
            embeddings = np.array(cached["embeddings"])
        else:
            print("Cache is stale; re-embedding.")
            cached = None

    if cached is None:
        print(f"Embedding {len(ideas)} ideas via {EMBED_MODEL}...")
        texts = [i["idea_text"] for i in ideas]
        embeddings = get_gemini_embeddings(texts)
        if embeddings is None:
            print("Falling back to TF-IDF")
            embeddings = get_tfidf_embeddings(texts)
        cache_path.write_text(json.dumps({
            "model": EMBED_MODEL,
            "executor_ids": [i["executor_id"] for i in ideas],
            "island_ids": [i["island_id"] for i in ideas],
            "scores": [i["score"] for i in ideas],
            "embeddings": embeddings.tolist(),
        }))
        print(f"  cached -> {cache_path}")

    plot_island_embedding_map(
        ideas, embeddings, figure_path,
        title_suffix=f"(run: {exp_id})",
    )


if __name__ == "__main__":
    main()
