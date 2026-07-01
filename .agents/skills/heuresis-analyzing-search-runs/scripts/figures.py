#!/usr/bin/env python3
"""Generate comparison figures from analysis caches.

Usage: python figures.py cache1.json cache2.json [--output-dir figures/]

Produces:
  score_distribution.png   — violin/box plot of scores per strategy
  novelty_distribution.png — grouped bar chart of novelty counts
  quality_vs_novelty.png   — scatter plot with Pareto front
  technique_coverage.png   — side-by-side component × approach heatmaps
  diversity_map.png        — UMAP of Gemini-embedded ideas + pairwise distances
  fitness_curve.png        — running best score vs. valid-solution index, one line per strategy
"""

import argparse
import json
import os
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Allow importing the aggregate module (same directory) for metric recomputation
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import (  # noqa: E402
    compute_diversity_metrics,
    compute_novelty_metrics,
    compute_quality_metrics,
)
from _rubric import (  # noqa: E402
    DEFAULT_RUBRIC,
    is_novel_score,
    level_range,
    load_rubric,
)


def cache_rubric(cache: dict) -> dict:
    """Resolve the rubric for a cache. Trusts the cache's `rubric` stamp;
    falls back to the legacy `rubric_version` field (`"1to4"` → nanogpt_1to4),
    then to the package default (G&P).
    """
    name = cache.get("rubric")
    if not name:
        legacy = cache.get("rubric_version")
        if legacy in ("1to4", "1to3"):
            name = "nanogpt_1to4"
        else:
            name = DEFAULT_RUBRIC
    return load_rubric(name)


# ---------------------------------------------------------------------------
# Styling — sourced from analysis/libs/paper_theme.py so the same palette
# and rcParams are reused by ad-hoc analysis scripts and the paper repo.
# ---------------------------------------------------------------------------

_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()),
    Path(__file__).resolve().parents[4],
)
sys.path.insert(0, str(_REPO_ROOT / "analysis"))
from libs.paper_theme import (  # noqa: E402
    BASELINE_COLOR,
    apply_paper_rcparams,
    color_for,
)

apply_paper_rcparams()


def load_cache(path: str) -> dict:
    return json.loads(Path(path).read_text())


def label_for(cache: dict) -> str:
    """Short human-readable label."""
    strategy = cache.get("strategy_type", "unknown")
    return strategy.replace("mapelites", "MAP-Elites").replace("island", "Island")


# ---------------------------------------------------------------------------
# Balancing: truncate caches to a common number of ideas
# ---------------------------------------------------------------------------

def _executor_num(idea: dict) -> int:
    """Parse a numeric executor index from 'executor_NNN' for chronological sort."""
    eid = idea.get("executor_id", "")
    try:
        return int(eid.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return 0


def balance_caches(
    caches: list[dict], target_n: int | None = None
) -> tuple[list[dict], int]:
    """Truncate each cache to the same number of ideas and recompute metrics.

    Takes the first N valid ideas by executor_id order (chronological). This
    ensures that diversity, novelty, and quality metrics compared across runs
    correspond to the same ideation budget. Metric recomputation uses the same
    helpers as aggregate.py so the numbers stay consistent with full-run caches.

    The success_rate is rescaled using the largest executor number within the
    truncated window as a proxy for total attempts, so it reflects the local
    success rate over the first N valid ideas rather than the full run.

    Returns (balanced_caches, n_used).
    """
    if not caches:
        return [], 0
    if target_n is None:
        target_n = min(len(c.get("ideas", [])) for c in caches)
    if target_n <= 0:
        return list(caches), 0

    balanced: list[dict] = []
    for cache in caches:
        new_cache = deepcopy(cache)
        ideas_sorted = sorted(cache.get("ideas", []), key=_executor_num)
        truncated = ideas_sorted[:target_n]

        direction = cache.get("metric_direction", "lower_is_better")
        original_valid = cache.get("metrics", {}).get("quality", {}).get(
            "valid_count", len(cache.get("ideas", []))
        )
        original_total = cache.get("metrics", {}).get("quality", {}).get(
            "total_executors", original_valid
        )

        # Estimate total attempts within the truncated window. The executor
        # numbers increase monotonically with iteration, so the largest number
        # in the window is a good proxy for "attempts up to the Nth valid idea".
        if truncated:
            max_eid = max(_executor_num(i) for i in truncated)
            min_eid = min(_executor_num(i) for i in truncated)
            window_total = max(max_eid - min_eid + 1, len(truncated))
        else:
            window_total = 0

        rubric = cache_rubric(cache)
        new_cache["ideas"] = truncated
        new_cache["metrics"] = {
            "quality": compute_quality_metrics(truncated, window_total, direction),
            "diversity": compute_diversity_metrics(truncated),
            "novelty": compute_novelty_metrics(truncated, rubric, direction),
        }
        new_cache["balanced"] = {
            "target_n": target_n,
            "original_valid_count": original_valid,
            "original_total_executors": original_total,
        }
        balanced.append(new_cache)

    return balanced, target_n


def write_balanced_metrics(
    caches: list[dict], target_n: int, output: Path
) -> None:
    """Save balanced metrics to JSON for downstream synthesis (report writing)."""
    payload = {
        "balanced_n": target_n,
        "note": (
            "Each run was truncated to the first N valid ideas (by executor_id "
            "order). Metrics recomputed on the truncated set. Use these numbers "
            "for apples-to-apples comparisons; see analysis_cache.json for the "
            "full per-run data."
        ),
        "runs": [
            {
                "run_id": cache.get("run_id"),
                "strategy_type": cache.get("strategy_type"),
                "label": label_for(cache),
                "balanced": cache.get("balanced", {}),
                "metrics": cache.get("metrics", {}),
            }
            for cache in caches
        ],
    }
    output.write_text(json.dumps(payload, indent=2))
    print(f"  \u2192 {output}")


# ---------------------------------------------------------------------------
# Outlier filtering
# ---------------------------------------------------------------------------

def filter_outliers_iqr(scores: list[float], factor: float = 3.0) -> list[float]:
    """Remove extreme outliers beyond factor*IQR above Q3."""
    arr = np.array(scores)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    upper = q3 + factor * iqr
    return arr[arr <= upper].tolist()


def outlier_cutoff(caches: list[dict], factor: float = 3.0) -> float:
    """Compute a shared y-axis upper limit from all caches."""
    all_scores = []
    for cache in caches:
        all_scores.extend(cache["metrics"]["quality"]["scores"])
    arr = np.array(all_scores)
    q1, q3 = np.percentile(arr, [25, 75])
    return q3 + factor * (q3 - q1)


# ---------------------------------------------------------------------------
# Figure 1: Score distribution
# ---------------------------------------------------------------------------

def plot_score_distribution(caches: list[dict], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))

    cutoff = outlier_cutoff(caches, factor=3.0)

    scores_list = []
    labels = []
    for cache in caches:
        raw = cache["metrics"]["quality"]["scores"]
        filtered = [s for s in raw if s <= cutoff]
        scores_list.append(filtered)
        labels.append(label_for(cache))

    parts = ax.violinplot(scores_list, showmeans=False, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        c = color_for(labels[i], fallback_index=i)
        pc.set_facecolor(c)
        pc.set_alpha(0.65)
        pc.set_edgecolor(c)

    # Overlay box plots
    ax.boxplot(scores_list, widths=0.12, patch_artist=True,
               showfliers=False,
               medianprops={"color": "black", "linewidth": 1.5},
               boxprops={"facecolor": "white", "edgecolor": "gray", "alpha": 0.8},
               whiskerprops={"color": "gray"}, capprops={"color": "gray"})

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=13)
    ax.grid(axis="y", which="major")  # categorical x → suppress vertical grid

    metric_name = caches[0].get("metric_name", "score")
    direction = caches[0].get("metric_direction", "lower_is_better")
    arrow = "\u2193" if direction == "lower_is_better" else "\u2191"
    ax.set_ylabel(f"{metric_name} ({arrow} better)", fontsize=12)
    ax.set_title("Score Distribution", fontsize=14, pad=12)

    # Best-score star markers
    for i, cache in enumerate(caches):
        q = cache["metrics"]["quality"]
        ax.plot(i + 1, q["best"], "*", color=color_for(labels[i], fallback_index=i),
                markersize=14, markeredgecolor="black", markeredgewidth=0.5, zorder=5)

    # Baseline reference line
    ax.axhline(y=0.998, color=BASELINE_COLOR, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(len(caches) + 0.55, 0.998, "baseline", fontsize=9, color=BASELINE_COLOR, va="bottom")

    ax.set_ylim(top=cutoff * 1.02)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"  \u2192 {output}")


# ---------------------------------------------------------------------------
# Figure 2: Novelty distribution
# ---------------------------------------------------------------------------

def plot_novelty_distribution(caches: list[dict], output: Path) -> None:
    """Grouped bar chart: x-axis = strategy, bars within = novelty levels.

    The level set, names, and colors are sourced from the cache's rubric.
    All caches are expected to share a rubric (validated up front; if they
    differ, we use the first cache's rubric and warn).
    """
    rubrics = [cache_rubric(c) for c in caches]
    rubric = rubrics[0]
    if any(r["name"] != rubric["name"] for r in rubrics[1:]):
        print(f"  ! Mixed rubrics across caches; using {rubric['name']} for the figure",
              file=sys.stderr)

    # Reverse so the bar order within each strategy reads "less novel → more novel"
    # (5 → 1 under Gupta-Pruthi where higher = more plagiarized; the reader's eye
    # tracks novelty from the warm/dark side toward the original/novel side).
    levels = list(reversed(level_range(rubric)))
    level_lookup = rubric.get("levels", {}) or {}
    fallback_colors = ["#bbbbbb", "#6c8ebf", "#82b366", "#d6635c", "#7c4a8a"]

    n_strategies = len(caches)
    n_levels = len(levels)
    fig, ax = plt.subplots(figsize=(max(6.5, 1.6 * n_strategies + 2.5), 4.8))

    group_width = 0.8
    bar_w = group_width / max(n_levels, 1)
    x = np.arange(n_strategies)

    for j, lvl in enumerate(levels):
        offset = (j - (n_levels - 1) / 2) * bar_w
        counts = [c["metrics"]["novelty"].get("distribution", {}).get(lvl, 0) for c in caches]
        entry = level_lookup.get(lvl, {}) if isinstance(level_lookup, dict) else {}
        name = entry.get("name", lvl) if isinstance(entry, dict) else str(entry)
        color = entry.get("color") if isinstance(entry, dict) else None
        if not color:
            color = fallback_colors[j % len(fallback_colors)]
        bars = ax.bar(x + offset, counts, bar_w,
                      label=f"{lvl} — {name}",
                      color=color, edgecolor="black", linewidth=0.6)
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        str(count), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([label_for(c) for c in caches], fontsize=11)
    ax.set_ylabel("Number of Ideas", fontsize=12)
    direction_note = (
        " — higher = more novel" if rubric["direction"] == "higher_is_more_novel"
        else " — higher = more plagiarized"
    )
    title_suffix = f"  ({rubric['name']}{direction_note})"
    ax.set_title("Novelty Distribution" + title_suffix,
                 fontsize=13, pad=10)
    ax.legend(title="Novelty", fontsize=10, framealpha=0.9, loc="upper left")
    ax.grid(axis="y", which="major")  # categorical x → suppress vertical grid
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)

    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    print(f"  \u2192 {output}")

# ---------------------------------------------------------------------------
# Figure 3: Quality vs Novelty scatter
# ---------------------------------------------------------------------------

def plot_quality_vs_novelty(caches: list[dict], output: Path) -> None:
    """Scatter plot of quality vs novelty per cache, with rubric-aware Pareto.

    Pareto front = top-10 quality intersected with the "novel" side of the
    rubric (rubric.novel_threshold + rubric.direction). Y-axis tick labels
    use the rubric's level names. Spearman correlation is computed on raw
    scores (no jitter / outlier filtering) per cache.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    cutoff = outlier_cutoff(caches, factor=3.0)

    rubric = cache_rubric(caches[0])
    score_lo, score_hi = rubric["score_range"]
    novel_threshold = rubric["novel_threshold"]
    novel_dir = rubric["direction"]
    levels = level_range(rubric)
    level_lookup = rubric.get("levels", {}) or {}

    pareto_op = ">=" if novel_dir == "higher_is_more_novel" else "<="

    for i, cache in enumerate(caches):
        scores = []
        novelties = []
        pareto_scores = []
        pareto_novelties = []
        verified_scores = []
        verified_novelties = []
        verified_labels = []

        direction = cache.get("metric_direction", "lower_is_better")

        verified_ids = {v["executor_id"] for v in cache.get("verified_ideas", [])}
        verified_lookup = {v["executor_id"]: v for v in cache.get("verified_ideas", [])}

        ideas = sorted(cache["ideas"],
                       key=lambda x: x["score"],
                       reverse=(direction == "higher_is_better"))
        top10_ids = {idea["executor_id"] for idea in ideas[:10]}

        raw_scores_for_corr = []
        raw_novs_for_corr = []

        for idea in ideas:
            nov = idea.get("novelty", {}).get("score")
            if nov is None:
                continue
            raw_scores_for_corr.append(idea["score"])
            raw_novs_for_corr.append(nov)

            if idea["score"] > cutoff:
                continue

            jittered_nov = nov + np.random.uniform(-0.18, 0.18)
            scores.append(idea["score"])
            novelties.append(jittered_nov)

            if idea["executor_id"] in top10_ids and is_novel_score(rubric, nov):
                pareto_scores.append(idea["score"])
                pareto_novelties.append(jittered_nov)

            if idea["executor_id"] in verified_ids:
                verified_scores.append(idea["score"])
                verified_novelties.append(jittered_nov)
                v = verified_lookup[idea["executor_id"]]
                verified_labels.append(
                    f'{idea["executor_id"]}\n{v["original_score"]}→{v["verified_score"]}'
                )

        c = color_for(label_for(cache), fallback_index=i)
        ax.scatter(scores, novelties, alpha=0.5, s=40,
                   color=c, label=label_for(cache),
                   edgecolors="white", linewidths=0.3)

        if pareto_scores:
            ax.scatter(pareto_scores, pareto_novelties, s=120, facecolors="none",
                       edgecolors=c, linewidths=2.2, zorder=5)

        if verified_scores:
            ax.scatter(verified_scores, verified_novelties, s=90, marker="D",
                       color=c, edgecolors="black",
                       linewidths=1.2, zorder=6)
            for vs, vn, vl in zip(verified_scores, verified_novelties, verified_labels):
                ax.annotate(vl, xy=(vs, vn), xytext=(8, 8), textcoords="offset points",
                            fontsize=7.5, color=c,
                            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white",
                                  "edgecolor": c, "alpha": 0.85},
                            arrowprops={"arrowstyle": "->", "color": c,
                                        "lw": 0.8})

        if len(raw_scores_for_corr) > 5:
            from scipy.stats import spearmanr
            rho, pval = spearmanr(raw_scores_for_corr, raw_novs_for_corr)
            sig = "*" if pval < 0.05 else ""
            sig = "**" if pval < 0.01 else sig
            label_text = f"{label_for(cache)}: ρ={rho:.2f} (p={pval:.3f}){sig}"
            ax.annotate(label_text,
                        xy=(0.02, 0.04 + i * 0.06), xycoords="axes fraction",
                        fontsize=10, color=c,
                        ha="left", va="bottom",
                        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                              "edgecolor": c, "alpha": 0.9})

    ax.axvline(x=0.998, color=BASELINE_COLOR, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(0.998, score_hi + 0.3, "baseline",
            fontsize=8, color=BASELINE_COLOR, ha="left", va="bottom")

    metric_name = caches[0].get("metric_name", "score")
    direction = caches[0].get("metric_direction", "lower_is_better")
    if direction == "lower_is_better":
        ax.invert_xaxis()
    ax.set_xlabel(f"{metric_name}  (better →)", fontsize=12)
    ax.set_ylabel(f"Novelty Score  ({rubric['name']})", fontsize=12)
    yticks = [int(lvl) for lvl in levels]
    ytick_labels = []
    for lvl in levels:
        entry = level_lookup.get(lvl, {})
        name = entry.get("name", lvl) if isinstance(entry, dict) else str(entry)
        ytick_labels.append(f"{lvl} ({name})")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize=11)
    ax.set_ylim(score_lo - 0.5, score_hi + 0.5)
    ax.set_title("Quality vs Novelty", fontsize=14, pad=12)

    ax.scatter([], [], s=120, facecolors="none", edgecolors="gray", linewidths=2,
               label=f"Pareto (top-10 ∩ novelty {pareto_op} {novel_threshold})")
    any_verified = any(cache.get("verified_ideas") for cache in caches)
    if any_verified:
        ax.scatter([], [], s=90, marker="D", color="gray", edgecolors="black",
                   linewidths=1.2, label="Verified (2nd-round review)")
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    print(f"  \u2192 {output}")

# ---------------------------------------------------------------------------
# Figure 4: Technique coverage heatmap
# ---------------------------------------------------------------------------

TOP_N_COMPONENTS = 12
TOP_N_APPROACHES = 10


def plot_technique_coverage(caches: list[dict], output: Path) -> None:
    # Aggregate component/approach counts across all caches to find top-N
    global_comp_counts: Counter = Counter()
    global_app_counts: Counter = Counter()
    for cache in caches:
        for idea in cache["ideas"]:
            cls = idea.get("classification", {})
            for c in cls.get("components", []):
                global_comp_counts[c] += 1
            for a in cls.get("approaches", []):
                global_app_counts[a] += 1

    if not global_comp_counts or not global_app_counts:
        print("  \u2192 Skipping technique_coverage.png (no classification data)")
        return

    # Take top-N most frequent
    components = [c for c, _ in global_comp_counts.most_common(TOP_N_COMPONENTS)]
    approaches = [a for a, _ in global_app_counts.most_common(TOP_N_APPROACHES)]

    n = len(caches)
    cell_h = 0.45
    cell_w = 0.55
    fig_h = max(5, len(components) * cell_h + 2.5)
    fig_w = max(6, len(approaches) * cell_w + 2) * n + 1
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), squeeze=False)

    # Shared color scale
    vmax = 0
    matrices = []
    for cache in caches:
        matrix = np.zeros((len(components), len(approaches)))
        for idea in cache["ideas"]:
            cls = idea.get("classification", {})
            idea_comps = set(cls.get("components", []))
            idea_apps = set(cls.get("approaches", []))
            for ci, comp in enumerate(components):
                for ai, app in enumerate(approaches):
                    if comp in idea_comps and app in idea_apps:
                        matrix[ci, ai] += 1
        matrices.append(matrix)
        vmax = max(vmax, matrix.max())

    for idx, (cache, matrix) in enumerate(zip(caches, matrices)):
        ax = axes[0][idx]
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", interpolation="nearest",
                        vmin=0, vmax=vmax)

        ax.set_xticks(range(len(approaches)))
        ax.set_xticklabels(approaches, rotation=40, ha="right", fontsize=10)
        ax.set_yticks(range(len(components)))
        ax.set_yticklabels(components, fontsize=10)
        ax.set_title(label_for(cache), fontsize=13, pad=8)

        # Annotate cells
        for ci in range(len(components)):
            for ai in range(len(approaches)):
                val = int(matrix[ci, ai])
                if val > 0:
                    text_color = "white" if val > vmax * 0.55 else "black"
                    ax.text(ai, ci, str(val), ha="center", va="center",
                            fontsize=9, color=text_color)

        # Grid lines between cells (white minor only; suppress theme major grid)
        ax.set_xticks(np.arange(-0.5, len(approaches)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(components)), minor=True)
        ax.grid(False, which="major")
        ax.grid(True, which="minor", color="white", linewidth=1)
        ax.tick_params(which="minor", size=0)

    # Shared colorbar
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02, pad=0.03, label="Co-occurrence count")

    fig.suptitle("Technique Coverage (top components \u00d7 approaches)",
                 fontsize=14)
    fig.subplots_adjust(top=0.92, bottom=0.18, wspace=0.3)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"  \u2192 {output}")


# ---------------------------------------------------------------------------
# Figure 5: Diversity map (UMAP of Gemini-embedded idea texts)
# ---------------------------------------------------------------------------

_EMBED_MODEL = "gemini-embedding-001"
_EMBED_BATCH_SIZE = 20  # API batch limit


def _gemini_keys_file() -> Path | None:
    """Optional explicit keys file via env; otherwise repo-root gemini-keys.txt.

    Returns None when neither is present — the embedder then loads keys from
    ``GEMINI_API_KEYS`` / ``GEMINI_API_KEY`` env (see heuresis.api_keys).
    """
    env = os.environ.get("GEMINI_KEYS_FILE") or os.environ.get("HEURESIS_GEMINI_KEYS_FILE")
    if env and Path(env).is_file():
        return Path(env)
    cand = _REPO_ROOT / "gemini-keys.txt"
    return cand if cand.is_file() else None


def _get_gemini_embeddings(texts: list[str]) -> np.ndarray | None:
    """Embed texts via the shared GeminiEmbedder. Returns None on total failure."""
    try:
        from heuresis.qd.core.embedding import GeminiEmbedder
    except ImportError:
        print("    heuresis.qd.core.embedding not importable, falling back to TF-IDF")
        return None

    try:
        # Pass an explicit keys file if we have one; otherwise GeminiEmbedder
        # loads keys from the GEMINI_API_KEYS / GEMINI_API_KEY environment.
        emb = GeminiEmbedder(
            api_keys_file=_gemini_keys_file(), batch_size=_EMBED_BATCH_SIZE
        )
        return emb.embed(texts)
    except Exception as exc:
        print(f"    Embedder unavailable ({exc}), falling back to TF-IDF")
        return None


def _get_tfidf_embeddings(texts: list[str]) -> np.ndarray:
    """Fallback TF-IDF embeddings."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(
        max_features=1000, stop_words="english", sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b",
    ).fit_transform(texts).toarray()


def plot_diversity_map(caches: list[dict], output: Path) -> None:
    """UMAP projection of idea texts using Gemini embeddings."""
    try:
        import umap
    except ImportError:
        print("  \u2192 Skipping diversity_map.png (umap-learn not installed)")
        return

    # --- Collect idea texts ---
    idea_texts: list[str] = []
    strategy_labels: list[str] = []
    novelty_scores: list[float] = []
    quality_scores: list[float] = []

    for cache in caches:
        strategy = label_for(cache)
        extracted_path = Path(cache["run_dir"]) / "extracted_ideas.json"
        if extracted_path.exists():
            extracted = json.loads(extracted_path.read_text())
            idea_lookup = {i["executor_id"]: i["idea_text"] for i in extracted["ideas"]}
        else:
            idea_lookup = {}

        for idea in cache["ideas"]:
            text = idea_lookup.get(idea["executor_id"], "")
            if not text:
                continue
            idea_texts.append(text[:2500])  # truncate for API
            strategy_labels.append(strategy)
            novelty_scores.append(idea.get("novelty", {}).get("score", 1))
            quality_scores.append(idea["score"])

    if len(idea_texts) < 10:
        print("  \u2192 Skipping diversity_map.png (too few ideas)")
        return

    # --- Embed ---
    print(f"    Embedding {len(idea_texts)} ideas via {_EMBED_MODEL}...")
    embeddings = _get_gemini_embeddings(idea_texts)
    if embeddings is None:
        print("    Falling back to TF-IDF embeddings")
        embeddings = _get_tfidf_embeddings(idea_texts)
    else:
        print(f"    Got {embeddings.shape} embeddings")

    # --- UMAP projection ---
    n_neighbors = min(15, len(idea_texts) - 1)
    projected = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=0.2, metric="cosine", random_state=42,
    ).fit_transform(embeddings)

    # --- Compute pairwise cosine distances in raw embedding space ---
    from sklearn.metrics.pairwise import cosine_distances

    strategy_set = list(dict.fromkeys(strategy_labels))
    strat_distances: dict[str, np.ndarray] = {}
    for strat in strategy_set:
        mask = np.array([s == strat for s in strategy_labels])
        strat_emb = embeddings[mask]
        if len(strat_emb) < 2:
            continue
        dist_matrix = cosine_distances(strat_emb)
        # Upper triangle (no diagonal)
        triu_idx = np.triu_indices(len(strat_emb), k=1)
        strat_distances[strat] = dist_matrix[triu_idx]

    # --- Plot: UMAP (left) + pairwise distance violin (right) ---
    # Legend goes outside/below via bbox_to_anchor on the figure legend.
    fig, (ax_map, ax_dist) = plt.subplots(1, 2, figsize=(11, 5.8),
                                           gridspec_kw={"width_ratios": [1.4, 1]})

    # Left panel: UMAP scatter (one marker per strategy, no novelty differentiation).
    for si, strat in enumerate(strategy_set):
        strat_mask = np.array([s == strat for s in strategy_labels])
        if not strat_mask.any():
            continue
        ax_map.scatter(
            projected[strat_mask, 0], projected[strat_mask, 1],
            c=color_for(strat, fallback_index=si), marker="o",
            s=42, alpha=0.6, edgecolors="white", linewidths=0.4,
            label=strat,
        )

    # Highlight best per strategy (star) + overall best (crown/outlined star)
    lower_is_better = caches[0].get("metric_direction", "lower_is_better") == "lower_is_better"
    quality_arr = np.array(quality_scores)
    strategy_arr = np.array(strategy_labels)
    best_per_strategy: dict[str, int] = {}
    for strat in strategy_set:
        mask = strategy_arr == strat
        if not mask.any():
            continue
        scores = np.where(mask, quality_arr, np.inf if lower_is_better else -np.inf)
        best_per_strategy[strat] = int(np.argmin(scores) if lower_is_better else np.argmax(scores))
    overall_best_idx = (int(np.argmin(quality_arr)) if lower_is_better
                       else int(np.argmax(quality_arr)))

    for strat, idx in best_per_strategy.items():
        si = strategy_set.index(strat)
        ax_map.scatter(projected[idx, 0], projected[idx, 1],
                       marker="*", s=380, c=color_for(strat, fallback_index=si),
                       edgecolors="black", linewidths=1.3, zorder=5)
    ax_map.scatter(projected[overall_best_idx, 0], projected[overall_best_idx, 1],
                   marker="*", s=720, facecolors="none", edgecolors="black",
                   linewidths=2.0, zorder=6)

    # Shared figure-level legend placed below the plot, horizontal.
    handles, labels = ax_map.get_legend_handles_labels()
    handles.append(plt.scatter([], [], marker="*", s=180, c="gray",
                               edgecolors="black", linewidths=1.0))
    labels.append("Best per strategy")
    handles.append(plt.scatter([], [], marker="*", s=260, facecolors="none",
                               edgecolors="black", linewidths=1.5))
    labels.append("Overall best")
    fig.legend(handles, labels, fontsize=9, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=len(handles), framealpha=0.9)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    ax_map.grid(False)
    ax_map.set_xlabel("UMAP 1", fontsize=10)
    ax_map.set_ylabel("UMAP 2", fontsize=10)
    ax_map.set_title("Idea Space (UMAP)", fontsize=13, pad=8)

    # Right panel: pairwise cosine distance violin
    dist_data = []
    dist_labels = []
    for si, strat in enumerate(strategy_set):
        if strat in strat_distances:
            dist_data.append(strat_distances[strat])
            dist_labels.append(strat)

    if dist_data:
        parts = ax_dist.violinplot(dist_data, showmeans=True, showmedians=True,
                                    showextrema=False)
        for i, pc in enumerate(parts["bodies"]):
            c = color_for(dist_labels[i], fallback_index=i)
            pc.set_facecolor(c)
            pc.set_alpha(0.5)
            pc.set_edgecolor(c)
        parts["cmeans"].set_color("black")
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linestyle("--")

        # Overlay box plot
        ax_dist.boxplot(dist_data, widths=0.1, patch_artist=True, showfliers=False,
                        medianprops={"color": "black", "linewidth": 1.5},
                        boxprops={"facecolor": "white", "edgecolor": "gray", "alpha": 0.8},
                        whiskerprops={"color": "gray"}, capprops={"color": "gray"})

        ax_dist.set_xticks(range(1, len(dist_labels) + 1))
        ax_dist.set_xticklabels(dist_labels, fontsize=12)
        ax_dist.set_ylabel("Pairwise Cosine Distance", fontsize=11)
        ax_dist.set_title("Raw Diversity", fontsize=13, pad=8)
        ax_dist.grid(False)

    fig.suptitle("Idea Diversity (Gemini embeddings)",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))  # leave room for bottom legend
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"  \u2192 {output}")


# ---------------------------------------------------------------------------
# Figure 6: Fitness curve (running best over valid-solution index)
# ---------------------------------------------------------------------------

def plot_fitness_curve(caches: list[dict], output: Path) -> None:
    """Running best score as a function of the number of valid solutions seen.

    One curve per strategy. Ideas are ordered chronologically via executor_id;
    y tracks the running min (lower_is_better) or max (higher_is_better) so the
    curve is monotone by construction. A dashed baseline line is drawn, and
    each curve's terminal best is marked with a star.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    direction = caches[0].get("metric_direction", "lower_is_better")
    lower_is_better = direction == "lower_is_better"

    all_bests: list[float] = []
    for i, cache in enumerate(caches):
        ideas_sorted = sorted(cache.get("ideas", []), key=_executor_num)
        scores = [idea["score"] for idea in ideas_sorted
                  if idea.get("score") is not None]
        if not scores:
            continue

        running_best: list[float] = []
        best = scores[0]
        for s in scores:
            best = min(best, s) if lower_is_better else max(best, s)
            running_best.append(best)
        all_bests.extend(running_best)

        x = np.arange(1, len(running_best) + 1)
        color = color_for(label_for(cache), fallback_index=i)
        label = (f"{label_for(cache)}  (n={len(scores)}, "
                 f"best={running_best[-1]:.4f})")
        ax.plot(x, running_best, color=color, linewidth=2, alpha=0.9, label=label)
        ax.plot(x[-1], running_best[-1], "*", color=color, markersize=14,
                markeredgecolor="black", markeredgewidth=0.5, zorder=5)

    # Baseline reference
    ax.axhline(y=0.998, color=BASELINE_COLOR, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(1.0, 0.998, " baseline", fontsize=9, color=BASELINE_COLOR,
            ha="left", va="center", transform=ax.get_yaxis_transform())

    # Cap y-range so outlier early scores don't squash the interesting region
    if all_bests:
        arr = np.array(all_bests)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        if lower_is_better:
            upper = q3 + 3.0 * iqr
            ax.set_ylim(top=min(ax.get_ylim()[1], upper * 1.02))
        else:
            lower = q1 - 3.0 * iqr
            ax.set_ylim(bottom=max(ax.get_ylim()[0], lower))

    metric_name = caches[0].get("metric_name", "score")
    arrow = "\u2193" if lower_is_better else "\u2191"
    ax.set_xlabel("Valid solutions", fontsize=12)
    ax.set_ylabel(f"Running best {metric_name} ({arrow} better)", fontsize=12)
    ax.set_title("Fitness Curve", fontsize=14, pad=12)
    ax.legend(fontsize=10, framealpha=0.9,
              loc="upper right" if lower_is_better else "lower right")

    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    print(f"  \u2192 {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comparison figures")
    parser.add_argument("caches", nargs="+", help="Paths to analysis_cache.json files")
    parser.add_argument("--output-dir", "-o", default="figures/",
                        help="Output directory for figures")
    parser.add_argument("--no-balance", action="store_true",
                        help="Disable balancing to the minimum number of ideas across runs")
    parser.add_argument("--balance-n", "--max-ideas", dest="balance_n", type=int, default=None,
                        help="Explicit target N for balancing (default: min across caches). "
                             "Alias: --max-ideas")
    args = parser.parse_args()

    if len(args.caches) < 2:
        print("Warning: comparison figures work best with 2+ caches", file=sys.stderr)

    caches = [load_cache(p) for p in args.caches]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Balance caches to the minimum number of ideas so diversity / novelty /
    # quality metrics are directly comparable across runs with different
    # ideation budgets.
    if len(caches) >= 2 and not args.no_balance:
        original_ns = [len(c.get("ideas", [])) for c in caches]
        caches, n_used = balance_caches(caches, target_n=args.balance_n)
        labels = [label_for(c) for c in caches]
        print(
            "Balancing ideas across runs: "
            + ", ".join(f"{lbl}={orig}" for lbl, orig in zip(labels, original_ns))
            + f" \u2192 {n_used} each (first N by executor order)"
        )
        write_balanced_metrics(caches, n_used, out_dir / "balanced_metrics.json")
    elif args.no_balance and len(caches) >= 2:
        print("Skipping balancing (--no-balance); figures use full per-run data.")

    print("Generating figures...")
    plot_score_distribution(caches, out_dir / "score_distribution.png")
    plot_novelty_distribution(caches, out_dir / "novelty_distribution.png")
    plot_quality_vs_novelty(caches, out_dir / "quality_vs_novelty.png")
    plot_technique_coverage(caches, out_dir / "technique_coverage.png")
    plot_diversity_map(caches, out_dir / "diversity_map.png")
    plot_fitness_curve(caches, out_dir / "fitness_curve.png")
    write_reproduce_script(out_dir, args)
    print(f"Done \u2014 {len(list(out_dir.glob('*.png')))} figures in {out_dir}")


def write_reproduce_script(out_dir: Path, args: argparse.Namespace) -> None:
    """Write a reproduce.sh that regenerates the figures with the same invocation."""
    script_path = Path(__file__).resolve()
    cache_paths = [str(Path(p).resolve()) for p in args.caches]
    flags = []
    if args.no_balance:
        flags.append("--no-balance")
    if args.balance_n is not None:
        flags.append(f"--max-ideas {args.balance_n}")
    out_abs = out_dir.resolve()

    lines = [
        "#!/usr/bin/env bash",
        "# Auto-generated by figures.py. Regenerates the figures in this directory.",
        "set -euo pipefail",
        "",
        f'python "{script_path}" \\',
    ]
    for c in cache_paths:
        lines.append(f'  "{c}" \\')
    lines.append(f'  --output-dir "{out_abs}"' + (" \\" if flags else ""))
    for i, f in enumerate(flags):
        suffix = " \\" if i < len(flags) - 1 else ""
        lines.append(f"  {f}{suffix}")
    lines.append("")

    repro = out_dir / "reproduce.sh"
    repro.write_text("\n".join(lines))
    repro.chmod(0o755)
    print(f"  \u2192 {repro}")


if __name__ == "__main__":
    main()
