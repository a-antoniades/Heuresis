#!/usr/bin/env python3
"""Aggregate classifications into metrics and produce analysis_cache.json.

Usage: python aggregate.py <run_dir> [--rubric <name>]

Reads <run_dir>/extracted_ideas.json and <run_dir>/classifications.json,
computes quality/diversity/novelty metrics, writes <run_dir>/analysis_cache.json.

The active rubric is taken from the extracted_ideas.json `rubric` field if
present, then the --rubric flag, then the package default. The rubric drives
distribution buckets, the Pareto threshold, and which side of the score range
counts as "novel".

If a store.db is available, writes novelty_score, novelty_explanation, and
analysis_classification back into the executor run's metadata column.
"""

import argparse
import json
import math
import random
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from _rubric import (
    DEFAULT_RUBRIC,
    is_novel_score,
    level_range,
    load_rubric,
)


def compute_quality_metrics(
    ideas: list[dict], total_executors: int, direction: str
) -> dict:
    """Compute quality metrics from scored ideas."""
    scores = [i["score"] for i in ideas]
    if not scores:
        return {"best": None, "top5_mean": None, "top10_mean": None,
                "median": None, "success_rate": 0.0,
                "total_executors": total_executors, "valid_count": 0, "scores": []}

    # Sort: best first
    if direction == "lower_is_better":
        sorted_scores = sorted(scores)
    else:
        sorted_scores = sorted(scores, reverse=True)

    n = len(sorted_scores)
    median = sorted_scores[n // 2] if n % 2 == 1 else (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2

    return {
        "best": sorted_scores[0],
        "top5_mean": sum(sorted_scores[:5]) / min(5, n),
        "top10_mean": sum(sorted_scores[:10]) / min(10, n),
        "median": median,
        "success_rate": round(n / total_executors, 4) if total_executors > 0 else 0.0,
        "total_executors": total_executors,
        "valid_count": n,
        "scores": sorted_scores,
    }


def compute_diversity_metrics(classified_ideas: list[dict], max_pairs: int = 200) -> dict:
    """Compute diversity metrics from classified ideas."""
    all_technique_tags: list[list[str]] = []
    component_counter: Counter = Counter()
    approach_counter: Counter = Counter()
    tag_counter: Counter = Counter()

    for idea in classified_ideas:
        cls = idea.get("classification", {})
        tags = cls.get("technique_tags", [])
        all_technique_tags.append(tags)

        for c in cls.get("components", []):
            component_counter[c] += 1
        for a in cls.get("approaches", []):
            approach_counter[a] += 1
        for t in tags:
            tag_counter[t] += 1

    unique_techniques = len(tag_counter)

    # Shannon entropy over tag frequency
    total_tag_occurrences = sum(tag_counter.values())
    entropy = 0.0
    if total_tag_occurrences > 0:
        for count in tag_counter.values():
            p = count / total_tag_occurrences
            if p > 0:
                entropy -= p * math.log2(p)

    # Mean pairwise Jaccard distance
    n = len(all_technique_tags)
    if n < 2:
        mean_jaccard_distance = 0.0
    else:
        if n * (n - 1) // 2 > max_pairs:
            pairs = set()
            while len(pairs) < max_pairs:
                i, j = random.sample(range(n), 2)
                pairs.add((min(i, j), max(i, j)))
        else:
            pairs = {(i, j) for i in range(n) for j in range(i + 1, n)}

        distances = []
        for i, j in pairs:
            set_i = set(all_technique_tags[i])
            set_j = set(all_technique_tags[j])
            union = set_i | set_j
            if not union:
                distances.append(1.0)
            else:
                jaccard_sim = len(set_i & set_j) / len(union)
                distances.append(1.0 - jaccard_sim)
        mean_jaccard_distance = sum(distances) / len(distances) if distances else 0.0

    return {
        "unique_technique_count": unique_techniques,
        "technique_entropy": round(entropy, 4),
        "component_distribution": dict(component_counter.most_common()),
        "approach_distribution": dict(approach_counter.most_common()),
        "mean_pairwise_jaccard_distance": round(mean_jaccard_distance, 4),
    }


def compute_novelty_metrics(
    classified_ideas: list[dict], rubric: dict, direction: str
) -> dict:
    """Compute novelty metrics from classified ideas under the active rubric.

    Distribution buckets cover the rubric's full score range (entries with
    zero counts are kept so plotting is consistent). `top5_quality_mean_novelty`
    / `bottom5_quality_mean_novelty` use the metric `direction` to pick best/
    worst quality — the novelty interpretation is rubric-dependent.
    """
    novelty_scores: list[int] = []
    score_novelty_pairs: list[tuple[float, int]] = []
    mechanism_tags: list[str] = []
    novel_count = 0  # count on the "novel" side of the rubric

    for idea in classified_ideas:
        nov = idea.get("novelty", {}) or {}
        ns = nov.get("score")
        if ns is not None:
            novelty_scores.append(ns)
            if is_novel_score(rubric, ns):
                novel_count += 1
            if idea.get("score") is not None:
                score_novelty_pairs.append((idea["score"], ns))
        tag = nov.get("mechanism_tag")
        if tag:
            mechanism_tags.append(tag)

    if not novelty_scores:
        return {"mean_score": None, "distribution": {},
                "mechanism_tag_distribution": {},
                "novel_count": 0,
                "novel_fraction": None,
                "top5_quality_mean_novelty": None,
                "bottom5_quality_mean_novelty": None}

    raw_dist = Counter(novelty_scores)
    # Render distribution over full range (zero-fill missing buckets)
    dist = {lvl: raw_dist.get(int(lvl), 0) for lvl in level_range(rubric)}
    tag_dist = Counter(mechanism_tags)

    if direction == "lower_is_better":
        score_novelty_pairs.sort(key=lambda x: x[0])
    else:
        score_novelty_pairs.sort(key=lambda x: x[0], reverse=True)

    top5_novelty = [ns for _, ns in score_novelty_pairs[:5]] if len(score_novelty_pairs) >= 5 else [ns for _, ns in score_novelty_pairs]
    bottom5_novelty = [ns for _, ns in score_novelty_pairs[-5:]] if len(score_novelty_pairs) >= 5 else [ns for _, ns in score_novelty_pairs]

    return {
        "mean_score": round(sum(novelty_scores) / len(novelty_scores), 4),
        "distribution": dist,
        "mechanism_tag_distribution": dict(sorted(tag_dist.items())),
        "novel_count": novel_count,
        "novel_fraction": round(novel_count / len(novelty_scores), 4),
        "top5_quality_mean_novelty": round(sum(top5_novelty) / len(top5_novelty), 4) if top5_novelty else None,
        "bottom5_quality_mean_novelty": round(sum(bottom5_novelty) / len(bottom5_novelty), 4) if bottom5_novelty else None,
    }


# ---------------------------------------------------------------------------
# Store write-back
# ---------------------------------------------------------------------------

def write_back_to_store(
    extracted: dict,
    classifications: list[dict],
    rubric_name: str,
) -> int:
    """Write novelty + classification data into the store's metadata column.

    Stamps `rubric` (the new field) on every updated row. Drops the legacy
    `rubric_version` heuristic — the rubric is now an explicit input, not
    inferred from observed scores.

    Returns the number of rows updated.
    """
    db_path_str = extracted.get("store_db")
    experiment_id = extracted.get("experiment_id")
    if not db_path_str or not experiment_id:
        return 0

    db_path = Path(db_path_str)
    if not db_path.exists():
        return 0

    cls_lookup = {c["executor_id"]: c for c in classifications}
    updated = 0

    try:
        with sqlite3.connect(str(db_path), timeout=30) as conn:
            conn.row_factory = sqlite3.Row
            for eid, cls_data in cls_lookup.items():
                row = conn.execute(
                    "SELECT metadata FROM runs WHERE experiment_id = ? AND run_id = ?",
                    (experiment_id, eid),
                ).fetchone()
                if row is None:
                    continue

                metadata = json.loads(row["metadata"]) if row["metadata"] else {}

                # Write novelty data
                nov = cls_data.get("novelty", {}) or {}
                if "score" in nov:
                    metadata["novelty_score"] = nov["score"]
                if "label" in nov:
                    metadata["novelty_label"] = nov["label"]
                if "mechanism_tag" in nov:
                    metadata["novelty_mechanism_tag"] = nov["mechanism_tag"]
                if "explanation" in nov:
                    metadata["novelty_explanation"] = nov["explanation"]
                if "evidence" in nov:
                    metadata["novelty_evidence"] = nov["evidence"]
                # Stamp the rubric explicitly so consumers can interpret the
                # novelty_score correctly (range + direction depend on it).
                metadata["rubric"] = rubric_name

                # Write classification data
                cls = cls_data.get("classification", {})
                if cls:
                    metadata["analysis_classification"] = cls

                conn.execute(
                    "UPDATE runs SET metadata = ? WHERE experiment_id = ? AND run_id = ?",
                    (json.dumps(metadata), experiment_id, eid),
                )
                updated += 1

        print(f"  Store write-back: updated {updated} rows in {db_path}")
    except Exception as exc:
        print(f"  Store write-back failed: {exc}", file=sys.stderr)

    return updated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(run_dir: str, rubric_name: str | None = None) -> None:
    run_path = Path(run_dir).resolve()
    extracted_path = run_path / "extracted_ideas.json"
    classifications_path = run_path / "classifications.json"
    output_path = run_path / "analysis_cache.json"

    if output_path.exists():
        print(f"Already aggregated: {output_path}")
        return

    if not extracted_path.exists():
        print(f"Error: {extracted_path} not found. Run extract_run.py first.", file=sys.stderr)
        sys.exit(1)
    if not classifications_path.exists():
        print(f"Error: {classifications_path} not found. Run classification agents first.", file=sys.stderr)
        sys.exit(1)

    extracted = json.loads(extracted_path.read_text())
    classifications = json.loads(classifications_path.read_text())

    # Resolve rubric: explicit flag > extraction stamp > default.
    chosen = rubric_name or extracted.get("rubric") or DEFAULT_RUBRIC
    rubric = load_rubric(chosen)
    print(f"Using rubric: {rubric['name']}  (range {rubric['score_range']}, "
          f"direction {rubric['direction']})")

    # Build lookup: executor_id → classification
    cls_lookup = {c["executor_id"]: c for c in classifications}

    # Merge ideas with classifications
    merged_ideas = []
    for idea in extracted["ideas"]:
        eid = idea["executor_id"]
        cls_data = cls_lookup.get(eid, {})
        merged_ideas.append({
            "executor_id": eid,
            "score": idea["score"],
            "classification": cls_data.get("classification", {}),
            "novelty": cls_data.get("novelty", {}),
        })

    direction = extracted.get("metric_direction", "lower_is_better")

    # Compute metrics
    quality = compute_quality_metrics(merged_ideas, extracted["total_executors"], direction)
    diversity = compute_diversity_metrics(merged_ideas)
    novelty = compute_novelty_metrics(merged_ideas, rubric, direction)

    # Collect verified ideas (those that went through novelty verification pass)
    verified_ideas = []
    for cls_data in classifications:
        nov = cls_data.get("novelty", {})
        if nov.get("verified"):
            verified_ideas.append({
                "executor_id": cls_data["executor_id"],
                "original_score": nov.get("original_score"),
                "verified_score": nov.get("score"),
                "explanation": nov.get("explanation", ""),
            })

    cache = {
        "run_id": extracted["run_id"],
        "run_dir": extracted["run_dir"],
        "strategy_type": extracted.get("strategy_type", "unknown"),
        "task_name": extracted.get("task_name"),
        "rubric": rubric["name"],
        "rubric_score_range": rubric["score_range"],
        "rubric_direction": rubric["direction"],
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "metric_name": extracted.get("metric_name", "score"),
        "metric_direction": direction,
        "ideas": merged_ideas,
        "metrics": {
            "quality": quality,
            "diversity": diversity,
            "novelty": novelty,
        },
        "verified_ideas": verified_ideas,
    }

    output_path.write_text(json.dumps(cache, indent=2))
    print(f"Analysis cache written → {output_path}")
    print(f"  Quality: best={quality['best']}, top5={quality['top5_mean']}, success={quality['success_rate']}")
    print(f"  Diversity: {diversity['unique_technique_count']} techniques, entropy={diversity['technique_entropy']}")
    print(f"  Novelty: mean={novelty['mean_score']}, dist={novelty['distribution']}, "
          f"novel_fraction={novelty['novel_fraction']}")

    # Write classifications back to store if available
    write_back_to_store(extracted, classifications, rubric["name"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate classifications into analysis_cache.json")
    parser.add_argument("run_dir", help="Path to the run directory")
    parser.add_argument("--rubric", default=None,
                        help=f"Rubric name (overrides extraction stamp; default: {DEFAULT_RUBRIC})")
    args = parser.parse_args()
    main(args.run_dir, rubric_name=args.rubric)
