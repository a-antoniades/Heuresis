#!/usr/bin/env python3
"""Extract ideas and scores from a research-agent run directory.

Usage: python extract_run.py <run_dir> [--rubric <name>]

Produces <run_dir>/extracted_ideas.json with task preamble, metric info,
and all ideas that have valid scores. The rubric stamp tells downstream
classification + aggregation which scoring scheme to apply.

Tries the SQLite store first (store.db in the run dir or project root),
falls back to filesystem parsing if unavailable.
"""

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

from _rubric import DEFAULT_RUBRIC, load_rubric


# ---------------------------------------------------------------------------
# Novelty anchor (task-relativized reference frame for anchor-using rubrics)
# ---------------------------------------------------------------------------

def _novelty_anchor_markdown(task_name: str | None) -> str:
    """Return the task's novelty_anchor as a markdown block, or empty string.

    Looked up from ``src/heuresis/tasks/<task_name>/task_config.yaml``.
    Only injected when the active rubric has ``uses_anchor: true`` (the caller
    decides). If the helper cannot be imported, returns "". Absence is
    non-fatal: an anchor-using rubric will fall back to the implicit "all of
    ML literature" reference frame.
    """
    if not task_name:
        return ""
    try:
        # Locate the repo root by walking up for pyproject.toml.
        here = Path(__file__).resolve()
        repo = None
        for p in here.parents:
            if (p / "pyproject.toml").is_file():
                repo = p
                break
        if repo is None:
            return ""
        sys.path.insert(0, str(repo / "src"))
        try:
            from heuresis.tasks.config import (
                novelty_anchor_markdown, task_dir,
            )
            return novelty_anchor_markdown(task_dir(task_name))
        finally:
            sys.path.pop(0)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Store-based extraction
# ---------------------------------------------------------------------------

def _find_store_db(run_path: Path) -> Path | None:
    """Locate store.db: check run dir, then walk up to project root."""
    # Run-local store
    local = run_path / "store.db"
    if local.exists():
        return local
    # Walk up to find project-level store
    for parent in run_path.parents:
        candidate = parent / "store.db"
        if candidate.exists():
            return candidate
        # Stop at project root (has pyproject.toml or .git)
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            break
    return None


def _detect_experiment_id(run_path: Path, db_path: Path) -> str | None:
    """Match run directory to an experiment_id in the store."""
    run_name = run_path.name
    try:
        with sqlite3.connect(str(db_path), timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            # Try exact match on directory
            row = conn.execute(
                "SELECT experiment_id FROM experiments WHERE dir = ?",
                (str(run_path),),
            ).fetchone()
            if row:
                return row["experiment_id"]
            # Try matching by experiment_id prefix (run dir name often IS the id)
            row = conn.execute(
                "SELECT experiment_id FROM experiments WHERE experiment_id = ?",
                (run_name,),
            ).fetchone()
            if row:
                return row["experiment_id"]
            # Try substring match
            row = conn.execute(
                "SELECT experiment_id FROM experiments WHERE dir LIKE ?",
                (f"%{run_name}%",),
            ).fetchone()
            if row:
                return row["experiment_id"]
    except Exception:
        pass
    return None


def _lookup_task(db_path: Path, experiment_id: str) -> str | None:
    try:
        with sqlite3.connect(str(db_path), timeout=10) as conn:
            row = conn.execute(
                "SELECT task FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
            return row[0] if row else None
    except Exception:
        return None


def extract_from_store(
    run_path: Path, db_path: Path, experiment_id: str, rubric: dict
) -> dict | None:
    """Extract ideas and scores from the SQLite store."""
    try:
        with sqlite3.connect(str(db_path), timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            # Fetch all valid scored runs; sorting happens after metric detection
            rows = conn.execute(
                """SELECT run_id, idea, score, valid, metadata
                   FROM runs
                   WHERE experiment_id = ? AND run_type = 'executor'
                   AND score IS NOT NULL AND valid = 1""",
                (experiment_id,),
            ).fetchall()

        if not rows:
            return None

        # Count total executors
        with sqlite3.connect(str(db_path), timeout=10) as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE experiment_id = ? AND run_type = 'executor'",
                (experiment_id,),
            ).fetchone()[0]

        ideas = []
        for row in rows:
            idea_text = row["idea"] or ""
            # Truncate to strategy section if present
            for marker in ("## Strategy", "## Approach"):
                idx = idea_text.find(marker)
                if idx >= 0:
                    idea_text = idea_text[idx:]
                    break
            idea_text = idea_text[:3000]

            metadata = json.loads(row["metadata"]) if row["metadata"] else {}

            entry: dict = {
                "executor_id": row["run_id"],
                "score": row["score"],
                "idea_text": idea_text,
            }
            # Include existing classification + novelty if present in store
            if "novelty_score" in metadata:
                entry["novelty_score"] = metadata["novelty_score"]
                entry["novelty_explanation"] = metadata.get("novelty_explanation", "")
                entry["novelty_evidence"] = metadata.get("novelty_evidence", [])
            if "analysis_classification" in metadata:
                entry["classification"] = metadata["analysis_classification"]

            ideas.append(entry)

        # Get task preamble from first executor's prompt file (store doesn't have it).
        # Accept both legacy "executor_NNN/" and post-refactor "exec_NNN/" layouts.
        task_preamble = ""
        for d in sorted(run_path.iterdir()):
            if d.is_dir() and (d.name.startswith("executor_") or d.name.startswith("exec_")):
                prompt_file = d / ".prompt.txt"
                if prompt_file.exists():
                    task_preamble = _extract_task_preamble(
                        prompt_file.read_text(errors="replace")
                    )
                    if task_preamble:
                        break

        metric_name, metric_direction = _detect_metric(task_preamble)

        # Sort: best first
        reverse = metric_direction == "higher_is_better"
        ideas.sort(key=lambda x: x["score"], reverse=reverse)

        # Task-anchored novelty reference frame — only injected when the
        # active rubric uses it (e.g. nanogpt_1to4). Web-search-only rubrics
        # like gupta_pruthi_2025 deliberately skip the anchor.
        task_name = _lookup_task(db_path, experiment_id)
        if rubric.get("uses_anchor"):
            anchor_md = _novelty_anchor_markdown(task_name)
            if anchor_md:
                task_preamble = (task_preamble + "\n\n" + anchor_md).strip()

        return {
            "run_id": run_path.name,
            "run_dir": str(run_path),
            "strategy_type": _detect_strategy(run_path.name),
            "task_name": task_name,
            "task_preamble": task_preamble,
            "metric_name": metric_name,
            "metric_direction": metric_direction,
            "rubric": rubric["name"],
            "total_executors": total,
            "valid_count": len(ideas),
            "source": "store",
            "store_db": str(db_path),
            "experiment_id": experiment_id,
            "ideas": ideas,
        }
    except Exception as exc:
        print(f"Store extraction failed: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Filesystem-based extraction (fallback)
# ---------------------------------------------------------------------------

def _extract_score(run_log: Path) -> float | None:
    """Extract the last val_bpb (or similar metric) from a run log."""
    if not run_log.exists():
        return None
    text = run_log.read_text(errors="replace")
    matches = re.findall(r"val_bpb[=: ]+([0-9]+\.[0-9]+)", text)
    if matches:
        return float(matches[-1])
    return None


def _extract_task_preamble(prompt_text: str) -> str:
    """Extract task context from a prompt (up to the approach section)."""
    for marker in ["## Approach", "## Strategy"]:
        idx = prompt_text.find(marker)
        if idx > 0:
            return prompt_text[:idx].strip()
    return prompt_text[:1500].strip()


def _extract_idea_text(prompt_text: str) -> str:
    """Extract the idea section from an executor prompt."""
    for marker in ["## Strategy", "## Approach"]:
        idx = prompt_text.find(marker)
        if idx >= 0:
            idea = prompt_text[idx:]
            second_start = idea.find("You are an expert", 10)
            if second_start > 0:
                idea = idea[:second_start].strip()
            return idea[:3000]
    return prompt_text[:3000]


def _detect_metric(preamble: str) -> tuple[str, str]:
    """Detect metric name and direction from the task preamble."""
    text = preamble.lower()
    if "val_bpb" in text:
        return "val_bpb", "lower_is_better"
    if "mean_log_gap" in text:
        return "mean_log_gap", "lower_is_better"
    if "accuracy" in text:
        return "accuracy", "higher_is_better"
    if "rmse" in text or "mse" in text:
        return "rmse", "lower_is_better"
    return "score", "higher_is_better"


def _detect_strategy(dirname: str) -> str:
    """Detect search strategy type from directory name."""
    name = dirname.lower()
    if "island" in name:
        return "island"
    if "mapelites" in name or "map-elites" in name or "map_elites" in name:
        return "mapelites"
    if "omni-epic" in name or "omni_epic" in name or "omniepic" in name:
        return "omni-epic"
    if "curiosity" in name:
        return "curiosity"
    if "linear" in name:
        return "linear"
    if "greedy" in name:
        return "greedy"
    if "baseline" in name:
        return "baseline"
    return "unknown"


def extract_from_filesystem(run_path: Path, rubric: dict) -> dict:
    """Extract ideas and scores by parsing executor directories."""
    executor_dirs = sorted(
        [d for d in run_path.iterdir()
         if d.is_dir() and d.name.startswith("executor_") and "_broken" not in d.name],
        key=lambda d: d.name,
    )

    if not executor_dirs:
        print(f"Error: No executor directories found in {run_path}", file=sys.stderr)
        sys.exit(1)

    # Task preamble from first prompt
    task_preamble = ""
    for d in executor_dirs:
        prompt_file = d / ".prompt.txt"
        if prompt_file.exists():
            task_preamble = _extract_task_preamble(prompt_file.read_text(errors="replace"))
            break

    metric_name, metric_direction = _detect_metric(task_preamble)

    ideas = []
    for d in executor_dirs:
        score = _extract_score(d / "run.log")
        if score is None:
            continue

        prompt_file = d / ".prompt.txt"
        if not prompt_file.exists():
            continue

        prompt_text = prompt_file.read_text(errors="replace")
        idea_text = _extract_idea_text(prompt_text)

        ideas.append({
            "executor_id": d.name,
            "score": score,
            "idea_text": idea_text,
        })

    reverse = metric_direction == "higher_is_better"
    ideas.sort(key=lambda x: x["score"], reverse=reverse)

    # Task-anchored novelty reference frame — only injected when the active
    # rubric uses it. No task metadata in filesystem fallback mode, so guess
    # "nanogpt" when the run name looks like nanogpt.
    task_name = None
    if "nanogpt" in run_path.name.lower():
        task_name = "nanogpt"
    if rubric.get("uses_anchor"):
        anchor_md = _novelty_anchor_markdown(task_name)
        if anchor_md:
            task_preamble = (task_preamble + "\n\n" + anchor_md).strip()

    return {
        "run_id": run_path.name,
        "run_dir": str(run_path),
        "strategy_type": _detect_strategy(run_path.name),
        "task_name": task_name,
        "task_preamble": task_preamble,
        "metric_name": metric_name,
        "metric_direction": metric_direction,
        "rubric": rubric["name"],
        "total_executors": len(executor_dirs),
        "valid_count": len(ideas),
        "source": "filesystem",
        "ideas": ideas,
    }


# ---------------------------------------------------------------------------
# Staleness check
# ---------------------------------------------------------------------------

_CLASSIFICATION_KEYS = (
    "classification",
    "novelty_score",
    "novelty_explanation",
    "novelty_evidence",
)


def _count_current_valid(
    run_path: Path, db_path: Path | None, experiment_id: str | None
) -> int | None:
    """Cheaply count currently-valid scored executors.

    Uses the store when available, otherwise scans executor run.log files for
    a val_bpb match. Returns None only if we can't determine the count.
    """
    if db_path and experiment_id:
        try:
            with sqlite3.connect(str(db_path), timeout=10) as conn:
                row = conn.execute(
                    """SELECT COUNT(*) FROM runs
                       WHERE experiment_id = ? AND run_type = 'executor'
                       AND score IS NOT NULL AND valid = 1""",
                    (experiment_id,),
                ).fetchone()
                if row is not None:
                    return int(row[0])
        except Exception:
            pass
    # Filesystem fallback
    count = 0
    for d in sorted(run_path.iterdir()):
        if not (d.is_dir() and d.name.startswith("executor_") and "_broken" not in d.name):
            continue
        run_log = d / "run.log"
        if not run_log.exists():
            continue
        try:
            if re.search(r"val_bpb[=: ]+[0-9]+\.[0-9]+", run_log.read_text(errors="replace")):
                count += 1
        except Exception:
            continue
    return count


def _merge_existing_classifications(
    result: dict, old_extracted: dict | None, run_path: Path
) -> int:
    """Carry classification/novelty fields forward from prior work.

    Sources, in priority order:
      1. Old extracted_ideas.json (may already contain classifications)
      2. classifications.json (authoritative Phase 2 output if present)

    Store-backed extraction already pulls classifications from the store for
    ideas that were written back in Phase 3; this helper covers the gap for
    Phase 2 output that hasn't been aggregated yet and filesystem-only runs.
    Returns the number of ideas that picked up at least one preserved field.
    """
    merged: dict[str, dict] = {}
    if old_extracted:
        for idea in old_extracted.get("ideas", []):
            merged[idea["executor_id"]] = {
                k: idea[k] for k in _CLASSIFICATION_KEYS if k in idea
            }

    classifications_path = run_path / "classifications.json"
    if classifications_path.exists():
        try:
            raw = json.loads(classifications_path.read_text())
        except Exception:
            raw = []
        for entry in raw:
            eid = entry.get("executor_id")
            if not eid:
                continue
            slot = merged.setdefault(eid, {})
            cls = entry.get("classification")
            if cls:
                slot["classification"] = cls
            nov = entry.get("novelty") or {}
            if "score" in nov:
                slot["novelty_score"] = nov["score"]
            if "explanation" in nov:
                slot["novelty_explanation"] = nov["explanation"]
            if "evidence" in nov:
                slot["novelty_evidence"] = nov["evidence"]

    picked_up = 0
    for new_idea in result["ideas"]:
        preserved = merged.get(new_idea["executor_id"])
        if not preserved:
            continue
        touched = False
        for key, value in preserved.items():
            if key not in new_idea:
                new_idea[key] = value
                touched = True
        if touched:
            picked_up += 1
    return picked_up


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(run_dir: str, rubric_name: str = DEFAULT_RUBRIC) -> None:
    run_path = Path(run_dir).resolve()
    if not run_path.is_dir():
        print(f"Error: {run_path} is not a directory", file=sys.stderr)
        sys.exit(1)

    rubric = load_rubric(rubric_name)
    print(f"Using rubric: {rubric['name']}  (range {rubric['score_range']}, "
          f"direction {rubric['direction']}, anchor {rubric.get('uses_anchor', False)})")

    output_path = run_path / "extracted_ideas.json"

    # Resolve store + experiment id up-front so we can reuse them for the
    # staleness check and the extraction itself.
    db_path = _find_store_db(run_path)
    experiment_id = _detect_experiment_id(run_path, db_path) if db_path else None

    # Check whether an existing extraction is still up-to-date with the live
    # run. A run that's still going will have more valid executors on disk (or
    # in the store) than the cache captured; in that case we re-extract,
    # preserving previously-classified ideas so Phase 2 only processes the gap.
    old_extracted: dict | None = None
    if output_path.exists():
        try:
            old_extracted = json.loads(output_path.read_text())
        except Exception as exc:
            print(f"Warning: could not read existing cache ({exc}); re-extracting")
            old_extracted = None
        old_count = (old_extracted or {}).get("valid_count", 0)
        current_count = _count_current_valid(run_path, db_path, experiment_id)
        if current_count is None:
            print(f"Already extracted: {output_path} (could not verify freshness)")
            return
        if current_count <= old_count:
            print(f"Already extracted: {output_path} "
                  f"({old_count} valid ideas, up-to-date)")
            return
        print(f"Cache is stale: {old_count} ideas cached, {current_count} on disk "
              f"\u2192 re-extracting and preserving prior classifications")
        # Downstream caches are derived from this extraction, so drop them.
        analysis_cache = run_path / "analysis_cache.json"
        if analysis_cache.exists():
            analysis_cache.unlink()
            print(f"  Removed stale {analysis_cache.name}")

    # Try store first
    result = None
    if db_path and experiment_id:
        print(f"Found store: {db_path} (experiment: {experiment_id})")
        result = extract_from_store(run_path, db_path, experiment_id, rubric)

    # Fall back to filesystem
    if result is None:
        print("No store found or store extraction failed, using filesystem")
        result = extract_from_filesystem(run_path, rubric)

    # Preserve prior classification work so Phase 2 only processes new ideas.
    preserved = _merge_existing_classifications(result, old_extracted, run_path)
    if preserved:
        print(f"  Preserved classifications for {preserved} previously-processed ideas")

    output_path.write_text(json.dumps(result, indent=2))
    print(f"Extracted {result['valid_count']}/{result['total_executors']} ideas"
          f" with scores ({result.get('source', 'unknown')}) \u2192 {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ideas from a research-agent run")
    parser.add_argument("run_dir", help="Path to the run directory")
    parser.add_argument("--rubric", default=DEFAULT_RUBRIC,
                        help=f"Rubric name from rubrics/<name>.md (default: {DEFAULT_RUBRIC})")
    args = parser.parse_args()
    main(args.run_dir, rubric_name=args.rubric)
