"""Workspace parsing — extract scores, tokens, and run metadata."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def parse_workspace(
    workspace: Path,
    *,
    lower_is_better: bool = False,
) -> dict[str, Any]:
    """Read a completed workspace and return everything worth logging.

    Combines agent log stats (tokens, cost) with grading attempt history
    (scores, files submitted). Returns a single dict suitable for passing
    to ``exp.save(..., metadata=info)``.
    """
    info: dict[str, Any] = {}

    info.update(_parse_agent_log(workspace / "agent.log"))
    info.update(_parse_attempts(workspace / "attempts", lower_is_better=lower_is_better))

    return info


def _parse_agent_log(log_path: Path) -> dict[str, Any]:
    """Parse token counts and cost from a JSONL agent log."""
    stats: dict[str, Any] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "total_cost": 0.0,
    }
    if not log_path.exists():
        return stats
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") != "step_finish":
                    continue
                part = event.get("part", event)
                stats["total_cost"] += part.get("cost", 0.0)
                tokens = part.get("tokens", {})
                cache = tokens.get("cache", {})
                stats["input_tokens"] += tokens.get("input", 0)
                stats["output_tokens"] += tokens.get("output", 0)
                stats["reasoning_tokens"] += tokens.get("reasoning", 0)
                stats["cache_read_tokens"] += cache.get("read", 0)
                stats["cache_write_tokens"] += cache.get("write", 0)
    except OSError:
        logger.warning("Could not read agent log at %s", log_path)
    return stats


def _parse_attempts(
    attempts_dir: Path,
    *,
    lower_is_better: bool = False,
) -> dict[str, Any]:
    """Parse grading attempts from the attempts/ directory."""
    result: dict[str, Any] = {
        "attempts": [],
        "best_score": None,
        "num_attempts": 0,
        "valid": False,
    }

    if not attempts_dir.is_dir():
        return result

    attempt_dirs = sorted(
        [d for d in attempts_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    attempts: list[dict[str, Any]] = []
    best_score: float | None = None
    any_valid = False

    for attempt_dir in attempt_dirs:
        result_file = attempt_dir / "result.json"
        if not result_file.exists():
            continue

        try:
            data = json.loads(result_file.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not parse %s", result_file)
            continue

        timestamp_str = attempt_dir.name
        try:
            dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S_%f")
            timestamp = dt.isoformat()
        except ValueError:
            timestamp = timestamp_str

        score = data.get("score")
        valid = data.get("valid", False)

        entry = {
            "timestamp": timestamp,
            "score": score,
            "valid": valid,
            "files": data.get("files", []),
            "details": data.get("details", {}),
        }
        attempts.append(entry)

        if valid:
            any_valid = True
            if score is not None:
                is_better = (
                    best_score is None
                    or (lower_is_better and score < best_score)
                    or (not lower_is_better and score > best_score)
                )
                if is_better:
                    best_score = score

    result["attempts"] = attempts
    result["num_attempts"] = len(attempts)
    result["best_score"] = best_score
    result["valid"] = any_valid

    return result
