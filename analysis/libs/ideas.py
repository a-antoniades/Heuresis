"""Idea extraction and display utilities for experiment analysis.

Quick ways to pull idea summaries from the SQLite store and display
them as formatted tables for monitoring runs.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

from .data import DB_PATH


def _extract_strategy(idea: str) -> str:
    """Extract the '## Strategy' section content from an idea document."""
    lines = idea.split("\n")
    in_strategy = False
    result: list[str] = []
    for line in lines:
        if line.strip().lower().startswith("## strategy"):
            in_strategy = True
            # Check if content is on the same line
            after = line.strip()[len("## strategy") :].strip().lstrip(":").strip()
            if after:
                result.append(after)
            continue
        if in_strategy:
            if line.strip().startswith("## "):
                break
            if line.strip():
                result.append(line.strip())
    return " ".join(result) if result else ""


def _first_sentence(text: str, max_len: int = 150) -> str:
    """Return the first sentence, truncated to max_len."""
    # Split on sentence boundaries
    m = re.match(r"^(.+?[.!?])\s", text)
    sentence = m.group(1) if m else text
    if len(sentence) > max_len:
        return sentence[: max_len - 3] + "..."
    return sentence


def load_ideas(
    experiment_id: str | None = None,
    db_path: Path = DB_PATH,
    run_type: str = "executor",
) -> pd.DataFrame:
    """Load ideas with scores from the store.

    If *experiment_id* is None, uses the most recent experiment.

    Returns a DataFrame with columns:
        run_id, iteration, score, valid, strategy, idea_full
    sorted by iteration.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    if experiment_id is None:
        row = conn.execute(
            "SELECT experiment_id FROM experiments ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            conn.close()
            return pd.DataFrame()
        experiment_id = row["experiment_id"]

    rows = conn.execute(
        """
        SELECT run_id, iteration, score, valid, idea, duration_s
        FROM runs
        WHERE experiment_id = ? AND run_type = ?
        ORDER BY iteration
        """,
        (experiment_id, run_type),
    ).fetchall()
    conn.close()

    records = []
    for r in rows:
        idea_text = r["idea"] or ""
        strategy = _extract_strategy(idea_text)
        records.append(
            {
                "run_id": r["run_id"],
                "iteration": r["iteration"],
                "score": r["score"],
                "valid": bool(r["valid"]),
                "duration_m": round(r["duration_s"] / 60, 1) if r["duration_s"] else None,
                "strategy": _first_sentence(strategy),
                "idea_full": idea_text,
            }
        )
    return pd.DataFrame(records)


def ideas_table(
    experiment_id: str | None = None,
    db_path: Path = DB_PATH,
    sort_by: str = "score",
    top_n: int | None = None,
    show_failed: bool = True,
) -> pd.DataFrame:
    """Return a concise table of ideas suitable for display.

    Args:
        experiment_id: Experiment ID (None = latest).
        db_path: Path to SQLite store.
        sort_by: Column to sort by ("score", "iteration").
        top_n: Limit to top N rows (by score, ascending for BPB).
        show_failed: Include runs with score=None.
    """
    df = load_ideas(experiment_id, db_path)
    if df.empty:
        return df

    if not show_failed:
        df = df[df["score"].notna()]

    if sort_by == "score":
        df = df.sort_values("score", ascending=True, na_position="last")
    else:
        df = df.sort_values(sort_by)

    if top_n:
        df = df.head(top_n)

    return df[["run_id", "iteration", "score", "valid", "duration_m", "strategy"]].reset_index(
        drop=True
    )


def print_ideas(
    experiment_id: str | None = None,
    db_path: Path = DB_PATH,
    sort_by: str = "score",
    top_n: int | None = None,
    show_failed: bool = True,
) -> None:
    """Print a formatted ideas table to stdout."""
    df = ideas_table(experiment_id, db_path, sort_by, top_n, show_failed)
    if df.empty:
        print("No runs found.")
        return
    # Format score column
    df["score"] = df["score"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    df["valid"] = df["valid"].apply(lambda x: "Y" if x else "N")
    with pd.option_context("display.max_colwidth", 100, "display.width", 200):
        print(df.to_string(index=False))


def latest_experiment_id(db_path: Path = DB_PATH) -> str | None:
    """Return the most recent experiment ID."""
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT experiment_id FROM experiments ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return row[0] if row else None
