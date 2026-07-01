"""Data loading utilities for MAP-Elites experiment analysis.

Loads run data from the heuresis SQLite store and parses
QD metadata into convenient DataFrame columns.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

# Paths relative to heuresis/
RESEARCH_AGENT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = RESEARCH_AGENT_ROOT / "runs" / "nanogpt" / "store.db"
LEGACY_DB_PATH = RESEARCH_AGENT_ROOT / "runs" / "_legacy" / "store.db"
QD_SRC_PATH = RESEARCH_AGENT_ROOT / "src"


def load_runs(
    experiment_id: str,
    db_path: Path = DB_PATH,
    run_type: str = "executor",
) -> pd.DataFrame:
    """Load executor runs for an experiment from the SQLite store.

    Parses the JSON ``metadata`` column and extracts QD features into
    top-level columns: ``mod_target``, ``int_style``, ``mod_target_name``,
    ``int_style_name``.

    Returns a DataFrame sorted by iteration.
    """
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(
        "SELECT * FROM runs WHERE experiment_id = ? AND run_type = ? ORDER BY iteration",
        conn,
        params=(experiment_id, run_type),
    )
    conn.close()

    # Parse metadata JSON
    def _parse_meta(row: pd.Series) -> pd.Series:
        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        qd = meta.get("qd_features", {})
        names = meta.get("feature_names", {})
        return pd.Series({
            "mod_target": qd.get("modification_target"),
            "int_style": qd.get("intervention_style"),
            "mod_target_name": names.get("modification_target"),
            "int_style_name": names.get("intervention_style"),
        })

    if not df.empty:
        extras = df.apply(_parse_meta, axis=1)
        df = pd.concat([df, extras], axis=1)

    return df


def filter_scores(
    df: pd.DataFrame,
    min_bpb: float = 0.5,
    max_bpb: float = 10.0,
) -> pd.DataFrame:
    """Add a ``valid_filtered`` column excluding diverged/invalid scores.

    Keeps original ``valid`` column intact. ``valid_filtered`` is True only
    when the run is valid AND the score falls within [min_bpb, max_bpb].
    """
    df = df.copy()
    valid_mask = df["valid"].astype(bool) & df["score"].notna()
    valid_mask &= (df["score"] >= min_bpb) & (df["score"] <= max_bpb)
    df["valid_filtered"] = valid_mask
    return df


def classify_runs_posthoc(
    df: pd.DataFrame,
    runs_dir: Path,
    *,
    use_llm: bool = True,
    api_keys_file: Path | None = None,
) -> pd.DataFrame:
    """Post-hoc classify runs that lack QD features using the nanoGPT classifier.

    Reads each executor's ``notes.md`` (or falls back to the corresponding
    ``ideator_NNN/idea.md``) and classifies via the task's feature classifier
    (LLM with keyword fallback). The first feature axis fills ``mod_target``,
    the second fills ``int_style``; labels come from each axis's ``bin_names``.

    Only fills in rows where ``mod_target`` is NaN.
    """
    import sys

    if str(QD_SRC_PATH) not in sys.path:
        sys.path.insert(0, str(QD_SRC_PATH))

    from heuresis.qd import feature_namer
    from heuresis.tasks.nanogpt.features import make_classifier

    classifier = make_classifier(use_llm=use_llm, api_keys_file=api_keys_file)
    namer = feature_namer(classifier.features)
    target_axis = classifier.features[0].name
    style_axis = classifier.features[1].name

    df = df.copy()
    needs_classification = df["mod_target"].isna()
    total = needs_classification.sum()

    for count, idx in enumerate(df.index[needs_classification], 1):
        run_id = df.loc[idx, "run_id"]
        iteration = int(df.loc[idx, "iteration"])

        # Try notes.md first, then ideator idea.md
        notes_path = runs_dir / run_id / "notes.md"
        idea_path = runs_dir / f"ideator_{iteration:03d}" / "idea.md"
        # Some runs use ideator_N (no zero-padding)
        idea_path_alt = runs_dir / f"ideator_{iteration}" / "idea.md"

        plan = ""
        if notes_path.exists():
            plan = notes_path.read_text(errors="replace")
        elif idea_path.exists():
            plan = idea_path.read_text(errors="replace")
        elif idea_path_alt.exists():
            plan = idea_path_alt.read_text(errors="replace")

        if not plan:
            continue

        features = classifier.classify(plan)
        names = namer(features)

        df.loc[idx, "mod_target"] = features[target_axis]
        df.loc[idx, "int_style"] = features[style_axis]
        df.loc[idx, "mod_target_name"] = names[target_axis]
        df.loc[idx, "int_style_name"] = names[style_axis]

        if use_llm and count % 10 == 0:
            print(f"    classified {count}/{total}...")

    return df
