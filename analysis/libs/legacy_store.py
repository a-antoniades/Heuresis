"""Shim for reading pre-foundation-refactor store.db schema.

The old schema had many first-class columns (iteration, run_type, idea,
duration_s, etc.) plus run_files, archive_events tables. We restored most
columns in the new schema but there are edge cases (e.g., old `verification`
column, no `novelty_reviews` table). This shim presents the old DB as
RunRecord objects for backward-compat analysis.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from heuresis.models import RunRecord


class LegacyStore:
    """Read-only view of a pre-refactor SQLite store.

    Default path: runs/_legacy/store.db
    """

    def __init__(self, db_path: Path | str = "runs/_legacy/store.db") -> None:
        self._db = Path(db_path)
        if not self._db.exists():
            raise FileNotFoundError(f"No legacy store at {self._db}")

    def experiments(self) -> list[dict[str, Any]]:
        with sqlite3.connect(str(self._db)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT experiment_id, name, task, started_at, dir FROM experiments "
                "ORDER BY started_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def runs(self, experiment_id: str) -> list[RunRecord]:
        with sqlite3.connect(str(self._db)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM runs WHERE experiment_id = ? ORDER BY iteration",
                (experiment_id,),
            ).fetchall()
        return [self._to_record(r) for r in rows]

    def run_files(self, experiment_id: str, run_id: str) -> dict[str, str]:
        try:
            with sqlite3.connect(str(self._db)) as conn:
                rows = conn.execute(
                    "SELECT filename, content FROM run_files "
                    "WHERE experiment_id = ? AND run_id = ?",
                    (experiment_id, run_id),
                ).fetchall()
            return {fn: c for fn, c in rows}
        except sqlite3.OperationalError:
            return {}  # old db may not have run_files

    def _to_record(self, row: sqlite3.Row) -> RunRecord:
        meta_raw = row["metadata"] if "metadata" in row.keys() else ""
        metadata = json.loads(meta_raw) if meta_raw else {}
        pids_raw = row["parent_ids"] if "parent_ids" in row.keys() else ""
        parent_ids = [p for p in (pids_raw or "").split(",") if p]
        return RunRecord(
            experiment_id=row["experiment_id"],
            run_id=row["run_id"],
            score=row["score"],
            workspace=Path(row["workspace_path"] or "."),
            metadata=metadata,
            iteration=row["iteration"],
            run_type=row["run_type"],
            valid=bool(row["valid"]) if row["valid"] is not None else None,
            started_at=row["started_at"],
            duration_s=row["duration_s"],
            exit_code=row["exit_code"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            total_cost=row["total_cost"],
            parent_ids=parent_ids,
            generation=row["generation"] or 0,
            idea=row["idea"],
        )
