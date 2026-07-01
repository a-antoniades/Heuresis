"""SQLite result store for experiment history and analysis."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from heuresis.models import RunRecord, RunResult

_CONN_TIMEOUT = 30.0  # seconds — tolerates concurrent writes from multi-thread islands

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id   TEXT PRIMARY KEY,
    name            TEXT,
    task            TEXT,
    started_at      TEXT,
    dir             TEXT,
    config          TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    experiment_id   TEXT,
    run_id          TEXT,
    iteration       INTEGER,
    run_type        TEXT,
    score           REAL,
    valid           BOOLEAN,
    workspace_path  TEXT,
    started_at      TEXT,
    duration_s      REAL,
    exit_code       INTEGER,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    total_cost      REAL,
    parent_ids      TEXT,
    generation      INTEGER DEFAULT 0,
    idea            TEXT,
    metadata        TEXT,
    PRIMARY KEY (experiment_id, run_id)
);

CREATE TABLE IF NOT EXISTS run_files (
    experiment_id   TEXT,
    run_id          TEXT,
    filename        TEXT,
    content         TEXT,
    PRIMARY KEY (experiment_id, run_id, filename)
);

CREATE TABLE IF NOT EXISTS archive_events (
    experiment_id   TEXT,
    cell_key        TEXT,
    timestamp       TEXT,
    new_id          TEXT,
    old_id          TEXT,
    new_fitness     REAL,
    old_fitness     REAL
);

CREATE TABLE IF NOT EXISTS novelty_reviews (
    experiment_id   TEXT,
    review_id       TEXT,
    run_id          TEXT,
    iteration       INTEGER,
    attempt         INTEGER,
    novelty_score   INTEGER,
    accepted        BOOLEAN,
    explanation     TEXT,
    raw_response    TEXT,
    started_at      TEXT,
    duration_s      REAL,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    total_cost      REAL,
    PRIMARY KEY (experiment_id, review_id)
);

CREATE TABLE IF NOT EXISTS run_embeddings (
    experiment_id   TEXT NOT NULL,
    run_id          TEXT NOT NULL,
    embedder        TEXT NOT NULL,
    text_kind       TEXT NOT NULL,
    embedding       BLOB NOT NULL,
    dim             INTEGER NOT NULL,
    text_hash       TEXT NOT NULL,
    normalized      INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (experiment_id, run_id, embedder, text_kind)
);

CREATE TABLE IF NOT EXISTS meta_test_results (
    experiment_id   TEXT,
    run_id          TEXT,
    score           REAL,
    per_dataset     TEXT,
    evaluated_at    TEXT,
    PRIMARY KEY (experiment_id, run_id)
);

CREATE TABLE IF NOT EXISTS judge_reviews (
    experiment_id   TEXT NOT NULL,
    run_id          TEXT NOT NULL,
    decision        TEXT NOT NULL,
    reasoning       TEXT,
    evidence_refs   TEXT,
    raw_response    TEXT,
    errored         INTEGER NOT NULL DEFAULT 0,
    duration_s      REAL,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (experiment_id, run_id)
);
"""

EXPERIMENT_ID_KEYED_TABLES: tuple[str, ...] = (
    "experiments",
    "runs",
    "run_files",
    "archive_events",
    "judge_reviews",
    "novelty_reviews",
    "run_embeddings",
    "meta_test_results",
)
"""Tuple of all tables whose primary or foreign key includes experiment_id.

Used by the cross-host campaign-migration tooling (shard export/merge) to copy
one experiment's rows between store.db files. The tables are listed in
dependency order (parent tables first). Adding a new experiment_id-keyed table
to the schema requires updating this tuple, or shard export/merge will silently
skip the new table's rows.
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(db_path), timeout=_CONN_TIMEOUT)


class Experiment:
    """Scoped handle to a single experiment."""

    def __init__(
        self,
        experiment_id: str,
        name: str,
        directory: Path,
        db_path: Path,
        task: str = "",
    ) -> None:
        self.id = experiment_id
        self.name = name
        self.task = task
        self.dir = directory
        self._db_path = db_path

    def save(
        self,
        run_id: str,
        *,
        result: RunResult,
        iteration: int | None = None,
        run_type: str = "executor",
        valid: bool | None = None,
        idea: str | None = None,
        parent_ids: list[str] | None = None,
        generation: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist a run row. Columns lift well-known fields; metadata JSON keeps the full dict."""
        all_meta: dict[str, Any] = dict(result.stats)
        if metadata:
            all_meta.update(metadata)

        score = all_meta.get("best_score")
        duration_s = all_meta.get("duration")
        input_tokens = all_meta.get("input_tokens")
        output_tokens = all_meta.get("output_tokens")
        total_cost = all_meta.get("total_cost")
        if valid is None:
            valid = all_meta.get("valid")

        with _connect(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO runs
                   (experiment_id, run_id, iteration, run_type, score, valid,
                    workspace_path, started_at, duration_s, exit_code,
                    input_tokens, output_tokens, total_cost,
                    parent_ids, generation, idea, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.id, run_id, iteration, run_type, score, valid,
                    str(result.workspace), datetime.now().isoformat(),
                    duration_s, result.exit_code,
                    input_tokens, output_tokens, total_cost,
                    ",".join(parent_ids or []),
                    generation,
                    idea,
                    json.dumps(all_meta),
                ),
            )

    def runs(self, *, run_type: str | None = None) -> list[RunRecord]:
        """Return persisted runs, optionally filtered by run_type."""
        q = "SELECT * FROM runs WHERE experiment_id = ?"
        params: tuple[Any, ...] = (self.id,)
        if run_type is not None:
            q += " AND run_type = ?"
            params = (self.id, run_type)
        q += " ORDER BY iteration, started_at"
        with _connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(q, params).fetchall()
        return [_row_to_record(r) for r in rows]

    def get_run(self, run_id: str) -> RunRecord | None:
        """Return a single run by id, or None if not found."""
        with _connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM runs WHERE experiment_id = ? AND run_id = ?",
                (self.id, run_id),
            ).fetchone()
        return _row_to_record(row) if row else None

    def best(
        self,
        *,
        lower_is_better: bool = False,
        run_type: str | None = None,
    ) -> RunRecord | None:
        """Return the best-scoring run.

        If ``run_type`` is given, filters by that type (e.g., ``"executor"``).
        If ``None`` (default), considers runs of all types.
        """
        scored = [r for r in self.runs(run_type=run_type) if r.score is not None]
        if not scored:
            return None

        def key(r: RunRecord) -> float:
            return r.score  # type: ignore[return-value]

        return min(scored, key=key) if lower_is_better else max(scored, key=key)

    # ------------------------------------------------------------------
    # run_files helpers
    # ------------------------------------------------------------------

    def save_file(self, run_id: str, filename: str, content: str) -> None:
        """Snapshot a file's textual content into the DB."""
        with _connect(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO run_files
                   (experiment_id, run_id, filename, content) VALUES (?, ?, ?, ?)""",
                (self.id, run_id, filename, content),
            )

    def files(self, run_id: str) -> dict[str, str]:
        """Return all snapshotted files for a run as {filename: content}."""
        with _connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT filename, content FROM run_files WHERE experiment_id = ? AND run_id = ?",
                (self.id, run_id),
            ).fetchall()
        return {fn: c for fn, c in rows}

    # ------------------------------------------------------------------
    # archive_events helpers
    # ------------------------------------------------------------------

    def log_archive_event(
        self,
        cell_key: str,
        *,
        new_id: str,
        new_fitness: float,
        old_id: str | None = None,
        old_fitness: float | None = None,
    ) -> None:
        """Log an archive cell replacement or first fill."""
        with _connect(self._db_path) as conn:
            conn.execute(
                """INSERT INTO archive_events
                   (experiment_id, cell_key, timestamp, new_id, old_id, new_fitness, old_fitness)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (self.id, cell_key, datetime.now().isoformat(),
                 new_id, old_id, new_fitness, old_fitness),
            )

    def archive_events(self) -> list[dict[str, Any]]:
        """Return all archive events for this experiment ordered by timestamp."""
        with _connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM archive_events WHERE experiment_id = ? ORDER BY timestamp",
                (self.id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # novelty_reviews helpers
    # ------------------------------------------------------------------

    def save_review(
        self,
        review_id: str,
        *,
        run_id: str | None,
        iteration: int,
        attempt: int,
        novelty_score: int,
        accepted: bool,
        explanation: str,
        raw_response: str = "",
        duration_s: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_cost: float | None = None,
    ) -> None:
        """Persist a novelty review attempt (accepted OR rejected)."""
        with _connect(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO novelty_reviews
                   (experiment_id, review_id, run_id, iteration, attempt,
                    novelty_score, accepted, explanation, raw_response,
                    started_at, duration_s, input_tokens, output_tokens, total_cost)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (self.id, review_id, run_id, iteration, attempt,
                 novelty_score, accepted, explanation, raw_response,
                 datetime.now().isoformat(), duration_s,
                 input_tokens, output_tokens, total_cost),
            )

    def reviews(self) -> list[dict[str, Any]]:
        """Return all novelty reviews for this experiment ordered by iteration and attempt."""
        with _connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM novelty_reviews WHERE experiment_id = ? ORDER BY iteration, attempt",
                (self.id,),
            ).fetchall()
        result = []
        for r in rows:
            row = dict(r)
            if row["accepted"] is not None:
                row["accepted"] = bool(row["accepted"])
            result.append(row)
        return result

    # ------------------------------------------------------------------
    # meta_test_results helpers
    # ------------------------------------------------------------------

    def save_meta_test_result(
        self,
        run_id: str,
        *,
        score: float,
        per_dataset: dict[str, Any],
    ) -> None:
        """Persist a meta-test evaluation result for an archive elite.

        Args:
            run_id: The elite's original run_id from the QD loop.
            score: Aggregated normalized score on held-out test datasets.
            per_dataset: Per-dataset breakdown (JSON-serialized for storage).
        """
        with _connect(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO meta_test_results
                   (experiment_id, run_id, score, per_dataset, evaluated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (self.id, run_id, score, json.dumps(per_dataset),
                 datetime.now().isoformat()),
            )

    def meta_test_results(self) -> list[dict[str, Any]]:
        """Return all meta-test results for this experiment, ordered by run_id."""
        with _connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM meta_test_results WHERE experiment_id = ? ORDER BY run_id",
                (self.id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # judge_reviews helpers
    # ------------------------------------------------------------------

    def save_judge_review(self, run_id: str, verdict: Any) -> None:
        """Persist a hacker-judge verdict for ``run_id``.

        ``verdict`` is duck-typed as ``HackerVerdict`` (avoids a hard import
        cycle since ``judge`` re-imports from ``heuresis``).
        """
        import json as _json

        with _connect(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO judge_reviews
                   (experiment_id, run_id, decision, reasoning, evidence_refs,
                    raw_response, errored, duration_s, input_tokens, output_tokens,
                    created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.id,
                    run_id,
                    verdict.decision,
                    verdict.reasoning,
                    _json.dumps(verdict.evidence_refs),
                    verdict.raw_response,
                    1 if verdict.errored else 0,
                    verdict.duration_s,
                    verdict.input_tokens,
                    verdict.output_tokens,
                    datetime.now().isoformat(),
                ),
            )

    # ------------------------------------------------------------------
    # run_embeddings helpers
    # ------------------------------------------------------------------

    def save_embedding(
        self,
        run_id: str,
        *,
        text_kind: str,
        embedder: str,
        vector: np.ndarray,
        text_hash: str,
        normalized: bool = True,
    ) -> None:
        """Persist a single embedding for a run."""
        import numpy as np

        vec = np.asarray(vector, dtype=np.float32)
        if vec.ndim != 1:
            raise ValueError(f"vector must be 1-D, got shape {vec.shape}")
        with _connect(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO run_embeddings
                   (experiment_id, run_id, embedder, text_kind,
                    embedding, dim, text_hash, normalized, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.id, run_id, embedder, text_kind,
                    vec.tobytes(), int(vec.shape[0]), text_hash,
                    1 if normalized else 0,
                    datetime.now().isoformat(),
                ),
            )

    def get_embeddings(
        self,
        *,
        embedder: str,
        text_kind: str,
    ) -> dict[str, np.ndarray]:
        """Return {run_id -> vector} for the given embedder and text_kind."""
        import numpy as np

        with _connect(self._db_path) as conn:
            rows = conn.execute(
                """SELECT run_id, embedding, dim FROM run_embeddings
                   WHERE experiment_id = ? AND embedder = ? AND text_kind = ?""",
                (self.id, embedder, text_kind),
            ).fetchall()
        out: dict[str, np.ndarray] = {}
        for run_id, blob, dim in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.shape[0] != dim:
                raise ValueError(
                    f"embedding dim mismatch for {run_id}: blob={vec.shape[0]} expected={dim}"
                )
            out[run_id] = vec.copy()  # frombuffer returns read-only; copy for mutability
        return out


class ResultStore:
    """Experiment persistence backed by SQLite."""

    def __init__(self, db_path: Path | str = "store.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with _connect(self._db_path) as conn:
            conn.executescript(_SCHEMA)

    def experiment(
        self,
        name: str = "",
        *,
        task: str = "",
        config: dict[str, Any] | None = None,
        root: Path | None = None,
    ) -> Experiment:
        """Create a new experiment and register it in the DB."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") if name else ""
        experiment_id = f"{timestamp}_{slug}" if slug else timestamp

        exp_root = root or Path("runs")
        exp_dir = exp_root / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        with _connect(self._db_path) as conn:
            conn.execute(
                """INSERT INTO experiments
                   (experiment_id, name, task, started_at, dir, config)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (experiment_id, name, task, datetime.now().isoformat(),
                 str(exp_dir), json.dumps(config) if config else None),
            )

        return Experiment(
            experiment_id=experiment_id,
            name=name,
            task=task,
            directory=exp_dir,
            db_path=self._db_path,
        )

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        with _connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
        if row is None:
            return None
        return Experiment(
            experiment_id=row["experiment_id"],
            name=row["name"],
            task=row["task"] or "",
            directory=Path(row["dir"]),
            db_path=self._db_path,
        )

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with _connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


def _row_to_record(row: sqlite3.Row) -> RunRecord:
    m = row["metadata"]
    metadata = json.loads(m) if m else {}
    ws = row["workspace_path"]
    pids_raw = row["parent_ids"] or ""
    parent_ids = [p for p in pids_raw.split(",") if p]
    return RunRecord(
        experiment_id=row["experiment_id"],
        run_id=row["run_id"],
        score=row["score"],
        workspace=Path(ws) if ws else Path("."),
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
