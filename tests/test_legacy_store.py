"""Tests for legacy_store shim — reading pre-refactor store.db schema."""
from __future__ import annotations

import sqlite3

import pytest

from analysis.libs.legacy_store import LegacyStore


@pytest.fixture
def legacy_db(tmp_path):
    db = tmp_path / "legacy.db"
    with sqlite3.connect(db) as conn:
        # Pre-refactor schema (simplified)
        conn.executescript("""
            CREATE TABLE experiments (
                experiment_id TEXT PRIMARY KEY, name TEXT, task TEXT,
                started_at TEXT, dir TEXT, config TEXT
            );
            CREATE TABLE runs (
                experiment_id TEXT, run_id TEXT, iteration INTEGER,
                run_type TEXT, idea TEXT, score REAL, valid BOOLEAN,
                started_at TEXT, duration_s REAL, exit_code INTEGER,
                input_tokens INTEGER, output_tokens INTEGER, total_cost REAL,
                workspace_path TEXT, log_dir TEXT, verification TEXT,
                metadata TEXT, parent_ids TEXT, generation INTEGER DEFAULT 0,
                PRIMARY KEY (experiment_id, run_id)
            );
        """)
        conn.execute("INSERT INTO experiments VALUES ('exp1', 'old', 'nanogpt', '2026-03-01', '/old', '{}')")
        conn.execute(
            """INSERT INTO runs VALUES ('exp1', 'exec_000', 0, 'executor',
               'use rope', 0.95, 1, '2026-03-01', 1200.0, 0,
               1000, 500, 0.01, '/old/exec_000', '/old/logs', NULL,
               '{"best_score": 0.95}', 'prev', 1)"""
        )
    return db


def test_legacy_store_returns_runrecords(legacy_db):
    store = LegacyStore(db_path=legacy_db)
    runs = store.runs("exp1")
    assert len(runs) == 1
    r = runs[0]
    assert r.run_id == "exec_000"
    assert r.score == 0.95
    assert r.iteration == 0
    assert r.run_type == "executor"
    assert r.idea == "use rope"
    assert r.parent_ids == ["prev"]
    assert r.generation == 1


def test_legacy_store_lists_experiments(legacy_db):
    store = LegacyStore(db_path=legacy_db)
    exps = store.experiments()
    assert [e["experiment_id"] for e in exps] == ["exp1"]
