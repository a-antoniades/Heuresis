"""Tests for the run_embeddings table and Experiment save/get API."""
from __future__ import annotations

import numpy as np
import pytest

from heuresis.models import RunResult
from heuresis.store import ResultStore


@pytest.fixture
def exp(tmp_path):
    store = ResultStore(db_path=tmp_path / "store.db")
    e = store.experiment("test", root=tmp_path / "runs")
    # Insert a run so the FK-ish relationship is plausible
    e.save(
        "exec_000",
        result=RunResult(workspace=tmp_path, exit_code=0, stats={}),
        run_type="executor",
    )
    return e


def test_save_embedding_roundtrip(exp):
    vec = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    exp.save_embedding(
        "exec_000",
        text_kind="plan",
        embedder="fake-v1",
        vector=vec,
        text_hash="abc123",
    )
    out = exp.get_embeddings(embedder="fake-v1", text_kind="plan")
    assert "exec_000" in out
    np.testing.assert_allclose(out["exec_000"], vec)


def test_save_embedding_overwrite_same_key(exp):
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    exp.save_embedding("exec_000", text_kind="plan", embedder="fake-v1",
                       vector=v1, text_hash="h1")
    exp.save_embedding("exec_000", text_kind="plan", embedder="fake-v1",
                       vector=v2, text_hash="h2")
    out = exp.get_embeddings(embedder="fake-v1", text_kind="plan")
    np.testing.assert_allclose(out["exec_000"], v2)


def test_text_kind_separation(exp):
    plan_vec = np.array([1.0, 0.0], dtype=np.float32)
    code_vec = np.array([0.0, 1.0], dtype=np.float32)
    exp.save_embedding("exec_000", text_kind="plan", embedder="fake-v1",
                       vector=plan_vec, text_hash="p")
    exp.save_embedding("exec_000", text_kind="code_diff", embedder="fake-v1",
                       vector=code_vec, text_hash="c")
    plans = exp.get_embeddings(embedder="fake-v1", text_kind="plan")
    codes = exp.get_embeddings(embedder="fake-v1", text_kind="code_diff")
    np.testing.assert_allclose(plans["exec_000"], plan_vec)
    np.testing.assert_allclose(codes["exec_000"], code_vec)


def test_embedder_separation(exp):
    v_a = np.array([1.0, 0.0], dtype=np.float32)
    v_b = np.array([0.0, 1.0], dtype=np.float32)
    exp.save_embedding("exec_000", text_kind="plan", embedder="gemini-a",
                       vector=v_a, text_hash="h")
    exp.save_embedding("exec_000", text_kind="plan", embedder="gemini-b",
                       vector=v_b, text_hash="h")
    a = exp.get_embeddings(embedder="gemini-a", text_kind="plan")
    b = exp.get_embeddings(embedder="gemini-b", text_kind="plan")
    np.testing.assert_allclose(a["exec_000"], v_a)
    np.testing.assert_allclose(b["exec_000"], v_b)


def test_get_embeddings_empty(exp):
    out = exp.get_embeddings(embedder="nonexistent", text_kind="plan")
    assert out == {}


def test_save_embedding_stores_metadata(exp):
    vec = np.array([0.5, 0.5], dtype=np.float32)
    exp.save_embedding("exec_000", text_kind="plan", embedder="fake-v1",
                       vector=vec, text_hash="abc123", normalized=False)
    # Verify via raw SQL that text_hash, dim, normalized are persisted
    import sqlite3
    conn = sqlite3.connect(str(exp._db_path))
    row = conn.execute(
        "SELECT text_hash, dim, normalized FROM run_embeddings "
        "WHERE experiment_id=? AND run_id=? AND embedder=? AND text_kind=?",
        (exp.id, "exec_000", "fake-v1", "plan"),
    ).fetchone()
    conn.close()
    assert row[0] == "abc123"
    assert row[1] == 2
    assert row[2] == 0
