"""Tests for nanogpt task preflight."""
from __future__ import annotations

from unittest.mock import patch

from heuresis.tasks.nanogpt.preflight import check_nanogpt


def test_missing_data_reported(tmp_path):
    with patch("heuresis.tasks.nanogpt.preflight.CACHE_DIR", tmp_path / "no_exist"):
        errors = check_nanogpt(gpus=[])
    assert any("data" in e.lower() for e in errors)


def test_all_ok_when_cache_present(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "shard_00000.parquet").write_bytes(b"x")
    (tmp_path / "tokenizer").mkdir()
    (tmp_path / "tokenizer" / "tokenizer.json").write_text("{}")

    with patch("heuresis.tasks.nanogpt.preflight.CACHE_DIR", tmp_path):
        errors = check_nanogpt(gpus=[])
    # Still may have GPU/agent errors, but no data errors
    assert not any("data" in e.lower() for e in errors)
