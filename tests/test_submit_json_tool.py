"""Tests for the generic submit_json tool."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "src" / "heuresis" / "tools" / "submit_json.py"

# Matches src/heuresis/judge/verdict.schema.json but kept inline
# so the tool's behavior is tested independently of any caller.
VERDICT_SCHEMA = {
    "type": "object",
    "required": ["decision", "reasoning", "evidence_refs"],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["valid", "suspicious_evidence", "invalid_idea"],
        },
        "reasoning": {"type": "string", "minLength": 10},
        "evidence_refs": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
    },
}


def _run(payload: str, schema_path: Path, out_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), "--schema", str(schema_path), "--out", str(out_path)],
        input=payload,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def schema_file(tmp_path: Path) -> Path:
    p = tmp_path / "schema.json"
    p.write_text(json.dumps(VERDICT_SCHEMA))
    return p


def test_valid_payload_writes_file(schema_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "sub" / "out.json"
    payload = json.dumps({
        "decision": "valid",
        "reasoning": "A completely unremarkable run.",
        "evidence_refs": ["run.log:tail"],
    })
    r = _run(payload, schema_file, out)
    assert r.returncode == 0, r.stderr
    assert out.read_text()
    data = json.loads(out.read_text())
    assert data["decision"] == "valid"


def test_invalid_decision_fails(schema_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    payload = json.dumps({
        "decision": "bananas",
        "reasoning": "something something",
        "evidence_refs": ["x:1"],
    })
    r = _run(payload, schema_file, out)
    assert r.returncode == 1
    assert "not in enum" in r.stderr
    assert not out.exists()


def test_missing_required_key_fails(schema_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    payload = json.dumps({"decision": "valid"})  # missing reasoning + evidence_refs
    r = _run(payload, schema_file, out)
    assert r.returncode == 1
    assert "missing required key 'reasoning'" in r.stderr
    assert "missing required key 'evidence_refs'" in r.stderr
    assert not out.exists()


def test_empty_evidence_refs_fails(schema_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    payload = json.dumps({
        "decision": "valid",
        "reasoning": "ok enough",
        "evidence_refs": [],
    })
    r = _run(payload, schema_file, out)
    assert r.returncode == 1
    assert "minItems" in r.stderr


def test_short_reasoning_fails(schema_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    payload = json.dumps({
        "decision": "valid",
        "reasoning": "no",
        "evidence_refs": ["x:1"],
    })
    r = _run(payload, schema_file, out)
    assert r.returncode == 1
    assert "minLength" in r.stderr


def test_malformed_stdin_fails(schema_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    r = _run("{not json", schema_file, out)
    assert r.returncode == 1
    assert "not valid JSON" in r.stderr


def test_empty_stdin_fails(schema_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    r = _run("", schema_file, out)
    assert r.returncode == 2
    assert "stdin is empty" in r.stderr


def test_missing_schema_fails(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    r = _run('{"decision": "valid"}', tmp_path / "nope.json", out)
    assert r.returncode == 2
    assert "schema not found" in r.stderr


def test_nested_array_items_validated(tmp_path: Path) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({
        "type": "object",
        "required": ["items"],
        "properties": {
            "items": {"type": "array", "items": {"type": "integer"}},
        },
    }))
    out = tmp_path / "out.json"
    # int array OK
    r = _run(json.dumps({"items": [1, 2, 3]}), schema, out)
    assert r.returncode == 0
    # mixed fails
    r = _run(json.dumps({"items": [1, "two", 3]}), schema, out)
    assert r.returncode == 1
    assert "items[1]" in r.stderr
    assert "expected integer" in r.stderr
