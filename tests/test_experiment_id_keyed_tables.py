"""Tests for EXPERIMENT_ID_KEYED_TABLES constant in store.py.

Pins two contracts:
1. The constant is a non-empty tuple of strings.
2. It matches exactly the set of tables in _SCHEMA whose CREATE TABLE SQL
   contains the substring "experiment_id" - so schema drift causes a failure
   rather than a silent data-loss gap for any consumer that iterates the
   experiment-id-keyed tables.
"""
from __future__ import annotations

import re


from heuresis.store import EXPERIMENT_ID_KEYED_TABLES, _SCHEMA


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_constant_is_nonempty_tuple_of_strings() -> None:
    assert isinstance(EXPERIMENT_ID_KEYED_TABLES, tuple)
    assert len(EXPERIMENT_ID_KEYED_TABLES) > 0
    for name in EXPERIMENT_ID_KEYED_TABLES:
        assert isinstance(name, str), f"expected str, got {type(name)!r}: {name!r}"


# ---------------------------------------------------------------------------
# Schema cross-check
# ---------------------------------------------------------------------------


def _tables_with_experiment_id_in_schema() -> set[str]:
    """Parse _SCHEMA and return table names whose CREATE TABLE SQL contains
    the substring 'experiment_id'."""
    # Extract each CREATE TABLE block: from "CREATE TABLE" to the closing ");"
    pattern = re.compile(
        r"CREATE TABLE(?:\s+IF NOT EXISTS)?\s+(\w+)\s*\((.+?)\);",
        re.DOTALL | re.IGNORECASE,
    )
    found: set[str] = set()
    for match in pattern.finditer(_SCHEMA):
        table_name = match.group(1)
        column_block = match.group(2)
        if "experiment_id" in column_block:
            found.add(table_name)
    return found


def test_constant_matches_schema_tables_with_experiment_id() -> None:
    """Every table in _SCHEMA that contains experiment_id must appear in the
    constant, and every name in the constant must actually appear in _SCHEMA
    as an experiment_id-keyed table."""
    schema_tables = _tables_with_experiment_id_in_schema()

    constant_set = set(EXPERIMENT_ID_KEYED_TABLES)

    missing_from_constant = schema_tables - constant_set
    extra_in_constant = constant_set - schema_tables

    assert not missing_from_constant, (
        f"Tables in _SCHEMA with experiment_id that are NOT in "
        f"EXPERIMENT_ID_KEYED_TABLES: {sorted(missing_from_constant)}. "
        "Add them to the constant to prevent silent data loss during shard migration."
    )
    assert not extra_in_constant, (
        f"Names in EXPERIMENT_ID_KEYED_TABLES that do NOT match an "
        f"experiment_id-keyed table in _SCHEMA: {sorted(extra_in_constant)}. "
        "Either the table was renamed/removed or the constant has a typo."
    )
