#!/usr/bin/env python3
"""submit_json — submit a JSON object validated against a schema.

Reads JSON on stdin. Validates against the schema at ``--schema``. On success,
writes the payload to ``--out`` (JSON-pretty-printed) and exits 0. On schema
violation, prints a human-readable error list to stderr and exits 1. On
infrastructure error (missing schema, bad stdin), exits 2.

Supports the subset of JSON Schema we need for agentic-reviewer verdicts:
  - ``type``: ``object`` | ``array`` | ``string`` | ``integer`` |
              ``number`` | ``boolean`` | ``null``
  - ``enum``: [...]
  - ``required``: [...]
  - ``properties``: {name: subschema}
  - ``items``: subschema
  - ``minLength``: int (strings)
  - ``minItems``: int (arrays)

Extend the validator here if a richer schema ever becomes necessary; prefer
keeping it stdlib-only so the tool runs in any minimal sandbox venv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def validate(data: Any, schema: dict, path: str = "") -> list[str]:
    """Validate ``data`` against ``schema``. Returns an error list (empty = OK)."""
    errors: list[str] = []
    loc = path or "root"
    t = schema.get("type")

    if t == "object":
        if not isinstance(data, dict):
            errors.append(f"{loc}: expected object, got {type(data).__name__}")
            return errors
        for key in schema.get("required", []):
            if key not in data:
                errors.append(f"{loc}: missing required key '{key}'")
        for key, subschema in schema.get("properties", {}).items():
            if key in data:
                child = f"{path}.{key}" if path else key
                errors.extend(validate(data[key], subschema, path=child))
    elif t == "array":
        if not isinstance(data, list):
            errors.append(f"{loc}: expected array, got {type(data).__name__}")
            return errors
        min_items = schema.get("minItems")
        if min_items is not None and len(data) < min_items:
            errors.append(f"{loc}: array has {len(data)} items, minItems={min_items}")
        item_schema = schema.get("items")
        if item_schema is not None:
            for i, item in enumerate(data):
                errors.extend(validate(item, item_schema, path=f"{loc}[{i}]"))
    elif t == "string":
        if not isinstance(data, str):
            errors.append(f"{loc}: expected string, got {type(data).__name__}")
            return errors
        min_len = schema.get("minLength")
        if min_len is not None and len(data) < min_len:
            errors.append(f"{loc}: string length {len(data)} < minLength {min_len}")
    elif t == "integer":
        if not isinstance(data, int) or isinstance(data, bool):
            errors.append(f"{loc}: expected integer, got {type(data).__name__}")
    elif t == "number":
        if not isinstance(data, (int, float)) or isinstance(data, bool):
            errors.append(f"{loc}: expected number, got {type(data).__name__}")
    elif t == "boolean":
        if not isinstance(data, bool):
            errors.append(f"{loc}: expected boolean, got {type(data).__name__}")
    elif t == "null":
        if data is not None:
            errors.append(f"{loc}: expected null, got {type(data).__name__}")
    # else: no type constraint at this node

    enum = schema.get("enum")
    if enum is not None and data not in enum:
        errors.append(f"{loc}: value {data!r} not in enum {enum}")

    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description="Submit JSON validated against a schema.")
    ap.add_argument("--schema", required=True, type=Path, help="Path to JSON Schema file.")
    ap.add_argument("--out", required=True, type=Path, help="Path to write validated JSON.")
    args = ap.parse_args()

    raw = sys.stdin.read()
    if not raw.strip():
        print("submit_json: stdin is empty (expected JSON on stdin)", file=sys.stderr)
        return 2

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"submit_json: stdin is not valid JSON: {e}", file=sys.stderr)
        return 1

    try:
        schema = json.loads(args.schema.read_text())
    except FileNotFoundError:
        print(f"submit_json: schema not found: {args.schema}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as e:
        print(f"submit_json: schema file is not valid JSON: {e}", file=sys.stderr)
        return 2

    errors = validate(data, schema)
    if errors:
        print("submit_json: schema violations:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2))
    print(f"submit_json: wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
