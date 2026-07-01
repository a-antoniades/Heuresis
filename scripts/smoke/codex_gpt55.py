#!/usr/bin/env python3
"""Minimal smoke: invoke codex + gpt-5.5 via the Harness.

Asks the agent to write a single file 'hello.txt' with the word READY,
then verifies the file exists. Exercises the full Harness + bwrap path.

Usage:
    uv run scripts/smoke/codex_gpt55.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from heuresis import Harness, Workspace


PROMPT = """\
Write a single file named hello.txt in the current working directory.
The file must contain exactly one line: READY
Then stop. Do not create any other files.
"""


def main() -> int:
    ws = Workspace(prompt=PROMPT)
    h = Harness(agent="codex", model="gpt-5.5", gpus=[])

    with tempfile.TemporaryDirectory(prefix="codex_smoke_") as tmp:
        path = Path(tmp) / "ws"
        print(f"workspace: {path}")
        future = h.run(ws, prompt={}, path=path)
        result = future.result(timeout=300)

        print(f"exit_code: {result.exit_code}")
        dur = result.stats.get("duration_s")
        print(f"duration_s: {dur}")
        hello = path / "hello.txt"
        if hello.exists():
            content = hello.read_text().strip()
            print(f"hello.txt: {content!r}")
            if "READY" in content:
                print("SMOKE PASSED")
                return 0
            print("SMOKE FAILED: hello.txt did not contain READY")
            return 1

        print("SMOKE FAILED: hello.txt not created")
        print("--- agent.log (last 40 lines) ---")
        log = path / "agent.log"
        if log.exists():
            lines = log.read_text().splitlines()
            for line in lines[-40:]:
                print(line)
        return 1


if __name__ == "__main__":
    sys.exit(main())
