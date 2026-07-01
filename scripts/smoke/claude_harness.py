#!/usr/bin/env python3
"""T1 smoke: verify the Claude Code agent profile works inside bwrap.

Tests:
  - claude binary mounted into sandbox via _mount_agent_binary
  - ~/.claude/ bind-mount carries OAuth credentials in
  - --session-id (dashed UUID) accepted by Claude
  - --dangerously-skip-permissions lets the agent write files
  - ANTHROPIC_API_KEY auto-stripped (OAuth path used)
  - agent.log produced; exit code 0; requested file exists

Usage:
    uv run scripts/smoke/claude_harness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from heuresis import Harness, Workspace


def main() -> int:
    ws = Workspace()  # no tools, no venv, no seed files
    harness = Harness("claude", model="claude-sonnet-4-6")

    errors = harness.preflight()
    if errors:
        print("Preflight failed:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("Preflight OK")

    sandbox_dir = Path(__file__).resolve().parents[2] / "runs" / "_smoke" / "claude_harness"
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    prompt = (
        "Write the single word 'hello' (no newline, no quotes) to the file "
        "'/workspace/hello.txt'. Then end your turn."
    )

    print(f"Launching Claude in {sandbox_dir} ...")
    result = harness.run(ws, prompt, path=sandbox_dir, timeout=120).result()
    print(f"  exit_code={result.exit_code} duration={result.stats.get('duration'):.1f}s")

    hello = sandbox_dir / "hello.txt"
    log = sandbox_dir / "agent.log"
    print(f"  agent.log exists: {log.exists()} ({log.stat().st_size if log.exists() else 0} bytes)")
    print(f"  hello.txt exists: {hello.exists()}")
    if hello.exists():
        print(f"  hello.txt contents: {hello.read_text()!r}")

    ok = result.exit_code == 0 and hello.exists() and "hello" in hello.read_text()
    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
