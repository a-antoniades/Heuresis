#!/usr/bin/env python3
"""Probe Claude Code's Bash-tool default timeout from inside our sandbox.

Hypothesis: Claude's Bash tool defaults to ~120s and caps around 600s,
which would explain why the nanogpt executor never produced run.log
during the islands T2 smoke (real TIME_BUDGET is 1200s).

This test asks Claude to run ONE foreground command that sleeps 130s
(just over the suspected default) and write a notes.md describing what
it actually observed. Harness timeout is 360s so the harness won't be
the thing that kills it.

Usage:
    uv run scripts/smoke/claude_bash_timeout.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from heuresis import Harness, Workspace


PROMPT = """\
Run this single bash command in the FOREGROUND with default settings
(do NOT background it, do NOT pass any custom timeout argument):

    date +%s > start.txt && sleep 130 && date +%s > end.txt && echo OK > done.txt

Then, regardless of whether that command succeeded or failed:

1. Write notes.md describing exactly what happened: did the bash call
   return successfully? Did you see a timeout / interruption message?
   Quote the exact error string if any. State the elapsed wall time
   you observed.
2. Do NOT retry. Do NOT use nohup, &, or any backgrounding.
3. End your turn.
"""


def main() -> int:
    ws = Workspace()
    harness = Harness("claude", model="claude-sonnet-4-6")
    if errors := harness.preflight():
        for e in errors:
            print(f"  - {e}")
        return 1

    sandbox = (
        Path(__file__).resolve().parents[2]
        / "runs" / "_smoke" / "claude_bash_timeout"
    )
    if sandbox.exists():
        import shutil
        shutil.rmtree(sandbox)
    sandbox.mkdir(parents=True)

    print(f"Launching Claude (harness timeout=360s) in {sandbox} ...")
    result = harness.run(ws, PROMPT, path=sandbox, timeout=360).result()
    print(f"  exit_code={result.exit_code} duration={result.stats.get('duration'):.1f}s")
    print()

    for fn in ("start.txt", "end.txt", "done.txt", "notes.md"):
        p = sandbox / fn
        if p.exists():
            txt = p.read_text().strip()
            short = txt if len(txt) < 600 else txt[:600] + "..."
            print(f"--- {fn} ({p.stat().st_size} bytes) ---")
            print(short)
        else:
            print(f"--- {fn}: MISSING ---")
        print()

    start = sandbox / "start.txt"
    end = sandbox / "end.txt"
    if start.exists() and end.exists():
        try:
            elapsed = int(end.read_text().strip()) - int(start.read_text().strip())
            print(f"Observed sleep elapsed: {elapsed}s (expected 130s if uninterrupted)")
        except ValueError:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
