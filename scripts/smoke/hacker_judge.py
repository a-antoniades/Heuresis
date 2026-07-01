#!/usr/bin/env python3
"""Live smoke test for HackerJudge against known reward-hacking workspaces.

Takes an executor workspace (a real ``exec_NNN/`` directory from a past run),
runs the HackerJudge on it, and prints the verdict. Designed for quick spot-
checks on known-fabricated workspaces. The curated cases and incident notes are
maintainer-only research artifacts.

Usage:

    # With explicit idea + reported score:
    uv run scripts/smoke/hacker_judge.py \\
        runs/nanogpt/2026-04-17_185209_nanogpt-islands/exec_010 \\
        --score 0.85 --idea "DEPTH 8->13, disable value embeddings"

    # Auto-detect idea + score from store.db:
    uv run scripts/smoke/hacker_judge.py \\
        runs/nanogpt/2026-04-17_185209_nanogpt-islands/exec_010

Env:
    Credentials only. Judge configuration is passed with CLI flags.

Known fabricated workspaces (baseline for smoke):
    runs/nanogpt/2026-04-17_185209_nanogpt-islands/exec_010   # 222 B run.log, val_bpb=0.85
    runs/nanogpt/2026-04-17_185209_nanogpt-islands/exec_017   # 122 KB + appended fake, val_bpb=0.931
    runs/nanogpt/2026-04-14_224007_nanogpt-omni-epic/exec_1800  # 223 B, val_bpb=0.900000

Expected verdict on all three: ``suspicious_evidence``.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from heuresis import Harness, HackerJudge, ResultStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = PROJECT_ROOT / "src" / "heuresis" / "tasks" / "nanogpt"
STORE_DB = PROJECT_ROOT / "runs" / "nanogpt" / "store.db"
SMOKE_ROOT = PROJECT_ROOT / "runs" / "smoke_hacker_judge"


def autofill_from_store(exec_workspace: Path) -> tuple[str, float | None]:
    """Look up (idea, score) from store.db for the run whose workspace path matches."""
    if not STORE_DB.is_file():
        return "", None
    store = ResultStore(db_path=STORE_DB)
    # The store records `workspace_path` as an absolute path string. Match on that
    # OR by run_id (last path segment) as a fallback.
    rows = store.query(
        """SELECT idea, score FROM runs
           WHERE workspace_path = ? OR run_id = ?""",
        (str(exec_workspace), exec_workspace.name),
    )
    if not rows:
        return "", None
    row = rows[0]
    return (row.get("idea") or "", row.get("score"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exec_workspace", type=Path,
                    help="Path to a past exec_NNN/ workspace (absolute or relative to repo root).")
    ap.add_argument("--idea", type=str, default=None,
                    help="The proposed idea text. If omitted, try to look it up in store.db.")
    ap.add_argument("--score", type=float, default=None,
                    help="Reported score. If omitted, try to look it up in store.db.")
    ap.add_argument("--timed-out", action="store_true", help="Set reported timed_out=True in info.")
    ap.add_argument("--agent", default="opencode", help="Agent binary/profile to use for judging.")
    ap.add_argument("--judge-model", default="claude-sonnet-4-6", help="Judge model name.")
    ap.add_argument("--judge-timeout", type=int, default=300, help="Judge timeout in seconds.")
    args = ap.parse_args()

    exec_workspace = args.exec_workspace.resolve()
    if not exec_workspace.is_dir():
        print(f"ERROR: not a directory: {exec_workspace}", file=sys.stderr)
        return 2

    idea = args.idea
    score = args.score
    if idea is None or score is None:
        auto_idea, auto_score = autofill_from_store(exec_workspace)
        if idea is None:
            idea = auto_idea
        if score is None:
            score = auto_score

    if score is None:
        print("ERROR: no --score given and none found in store.db", file=sys.stderr)
        print("       Pass --score <float> to supply the reported value manually.", file=sys.stderr)
        return 2

    print(f"Judge model:    {args.judge_model}")
    print(f"Judge timeout:  {args.judge_timeout}s")
    print(f"Agent:          {args.agent}")
    print(f"Exec workspace: {exec_workspace}")
    print(f"Reported score: {score}")
    print(f"Idea chars:     {len(idea)}")
    print()

    harness = Harness(args.agent, model=args.judge_model, gpus=[])
    judge = HackerJudge(harness, TASK_DIR, timeout=args.judge_timeout)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    judge_dir = SMOKE_ROOT / f"{exec_workspace.name}_{timestamp}"
    print(f"Judge workspace: {judge_dir}")
    print()

    info = {
        "best_score": score,
        "valid": True,
        "timed_out": args.timed_out,
        "exit_code": 0,
        "duration": 1800.0,
    }

    print("Running HackerJudge.review()...", flush=True)
    verdict = judge.review(
        exec_workspace=exec_workspace,
        judge_dir=judge_dir,
        idea=idea,
        info=info,
    )

    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    payload = asdict(verdict)
    # Truncate raw_response for terminal readability
    if payload.get("raw_response") and len(payload["raw_response"]) > 600:
        payload["raw_response"] = payload["raw_response"][:600] + "...[truncated]"
    print(json.dumps(payload, indent=2))
    print()
    print(f"Full judge workspace: {judge_dir}")
    print(f"  judge.json:  {judge_dir / 'judge.json'}")
    print(f"  agent.log:   {judge_dir / 'agent.log'}")

    if verdict.errored:
        return 3  # judge itself errored — infra problem, not a clean signal
    return 0


if __name__ == "__main__":
    sys.exit(main())
