#!/usr/bin/env python3
"""End-to-end smoke test for the MemoryStore primitive.

Runs without any real coding agent: spins up a MemoryStore, exercises
ingest_experiment via record_run with synthetic RunResults, then
subprocess-invokes ``memory append / search / read`` against the real
Unix socket exactly the way an in-sandbox agent would.

The test ingests a *family* of experiments and learnings (not just one
of each) so we can check that semantic search actually *orders* results
correctly, unrelated queries return the unrelated items last, and
multiple authors' learnings get attributed properly. Real Gemini calls
for every embedding — this is a live smoke test, not a unit test.

Requires:
  - GEMINI_API_KEY or GOOGLE_GENERATIVE_AI_API_KEY in the environment
  - sqlite-vec installed in the Python that uv runs

Usage:
    uv run scripts/smoke/smoke_memory.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heuresis.experiment import record_run  # noqa: E402
from heuresis.memory import MemoryStore  # noqa: E402


@dataclass
class _FakeResult:
    """Minimal RunResult stand-in. record_run only reads .workspace."""
    workspace: Path
    exit_code: int = 0
    stats: dict | None = None


class _FakeExperiment:
    """Just enough of the ResultStore Experiment surface to exercise record_run."""

    def __init__(self, root: Path) -> None:
        self.dir = root
        self._rows: list[dict] = []

    def save(self, run_id, *, result, iteration, run_type, valid, idea,
             parent_ids, generation, metadata):
        self._rows.append({
            "run_id": run_id, "iteration": iteration, "valid": valid,
            "score": (metadata or {}).get("best_score"),
            "idea": idea, "generation": generation,
        })

    def save_file(self, *_a, **_k) -> None:
        pass

    def log_archive_event(self, *_a, **_k) -> None:
        pass


def _write_workspace(path: Path, *, role: str) -> str:
    """Set up a fake workspace with the minimum markers record_run reads."""
    path.mkdir(parents=True, exist_ok=True)
    wsid = uuid.uuid4().hex[:12]
    (path / ".workspace_id").write_text(wsid)
    (path / ".workspace_role").write_text(role)
    return wsid


def _memory_cli(socket_path: Path, workspace_id: str, role: str, *args: str) -> dict:
    """Invoke the memory CLI the same way the in-sandbox agent would.

    Uses the env-var path for author identity (MEMORY_SOCKET / WORKSPACE_ID /
    WORKSPACE_ROLE) since we run outside bwrap and can't write the
    ``/workspace/.workspace_role`` marker the CLI otherwise reads.
    """
    env = os.environ.copy()
    env["MEMORY_SOCKET"] = str(socket_path)
    env["WORKSPACE_ID"] = workspace_id
    env["WORKSPACE_ROLE"] = role

    cli = ROOT / "src" / "heuresis" / "tools" / "memory.py"
    proc = subprocess.run(
        [sys.executable, str(cli), *args],
        env=env, capture_output=True, text=True, timeout=60,
    )
    if proc.returncode not in (0, 1):
        print("STDOUT:", proc.stdout, file=sys.stderr)
        print("STDERR:", proc.stderr, file=sys.stderr)
        raise RuntimeError(f"memory CLI failed with code {proc.returncode}")
    stdout = proc.stdout.strip()
    return json.loads(stdout) if stdout else {}


# ---------------------------------------------------------------------------
# Content — a small, semantically structured corpus so we can check ordering.
# ---------------------------------------------------------------------------

# Two clusters: trust-region-ish optimizers and CMA-ES-ish optimizers.
# A related query in one cluster should rank that cluster's members above
# the other cluster. An unrelated query ("matrix factorization") should
# rank all of these below... itself (we don't have an unrelated row, so we
# just check that distances are noticeably higher than the cluster case).

IDEAS = [
    ("exec_001",
     "Trust-region method with random restarts every 200 FEvals. "
     "Initial trust radius 0.5, shrinking by 0.7 on failure.",
     "Restart reset the radius; final mean_log_gap=1.2.", 1.2),
    ("exec_002",
     "Trust-region method with adaptive radius shrinkage based on "
     "success ratio; BFGS local refinement inside the region.",
     "BFGS helped on Rosenbrock but hurt on Rastrigin.", 1.3),
    ("exec_003",
     "Full CMA-ES with rank-one covariance updates and sigma adaptation. "
     "Population 4+floor(3*log(dim)).", None, 0.9),
    ("exec_004",
     "sep-CMA-ES (diagonal-only covariance) for high-dim efficiency. "
     "Same sigma adaptation as full CMA-ES.", None, 1.0),
]

LEARNINGS = [
    ("content", "tags", "author_role_idx"),
    # (content, tags, which_author — 0=executor, 1=ideator)
    ("Nelder-Mead simplex collapses on Rastrigin past 200 FEvals; "
     "Powell method more robust at that budget.",
     "nelder-mead,rastrigin,numerical", 0),
    ("CMA-ES sigma grows unboundedly on ill-conditioned Rosenbrock "
     "unless damping is tightened.",
     "cma-es,rosenbrock,numerical", 0),
    ("Trust-region radius should never go below 1e-6 or step norms underflow.",
     "trust-region,numerical", 1),  # ideator-recorded meta-learning
]


def _print_rows(rows: list[dict], keys: list[str]) -> None:
    for r in rows:
        vals = {k: r.get(k) for k in keys}
        print(f"    - {vals}")


def main() -> int:
    if not (os.environ.get("GEMINI_API_KEY") or
            os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")):
        print("ERROR: set GEMINI_API_KEY or GOOGLE_GENERATIVE_AI_API_KEY",
              file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="smoke-memory-") as tmpdir:
        root = Path(tmpdir)

        # Two ideators + four executors so we can test attribution too.
        ideator_a = root / "ideator_0"
        ideator_b = root / "ideator_1"
        ideator_a_id = _write_workspace(ideator_a, role="ideator")
        ideator_b_id = _write_workspace(ideator_b, role="ideator")

        executor_dirs: dict[str, Path] = {}
        executor_ids: dict[str, str] = {}
        for rid, *_ in IDEAS:
            p = root / rid
            executor_dirs[rid] = p
            executor_ids[rid] = _write_workspace(p, role="executor")

        db_path = root / "memory.db"
        with MemoryStore(db_path) as memory:
            print(f"MemoryStore up @ {memory.socket_path}")

            # -----------------------------------------------------------
            # 1. Bulk-ingest: 4 experiments, alternating which ideator
            #    "proposed" each, some with notes.md retrospectives.
            # -----------------------------------------------------------
            print("\n=== Ingesting 4 experiments via record_run ===")
            fake_exp = _FakeExperiment(root=root)
            for i, (rid, idea, notes, score) in enumerate(IDEAS):
                exec_dir = executor_dirs[rid]
                if notes:
                    (exec_dir / "notes.md").write_text(notes)
                ideator_dir = ideator_a if i % 2 == 0 else ideator_b
                record_run(
                    fake_exp, run_id=rid,
                    result=_FakeResult(workspace=exec_dir),
                    info={"valid": True, "best_score": score},
                    strategy_meta={"generation": 0, "features": None},
                    iteration=i, run_type="executor",
                    idea=idea, parent_ids=None,
                    memory=memory, ideator_workspace=ideator_dir,
                )
                print(f"  [ok] {rid} ingested (ideator={'A' if i%2==0 else 'B'}, "
                      f"notes={'yes' if notes else 'no'}, score={score})")
            time.sleep(0.1)

            # -----------------------------------------------------------
            # 2. Bulk-append: 3 learnings from 2 different authors.
            # -----------------------------------------------------------
            print("\n=== Appending 3 learnings from 2 authors ===")
            learning_ids: list[int] = []
            for content, tags, author_kind in [
                (LEARNINGS[1][0], LEARNINGS[1][1], 0),
                (LEARNINGS[2][0], LEARNINGS[2][1], 0),
                (LEARNINGS[3][0], LEARNINGS[3][1], 1),
            ]:
                if author_kind == 0:
                    author_id, role = executor_ids["exec_001"], "executor"
                else:
                    author_id, role = ideator_a_id, "ideator"
                res = _memory_cli(
                    memory.socket_path, author_id, role,
                    "append", content, "--tags", tags,
                )
                assert res.get("ok") is True, f"append failed: {res}"
                learning_ids.append(res["learning_id"])
                print(f"  [ok] learning {res['learning_id']} by {role} "
                      f"({author_id[:8]}...) tags=[{tags}]")

            # -----------------------------------------------------------
            # 3. Semantic search: related query should rank the trust-region
            #    cluster ABOVE the CMA-ES cluster.
            # -----------------------------------------------------------
            print("\n=== Search: 'trust region adaptive radius' ===")
            res = _memory_cli(
                memory.socket_path, ideator_a_id, "ideator",
                "search", "trust region adaptive radius",
                "--table", "experiments", "--k", "4",
            )
            assert res.get("ok") is True
            rows = res["rows"]
            _print_rows(rows, ["executor_id", "distance", "score"])
            top_two = {rows[0]["executor_id"], rows[1]["executor_id"]}
            tr_cluster = {executor_ids["exec_001"], executor_ids["exec_002"]}
            assert top_two == tr_cluster, (
                f"trust-region cluster should rank top-2; got {top_two}"
            )
            print("  [ok] top-2 are the trust-region cluster "
                  "(exec_001, exec_002); CMA-ES cluster ranked below")

            # -----------------------------------------------------------
            # 4. Semantic search: query from the OTHER cluster should flip
            #    the ordering.
            # -----------------------------------------------------------
            print("\n=== Search: 'CMA-ES covariance adaptation' ===")
            res = _memory_cli(
                memory.socket_path, ideator_a_id, "ideator",
                "search", "CMA-ES covariance adaptation",
                "--table", "experiments", "--k", "4",
            )
            assert res.get("ok") is True
            rows = res["rows"]
            _print_rows(rows, ["executor_id", "distance"])
            top_two = {rows[0]["executor_id"], rows[1]["executor_id"]}
            cma_cluster = {executor_ids["exec_003"], executor_ids["exec_004"]}
            assert top_two == cma_cluster, (
                f"CMA-ES cluster should rank top-2; got {top_two}"
            )
            print("  [ok] top-2 flipped to CMA-ES cluster "
                  "(exec_003, exec_004) on a CMA-ES query")

            # -----------------------------------------------------------
            # 5. Unrelated query: distances should be noticeably higher
            #    than the cluster-match case.
            # -----------------------------------------------------------
            print("\n=== Search (unrelated): 'french baroque architecture' ===")
            res = _memory_cli(
                memory.socket_path, ideator_a_id, "ideator",
                "search", "french baroque architecture",
                "--table", "experiments", "--k", "4",
            )
            rows = res["rows"]
            unrelated_top = rows[0]["distance"]
            # Re-query a related one to compare.
            res2 = _memory_cli(
                memory.socket_path, ideator_a_id, "ideator",
                "search", "trust region method",
                "--table", "experiments", "--k", "1",
            )
            related_top = res2["rows"][0]["distance"]
            print(f"    unrelated top distance = {unrelated_top:.4f}")
            print(f"    related   top distance = {related_top:.4f}")
            assert unrelated_top > related_top, (
                "unrelated query should have a larger top distance"
            )
            print(f"  [ok] unrelated distance ({unrelated_top:.3f}) "
                  f"> related ({related_top:.3f})")

            # -----------------------------------------------------------
            # 6. Learnings search: phrased differently than the stored
            #    content, should still find the right one.
            # -----------------------------------------------------------
            print("\n=== Search learnings: 'step size blows up on Rosenbrock' ===")
            res = _memory_cli(
                memory.socket_path, ideator_a_id, "ideator",
                "search", "step size blows up on Rosenbrock",
                "--table", "learnings", "--k", "3",
            )
            rows = res["rows"]
            _print_rows(rows, ["learning_id", "distance", "author_role"])
            # The CMA-ES sigma learning (learning_ids[1]) should top.
            assert rows[0]["learning_id"] == learning_ids[1], (
                f"expected learning {learning_ids[1]} (CMA-ES sigma), got {rows[0]}"
            )
            print("  [ok] paraphrased query matched the sigma-damping learning")

            # -----------------------------------------------------------
            # 7. Notes.md influence: exec_001 has a retrospective note
            #    about "restart reset the radius". Query for that phrase
            #    should find exec_001 above exec_002 (which doesn't have
            #    a similar note).
            # -----------------------------------------------------------
            print("\n=== Search: retrospective phrase -> notes.md effect ===")
            res = _memory_cli(
                memory.socket_path, ideator_a_id, "ideator",
                "search", "restart reset the radius",
                "--table", "experiments", "--k", "2",
            )
            rows = res["rows"]
            _print_rows(rows, ["executor_id", "distance"])
            assert rows[0]["executor_id"] == executor_ids["exec_001"], (
                "exec_001 (with matching notes.md) should win over exec_002"
            )
            print("  [ok] notes.md content folded into the embedding as expected")

            # -----------------------------------------------------------
            # 8. SQL read: join + aggregate over the views.
            # -----------------------------------------------------------
            print("\n=== Read: aggregate over views ===")
            res = _memory_cli(
                memory.socket_path, ideator_a_id, "ideator",
                "read",
                "SELECT COUNT(*) AS n, AVG(score) AS avg_score "
                "FROM memory_experiments_v WHERE valid = 1",
            )
            assert res.get("ok") is True
            row = res["rows"][0]
            print(f"    n={row['n']}, avg_score={row['avg_score']:.3f}")
            assert row["n"] == 4
            assert 0.9 <= row["avg_score"] <= 1.3

            print("\n=== Read: attribution join on learnings ===")
            res = _memory_cli(
                memory.socket_path, ideator_a_id, "ideator",
                "read",
                "SELECT author_role, COUNT(*) AS n FROM memory_learnings_v "
                "GROUP BY author_role ORDER BY author_role",
            )
            rows = res["rows"]
            _print_rows(rows, ["author_role", "n"])
            by_role = {r["author_role"]: r["n"] for r in rows}
            assert by_role == {"executor": 2, "ideator": 1}, by_role
            print("  [ok] 2 executor-authored + 1 ideator-authored learnings")

            # -----------------------------------------------------------
            # 9. Cross-author attribution: ideator appends and the author_id
            #    in the DB matches the ideator's workspace UUID.
            # -----------------------------------------------------------
            print("\n=== Ideator-side append: author_id matches ideator UUID ===")
            res = _memory_cli(
                memory.socket_path, ideator_b_id, "ideator",
                "append", "Early stopping on stagnating fitness saves budget.",
                "--tags", "meta,policy",
            )
            new_id = res["learning_id"]
            # Query back via read to verify author_id was stamped correctly.
            res = _memory_cli(
                memory.socket_path, ideator_a_id, "ideator",
                "read",
                f"SELECT author_id, author_role FROM memory_learnings_v "
                f"WHERE learning_id = {new_id}",
            )
            row = res["rows"][0]
            assert row["author_id"] == ideator_b_id, (
                f"expected author {ideator_b_id}, got {row['author_id']}"
            )
            assert row["author_role"] == "ideator"
            print(f"  [ok] learning {new_id} attributed to ideator_b "
                  f"({ideator_b_id[:8]}...)")

            # -----------------------------------------------------------
            # 10. Safety rails — every rejection path.
            # -----------------------------------------------------------
            print("\n=== Safety rails ===")
            for sql, why in [
                ("DELETE FROM experiments", "DELETE"),
                ("UPDATE experiments SET score=0", "UPDATE"),
                ("DROP TABLE experiments", "DROP"),
                ("SELECT * FROM experiments", "underlying table"),
                ("SELECT * FROM learnings", "underlying table"),
                ("SELECT 1 FROM memory_experiments_v; DROP TABLE experiments",
                 "multi-statement"),
            ]:
                bad = _memory_cli(
                    memory.socket_path, ideator_a_id, "ideator",
                    "read", sql,
                )
                assert bad.get("ok") is False, f"{why!r} should be rejected"
                print(f"  [ok] rejected: {why}")

            # Bad author role at append time.
            bad = _memory_cli(
                memory.socket_path, executor_ids["exec_001"], "admin",
                "append", "trying to impersonate",
            )
            assert bad.get("ok") is False
            print("  [ok] rejected: bad author_role")

            # Empty content.
            bad = _memory_cli(
                memory.socket_path, executor_ids["exec_001"], "executor",
                "append", "   ",
            )
            assert bad.get("ok") is False
            print("  [ok] rejected: empty content")

            # -----------------------------------------------------------
            # 11. Final DB state summary.
            # -----------------------------------------------------------
            print("\n=== Final DB state ===")
            res = _memory_cli(
                memory.socket_path, ideator_a_id, "ideator",
                "read",
                "SELECT "
                "(SELECT COUNT(*) FROM memory_experiments_v) AS experiments, "
                "(SELECT COUNT(*) FROM memory_learnings_v) AS learnings",
            )
            print(f"    {res['rows'][0]}")
            assert res["rows"][0] == {"experiments": 4, "learnings": 4}

        print("\nSMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
