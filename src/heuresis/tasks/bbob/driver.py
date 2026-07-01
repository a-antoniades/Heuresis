"""BBOB task driver. Sealed by the executor prompt — agent edits optimizer.py only.

Responsibilities:
 1. Load problem_spec.json (functions, dim, instances, seeds, budget, wallclock_cap_s).
 2. For each (func_id, instance_id, seed):
    - Instantiate f via make_problem.
    - Wrap with BudgetedF (enforces FEval cap).
    - Arm a SIGALRM for wallclock_cap_s; call optimizer.optimize(f, ...).
    - Catch BudgetExhausted / TimeoutError / Exception; record status.
    - Log one per-tuple line.
 3. Aggregate mean_log_gap + per-function breakdown; write summary block.

Run from workspace root:
    python driver.py > run.log 2>&1
"""
from __future__ import annotations

import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

# Workspace-local imports — driver.py and problems.py always live in the same dir.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from problems import BudgetedF, BudgetExhausted, make_problem  # noqa: E402


class _WallclockExceeded(Exception):
    """Raised from the SIGALRM handler so we can distinguish from other errors."""


def _alarm_handler(signum, frame):
    raise _WallclockExceeded("wallclock cap reached")


def _run_one(
    func_id: int,
    instance_id: int,
    seed: int,
    dim: int,
    budget: int,
    wallclock_cap_s: int,
    optimize_fn,
) -> dict:
    f_raw, f_opt, bounds = make_problem(func_id, dim, instance_id)
    wrapped = BudgetedF(f_raw, budget=budget, bounds=bounds)
    rng_seed = func_id * 1_000_000 + instance_id * 1_000 + seed
    status = "ok"
    t0 = time.monotonic()
    try:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(int(wallclock_cap_s))
        try:
            optimize_fn(wrapped, dim, bounds, budget, rng_seed)
        finally:
            signal.alarm(0)
    except BudgetExhausted:
        status = "budget_exhausted"
    except _WallclockExceeded:
        status = "timeout"
    except Exception as e:
        status = f"error:{type(e).__name__}"
    duration_s = time.monotonic() - t0
    best_y = wrapped.best_y if np.isfinite(wrapped.best_y) else float("inf")
    gap = best_y - f_opt
    return {
        "func_id": func_id,
        "instance_id": instance_id,
        "seed": seed,
        "best_y": best_y,
        "f_opt": f_opt,
        "gap": gap,
        "n_fevals": wrapped.calls,
        "duration_s": duration_s,
        "status": status,
    }


def main() -> None:
    spec_path = Path(__file__).resolve().parent / "problem_spec.json"
    spec = json.loads(spec_path.read_text())
    # Import the agent's optimizer lazily, after problems is on sys.path.
    from optimizer import optimize as optimize_fn  # noqa: WPS433 (by design)

    records: list[dict] = []
    total_t0 = time.monotonic()
    for func_id in spec["functions"]:
        for instance_id in spec["instances"]:
            for seed in spec["seeds"]:
                rec = _run_one(
                    func_id=int(func_id),
                    instance_id=int(instance_id),
                    seed=int(seed),
                    dim=int(spec["dim"]),
                    budget=int(spec["budget"]),
                    wallclock_cap_s=int(spec["wallclock_cap_s"]),
                    optimize_fn=optimize_fn,
                )
                records.append(rec)
                print(
                    f"func_id={rec['func_id']} instance_id={rec['instance_id']} "
                    f"seed={rec['seed']} best_y={rec['best_y']:.6f} "
                    f"f_opt={rec['f_opt']:.6f} gap={rec['gap']:.6f} "
                    f"n_fevals={rec['n_fevals']} duration_s={rec['duration_s']:.3f} "
                    f"status={rec['status']}",
                    flush=True,
                )
    total_duration_s = time.monotonic() - total_t0

    gaps = np.array([max(r["gap"], 1e-12) for r in records])
    log_gaps = np.log10(gaps)
    mean_log_gap = float(np.mean(log_gaps))
    median_log_gap = float(np.median(log_gaps))
    n_errors = sum(
        1 for r in records
        if r["status"].startswith("error") or r["status"] == "timeout"
    )

    per_func: dict[str, float] = {}
    for func_id in sorted(set(r["func_id"] for r in records)):
        mask = np.array([r["func_id"] == func_id for r in records])
        per_func[f"f{func_id}_mean_log_gap"] = float(np.mean(log_gaps[mask]))

    print("\n---")
    print(f"mean_log_gap: {mean_log_gap:.6f}")
    print(f"median_log_gap: {median_log_gap:.6f}")
    print(f"n_tuples: {len(records)}")
    print(f"n_errors: {n_errors}")
    for k in sorted(per_func):
        print(f"{k}: {per_func[k]:.6f}")
    print(f"total_duration_s: {total_duration_s:.3f}")


if __name__ == "__main__":
    main()
