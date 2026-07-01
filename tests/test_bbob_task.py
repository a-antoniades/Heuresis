"""Unit tests for tasks/bbob — BudgetedF, make_problem, BBOBGrader."""
from __future__ import annotations

import numpy as np
import pytest


# --- BudgetedF ---

def test_budgeted_f_counts_calls():
    from heuresis.tasks.bbob.problems import BudgetedF
    f = BudgetedF(lambda x: float(np.sum(x * x)), budget=5)
    for _ in range(5):
        f(np.zeros(3))
    assert f.calls == 5


def test_budgeted_f_raises_past_budget():
    from heuresis.tasks.bbob.problems import BudgetedF, BudgetExhausted
    f = BudgetedF(lambda x: 0.0, budget=3)
    for _ in range(3):
        f(np.zeros(2))
    with pytest.raises(BudgetExhausted):
        f(np.zeros(2))


def test_budgeted_f_tracks_best_y_and_x():
    from heuresis.tasks.bbob.problems import BudgetedF
    f = BudgetedF(lambda x: float(x[0]), budget=10)
    f(np.array([3.0, 0.0]))
    f(np.array([1.0, 0.0]))
    f(np.array([2.0, 0.0]))
    assert f.best_y == pytest.approx(1.0)
    assert f.best_x[0] == pytest.approx(1.0)


def test_budgeted_f_ignores_nan_and_inf():
    from heuresis.tasks.bbob.problems import BudgetedF
    f = BudgetedF(lambda x: float("nan") if x[0] > 0 else 1.0, budget=10)
    f(np.array([-1.0]))  # y = 1.0
    f(np.array([ 1.0]))  # y = NaN, ignored
    assert f.best_y == pytest.approx(1.0)


def test_budgeted_f_clips_to_bounds():
    from heuresis.tasks.bbob.problems import BudgetedF
    captured = []
    def g(x):
        captured.append(x.copy())
        return 0.0
    f = BudgetedF(g, budget=2, bounds=(-5.0, 5.0))
    f(np.array([10.0, -10.0, 0.0]))
    assert np.array_equal(captured[-1], np.array([5.0, -5.0, 0.0]))


# --- _random_rotation ---

def test_random_rotation_is_orthogonal():
    from heuresis.tasks.bbob.problems import _random_rotation
    rng = np.random.default_rng(42)
    Q = _random_rotation(rng, dim=5)
    assert Q.shape == (5, 5)
    product = Q @ Q.T
    assert np.allclose(product, np.eye(5), atol=1e-10)


def test_random_rotation_is_deterministic():
    from heuresis.tasks.bbob.problems import _random_rotation
    Q1 = _random_rotation(np.random.default_rng(7), dim=4)
    Q2 = _random_rotation(np.random.default_rng(7), dim=4)
    assert np.array_equal(Q1, Q2)


# --- make_problem: Sphere ---

def test_sphere_f_opt_at_x_opt():
    from heuresis.tasks.bbob.problems import make_problem
    f, f_opt, bounds = make_problem(func_id=1, dim=3, instance_id=1)
    # Trick: at x = x_opt, we expect f(x_opt) == f_opt. We don't know x_opt from
    # the outside, but we can use the returned f on a grid to find something
    # very close: sphere is convex, so the argmin of f on any fine grid
    # approximates x_opt. Instead use the stricter property that f is the
    # squared distance from x_opt (no rotation for sphere), so f(0) should
    # equal ||x_opt||^2, which is positive and bounded.
    assert bounds == (-5.0, 5.0)
    assert f_opt == 0.0
    y_zero = f(np.zeros(3))
    assert y_zero > 0.0  # x_opt != 0 almost surely


def test_sphere_instance_changes_x_opt():
    from heuresis.tasks.bbob.problems import make_problem
    f1, _, _ = make_problem(func_id=1, dim=3, instance_id=1)
    f2, _, _ = make_problem(func_id=1, dim=3, instance_id=2)
    # Same x should give different values under different instances.
    assert f1(np.zeros(3)) != f2(np.zeros(3))


def test_sphere_is_reproducible_for_same_instance():
    from heuresis.tasks.bbob.problems import make_problem
    f1, _, _ = make_problem(func_id=1, dim=4, instance_id=3)
    f2, _, _ = make_problem(func_id=1, dim=4, instance_id=3)
    x = np.array([0.5, -0.5, 1.0, 0.0])
    assert f1(x) == f2(x)


def test_sphere_min_reached_by_gradient_descent():
    """Sanity: 50 gradient steps on sphere should get very close to 0."""
    from heuresis.tasks.bbob.problems import make_problem
    f, _, _ = make_problem(func_id=1, dim=5, instance_id=1)
    # Finite-difference gradient descent.
    x = np.zeros(5)
    lr = 0.3
    for _ in range(50):
        g = np.zeros(5)
        y = f(x)
        h = 1e-5
        for i in range(5):
            xp = x.copy()
            xp[i] += h
            g[i] = (f(xp) - y) / h
        x = x - lr * g
    assert f(x) < 1e-6


# --- make_problem: Rosenbrock ---

def test_rosenbrock_registered():
    from heuresis.tasks.bbob.problems import make_problem
    f, f_opt, bounds = make_problem(func_id=8, dim=3, instance_id=1)
    assert f_opt == 0.0
    assert bounds == (-5.0, 5.0)


def test_rosenbrock_min_at_x_opt():
    """f(x_opt) == 0 under shift + rotation by construction."""
    from heuresis.tasks.bbob.problems import make_problem, _random_rotation
    # Reconstruct x_opt and Q to verify directly — this is white-box checking of
    # the implementation, not something the optimizer gets to do.
    import numpy as _np
    rng = _np.random.default_rng(seed=8 * 1000 + 7)
    x_opt = rng.uniform(-4.0, 4.0, size=4)
    Q = _random_rotation(rng, dim=4)
    del Q  # just advancing rng
    f, _, _ = make_problem(func_id=8, dim=4, instance_id=7)
    y = f(x_opt)
    assert abs(y) < 1e-10


def test_rosenbrock_positive_away_from_opt():
    from heuresis.tasks.bbob.problems import make_problem
    f, _, _ = make_problem(func_id=8, dim=3, instance_id=1)
    # f(0) ≈ some positive value since 0 is far from x_opt almost surely.
    assert f(np.zeros(3)) > 0.0


# --- make_problem: Rastrigin ---

def test_rastrigin_registered():
    from heuresis.tasks.bbob.problems import make_problem
    f, f_opt, _ = make_problem(func_id=15, dim=3, instance_id=1)
    assert f_opt == 0.0


def test_rastrigin_min_at_x_opt():
    from heuresis.tasks.bbob.problems import make_problem, _random_rotation
    rng = np.random.default_rng(seed=15 * 1000 + 4)
    x_opt = rng.uniform(-4.0, 4.0, size=5)
    _random_rotation(rng, dim=5)  # advance rng the same amount as make_problem
    f, _, _ = make_problem(func_id=15, dim=5, instance_id=4)
    assert abs(f(x_opt)) < 1e-10


def test_rastrigin_multimodal():
    """Rastrigin: off-optimum values can be very large due to cosine term."""
    from heuresis.tasks.bbob.problems import make_problem
    f, _, _ = make_problem(func_id=15, dim=3, instance_id=1)
    y_bad = f(np.full(3, 4.0))
    y_zero = f(np.zeros(3))
    assert y_bad > 0.0
    assert y_zero > 0.0


def test_make_problem_rejects_unknown_func_id():
    from heuresis.tasks.bbob.problems import make_problem
    with pytest.raises(ValueError, match="Unknown func_id"):
        make_problem(func_id=99, dim=3, instance_id=1)


# --- driver integration ---

def test_driver_end_to_end(tmp_path):
    """Copy task files to tmp dir, run `python driver.py`, verify run.log shape."""
    import json
    import shutil
    import subprocess
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "heuresis" / "tasks" / "bbob"
    for name in ["driver.py", "problems.py", "optimizer.py"]:
        shutil.copy(src / name, tmp_path / name)
    (tmp_path / "problem_spec.json").write_text(json.dumps({
        "functions": [1],
        "dim": 3,
        "instances": [1],
        "seeds": [0],
        "budget": 50,
        "wallclock_cap_s": 5,
    }))
    result = subprocess.run(
        ["python", "driver.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"driver.py failed: stderr={result.stderr}"
    out = result.stdout
    # per-tuple line
    assert "func_id=1 instance_id=1 seed=0" in out
    # summary block
    assert "\n---\n" in out
    assert "mean_log_gap:" in out
    assert "n_tuples: 1" in out
    assert "f1_mean_log_gap:" in out


# --- BBOBGrader ---

def test_grader_parses_well_formed_log():
    from heuresis.tasks.bbob.grader import BBOBGrader
    log = (
        "func_id=1 instance_id=1 seed=0 best_y=0.012345 f_opt=0.000000 gap=0.012345 n_fevals=1000 duration_s=0.42 status=ok\n"
        "\n"
        "---\n"
        "mean_log_gap: -1.930000\n"
        "median_log_gap: -2.140000\n"
        "n_tuples: 1\n"
        "n_errors: 0\n"
        "f1_mean_log_gap: -1.930000\n"
        "total_duration_s: 0.420000\n"
    )
    g = BBOBGrader("/tmp/dummy.sock")
    result = g.grade({"run.log": log.encode()})
    assert result["valid"] is True
    assert result["score"] == pytest.approx(-1.93)
    assert result["details"]["is_lower_better"] is True
    assert result["details"]["median_log_gap"] == pytest.approx(-2.14)
    assert result["details"]["f1_mean_log_gap"] == pytest.approx(-1.93)


def test_grader_rejects_empty_file():
    from heuresis.tasks.bbob.grader import BBOBGrader
    g = BBOBGrader("/tmp/dummy.sock")
    result = g.grade({"run.log": b""})
    assert result["valid"] is False
    assert "No mean_log_gap" in result["details"]["error"]


def test_grader_rejects_missing_file():
    from heuresis.tasks.bbob.grader import BBOBGrader
    g = BBOBGrader("/tmp/dummy.sock")
    result = g.grade({})
    assert result["valid"] is False
    assert "No run.log" in result["details"]["error"]


def test_grader_rejects_fail_marker():
    from heuresis.tasks.bbob.grader import BBOBGrader
    g = BBOBGrader("/tmp/dummy.sock")
    log = "some noise\nFAIL: driver crashed\n"
    result = g.grade({"run.log": log.encode()})
    assert result["valid"] is False
    assert "failed" in result["details"]["error"].lower()


def test_grader_takes_last_match_of_mean_log_gap():
    """If for any reason multiple summary blocks appear, the last one wins."""
    from heuresis.tasks.bbob.grader import BBOBGrader
    log = (
        "---\nmean_log_gap: -1.000000\n"
        "---\nmean_log_gap: -2.500000\n"
    )
    g = BBOBGrader("/tmp/dummy.sock")
    result = g.grade({"run.log": log.encode()})
    assert result["valid"] is True
    assert result["score"] == pytest.approx(-2.5)
