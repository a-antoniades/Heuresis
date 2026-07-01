"""Regressions for B1 (execute respects lower_is_better) + B2 (run-id seeding)."""
import inspect

from heuresis import execute, next_run_index


class _Run:
    def __init__(self, run_id):
        self.run_id = run_id


class _Exp:
    def __init__(self, run_ids):
        self._runs = [_Run(r) for r in run_ids]

    def runs(self, run_type=None):
        return self._runs


def test_execute_accepts_lower_is_better():
    # B1: execute must forward score direction (default True, overridable).
    sig = inspect.signature(execute)
    assert "lower_is_better" in sig.parameters
    assert sig.parameters["lower_is_better"].default is True


def test_next_run_index_from_max_suffix_not_iteration():
    # B2: one iteration can emit several exec_NNN (omni retries). The next index
    # must be (max suffix)+1, never derived from the iteration count.
    assert next_run_index(_Exp([])) == 0
    assert next_run_index(_Exp(["exec_000", "exec_001", "exec_002"])) == 3
    # gaps + non-exec ids are handled; the max suffix wins (not the row count)
    assert next_run_index(_Exp(["exec_000", "exec_005", "seed_x"])) == 6
