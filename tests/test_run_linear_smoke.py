import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LOOP_SMOKE") != "1",
    reason="needs agent + Gemini keys; set RUN_LOOP_SMOKE=1 to run",
)


def test_bbob_linear_one_iteration():
    from heuresis.loops import run_linear
    # 1 attempt, judge disabled, count all attempts so it terminates fast.
    # Flags verified against experiment.py: --count-total, --disable-judge.
    run_linear("bbob",
               argv=["--num-iterations", "1", "--count-total",
                     "--num-ideators", "1", "--disable-judge"])
    # If it returns without raising, the wiring is sound.
