"""BBOB grading server. Parses `mean_log_gap` from driver-emitted run.log.

Both the in-sandbox `grade run.log` invocation and the host-side fallback
(execute() reads run.log when the agent never called grade) route through
this class. Lower mean_log_gap = better.
"""
from __future__ import annotations

import re
from typing import Any

from heuresis.grading import GradingServer

_MEAN_LOG_GAP_RE = re.compile(r"^mean_log_gap:\s+([-\d.]+)", re.MULTILINE)
_METRIC_RE = re.compile(r"^(\w+):\s+([-\d.]+)", re.MULTILINE)


def _parse_run_log(text: str) -> dict[str, Any]:
    # Treat an explicit FAIL marker anywhere in the tail as a hard failure.
    if "FAIL" in "\n".join(text.split("\n")[-5:]):
        return {
            "score": None,
            "valid": False,
            "details": {"error": "Driver failed (FAIL marker)"},
        }

    matches = _MEAN_LOG_GAP_RE.findall(text)
    if not matches:
        return {
            "score": None,
            "valid": False,
            "details": {"error": "No mean_log_gap found in run.log"},
        }

    score = float(matches[-1])

    details: dict[str, Any] = {"is_lower_better": True}
    last_sep = text.rfind("\n---\n")
    if last_sep >= 0:
        for key, value in _METRIC_RE.findall(text[last_sep:]):
            if key != "mean_log_gap":
                try:
                    details[key] = float(value)
                except ValueError:
                    pass
    return {"score": score, "valid": True, "details": details}


class BBOBGrader(GradingServer):
    """Grader for BBOB-style optimizer runs.

    Score = mean_log_gap (lower is better). Also surfaces per-function breakdown
    and diagnostic counters from the summary block.
    """

    input_files = ["run.log"]

    def grade(self, files: dict[str, bytes]) -> dict[str, Any]:
        if "run.log" not in files:
            return {
                "score": None,
                "valid": False,
                "details": {"error": "No run.log found"},
            }
        try:
            text = files["run.log"].decode(errors="replace")
        except Exception as e:
            return {
                "score": None,
                "valid": False,
                "details": {"error": f"Could not decode run.log: {e}"},
            }
        return _parse_run_log(text)
