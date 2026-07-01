"""NanoGPT grading server.

Parses run.log from a GPT pretraining run to extract val_bpb and
other metrics. Lower val_bpb = better.

Both the socket path (agent calls ``grade run.log``) and the host-side
fallback (``execute()`` reads run.log directly when the agent never
called grade) route through the same ``grade()`` method. ``input_files``
declares which workspace files the fallback should read.
"""

from __future__ import annotations

import re
from typing import Any

from heuresis.grading import GradingServer

_VAL_BPB_RE = re.compile(r"^val_bpb:\s+([\d.]+)", re.MULTILINE)
_METRIC_RE = re.compile(r"^(\w+):\s+([\d.]+)", re.MULTILINE)


def _parse_run_log(text: str) -> dict[str, Any]:
    if "FAIL" in text.split("\n")[-5:]:
        return {
            "score": None,
            "valid": False,
            "details": {"error": "Training failed (loss exploded)"},
        }

    matches = _VAL_BPB_RE.findall(text)
    if not matches:
        return {
            "score": None,
            "valid": False,
            "details": {"error": "No val_bpb found in run.log. Training may have crashed."},
        }

    val_bpb = float(matches[-1])

    details: dict[str, Any] = {"is_lower_better": True}
    last_sep = text.rfind("\n---\n")
    if last_sep >= 0:
        for key, value in _METRIC_RE.findall(text[last_sep:]):
            if key != "val_bpb":
                details[key] = float(value)

    return {"score": val_bpb, "valid": True, "details": details}


class NanoGPTGrader(GradingServer):
    """Grades a run.log from nanoGPT training.

    Extracts val_bpb (bits per byte) as the score. Lower is better.
    Also parses training_seconds, peak_vram_mb, mfu_percent, etc.
    from the final summary block.
    """

    input_files = ["run.log"]

    def grade(self, files: dict[str, bytes]) -> dict[str, Any]:
        if "run.log" not in files:
            return {
                "score": None,
                "valid": False,
                "details": {"error": "No run.log found. Run: python train.py > run.log 2>&1"},
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
