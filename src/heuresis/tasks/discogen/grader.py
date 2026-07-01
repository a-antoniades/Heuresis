"""DiscoGen grading server.

Parses run_main.py JSON output, normalizes per-dataset scores
against baselines, returns aggregated baseline-normalized mean.
"""
from __future__ import annotations

import json
import math
from typing import Any

from heuresis.grading import GradingServer


class DiscoGenGrader(GradingServer):
    """Grades a run.log from discogen's run_main.py output.

    Extracts per-dataset return_mean values, normalizes each against
    its baseline score, and returns the mean of normalized scores.

    For ``objective="max"``, normalization is ``return_mean / baseline``.
    For ``objective="min"``, normalization is ``baseline / return_mean``
    (lower raw score = higher normalized score).

    Args:
        socket_path: Path for the Unix grading socket.
        baselines: Maps dataset path prefixes (e.g. ``"./MinAtar/Breakout"``)
            to baseline scores.
        objective: ``"max"`` or ``"min"`` — direction of improvement.
    """

    input_files = ["run.log"]

    def __init__(
        self,
        socket_path: Any,
        baselines: dict[str, float],
        objective: str = "max",
    ) -> None:
        super().__init__(socket_path)
        self.baselines = baselines
        self.objective = objective

    def grade(self, files: dict[str, bytes]) -> dict[str, Any]:
        if "run.log" not in files:
            return {
                "score": None,
                "valid": False,
                "details": {"error": "No run.log found. Run: python run_main.py > run.log 2>&1"},
            }

        try:
            text = files["run.log"].decode(errors="replace")
        except Exception as e:
            return {
                "score": None,
                "valid": False,
                "details": {"error": f"Could not decode run.log: {e}"},
            }

        metrics = self._parse_last_json(text)
        if metrics is None:
            return {
                "score": None,
                "valid": False,
                "details": {"error": "No JSON output found in run.log"},
            }

        per_dataset: dict[str, dict[str, float]] = {}
        normalized_scores: list[float] = []

        for dataset_path, baseline in self.baselines.items():
            if dataset_path not in metrics:
                return {
                    "score": None,
                    "valid": False,
                    "details": {"error": f"Missing dataset in output: {dataset_path}. Found: {list(metrics.keys())}"},
                }
            ds_metrics = metrics[dataset_path]
            return_mean = ds_metrics.get("return_mean")
            if return_mean is None or not isinstance(return_mean, (int, float)):
                return {
                    "score": None,
                    "valid": False,
                    "details": {"error": f"Missing or invalid return_mean for {dataset_path}: got {return_mean!r}"},
                }

            if self.objective == "max":
                if baseline == 0:
                    normalized = 0.0
                else:
                    normalized = return_mean / baseline
            else:
                if return_mean == 0:
                    normalized = 0.0
                else:
                    normalized = baseline / return_mean

            if math.isnan(normalized) or math.isinf(normalized):
                return {
                    "score": None,
                    "valid": False,
                    "details": {"error": f"Invalid normalized score for {dataset_path}: {normalized}"},
                }

            per_dataset[dataset_path] = {
                "return_mean": return_mean,
                "normalized": normalized,
            }
            normalized_scores.append(normalized)

        score = sum(normalized_scores) / len(normalized_scores)
        return {
            "score": score,
            "valid": True,
            "details": {
                "per_dataset": per_dataset,
                "is_lower_better": self.objective == "min",
            },
        }

    @staticmethod
    def _parse_last_json(text: str) -> dict | None:
        """Extract the last JSON object from text (searching from bottom up)."""
        for line in reversed(text.strip().split("\n")):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        return None
