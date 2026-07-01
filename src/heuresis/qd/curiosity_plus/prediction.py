"""Prediction step: ask the LLM to predict an idea's outcome before execution.

Uses the existing harness/workspace machinery — a lightweight ideator-style
call with a prediction-specific prompt. Parses structured JSON output.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from heuresis.harness import Harness
    from heuresis.workspace import Workspace

from heuresis.qd.curiosity_plus.surprise import Prediction


def predict_outcome(
    harness: Harness,
    workspace: Workspace,
    work_dir: Path,
    *,
    prompt_vars: dict[str, Any],
    timeout: int = 60,
    max_attempts: int = 2,
) -> Prediction | None:
    """Run the prediction step. Returns Prediction or None on parse failure.

    Expects the LLM to write a `prediction.json` file with fields:
      - "valid": bool
      - "fitness": float | null
      - "reasoning": str
      - "confidence": float | null  (optional, 0..1)

    ``max_attempts`` bounds the retry count when parsing fails (default 2 =
    one initial call + one retry). Retries cover two common failure modes:
    Gemini tail-latency timeouts (agent never emits) and single-shot parse
    failures from non-JSON output. On retry, the workspace's prediction
    file + agent log are wiped first so the next attempt starts clean.
    """
    pred_file = work_dir / "prediction.json"
    log_file = work_dir / "agent.log"

    for attempt in range(max_attempts):
        # Start each attempt clean: old prediction + agent log wiped.
        if pred_file.exists():
            pred_file.unlink()
        if attempt > 0 and log_file.exists():
            log_file.unlink()

        harness.run(
            workspace,
            prompt=prompt_vars,
            timeout=timeout,
            path=work_dir,
            stateful=False,
        ).result()

        if pred_file.exists():
            parsed = parse_prediction(pred_file.read_text())
            if parsed is not None:
                return parsed

        # Fallback: some agent runs emit the JSON inline in their chat text
        # instead of calling a write tool. parse_prediction already strips
        # markdown fences, so pull the last text event from agent.log.
        fallback = _extract_text_from_agent_log(log_file)
        if fallback is not None:
            parsed = parse_prediction(fallback)
            if parsed is not None:
                return parsed

    return None


def _extract_text_from_agent_log(log_path: Path) -> str | None:
    """Return the text of the last ``text`` event in an agent.log, or None.

    agent.log is JSONL; each line is an event. We only care about events
    whose ``type`` is ``"text"`` and whose ``part.text`` holds the model's
    assistant message.
    """
    if not log_path.exists():
        return None

    last_text: str | None = None
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "text":
            continue
        part = event.get("part") or {}
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            last_text = text
    return last_text


def parse_prediction(raw: str) -> Prediction | None:
    """Parse a Prediction from raw JSON text. Returns None on failure."""
    if not raw or not raw.strip():
        return None

    # Try direct JSON parse first
    text = raw.strip()
    data: dict[str, Any] | None = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON object from a noisier response
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

    if not isinstance(data, dict):
        return None

    valid_raw = data.get("valid")
    if isinstance(valid_raw, str):
        predicted_valid = valid_raw.strip().lower() in ("true", "yes", "1")
    elif isinstance(valid_raw, bool):
        predicted_valid = valid_raw
    else:
        return None

    fitness_raw = data.get("fitness")
    predicted_fitness: float | None
    if fitness_raw is None:
        predicted_fitness = None
    else:
        try:
            predicted_fitness = float(fitness_raw)
        except (TypeError, ValueError):
            predicted_fitness = None

    confidence_raw = data.get("confidence")
    confidence: float | None
    if confidence_raw is None:
        confidence = None
    else:
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = None

    return Prediction(
        predicted_valid=predicted_valid,
        predicted_fitness=predicted_fitness,
        reasoning=str(data.get("reasoning", "")),
        confidence=confidence,
    )
