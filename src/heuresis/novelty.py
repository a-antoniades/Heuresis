"""Agentic novelty reviewer using HuggingFace papers search.

Launches an agent that searches HF papers for related prior work,
reads relevant papers, and assesses the novelty of a proposed idea.

Usage in any experiment loop::

    reviewer = NoveltyReviewer(harness)

    # After ideation, before execution:
    assessment = reviewer.assess(idea)
    if assessment.novelty < 2:
        # Ask ideator to try again
        harness.run(ideator_ws, "Too similar to prior work.", ...)

The reviewer is just another harness.run() with a specific workspace.
It uses the same primitives as everything else.
"""

from __future__ import annotations

import json
import logging
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from heuresis.harness import Harness
from heuresis.tool import Tool
from heuresis.workspace import Mount, Workspace

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REVIEWER_VENV = _PROJECT_ROOT / "venvs" / "reviewer"

_HF_WRAPPER = """\
#!/bin/bash
exec /workspace/.venv/bin/python /workspace/.venv/bin/hf "$@"
"""

_REVIEW_PROMPT = """\
Evaluate the novelty of the research idea in `idea.md`.

Use this rubric:
- 1 = known technique: the idea is primarily a standard, already-common technique
- 2 = novel combination: the idea mostly combines known techniques in a non-trivial way
- 3 = novel: the core technique itself appears materially new or uncommon

## Workflow

1. Read `idea.md` carefully. Identify the CORE mechanism (not surface details).
2. Use `hf papers search "<keywords>"` to search for related prior work.
   Try 2-3 different keyword queries covering the main concepts.
3. For promising matches, use `hf papers read <paper_id>` to check details.
4. Assess: is the core mechanism already published? Is it a known combination?
5. Write `novelty.json` with EXACTLY this JSON format:

{"novelty": <integer 1-3>, "explanation": "<short explanation grounded in papers you found>"}

Requirements:
- `novelty` must be an integer 1, 2, or 3
- `explanation` must reference specific papers or techniques you found
- Be conservative: prefer 1 or 2 unless the core mechanism is genuinely new
- Do NOT wrap in markdown fences. Just the raw JSON object.
"""


@dataclass(frozen=True)
class NoveltyAssessment:
    """Result of a novelty review."""

    novelty: int
    explanation: str
    raw_response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    # Stats fields from the reviewer run (for persistence/analysis)
    duration_s: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_cost: float | None = None


class NoveltyReviewer:
    """Agentic novelty reviewer backed by HuggingFace papers search.

    Launches an agent inside a sandbox with the ``hf`` CLI available.
    The agent searches for related papers, reads them, and writes a
    structured novelty assessment.

    Works with any harness and can be plugged into any experiment loop.
    """

    def __init__(
        self,
        harness: Harness,
        *,
        timeout: int = 300,
        venv: Path | None = None,
    ) -> None:
        self.harness = harness
        self.timeout = timeout
        self._venv = venv or _REVIEWER_VENV
        self._hf_cache = Path.home() / ".cache" / "huggingface"

        hf_tool = self._create_hf_tool()
        self._workspace = Workspace(
            tools=[hf_tool],
            prompt=_REVIEW_PROMPT,
            venv=self._venv,
            project_extra="novelty",
        )

    @staticmethod
    def _create_hf_tool() -> Tool:
        """Create an hf CLI wrapper tool that works inside bwrap."""
        tools_dir = _PROJECT_ROOT / "src" / "heuresis" / "tools"
        wrapper_path = tools_dir / "hf_wrapper"
        if not wrapper_path.exists():
            tools_dir.mkdir(parents=True, exist_ok=True)
            wrapper_path.write_text(_HF_WRAPPER)
            wrapper_path.chmod(wrapper_path.stat().st_mode | stat.S_IEXEC)
        return Tool(
            name="hf",
            binary=wrapper_path,
            docs=(
                "HuggingFace CLI. Use 'hf papers search \"query\"' to find papers, "
                "'hf papers read <paper_id>' to read full paper content, "
                "'hf papers info <paper_id>' for metadata."
            ),
        )

    def assess(
        self,
        idea: str,
        *,
        workspace_path: Path | None = None,
        idea_id: str | None = None,
    ) -> NoveltyAssessment:
        """Assess the novelty of an idea by searching HF papers.

        Launches an agent that searches for prior work and returns
        a structured assessment. Blocks until the agent finishes.

        Parameters
        ----------
        idea:
            The full idea text to assess.
        workspace_path:
            Where to run the reviewer agent. Auto-generated if omitted.
        idea_id:
            Optional identifier for logging.
        """
        if workspace_path is None:
            workspace_path = Path(f"review_{idea_id or 'idea'}")

        workspace_path.mkdir(parents=True, exist_ok=True)
        (workspace_path / "idea.md").write_text(idea)

        mounts: list[Path | Mount] = []
        if self._hf_cache.exists():
            mounts.append(Mount(
                source=self._hf_cache,
                target="/workspace/.cache/huggingface",
                readonly=True,
            ))

        result = self.harness.run(
            self._workspace,
            prompt=_REVIEW_PROMPT,
            mounts=mounts,
            timeout=self.timeout,
            path=workspace_path,
        ).result()

        assessment = self._parse_result(workspace_path, result.exit_code)

        # Enrich with reviewer-run stats
        from heuresis.parsing import parse_workspace

        info = parse_workspace(result.workspace)
        # parse_workspace parses agent.log for tokens/cost; fall back to result.stats
        return NoveltyAssessment(
            novelty=assessment.novelty,
            explanation=assessment.explanation,
            raw_response=assessment.raw_response,
            metadata=assessment.metadata,
            duration_s=result.stats.get("duration"),
            input_tokens=info.get("input_tokens") or result.stats.get("input_tokens"),
            output_tokens=info.get("output_tokens") or result.stats.get("output_tokens"),
            total_cost=info.get("total_cost") or result.stats.get("total_cost"),
        )

    def _parse_result(self, workspace: Path, exit_code: int) -> NoveltyAssessment:
        novelty_path = workspace / "novelty.json"

        if not novelty_path.exists():
            logger.warning("Reviewer did not produce novelty.json at %s", workspace)
            return NoveltyAssessment(
                novelty=1,
                explanation="Reviewer failed to produce assessment",
                metadata={"error": "no novelty.json", "exit_code": exit_code},
            )

        raw = novelty_path.read_text(errors="replace").strip()
        assessment = _parse_novelty_response(raw)
        if assessment is None:
            logger.warning("Failed to parse novelty.json at %s", workspace)
            return NoveltyAssessment(
                novelty=1,
                explanation="Reviewer produced invalid assessment",
                raw_response=raw,
                metadata={"error": "parse_failed", "exit_code": exit_code},
            )

        return NoveltyAssessment(
            novelty=assessment["novelty"],
            explanation=assessment["explanation"],
            raw_response=raw,
            metadata={"exit_code": exit_code, "workspace": str(workspace)},
        )


def _parse_novelty_response(raw: str) -> dict[str, Any] | None:
    """Parse a novelty JSON response, handling markdown fences."""
    if not raw:
        return None

    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None
    if "novelty" not in data or "explanation" not in data:
        return None

    try:
        novelty = int(data["novelty"])
    except (TypeError, ValueError):
        return None

    if novelty not in (1, 2, 3):
        return None

    explanation = str(data["explanation"]).strip()
    if not explanation:
        return None

    return {"novelty": novelty, "explanation": explanation}
