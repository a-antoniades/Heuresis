"""MoI reviewer for the OMNI-EPIC search strategy.

The reviewer is a single-shot Gemini call that decides whether a candidate
research idea is "interesting" enough to execute, given K nearest accepted
entries from a shared ``ArchiveIndex``. It is stateless and has no retry
policy of its own; retry/loop wiring belongs to the Phase 3 strategy.

See ``docs/concepts.md`` for where OMNI-EPIC fits in the experiment flow.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from jinja2 import Environment, StrictUndefined

from heuresis.qd.core.archive_index import ArchiveIndex, Neighbor
from heuresis.tasks import baseline_scores, load_yaml, lower_is_better

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_PATH = Path(__file__).parent / "moi_prompt.j2"

@dataclass(frozen=True)
class MoIAssessment:
    """Result of a single MoI review.

    Attributes:
        interesting: The reviewer's binary verdict.
        reasoning: One to three sentences explaining the verdict.
        retrieved_ids: ``run_id``s of the K nearest accepted entries shown to
            the reviewer (empty during the seed phase).
        raw_response: Full model output, retained for debugging.
        duration_s: Wall-clock seconds spent in ``review()``.
        input_tokens: Prompt tokens reported by the API.
        output_tokens: Completion tokens reported by the API.
        total_cost: Estimated USD cost (zero during the seed phase).
    """

    interesting: bool
    reasoning: str
    retrieved_ids: list[str]
    raw_response: str
    duration_s: float
    input_tokens: int
    output_tokens: int
    total_cost: float


@dataclass(frozen=True)
class MoIContext:
    """Optional prompt-context overrides for runtime-generated tasks.

    Static tasks can continue relying on ``task_dir`` files. Dynamic tasks
    such as DiscoGen can pass generated task text and seed code without
    teaching the reviewer about their workspace layout.
    """

    task_name: str | None = None
    task_description: str | None = None
    domain_description: str | None = None
    problem_text: str | None = None
    seed_code: str | None = None
    metric: str | None = None
    baseline: float | None = None
    lower_is_better: bool | None = None


class MoIReviewError(Exception):
    """Raised when the reviewer cannot produce a valid assessment.

    Cases: all Gemini keys exhausted, ``response.text`` is None (safety
    filter), or structured output fails to parse. The Phase 3 strategy
    decides whether to fail-open (treat as interesting), fail-closed, or
    retry — this class never silently fabricates a verdict.
    """


class MoIReviewer:
    """Single-shot MoI gate over a shared ``ArchiveIndex``.

    Reads task metadata from YAMLs at construction time, then evaluates each
    candidate idea by retrieving K nearest accepted entries and asking
    Gemini to judge. Stateless; no retry policy of its own.
    """

    def __init__(
        self,
        archive_index: ArchiveIndex,
        task_dir: Path,
        *,
        model: str = "gemini-3.1-pro-preview",
        k: int = 10,
        min_archive_size: int = 10,
        api_keys: list[str] | None = None,
        api_keys_file: Path | None = None,
        prompt_path: Path | None = None,
        temperature: float = 1.0,
        client_factory: Callable[[str], Any] | None = None,
        context: MoIContext | None = None,
    ) -> None:
        self._archive_index = archive_index
        self._task_dir = task_dir
        self._model = model
        self._k = k
        self._min_archive_size = min_archive_size
        self._temperature = temperature

        # Load task config (fail fast on missing files / keys)
        cfg = load_yaml(task_dir, "task_config.yaml")
        scores = baseline_scores(task_dir)

        def override(field: str, fallback: Any) -> Any:
            value = getattr(context, field) if context is not None else None
            return fallback if value is None else value

        self._task_name: str = override("task_name", cfg["name"])
        self._task_description: str = override("task_description", cfg["description"])
        self._metric: str = override("metric", scores["metric"])
        self._baseline: float | None = override("baseline", scores.get("baseline"))
        self._lower_is_better: bool = override(
            "lower_is_better",
            lower_is_better(task_dir),
        )

        # description.md is optional but expected
        desc_path = task_dir / "description.md"
        domain_description = override("domain_description", None)
        self._domain_description: str = (
            domain_description
            if domain_description is not None
            else desc_path.read_text() if desc_path.is_file() else ""
        )

        # editable file (seed code shown to the reviewer)
        editable_name = cfg.get("editable")
        if context and context.seed_code is not None:
            self._seed_code = context.seed_code
        elif editable_name and (task_dir / editable_name).is_file():
            self._seed_code = (task_dir / editable_name).read_text()
        else:
            self._seed_code = ""

        # un-rendered problem template shown as raw text
        problem_name = cfg.get("templates", {}).get("problem")
        problem_text = override("problem_text", None)
        self._problem_text: str = (
            problem_text
            if problem_text is not None
            else (task_dir / problem_name).read_text() if problem_name else ""
        )

        # Jinja prompt template
        prompt_path = prompt_path or _DEFAULT_PROMPT_PATH
        env = Environment(undefined=StrictUndefined, keep_trailing_newline=True)
        self._prompt_template = env.from_string(prompt_path.read_text())

        # Gemini key setup (lazy: clients constructed on first call)
        if api_keys is not None:
            self._api_keys = list(api_keys)
        elif api_keys_file is not None:
            from heuresis.api_keys import read_keys_file

            self._api_keys = read_keys_file(api_keys_file)
        else:
            from heuresis.api_keys import load_api_keys

            self._api_keys = load_api_keys("gemini")
        if not self._api_keys:
            raise ValueError(
                "MoIReviewer requires at least one API key "
                "(set GEMINI_API_KEYS / GEMINI_API_KEY / GOOGLE_GENERATIVE_AI_API_KEY, "
                "or pass api_keys / api_keys_file)"
            )
        self._client_factory = client_factory or self._default_factory
        self._clients: dict[str, Any] = {}

    @staticmethod
    def _default_factory(api_key: str) -> Any:
        from google import genai  # type: ignore[import-not-found]
        return genai.Client(api_key=api_key)

    # --- Public API ---------------------------------------------------------

    def review(self, idea: str) -> MoIAssessment:
        """Judge whether ``idea`` is interesting given the current archive."""
        t0 = time.monotonic()

        # 1. Seed gate
        if self._archive_index.accepted_size < self._min_archive_size:
            return MoIAssessment(
                interesting=True,
                reasoning="seed phase: archive has < min_archive_size accepted entries",
                retrieved_ids=[],
                raw_response="",
                duration_s=time.monotonic() - t0,
                input_tokens=0,
                output_tokens=0,
                total_cost=0.0,
            )

        # 2. Retrieve
        neighbors: list[Neighbor] = self._archive_index.top_k_from_text(
            idea, k=self._k, bucket="accepted",
        )

        # 3. Render prompt
        prompt = self._prompt_template.render(
            task_name=self._task_name,
            task_description=self._task_description,
            domain_description=self._domain_description,
            problem=self._problem_text,
            seed_code=self._seed_code,
            metric=self._metric,
            metric_direction=(
                "lower is better" if self._lower_is_better else "higher is better"
            ),
            baseline=self._baseline,
            candidate=idea,
            examples=[
                {
                    "run_id": n.run_id,
                    "similarity": n.similarity,
                    "score": n.meta.get("score"),
                    "plan": n.plan,
                }
                for n in neighbors
            ],
        )

        # 4+5. Call Gemini with key rotation, retry on transient parse
        # failures. Gemini-3.1-pro-preview intermittently returns the object
        # wrapped in a list `[{...}]` or with invalid escape sequences; a
        # re-call typically succeeds. If all ``max_attempts`` fail we raise,
        # which surfaces a real infrastructure problem rather than silently
        # marking ideas rejected.
        max_attempts = 3
        response = None
        last_exc: str | None = None
        for attempt in range(1, max_attempts + 1):
            response = self._call_with_rotation(prompt)
            if response.text is None:
                last_exc = "empty response (likely safety filter)"
                logger.warning("MoI attempt %d: %s", attempt, last_exc)
                continue
            try:
                parsed = json.loads(response.text)
                if isinstance(parsed, list) and parsed:
                    parsed = parsed[0]   # unwrap single-object array
                if not isinstance(parsed, dict):
                    raise TypeError(f"expected dict, got {type(parsed).__name__}")
                interesting = bool(parsed["interesting"])
                reasoning = str(parsed["reasoning"])
                break  # success
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                last_exc = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "MoI attempt %d parse failed: %s; raw=%r",
                    attempt, last_exc, response.text[:200],
                )
                continue
        else:
            # All attempts exhausted; surface the real problem.
            raw_tail = response.text[:200] if (response and response.text) else "None"
            raise MoIReviewError(
                f"MoI review failed after {max_attempts} attempts; "
                f"last error: {last_exc}; last raw={raw_tail!r}"
            )

        # 6. Return (success path)
        return MoIAssessment(
            interesting=interesting,
            reasoning=reasoning,
            retrieved_ids=[n.run_id for n in neighbors],
            raw_response=response.text,
            duration_s=time.monotonic() - t0,
            input_tokens=int(response.usage_metadata.prompt_token_count),
            output_tokens=int(response.usage_metadata.candidates_token_count),
            total_cost=0.0,
        )

    def _get_client(self, api_key: str) -> Any:
        if api_key not in self._clients:
            self._clients[api_key] = self._client_factory(api_key)
        return self._clients[api_key]

    def _call_with_rotation(self, prompt: str) -> Any:
        """Try each API key until one succeeds. Raise MoIReviewError on full exhaustion."""
        try:
            from google.genai.types import GenerateContentConfig  # type: ignore[import-not-found]
            config: Any = GenerateContentConfig(
                temperature=self._temperature,
                response_mime_type="application/json",
            )
        except ImportError:
            config = None

        last_exc: Exception | None = None
        for attempt, api_key in enumerate(self._api_keys):
            try:
                client = self._get_client(api_key)
                return client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=config,
                )
            except Exception as exc:
                logger.info(
                    "MoIReviewer: key #%d failed (%s); rotating", attempt, exc,
                )
                last_exc = exc
                continue
        raise MoIReviewError(
            f"all {len(self._api_keys)} keys exhausted (last error: {last_exc})"
        ) from last_exc
