"""CuriosityPlusSearch — prediction-error curiosity with k-NN local regions.

Implements the SearchStrategy interface. Composes:
  - EmbeddingStore
  - Prediction + Surprise
  - LearningProgress
  - Selection (softmax + repetition penalty)
  - Seeding (farthest-point greedy)
  - Curiosity context for prompts

Two phases:
  - Seeding (store.size < n_seed): farthest-point greedy, no curiosity signal.
  - Steady state (store.size >= n_seed): predict → execute → surprise → store;
    anchor chosen via softmax over learning progress. Neighborhoods without
    enough data for a stable LP estimate fall back to raw surprise per-entry
    (handled inside ``select_anchor``), so no separate global phase is needed.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from heuresis.qd.core.base import (
    SearchStrategy,
    compute_generation,
    extract_summary,
)
from heuresis.qd.core.embedding import Embedder, text_hash
from heuresis.qd.curiosity_plus.curiosity_signals import (
    build_curiosity_context,
    build_prediction_context,
)
from heuresis.qd.curiosity_plus.embedding_store import EmbeddingStore
from heuresis.qd.curiosity_plus.selection import select_anchor
from heuresis.qd.curiosity_plus.surprise import (
    Prediction,
    SigmaTracker,
    apply_memory_discount,
    surprise as compute_surprise,
)

logger = logging.getLogger(__name__)

_EMBEDDING_TEXT_KIND = "curiosity_idea"


class CuriosityPlusSearch(SearchStrategy):
    """Prediction-error curiosity search.

    Two phases:
      - Seeding: callers drive cold-start via ``select_seed_candidate``;
        ``select_parents`` returns ``[]`` until the store has ``n_seed`` entries.
      - Steady state: ``select_parents`` samples one anchor via softmax over
        LP (with per-entry fallback to raw surprise for sparse neighborhoods).

    Persistent state:
      - ``store``: vectors + surprise + iteration timestamps
      - ``_anchor_history``: per-ideator FIFO of recent anchors for repetition penalty
      - ``_sigma_tracker``: running σ_f for surprise normalization
      - ``_generation_map``: lineage depth per run_id
    """

    def __init__(
        self,
        embedder: Embedder,
        *,
        k_neighbors: int = 10,
        candidate_window: int = 20,
        softmax_temperature: float = 1.0,
        anchor_history: int = 5,
        novelty_threshold: float = 0.90,
        n_seed: int = 10,
        sigma_window: int = 50,
        lambda_mismatch: float = 1.0,
        lower_is_better: bool = True,
        seed: int = 42,
        memory: bool = False,
        score_weight: float = 0.0,
        tag_classifier: Any | None = None,
        memory_strength: float = 0.0,
        memory_k: int = 10,
        memory_min_k: int = 3,
    ) -> None:
        self.store = EmbeddingStore(embedder)
        self.k = k_neighbors
        self.candidate_window = candidate_window
        self.tau = softmax_temperature
        self.M = anchor_history
        self.novelty_threshold = novelty_threshold
        self.n_seed = n_seed
        self.lambda_mismatch = lambda_mismatch
        self.lower_is_better = lower_is_better
        # Opt-in campaign memory flag (single source of truth read by run.py
        # to wire MemoryStore / MEMORY tool / prompt gating). The strategy
        # itself never uses memory directly.
        self.memory = memory
        # Quality pressure for anchor selection (Change A in curiosity-plus).
        # 0.0 preserves pure LP-softmax behavior; >0 blends in score-rank.
        self.score_weight = float(score_weight)
        # Optional inline tag classifier (Change B). When set, on_result
        # extracts {components, technique_tags} for each idea and stores
        # them on the StoredIdea. select_anchor's repetition penalty then
        # switches to tag-Jaccard distance, so the search measures
        # architectural diversity instead of vocabulary diversity.
        self._tag_classifier: Any | None = tag_classifier
        # Memory-aware surprise (Change E). α=0 disables; α>0 discounts
        # raw surprise by how well the curiosity store's K-NN consensus
        # would have predicted the outcome.
        # Defaults memory_k=10, memory_min_k=3 — separates "search width"
        # (K) from "data threshold" (min_scored). Earlier defaults of 5/5
        # required ALL 5 nearest to be scored, which almost never holds at
        # ~50% valid rate; empirically that fired ≤5% during early run.
        self._memory_strength = float(memory_strength)
        self._memory_k = int(memory_k)
        self._memory_min_k = int(memory_min_k)

        self._sigma_tracker = SigmaTracker(window=sigma_window)
        self._anchor_history: dict[int, list[str]] = {}
        self._current_anchor: dict[int, str | None] = {}
        self._generation_map: dict[str, int] = {}
        self._idea_summaries: dict[str, str] = {}
        self._predictions: dict[str, Prediction] = {}
        self._iteration_counter = 0
        self._counter_lock = threading.Lock()
        self._rng = np.random.default_rng(seed)
        self._experiment: Any | None = None

    def set_experiment(self, experiment: Any) -> None:
        """Bind an Experiment so embeddings persist to its run_embeddings table.

        Must be called before any `on_result` if you want resume to skip
        re-embedding prior ideas. Safe to leave unset (persistence becomes a
        no-op — matches pre-persistence behavior).
        """
        self._experiment = experiment

    # --- Phase ------------------------------------------------------------

    def current_iteration(self) -> int:
        with self._counter_lock:
            return self._iteration_counter

    def next_iteration(self) -> int:
        with self._counter_lock:
            i = self._iteration_counter
            self._iteration_counter += 1
        return i

    def is_seeding(self) -> bool:
        """True while the store is still below ``n_seed`` entries."""
        return self.store.size < self.n_seed

    # --- SearchStrategy interface -----------------------------------------

    def select_parents(self, *, ideator_id: int = 0) -> list[str]:
        """Sample an anchor from recent ideas via softmax(LP) + repetition penalty.

        Returns ``[]`` during the seeding phase — the run.py loop should call
        ``select_seed_candidate`` instead.
        """
        if self.is_seeding():
            self._current_anchor[ideator_id] = None
            return []

        history = self._anchor_history.get(ideator_id, [])
        anchor = select_anchor(
            self.store,
            candidate_window=self.candidate_window,
            tau=self.tau,
            anchor_history=history,
            M=self.M,
            k=self.k,
            rng=self._rng,
            score_weight=self.score_weight,
            lower_is_better=self.lower_is_better,
            tag_repetition=self._tag_classifier is not None,
        )
        self._current_anchor[ideator_id] = anchor
        if anchor is not None:
            history.append(anchor)
            self._anchor_history[ideator_id] = history[-self.M :]
        return [anchor] if anchor else []

    def context(self, *, ideator_id: int = 0) -> str:
        """Return curiosity context (anchor + neighborhood + LP signal)."""
        anchor = self._current_anchor.get(ideator_id)
        if anchor is None:
            return ""
        try:
            return build_curiosity_context(anchor, self.store, k=self.k)
        except KeyError:
            return ""

    def on_result(
        self,
        run_id: str,
        score: float | None,
        features: dict[str, float] | None = None,
        *,
        idea: str | None = None,
        parent_ids: list[str] | None = None,
        ideator_id: int = 0,
        prediction: Prediction | None = None,
        valid: bool | None = None,
    ) -> dict[str, Any]:
        """Record a completed run: embed idea, compute surprise, update store."""
        plan = idea or ""
        generation = compute_generation(parent_ids, self._generation_map)
        self._generation_map[run_id] = generation

        if idea:
            self._idea_summaries[run_id] = extract_summary(idea)

        actual_valid = bool(valid) if valid is not None else (score is not None)

        raw_s = compute_surprise(
            prediction,
            actual_valid=actual_valid,
            actual_fitness=score,
            sigma_tracker=self._sigma_tracker,
            lambda_mismatch=self.lambda_mismatch,
        )

        if prediction is not None:
            self._predictions[run_id] = prediction

        iteration = self.next_iteration()
        # We may overwrite the stored surprise with the memory-discounted
        # value after K-NN consensus below; capture the discount factors
        # for telemetry so analysis can recover the raw signal.
        s = raw_s
        mem_explanation = 0.0
        mem_pred_score: float | None = None
        mem_pred_valid: bool | None = None

        if plan:
            nvec = self.store.add(
                run_id,
                plan,
                iteration=iteration,
                surprise=raw_s,
                score=score,
                valid=actual_valid,
                prediction=prediction,
            )
            self._persist_embedding(run_id, plan, nvec)

            # Change E: memory-aware surprise. Consensus is the K-NN
            # weighted average from this run's curiosity store, excluding
            # the just-added entry. We only discount when α>0, the store
            # has enough scored neighbors, and the LLM made a prediction
            # we can compare against.
            if (
                raw_s is not None
                and self._memory_strength > 0.0
                and self.store.size > self._memory_min_k
                and prediction is not None
            ):
                consensus = self.store.memory_consensus(
                    nvec,
                    k=self._memory_k,
                    exclude_ids={run_id},
                    min_scored=self._memory_min_k,
                )
                if consensus is not None:
                    mem_pred_score, mem_pred_valid = consensus
                    s, mem_explanation = apply_memory_discount(
                        raw_s,
                        prediction=prediction,
                        mem_pred_score=mem_pred_score,
                        mem_pred_valid=mem_pred_valid,
                        actual_fitness=score,
                        actual_valid=actual_valid,
                        sigma=self._sigma_tracker.sigma,
                        alpha=self._memory_strength,
                        lambda_mismatch=self.lambda_mismatch,
                    )
                    # Update the stored entry's surprise so LP / fallback
                    # downstream see the discounted value.
                    if s != raw_s:
                        self.store.update_surprise(run_id, surprise=s)

        meta: dict[str, Any] = {
            "parent_ids": parent_ids or [],
            "generation": generation,
            "curiosity_phase": "seeding" if self.is_seeding() else "steady",
            "curiosity_iteration": iteration,
        }
        if idea is not None:
            meta["idea"] = idea
        if s is not None:
            meta["curiosity_surprise"] = s
        if mem_explanation > 0.0:
            meta["curiosity_raw_surprise"] = raw_s
            meta["curiosity_memory_explanation"] = mem_explanation
            meta["curiosity_memory_predicted_score"] = mem_pred_score
            meta["curiosity_memory_predicted_valid"] = mem_pred_valid
        if prediction is not None:
            meta["curiosity_predicted_valid"] = prediction.predicted_valid
            meta["curiosity_predicted_fitness"] = prediction.predicted_fitness
            meta["curiosity_prediction_reasoning"] = prediction.reasoning
            if prediction.confidence is not None:
                meta["curiosity_prediction_confidence"] = prediction.confidence

        # Change B: extract architectural tags inline so the next anchor
        # selection can use tag-Jaccard distance instead of cosine over
        # text embeddings. Failures are absorbed inside _tag_idea.
        if plan and self._tag_classifier is not None:
            self._tag_idea(run_id, plan, meta)

        return meta

    def rebuild(
        self, records: list[tuple[str, float | None, dict[str, Any]]]
    ) -> None:
        """Restore generation map + sigma tracker from prior records.

        The embedding store is rebuilt separately by the caller (needs idea
        text + embeddings, not just metadata).
        """
        for run_id, score, metadata in records:
            self._generation_map[run_id] = metadata.get("generation", 0)
            idea = metadata.get("idea")
            if idea:
                self._idea_summaries[run_id] = extract_summary(idea)
            if score is not None:
                self._sigma_tracker.observe(score)

    def summary(self) -> str:
        n = self.store.size
        scored = sum(1 for e in self.store.all_entries() if e.score is not None)
        with_surprise = sum(1 for e in self.store.all_entries() if e.surprise is not None)
        phase_str = "seeding" if self.is_seeding() else "steady"
        return (
            f"Curiosity: {phase_str} | store={n} ideas "
            f"({scored} scored, {with_surprise} w/ surprise) | "
            f"σ_f≈{self._sigma_tracker.sigma:.4f}"
        )

    # --- Curiosity-specific helpers ---------------------------------------

    def select_seed_candidate(
        self,
        candidates: list[str],
        *,
        reserve_run_id: str | None = None,
        reserve_iteration: int | None = None,
    ) -> tuple[int, str]:
        """Seeding phase: pick farthest-point candidate from a batch.

        When ``reserve_run_id`` is supplied, the chosen idea is inserted
        into the store under that id so concurrent seed picks see it.
        ``on_result`` later upserts score/surprise for the same id.
        """
        from heuresis.qd.curiosity_plus.seeding import select_seed
        return select_seed(
            self.store,
            candidates,
            reserve_run_id=reserve_run_id,
            reserve_iteration=reserve_iteration,
        )

    def prediction_context(self, *, max_history: int = 10) -> str:
        """Build past-predictions context for the prediction prompt."""
        return build_prediction_context(self.store, max_history=max_history)

    def is_novel(self, idea_text: str) -> bool:
        """Check whether an idea is novel enough vs the store (cosine threshold)."""
        if self.store.size == 0:
            return True
        vec = self.store.embed_texts([idea_text])[0]
        return self.store.novelty_check(vec, threshold=self.novelty_threshold)

    # --- Tagging (Change B) -----------------------------------------------

    def _tag_idea(self, run_id: str, idea: str, meta: dict[str, Any]) -> None:
        """Classify idea → tags → store + meta. Swallows classifier errors."""
        try:
            tagged = self._tag_classifier.classify(idea)
        except Exception as exc:
            logger.warning(
                "CuriosityPlusSearch: tag_classifier.classify(run_id=%s) failed: %s",
                run_id, exc,
            )
            return
        components = tagged.get("components") or []
        technique_tags = tagged.get("technique_tags") or []
        if not components and not technique_tags:
            return
        try:
            self.store.update_tags(
                run_id,
                components=components,
                technique_tags=technique_tags,
            )
        except KeyError:
            # Run is not in the store (e.g. plan was empty). Nothing to update.
            return
        meta["curiosity_components"] = list(components)
        meta["curiosity_technique_tags"] = list(technique_tags)

    # --- Persistence ------------------------------------------------------

    def _persist_embedding(
        self, run_id: str, idea: str, vector: np.ndarray
    ) -> None:
        """Upsert this run's embedding into the experiment's run_embeddings table.

        No-op when ``set_experiment`` was never called. Failures are logged
        and swallowed so persistence hiccups don't derail the search loop.
        """
        if self._experiment is None:
            return
        try:
            self._experiment.save_embedding(
                run_id,
                text_kind=_EMBEDDING_TEXT_KIND,
                embedder=self.store._embedder.model,
                vector=vector,
                text_hash=text_hash(idea),
                normalized=True,
            )
        except Exception as exc:
            logger.warning(
                "CuriosityPlusSearch: save_embedding(run_id=%s) failed: %s; "
                "in-memory entry kept, persistence skipped",
                run_id, exc,
            )

    def load_persisted_embeddings(self) -> dict[str, np.ndarray]:
        """Return {run_id -> vector} saved previously for this experiment.

        Returns {} if no experiment is bound or if the table is empty.
        Vectors come back as (D,) float32; `EmbeddingStore.add` re-normalizes
        defensively so drift is not a correctness concern.
        """
        if self._experiment is None:
            return {}
        try:
            return self._experiment.get_embeddings(
                embedder=self.store._embedder.model,
                text_kind=_EMBEDDING_TEXT_KIND,
            )
        except Exception as exc:
            logger.warning(
                "CuriosityPlusSearch: get_embeddings failed: %s; falling back to re-embed",
                exc,
            )
            return {}
