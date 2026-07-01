"""In-memory index over accepted / failed_moi / failed_train runs.

Each bucket holds normalized embedding vectors + text + per-bucket metadata.
KNN is brute-force cosine: matrix @ qvec on a pre-normalized (N, D) matrix.
Thread-safe for concurrent reads under RLock.

Persistence is delegated to an optional Experiment handle passed at
construction time; the index itself is pure in-memory. Callers rebuild
the index from a store via `rebuild_from_experiment(exp, embedder_name)`.
"""

from __future__ import annotations

import logging
import random as _random
import threading
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

from heuresis.qd.core.embedding import Embedder, canonicalize_text, text_hash

logger = logging.getLogger(__name__)

Bucket = Literal["accepted", "failed_moi", "failed_train"]
BUCKETS: tuple[Bucket, ...] = ("accepted", "failed_moi", "failed_train")


@dataclass
class Neighbor:
    """Result of a top_k query."""

    run_id: str
    similarity: float
    bucket: str
    plan: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Entry:
    """One row's bookkeeping — text + per-bucket metadata."""

    run_id: str
    plan: str
    bucket: Bucket
    text_hash: str
    # Per-bucket metadata (one of these is set depending on bucket):
    score: float | None = None        # accepted
    reasoning: str | None = None      # failed_moi
    failure_mode: str | None = None   # failed_train


@dataclass
class _BucketState:
    """In-memory matrix + parallel run_id list for one bucket."""

    matrix: np.ndarray          # (N, D), normalized
    run_ids: list[str]
    row_index: dict[str, int]   # run_id -> row

    @classmethod
    def empty(cls, dim: int) -> "_BucketState":
        return cls(
            matrix=np.zeros((0, dim), dtype=np.float32),
            run_ids=[],
            row_index={},
        )


class ArchiveIndex:
    """Bucketed in-memory index with KNN retrieval."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        experiment: Optional[Any] = None,  # Experiment; typed Any to avoid circular import
        text_kind: str = "plan",
    ) -> None:
        self._embedder = embedder
        self._experiment = experiment
        self._text_kind = text_kind
        self._lock = threading.RLock()
        self._buckets: dict[Bucket, _BucketState] = {
            b: _BucketState.empty(embedder.dim) for b in BUCKETS
        }
        self._entries: dict[str, Entry] = {}
        self._anchor_probs: list[float] = []    # parallel to accepted.run_ids
        self._rng = _random.Random(42)

    # --- public API ---------------------------------------------------

    @property
    def accepted_size(self) -> int:
        with self._lock:
            return len(self._buckets["accepted"].run_ids)

    def size(self, bucket: Bucket) -> int:
        with self._lock:
            return len(self._buckets[bucket].run_ids)

    def add_accepted(self, run_id: str, plan: str, score: float) -> None:
        self._add(run_id, plan, bucket="accepted", score=score)

    def add_rejected_moi(self, run_id: str, plan: str, reasoning: str) -> None:
        self._add(run_id, plan, bucket="failed_moi", reasoning=reasoning)

    def add_rejected_train(self, run_id: str, plan: str, failure_mode: str) -> None:
        self._add(run_id, plan, bucket="failed_train", failure_mode=failure_mode)

    def get_entry(self, run_id: str) -> Entry:
        with self._lock:
            return self._entries[run_id]

    def top_k_from_text(
        self,
        query_text: str,
        k: int,
        bucket: Bucket,
    ) -> list[Neighbor]:
        """Retrieve top-k neighbors in `bucket` by similarity to `query_text`."""
        qvec = self._embedder.embed_one(canonicalize_text(query_text))
        qvec = _normalize(qvec)
        return self._top_k_from_vec(qvec, k, bucket, exclude_ids=set())

    def top_k_from_run_id(
        self,
        anchor_run_id: str,
        k: int,
        bucket: Bucket,
        *,
        include_self: bool = False,
    ) -> list[Neighbor]:
        """Retrieve top-k neighbors of an existing run's embedding."""
        with self._lock:
            entry = self._entries[anchor_run_id]
            anchor_bucket = self._buckets[entry.bucket]
            row = anchor_bucket.row_index[anchor_run_id]
            qvec = anchor_bucket.matrix[row].copy()
        exclude = set() if include_self else {anchor_run_id}
        return self._top_k_from_vec(qvec, k, bucket, exclude_ids=exclude)

    def sample_anchor(self, *, mark_used: bool = True) -> str | None:
        """Sample a run_id from `accepted` weighted by `_anchor_probs`.

        Matches OMNI-EPIC's `taskgen_choose_probs`:
        - probs are incremented by 1 every call
        - when `mark_used` is True, the chosen entry's prob becomes 0 after
        """
        with self._lock:
            state = self._buckets["accepted"]
            if not state.run_ids:
                return None
            # Increment everyone by 1 (OMNI-EPIC's regrowth step)
            for i in range(len(self._anchor_probs)):
                self._anchor_probs[i] += 1.0
            total = sum(self._anchor_probs)
            # If all probs are zero for some reason, fall back to uniform
            if total <= 0.0:
                weights = [1.0] * len(state.run_ids)
            else:
                weights = list(self._anchor_probs)
            chosen = self._rng.choices(state.run_ids, weights=weights, k=1)[0]
            if mark_used:
                idx = state.row_index[chosen]
                self._anchor_probs[idx] = 0.0
            return chosen

    def anchor_probs(self) -> list[float]:
        """Snapshot of current anchor probabilities (parallel to accepted run_ids)."""
        with self._lock:
            return list(self._anchor_probs)

    def rebuild_from_experiment(self, exp: Any) -> None:
        """Hydrate this index from stored runs + embeddings.

        Reads ``runs`` to find metadata (bucket, score, reasoning, failure_mode, idea).
        Reads ``run_embeddings`` for the matching (embedder, text_kind) rows.
        Skips any run whose embedding is missing — it will be re-embedded next
        time the run is re-added.

        NOTE: This method filters by ``run_type="executor"`` only. Reviewer /
        ideator runs are intentionally excluded — they produce no scored
        artifacts to embed against. If future task families emit scoreable
        runs under other ``run_type`` values, broaden this filter and ensure
        the metadata routing below learns their bucket conventions.
        """
        records = exp.runs(run_type="executor")
        stored = exp.get_embeddings(
            embedder=self._embedder.model,
            text_kind=self._text_kind,
        )

        with self._lock:
            # Clear existing state
            for b in BUCKETS:
                self._buckets[b] = _BucketState.empty(self._embedder.dim)
            self._entries.clear()
            self._anchor_probs = []

            for r in records:
                vec = stored.get(r.run_id)
                if vec is None:
                    logger.info(
                        "rebuild: no embedding for %s (embedder=%s, text_kind=%s); skipping",
                        r.run_id, self._embedder.model, self._text_kind,
                    )
                    continue
                meta = r.metadata or {}
                bucket = meta.get("omniepic_bucket")
                if bucket not in BUCKETS:
                    # Legacy / non-OmniEpic runs: infer from validity + score
                    if r.valid and r.score is not None:
                        bucket = "accepted"
                    else:
                        bucket = "failed_train"
                plan = r.idea or ""

                state = self._buckets[bucket]
                nvec = np.asarray(vec, dtype=np.float32)
                # Defensive re-normalize in case stored vectors drift
                nvec = _normalize(nvec)
                if state.matrix.shape[0] == 0:
                    state.matrix = nvec.reshape(1, -1).astype(np.float32)
                else:
                    state.matrix = np.vstack([state.matrix, nvec[np.newaxis, :]])
                state.row_index[r.run_id] = len(state.run_ids)
                state.run_ids.append(r.run_id)
                if bucket == "accepted":
                    self._anchor_probs.append(1.0)

                self._entries[r.run_id] = Entry(
                    run_id=r.run_id,
                    plan=plan,
                    bucket=bucket,
                    text_hash=text_hash(plan),
                    score=r.score if bucket == "accepted" else None,
                    reasoning=meta.get("omniepic_reasoning") if bucket == "failed_moi" else None,
                    failure_mode=meta.get("omniepic_failure_mode") if bucket == "failed_train" else None,
                )

    # --- internal -----------------------------------------------------

    def _add(
        self,
        run_id: str,
        plan: str,
        *,
        bucket: Bucket,
        score: float | None = None,
        reasoning: str | None = None,
        failure_mode: str | None = None,
    ) -> None:
        canon = canonicalize_text(plan)
        th = text_hash(plan)
        vec = self._embedder.embed_one(canon)
        nvec = _normalize(vec)

        with self._lock:
            # Remove prior entry for this run_id regardless of bucket
            if run_id in self._entries:
                self._remove_locked(run_id)

            state = self._buckets[bucket]
            if state.matrix.shape[0] == 0:
                state.matrix = nvec.reshape(1, -1).astype(np.float32)
            else:
                state.matrix = np.vstack([state.matrix, nvec[np.newaxis, :]])
            state.row_index[run_id] = len(state.run_ids)
            state.run_ids.append(run_id)
            if bucket == "accepted":
                self._anchor_probs.append(1.0)

            self._entries[run_id] = Entry(
                run_id=run_id,
                plan=plan,
                bucket=bucket,
                text_hash=th,
                score=score,
                reasoning=reasoning,
                failure_mode=failure_mode,
            )

        if self._experiment is not None:
            try:
                self._experiment.save_embedding(
                    run_id,
                    text_kind=self._text_kind,
                    embedder=self._embedder.model,
                    vector=nvec,
                    text_hash=th,
                    normalized=True,
                )
            except Exception as exc:
                logger.warning(
                    "ArchiveIndex._add: save_embedding(run_id=%s) failed: %s; "
                    "in-memory entry kept, persistence skipped",
                    run_id, exc,
                )

    def _remove_locked(self, run_id: str) -> None:
        """Caller must hold self._lock."""
        entry = self._entries.pop(run_id)
        state = self._buckets[entry.bucket]
        row = state.row_index.pop(run_id)
        mask = np.ones(state.matrix.shape[0], dtype=bool)
        mask[row] = False
        state.matrix = state.matrix[mask]
        state.run_ids.pop(row)
        if entry.bucket == "accepted":
            self._anchor_probs.pop(row)
        # Re-index all rows (O(N) full rebuild after compaction)
        for i, rid in enumerate(state.run_ids):
            state.row_index[rid] = i

    def _top_k_from_vec(
        self,
        qvec: np.ndarray,
        k: int,
        bucket: Bucket,
        *,
        exclude_ids: set[str],
    ) -> list[Neighbor]:
        with self._lock:
            state = self._buckets[bucket]
            n = state.matrix.shape[0]
            if n == 0 or k <= 0:
                return []
            sims = state.matrix @ qvec  # (N,)
            # Rank descending; then apply exclusion and k cap.
            order = np.argsort(-sims)
            out: list[Neighbor] = []
            for idx in order:
                rid = state.run_ids[idx]
                if rid in exclude_ids:
                    continue
                entry = self._entries[rid]
                out.append(
                    Neighbor(
                        run_id=rid,
                        similarity=float(sims[idx]),
                        bucket=bucket,
                        plan=entry.plan,
                        meta=_entry_meta(entry),
                    )
                )
                if len(out) >= k:
                    break
            return out


def _normalize(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n == 0.0:
        return vec.astype(np.float32)
    return (vec / n).astype(np.float32)


def _entry_meta(entry: Entry) -> dict[str, Any]:
    m: dict[str, Any] = {}
    if entry.score is not None:
        m["score"] = entry.score
    if entry.reasoning is not None:
        m["reasoning"] = entry.reasoning
    if entry.failure_mode is not None:
        m["failure_mode"] = entry.failure_mode
    return m
