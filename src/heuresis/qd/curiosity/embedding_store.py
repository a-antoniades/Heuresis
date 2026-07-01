"""In-memory store of (idea, embedding, surprise, iteration) with kNN retrieval.

Thread-safe via RLock (multiple concurrent readers, single writer).
Brute-force cosine similarity — sufficient for ~400 vectors per run.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from heuresis.qd.core.embedding import Embedder
from heuresis.qd.curiosity.surprise import Prediction


@dataclass
class StoredIdea:
    """One entry in the embedding store."""

    run_id: str
    idea: str
    vector: np.ndarray        # (D,) normalized
    surprise: float | None    # None until outcome observed
    iteration: int
    score: float | None = None
    valid: bool | None = None
    prediction: Prediction | None = None   # LLM's pre-execution prediction


class EmbeddingStore:
    """Flat embedding store with kNN, farthest-point selection, and novelty check.

    All vectors are L2-normalized at insertion time so cosine similarity
    reduces to a dot product.
    """

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._lock = threading.RLock()
        self._entries: list[StoredIdea] = []
        self._matrix: np.ndarray = np.zeros((0, embedder.dim), dtype=np.float32)
        self._id_index: dict[str, int] = {}

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def dim(self) -> int:
        return self._embedder.dim

    def add(
        self,
        run_id: str,
        idea: str,
        iteration: int,
        *,
        surprise: float | None = None,
        score: float | None = None,
        valid: bool | None = None,
        vector: np.ndarray | None = None,
        prediction: Prediction | None = None,
    ) -> np.ndarray:
        """Embed and store an idea. Returns the normalized embedding vector.

        If ``vector`` is provided, it is used directly (must already be
        normalized). Otherwise the idea text is embedded via the embedder.
        """
        if vector is not None:
            nvec = _normalize(vector)
        elif run_id in self._id_index:
            # Upsert with no explicit vector — reuse the reserved embedding
            # instead of paying another embedder call for the same idea.
            nvec = self._matrix[self._id_index[run_id]].copy()
        else:
            nvec = _normalize(self._embedder.embed_one(idea))

        entry = StoredIdea(
            run_id=run_id,
            idea=idea,
            vector=nvec,
            surprise=surprise,
            iteration=iteration,
            score=score,
            valid=valid,
            prediction=prediction,
        )
        with self._lock:
            if run_id in self._id_index:
                row = self._id_index[run_id]
                self._entries[row] = entry
                self._matrix[row] = nvec
            else:
                self._id_index[run_id] = len(self._entries)
                self._entries.append(entry)
                if self._matrix.shape[0] == 0:
                    self._matrix = nvec.reshape(1, -1)
                else:
                    self._matrix = np.vstack([self._matrix, nvec[np.newaxis, :]])
        return nvec

    def update_surprise(
        self,
        run_id: str,
        surprise: float,
        *,
        score: float | None = None,
        valid: bool | None = None,
    ) -> None:
        """Set the surprise value for an already-stored idea (post-execution)."""
        with self._lock:
            row = self._id_index[run_id]
            entry = self._entries[row]
            entry.surprise = surprise
            if score is not None:
                entry.score = score
            if valid is not None:
                entry.valid = valid

    def get(self, run_id: str) -> StoredIdea:
        with self._lock:
            return self._entries[self._id_index[run_id]]

    def get_vector(self, run_id: str) -> np.ndarray:
        with self._lock:
            return self._matrix[self._id_index[run_id]].copy()

    def all_entries(self) -> list[StoredIdea]:
        with self._lock:
            return list(self._entries)

    def recent_entries(self, window: int) -> list[StoredIdea]:
        """Return the last ``window`` entries by insertion order."""
        with self._lock:
            return list(self._entries[-window:])

    # --- kNN ---------------------------------------------------------------

    def knn(
        self,
        query: np.ndarray,
        k: int,
        *,
        exclude_ids: set[str] | None = None,
    ) -> list[tuple[StoredIdea, float]]:
        """Return k nearest neighbors by cosine similarity.

        Returns list of (entry, similarity) sorted descending by similarity.
        """
        exclude = exclude_ids or set()
        with self._lock:
            n = self._matrix.shape[0]
            if n == 0 or k <= 0:
                return []
            qvec = _normalize(query)
            sims = self._matrix @ qvec  # (N,)
            order = np.argsort(-sims)
            out: list[tuple[StoredIdea, float]] = []
            for idx in order:
                entry = self._entries[idx]
                if entry.run_id in exclude:
                    continue
                out.append((entry, float(sims[idx])))
                if len(out) >= k:
                    break
            return out

    def knn_by_run_id(
        self,
        run_id: str,
        k: int,
        *,
        include_self: bool = False,
    ) -> list[tuple[StoredIdea, float]]:
        """kNN from an existing entry's embedding."""
        with self._lock:
            row = self._id_index[run_id]
            qvec = self._matrix[row].copy()
        exclude = set() if include_self else {run_id}
        return self.knn(qvec, k, exclude_ids=exclude)

    # --- Farthest-point greedy (Phase 1 seeding) --------------------------

    def farthest_point(
        self,
        candidate_vectors: np.ndarray,
    ) -> int:
        """Return the index of the candidate vector farthest from the store.

        Uses max-min cosine distance: picks the candidate whose nearest
        neighbor in the store is most distant.

        If the store is empty, returns 0 (first candidate).
        ``candidate_vectors`` shape: (B, D).
        """
        with self._lock:
            if self._matrix.shape[0] == 0:
                return 0
            candidates = _normalize_batch(candidate_vectors)
            # (B, N) similarity matrix
            sims = candidates @ self._matrix.T
            # For each candidate, find its maximum similarity to any stored vector
            max_sims = sims.max(axis=1)  # (B,)
            # Pick the candidate with the lowest max-similarity (most distant)
            return int(np.argmin(max_sims))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts. Returns (N, D) normalized."""
        if not texts:
            return np.zeros((0, self._embedder.dim), dtype=np.float32)
        return _normalize_batch(self._embedder.embed(texts))

    # --- Novelty check ----------------------------------------------------

    def novelty_check(
        self,
        vector: np.ndarray,
        threshold: float = 0.90,
    ) -> bool:
        """Return True if the vector is novel (max similarity < threshold)."""
        with self._lock:
            if self._matrix.shape[0] == 0:
                return True
            qvec = _normalize(vector)
            sims = self._matrix @ qvec
            return float(sims.max()) < threshold

    # --- Persistence (save/load) ------------------------------------------

    def save(self, path: Any) -> None:
        """Save store state to an npz file."""
        from pathlib import Path
        path = Path(path)
        with self._lock:
            run_ids = [e.run_id for e in self._entries]
            ideas = [e.idea for e in self._entries]
            surprises = np.array(
                [e.surprise if e.surprise is not None else np.nan for e in self._entries],
                dtype=np.float64,
            )
            iterations = np.array([e.iteration for e in self._entries], dtype=np.int64)
            scores = np.array(
                [e.score if e.score is not None else np.nan for e in self._entries],
                dtype=np.float64,
            )
            valids = np.array(
                [int(e.valid) if e.valid is not None else -1 for e in self._entries],
                dtype=np.int8,
            )
            np.savez(
                path,
                matrix=self._matrix,
                run_ids=np.array(run_ids, dtype=object),
                ideas=np.array(ideas, dtype=object),
                surprises=surprises,
                iterations=iterations,
                scores=scores,
                valids=valids,
            )

    def load(self, path: Any) -> None:
        """Restore store state from an npz file."""
        from pathlib import Path
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        with self._lock:
            self._matrix = data["matrix"].astype(np.float32)
            run_ids = data["run_ids"].tolist()
            ideas = data["ideas"].tolist()
            surprises = data["surprises"]
            iterations = data["iterations"]
            scores = data["scores"]
            valids = data["valids"]

            self._entries = []
            self._id_index = {}
            for i, run_id in enumerate(run_ids):
                s = float(surprises[i]) if not np.isnan(surprises[i]) else None
                sc = float(scores[i]) if not np.isnan(scores[i]) else None
                v = bool(valids[i]) if valids[i] != -1 else None
                entry = StoredIdea(
                    run_id=run_id,
                    idea=ideas[i],
                    vector=self._matrix[i],
                    surprise=s,
                    iteration=int(iterations[i]),
                    score=sc,
                    valid=v,
                )
                self._entries.append(entry)
                self._id_index[run_id] = i


def _normalize(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n == 0.0:
        return vec.astype(np.float32)
    return (vec / n).astype(np.float32)


def _normalize_batch(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vecs / norms).astype(np.float32)
