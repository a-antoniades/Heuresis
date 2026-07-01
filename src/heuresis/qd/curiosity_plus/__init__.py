"""Prediction-error curiosity search.

Core idea: steer the ideator toward regions of the idea space where its
predictions about outcomes are most wrong *and improving* — genuine
knowledge gaps, not just statistical rarity.

Components:
  - ``EmbeddingStore``: in-memory (vector, surprise, iteration) tuples with kNN.
  - ``Prediction`` / ``surprise`` / ``SigmaTracker``: prediction-error pipeline.
  - ``compute_lp``: split-halves learning progress over kNN neighborhoods.
  - ``select_anchor``: softmax + repetition penalty sampler.
  - ``select_seed`` / ``parse_candidates``: Phase 1 farthest-point seeding.
  - ``build_curiosity_context``: natural-language signal for the ideator prompt.
  - ``predict_outcome`` / ``parse_prediction``: LLM prediction step.
  - ``CuriosityPlusSearch``: SearchStrategy subclass wiring it all together.
"""

from heuresis.qd.curiosity_plus.curiosity_signals import (
    build_curiosity_context,
    build_prediction_context,
)
from heuresis.qd.curiosity_plus.embedding_store import EmbeddingStore, StoredIdea
from heuresis.qd.curiosity_plus.learning_progress import (
    compute_all_lp,
    compute_lp,
    compute_lp_from_vector,
)
from heuresis.qd.curiosity_plus.prediction import parse_prediction, predict_outcome
from heuresis.qd.curiosity_plus.search import CuriosityPlusSearch
from heuresis.qd.curiosity_plus.seeding import parse_candidates, select_seed
from heuresis.qd.curiosity_plus.selection import select_anchor
from heuresis.qd.curiosity_plus.surprise import (
    Prediction,
    SigmaTracker,
    surprise,
)

__all__ = [
    "CuriosityPlusSearch",
    "EmbeddingStore",
    "Prediction",
    "SigmaTracker",
    "StoredIdea",
    "build_curiosity_context",
    "build_prediction_context",
    "compute_all_lp",
    "compute_lp",
    "compute_lp_from_vector",
    "parse_candidates",
    "parse_prediction",
    "predict_outcome",
    "select_anchor",
    "select_seed",
    "surprise",
]
