"""Shared QD primitives: archives, selection, metrics, migration, embeddings.

These are algorithm-agnostic building blocks that individual search
strategies (linear, map_elites, islands, omni_epic) compose.
"""

from heuresis.qd.core.archive import (
    Archive,
    CVTArchive,
    EliteEntry,
    Feature,
    GridArchive,
)
from heuresis.qd.core.archive_index import ArchiveIndex, Bucket, Entry, Neighbor
from heuresis.qd.core.base import (
    SearchStrategy,
    compute_generation,
    default_feature_names,
    extract_summary,
    feature_namer,
    format_archive_context,
)
from heuresis.qd.core.embedding import Embedder, FakeEmbedder, GeminiEmbedder
from heuresis.qd.core.features import (
    FeatureClassifier,
    KeywordClassifier,
    LLMClassifier,
)
from heuresis.qd.core.metrics import best_fitness, coverage, qd_score
from heuresis.qd.core.migration import (
    MigrationEvent,
    ScoredSolution,
    fully_connected_neighbors,
    migrate,
    ring_neighbors,
)
from heuresis.qd.core.selection import Selector

__all__ = [
    "Archive",
    "ArchiveIndex",
    "Bucket",
    "CVTArchive",
    "Embedder",
    "EliteEntry",
    "Entry",
    "FakeEmbedder",
    "Feature",
    "FeatureClassifier",
    "GeminiEmbedder",
    "GridArchive",
    "KeywordClassifier",
    "LLMClassifier",
    "MigrationEvent",
    "Neighbor",
    "ScoredSolution",
    "SearchStrategy",
    "Selector",
    "best_fitness",
    "compute_generation",
    "coverage",
    "default_feature_names",
    "extract_summary",
    "feature_namer",
    "format_archive_context",
    "fully_connected_neighbors",
    "migrate",
    "qd_score",
    "ring_neighbors",
]
