"""Quality-Diversity algorithm primitives.

Layout:
  - ``core/``: shared primitives (archives, selection, metrics, migration,
    embeddings, ``SearchStrategy`` ABC).
  - ``linear/``: ``LinearSearch`` (top-K baseline).
  - ``map_elites/``: ``MapElitesSearch``, ``CellTargetedMapElitesSearch``,
    ``NoveltyMapElitesSearch`` + ``plot_archive``.
  - ``go_explore/``: ``GoExploreSearch`` (cell-targeted MAP-Elites with
    score/visit-weighted cell selection).
  - ``islands/``: ``IslandSearch`` + ``plot_islands``.
  - ``omni_epic/``: Phase 3 canonical OMNI-EPIC — ``OmniEpicSearch`` +
    ``MoIReviewer`` + prompts.
  - ``curiosity/``: base prediction-error curiosity.
  - ``curiosity_plus/``: curiosity with score pressure, tag repetition,
    and memory-aware surprise.

The public API is re-exported here so callers can write
``from heuresis.qd import MapElitesSearch`` without knowing the
subpackage layout.
"""

from heuresis.qd.core import (
    Archive,
    ArchiveIndex,
    Bucket,
    CVTArchive,
    Embedder,
    EliteEntry,
    Entry,
    FakeEmbedder,
    Feature,
    FeatureClassifier,
    GeminiEmbedder,
    GridArchive,
    KeywordClassifier,
    LLMClassifier,
    MigrationEvent,
    Neighbor,
    ScoredSolution,
    SearchStrategy,
    Selector,
    best_fitness,
    coverage,
    feature_namer,
    fully_connected_neighbors,
    migrate,
    qd_score,
    ring_neighbors,
)
from heuresis.qd.curiosity import CuriositySearch
from heuresis.qd.curiosity_plus import CuriosityPlusSearch
from heuresis.qd.go_explore import GoExploreSearch
from heuresis.qd.islands import IslandSearch, plot_islands
from heuresis.qd.linear import LinearSearch
from heuresis.qd.map_elites import (
    CellTargetedMapElitesSearch,
    IdeaReview,
    MapElitesSearch,
    NoveltyMapElitesSearch,
    plot_archive,
)
from heuresis.qd.omni_epic import (
    MoIAssessment,
    MoIContext,
    MoIReviewError,
    MoIReviewer,
    OmniEpicSearch,
)

__all__ = [
    "Archive",
    "ArchiveIndex",
    "Bucket",
    "CVTArchive",
    "CellTargetedMapElitesSearch",
    "CuriosityPlusSearch",
    "CuriositySearch",
    "Embedder",
    "EliteEntry",
    "Entry",
    "FakeEmbedder",
    "Feature",
    "FeatureClassifier",
    "GeminiEmbedder",
    "GoExploreSearch",
    "GridArchive",
    "IdeaReview",
    "IslandSearch",
    "KeywordClassifier",
    "LLMClassifier",
    "LinearSearch",
    "MapElitesSearch",
    "MigrationEvent",
    "MoIAssessment",
    "MoIContext",
    "MoIReviewError",
    "MoIReviewer",
    "Neighbor",
    "NoveltyMapElitesSearch",
    "OmniEpicSearch",
    "ScoredSolution",
    "SearchStrategy",
    "Selector",
    "best_fitness",
    "coverage",
    "feature_namer",
    "fully_connected_neighbors",
    "migrate",
    "plot_archive",
    "plot_islands",
    "qd_score",
    "ring_neighbors",
]
