"""Tests for MoIReviewer (Phase 2)."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from heuresis.qd.core.archive_index import ArchiveIndex
from heuresis.qd.core.embedding import FakeEmbedder
from heuresis.qd.omni_epic import MoIAssessment, MoIReviewError
from heuresis.qd.omni_epic.reviewer import MoIContext, MoIReviewer
from heuresis.tasks import task_dir


# --- fixtures --------------------------------------------------------------

@pytest.fixture
def empty_index() -> ArchiveIndex:
    return ArchiveIndex(embedder=FakeEmbedder(dim=16))


@pytest.fixture
def seeded_index() -> ArchiveIndex:
    """Archive with 12 accepted entries (above default seed threshold of 10)."""
    idx = ArchiveIndex(embedder=FakeEmbedder(dim=16))
    for i in range(12):
        idx.add_accepted(run_id=f"r{i}", plan=f"plan number {i}", score=0.9 + i * 0.001)
    return idx


@pytest.fixture
def nanogpt_dir() -> Path:
    return task_dir("nanogpt")


# --- seed-gate behavior ----------------------------------------------------

def test_seed_gate_returns_interesting_when_archive_empty(
    empty_index: ArchiveIndex, nanogpt_dir: Path,
) -> None:
    reviewer = MoIReviewer(
        empty_index, nanogpt_dir,
        api_keys=["fake-key"],
        client_factory=lambda k: MagicMock(),
    )
    out = reviewer.review("Replace AdamW with Muon optimizer.")
    assert isinstance(out, MoIAssessment)
    assert out.interesting is True
    assert out.retrieved_ids == []
    assert "seed phase" in out.reasoning.lower()
    assert out.input_tokens == 0
    assert out.output_tokens == 0
    assert out.total_cost == 0.0
    assert out.duration_s >= 0.0


def test_seed_gate_returns_interesting_just_below_threshold(nanogpt_dir: Path) -> None:
    idx = ArchiveIndex(embedder=FakeEmbedder(dim=16))
    for i in range(9):
        idx.add_accepted(run_id=f"r{i}", plan=f"plan {i}", score=0.9)
    factory_calls: list[str] = []
    reviewer = MoIReviewer(
        idx, nanogpt_dir,
        api_keys=["fake-key"],
        client_factory=lambda k: factory_calls.append(k) or MagicMock(),
    )
    out = reviewer.review("Use SGD instead of AdamW.")
    assert out.interesting is True
    assert out.retrieved_ids == []
    assert factory_calls == []   # no client built during seed phase


# --- task config loading ---------------------------------------------------

def test_loads_task_config_correctly(
    seeded_index: ArchiveIndex, nanogpt_dir: Path,
) -> None:
    reviewer = MoIReviewer(
        seeded_index, nanogpt_dir,
        api_keys=["fake-key"],
        client_factory=lambda k: MagicMock(),
    )
    assert reviewer._task_name == "nanogpt"
    assert reviewer._metric == "val_bpb"
    assert reviewer._lower_is_better is True
    assert reviewer._baseline == pytest.approx(0.992)
    # Editable file (train.py) should have been loaded.
    assert len(reviewer._seed_code) > 0
    # Long-form description was loaded.
    assert "GPT" in reviewer._domain_description or "nanogpt" in reviewer._domain_description.lower()


def test_context_overrides_avoid_static_editable_file_reads(
    seeded_index: ArchiveIndex, tmp_path: Path,
) -> None:
    """Runtime-generated tasks can provide reviewer prompt context directly."""
    (tmp_path / "task_config.yaml").write_text(
        "name: dynamic-task\n"
        "description: static description should be overridden\n"
        "editable: generated\n"
    )
    (tmp_path / "baseline_scores.yaml").write_text(
        "metric: static_metric\nobjective: min\nbaseline: 123\n"
    )
    client = MagicMock()
    client.models.generate_content.return_value = _make_response(
        '{"interesting": true, "reasoning": "uses generated context"}'
    )

    reviewer = MoIReviewer(
        seeded_index,
        tmp_path,
        api_keys=["fake-key"],
        client_factory=lambda k: client,
        context=MoIContext(
            task_name="discogen-OnPolicyRL",
            task_description="runtime Discogen task",
            domain_description="runtime generated description",
            problem_text="runtime generated problem",
            seed_code="# discovered/loss.py\nclass RuntimeOnly: pass",
            metric="baseline_normalized_score",
            baseline=1.0,
            lower_is_better=False,
        ),
    )

    assert reviewer._task_name == "discogen-OnPolicyRL"
    assert reviewer._task_description == "runtime Discogen task"
    assert reviewer._domain_description == "runtime generated description"
    assert reviewer._problem_text == "runtime generated problem"
    assert reviewer._seed_code == "# discovered/loss.py\nclass RuntimeOnly: pass"
    assert reviewer._metric == "baseline_normalized_score"
    assert reviewer._baseline == pytest.approx(1.0)
    assert reviewer._lower_is_better is False
    assert not (tmp_path / "generated").exists()

    out = reviewer.review("try the runtime-only discovered code")
    assert out.interesting is True
    prompt = client.models.generate_content.call_args.kwargs["contents"]
    assert "runtime generated description" in prompt
    assert "RuntimeOnly" in prompt
    assert "static description should be overridden" not in prompt


# --- Gemini-call behavior (mocked client) ---------------------------------

def _make_response(
    json_body: str, prompt_tokens: int = 100, completion_tokens: int = 30
) -> MagicMock:
    """Build a MagicMock that mimics google.genai response shape."""
    resp = MagicMock()
    resp.text = json_body
    resp.usage_metadata.prompt_token_count = prompt_tokens
    resp.usage_metadata.candidates_token_count = completion_tokens
    return resp


def test_happy_path_returns_assessment(
    seeded_index: ArchiveIndex, nanogpt_dir: Path,
) -> None:
    client = MagicMock()
    client.models.generate_content.return_value = _make_response(
        '{"interesting": true, "reasoning": "Different optimizer family from prior entries."}'
    )
    reviewer = MoIReviewer(
        seeded_index, nanogpt_dir,
        api_keys=["fake-key"],
        client_factory=lambda k: client,
    )
    out = reviewer.review("Use SGD with momentum.")
    assert out.interesting is True
    assert "optimizer" in out.reasoning.lower()
    assert len(out.retrieved_ids) == 10           # seeded_index has 12; k=10
    assert out.input_tokens == 100
    assert out.output_tokens == 30
    assert out.duration_s >= 0
    client.models.generate_content.assert_called_once()


def test_retrieves_fewer_when_archive_smaller_than_k(nanogpt_dir: Path) -> None:
    """k=15 against an archive of 12 returns 12 retrieved ids."""
    idx = ArchiveIndex(embedder=FakeEmbedder(dim=16))
    for i in range(12):
        idx.add_accepted(run_id=f"r{i}", plan=f"p{i}", score=0.9)
    client = MagicMock()
    client.models.generate_content.return_value = _make_response(
        '{"interesting": false, "reasoning": "trivial variant"}'
    )
    reviewer = MoIReviewer(
        idx, nanogpt_dir,
        api_keys=["k"], client_factory=lambda k: client,
        k=15,
    )
    out = reviewer.review("plan 0 again")
    assert len(out.retrieved_ids) == 12
    assert out.interesting is False


def test_parse_failure_raises_after_retries(
    seeded_index: ArchiveIndex, nanogpt_dir: Path,
) -> None:
    """All 3 attempts return malformed JSON → raise with retry-count message."""
    client = MagicMock()
    client.models.generate_content.return_value = _make_response("not json at all")
    reviewer = MoIReviewer(
        seeded_index, nanogpt_dir,
        api_keys=["k"], client_factory=lambda k: client,
    )
    with pytest.raises(MoIReviewError, match="failed after 3 attempts"):
        reviewer.review("anything")
    # 3 retries were attempted
    assert client.models.generate_content.call_count == 3


def test_empty_response_raises_after_retries(
    seeded_index: ArchiveIndex, nanogpt_dir: Path,
) -> None:
    """All 3 attempts return None text (safety filter) → raise."""
    client = MagicMock()
    resp = MagicMock()
    resp.text = None
    client.models.generate_content.return_value = resp
    reviewer = MoIReviewer(
        seeded_index, nanogpt_dir,
        api_keys=["k"], client_factory=lambda k: client,
    )
    with pytest.raises(MoIReviewError, match="empty response"):
        reviewer.review("anything")
    assert client.models.generate_content.call_count == 3


def test_array_wrapped_response_recovers(
    seeded_index: ArchiveIndex, nanogpt_dir: Path,
) -> None:
    """Gemini wraps the object in a single-element array — reviewer unwraps."""
    client = MagicMock()
    client.models.generate_content.return_value = _make_response(
        '[{"interesting": true, "reasoning": "genuinely different: sparse attention"}]'
    )
    reviewer = MoIReviewer(
        seeded_index, nanogpt_dir,
        api_keys=["k"], client_factory=lambda k: client,
    )
    out = reviewer.review("idea")
    assert out.interesting is True
    assert "sparse" in out.reasoning
    # Single call: success on first attempt, no retry needed
    assert client.models.generate_content.call_count == 1


def test_transient_parse_failure_then_success(
    seeded_index: ArchiveIndex, nanogpt_dir: Path,
) -> None:
    """Two bad responses (array → invalid-escape) then a good one: reviewer succeeds."""
    client = MagicMock()
    client.models.generate_content.side_effect = [
        _make_response('{"interesting": true'),                    # truncated JSON
        _make_response('{"interesting": false, "reasoning": "\\q"}'),  # invalid \q escape
        _make_response('{"interesting": false, "reasoning": "trivial variant"}'),
    ]
    reviewer = MoIReviewer(
        seeded_index, nanogpt_dir,
        api_keys=["k"], client_factory=lambda k: client,
    )
    out = reviewer.review("idea")
    assert out.interesting is False
    assert out.reasoning == "trivial variant"
    assert client.models.generate_content.call_count == 3


def test_empty_then_success(
    seeded_index: ArchiveIndex, nanogpt_dir: Path,
) -> None:
    """Safety-filter None followed by a valid response: recover on retry."""
    empty_resp = MagicMock()
    empty_resp.text = None
    good_resp = _make_response('{"interesting": true, "reasoning": "ok"}')
    client = MagicMock()
    client.models.generate_content.side_effect = [empty_resp, good_resp]
    reviewer = MoIReviewer(
        seeded_index, nanogpt_dir,
        api_keys=["k"], client_factory=lambda k: client,
    )
    out = reviewer.review("idea")
    assert out.interesting is True
    assert client.models.generate_content.call_count == 2


def test_key_rotation_on_failure(
    seeded_index: ArchiveIndex, nanogpt_dir: Path,
) -> None:
    """First key raises; second key succeeds."""
    client_a = MagicMock()
    client_a.models.generate_content.side_effect = RuntimeError("429 rate limit")
    client_b = MagicMock()
    client_b.models.generate_content.return_value = _make_response(
        '{"interesting": true, "reasoning": "ok"}'
    )
    def factory(api_key: str) -> Any:
        return client_a if api_key == "key1" else client_b
    reviewer = MoIReviewer(
        seeded_index, nanogpt_dir,
        api_keys=["key1", "key2"],
        client_factory=factory,
    )
    out = reviewer.review("anything")
    assert out.interesting is True
    client_a.models.generate_content.assert_called_once()
    client_b.models.generate_content.assert_called_once()


def test_all_keys_exhausted_raises(
    seeded_index: ArchiveIndex, nanogpt_dir: Path,
) -> None:
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("429")
    reviewer = MoIReviewer(
        seeded_index, nanogpt_dir,
        api_keys=["k1", "k2", "k3"],
        client_factory=lambda k: client,
    )
    with pytest.raises(MoIReviewError, match="all .* keys exhausted"):
        reviewer.review("anything")
    assert client.models.generate_content.call_count == 3


# --- synthetic-task construction errors ------------------------------------

def test_construction_raises_when_required_yaml_missing(tmp_path: Path) -> None:
    # Create a fake task dir with only baseline_scores, no task_config.yaml
    (tmp_path / "baseline_scores.yaml").write_text(
        "metric: x\nobjective: min\nbaseline: 0\n"
    )
    (tmp_path / "description.md").write_text("desc")
    idx = ArchiveIndex(embedder=FakeEmbedder(dim=16))
    with pytest.raises(FileNotFoundError):
        MoIReviewer(
            idx, tmp_path,
            api_keys=["fake-key"],
            client_factory=lambda k: MagicMock(),
        )
