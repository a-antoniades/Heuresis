"""Tests for the memory hook inside heuresis.experiment.record_run.

These tests exercise the framework side only: the ingestion path is
driven by a fake MemoryIngest. We verify:

- Ingestion fires only when memory + ideator_workspace + run_type=executor
  are ALL present AND an ``idea`` string is supplied.
- Workspace UUIDs are read from the ``.workspace_id`` markers on both
  sides (ideator and executor dirs).
- ``notes.md`` is folded in when present, omitted otherwise.
- Failure to read markers does not raise — the experiment loop must
  never crash because memory was misconfigured.
- Exceptions from ``ingest_experiment`` are swallowed (logged, not
  propagated) so a hiccup in the memory side doesn't sink the run.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path


from heuresis.experiment import record_run


# -- fakes ------------------------------------------------------------------


@dataclass
class _FakeResult:
    workspace: Path
    exit_code: int = 0
    stats: dict = field(default_factory=dict)


class _FakeExperiment:
    """Implements just what record_run calls on Experiment."""

    def __init__(self) -> None:
        self.saved: list[dict] = []
        self.files: list[tuple[str, str, str]] = []
        self.events: list[dict] = []

    def save(self, run_id, *, result, iteration, run_type, valid, idea,
             parent_ids, generation, metadata):
        self.saved.append({
            "run_id": run_id, "iteration": iteration, "run_type": run_type,
            "valid": valid, "idea": idea, "parent_ids": parent_ids,
            "generation": generation, "metadata": metadata,
        })

    def save_file(self, run_id, name, content):
        self.files.append((run_id, name, content))

    def log_archive_event(self, **kwargs):
        self.events.append(kwargs)


class _CapturingMemory:
    """Captures every ingest_experiment call."""

    def __init__(self, *, raise_on: int | None = None) -> None:
        self.calls: list[dict] = []
        self._raise_on = raise_on

    def ingest_experiment(self, **kwargs):
        self.calls.append(kwargs)
        if self._raise_on is not None and len(self.calls) == self._raise_on:
            raise RuntimeError("simulated gemini outage")


def _make_ws(tmp_path: Path, name: str, *, wsid: str | None = None,
             notes: str | None = None) -> Path:
    """Create a workspace dir with a .workspace_id marker."""
    p = tmp_path / name
    p.mkdir(parents=True, exist_ok=True)
    (p / ".workspace_id").write_text(wsid or uuid.uuid4().hex[:12])
    if notes is not None:
        (p / "notes.md").write_text(notes)
    return p


# -- happy path -------------------------------------------------------------


def test_ingests_executor_run_with_notes(tmp_path: Path):
    ideator_ws = _make_ws(tmp_path, "ideator_0", wsid="id000000aaaa")
    exec_ws = _make_ws(tmp_path, "exec_001", wsid="ex000000bbbb",
                       notes="Rastrigin broke the simplex")
    memory = _CapturingMemory()
    exp = _FakeExperiment()

    record_run(
        exp, "exec_001",
        result=_FakeResult(workspace=exec_ws),
        info={"valid": True, "best_score": 1.5},
        strategy_meta={"generation": 2, "qd_features": {"axis_a": 0.4}},
        iteration=4, run_type="executor",
        idea="trust region with restart",
        parent_ids=["p1", "p2"],
        memory=memory, ideator_workspace=ideator_ws,
    )

    assert len(memory.calls) == 1
    call = memory.calls[0]
    assert call["ideator_id"] == "id000000aaaa"
    assert call["executor_id"] == "ex000000bbbb"
    assert call["valid"] is True
    assert call["score"] == 1.5
    assert call["features"] == {"axis_a": 0.4}
    assert call["parent_ids"] == ["p1", "p2"]
    assert call["generation"] == 2
    assert call["idea_md"] == "trust region with restart"
    assert call["notes_md"] == "Rastrigin broke the simplex"


def test_features_use_qd_features_key_not_features(tmp_path: Path):
    # Regression: MapElitesSearch.on_result writes metadata["qd_features"], so
    # record_run must read that key. A stale "features" key (which no real
    # strategy emits) must NOT satisfy the consumer.
    ideator_ws = _make_ws(tmp_path, "ideator_0")
    exec_ws = _make_ws(tmp_path, "exec_001")
    memory = _CapturingMemory()
    exp = _FakeExperiment()

    record_run(
        exp, "exec_001",
        result=_FakeResult(workspace=exec_ws),
        info={"valid": True, "best_score": 0.9},
        strategy_meta={
            "generation": 0,
            "qd_features": {"axis_a": 0.7},
            "features": {"axis_a": 0.0},  # stale — must be ignored
        },
        iteration=0, run_type="executor", idea="idea",
        memory=memory, ideator_workspace=ideator_ws,
    )
    assert memory.calls[0]["features"] == {"axis_a": 0.7}


def test_ingests_without_notes_when_missing(tmp_path: Path):
    ideator_ws = _make_ws(tmp_path, "ideator_0")
    exec_ws = _make_ws(tmp_path, "exec_001")  # no notes.md
    memory = _CapturingMemory()
    exp = _FakeExperiment()

    record_run(
        exp, "exec_001",
        result=_FakeResult(workspace=exec_ws),
        info={"valid": True, "best_score": 0.9},
        strategy_meta={"generation": 0},
        iteration=0, run_type="executor", idea="idea",
        parent_ids=None,
        memory=memory, ideator_workspace=ideator_ws,
    )

    assert memory.calls[0]["notes_md"] is None


def test_ingests_with_invalid_run(tmp_path: Path):
    """Even invalid runs have a story worth embedding."""
    ideator_ws = _make_ws(tmp_path, "ideator_0")
    exec_ws = _make_ws(tmp_path, "exec_001")
    memory = _CapturingMemory()
    exp = _FakeExperiment()

    record_run(
        exp, "exec_001",
        result=_FakeResult(workspace=exec_ws),
        info={"valid": False, "best_score": None},
        strategy_meta={"generation": 0},
        iteration=0, run_type="executor", idea="thing",
        memory=memory, ideator_workspace=ideator_ws,
    )
    assert len(memory.calls) == 1
    assert memory.calls[0]["valid"] is False
    assert memory.calls[0]["score"] is None


# -- no-op conditions -------------------------------------------------------


def test_no_ingest_when_memory_is_none(tmp_path: Path):
    ideator_ws = _make_ws(tmp_path, "ideator_0")
    exec_ws = _make_ws(tmp_path, "exec_001")

    record_run(
        _FakeExperiment(), "exec_001",
        result=_FakeResult(workspace=exec_ws),
        info={"valid": True, "best_score": 1.0},
        strategy_meta={"generation": 0},
        iteration=0, run_type="executor", idea="idea",
        memory=None, ideator_workspace=ideator_ws,
    )
    # Nothing to assert beyond "did not raise"; the absence of a fake
    # memory to capture calls is the whole test.


def test_no_ingest_when_ideator_workspace_missing(tmp_path: Path):
    exec_ws = _make_ws(tmp_path, "exec_001")
    memory = _CapturingMemory()

    record_run(
        _FakeExperiment(), "exec_001",
        result=_FakeResult(workspace=exec_ws),
        info={"valid": True, "best_score": 1.0},
        strategy_meta={"generation": 0},
        iteration=0, run_type="executor", idea="idea",
        memory=memory, ideator_workspace=None,
    )
    assert memory.calls == []


def test_no_ingest_for_non_executor_run_type(tmp_path: Path):
    """OmniEpic's rejected-idea rows (run_type='idea_rejected') pass through
    safely — memory is silently skipped.
    """
    ideator_ws = _make_ws(tmp_path, "ideator_0")
    exec_ws = _make_ws(tmp_path, "exec_001")
    memory = _CapturingMemory()

    record_run(
        _FakeExperiment(), "exec_001",
        result=_FakeResult(workspace=exec_ws),
        info={"valid": False, "best_score": None},
        strategy_meta={"generation": 0},
        iteration=0, run_type="idea_rejected", idea="the rejected idea",
        memory=memory, ideator_workspace=ideator_ws,
    )
    assert memory.calls == []


def test_no_ingest_without_idea(tmp_path: Path):
    """record_run needs an idea string to have something to embed."""
    ideator_ws = _make_ws(tmp_path, "ideator_0")
    exec_ws = _make_ws(tmp_path, "exec_001")
    memory = _CapturingMemory()

    record_run(
        _FakeExperiment(), "exec_001",
        result=_FakeResult(workspace=exec_ws),
        info={"valid": True, "best_score": 1.0},
        strategy_meta={"generation": 0},
        iteration=0, run_type="executor", idea=None,
        memory=memory, ideator_workspace=ideator_ws,
    )
    assert memory.calls == []


# -- failure modes: must NOT raise ------------------------------------------


def test_missing_workspace_id_marker_does_not_raise(tmp_path: Path):
    """If someone forgets workspace.setup(), record_run should log + skip."""
    ideator_ws = tmp_path / "ideator_0"
    ideator_ws.mkdir()  # no .workspace_id
    exec_ws = _make_ws(tmp_path, "exec_001")
    memory = _CapturingMemory()
    exp = _FakeExperiment()

    # Should NOT raise
    record_run(
        exp, "exec_001",
        result=_FakeResult(workspace=exec_ws),
        info={"valid": True, "best_score": 1.0},
        strategy_meta={"generation": 0},
        iteration=0, run_type="executor", idea="idea",
        memory=memory, ideator_workspace=ideator_ws,
    )
    assert memory.calls == []
    # Normal save still happened.
    assert len(exp.saved) == 1


def test_ingest_exception_does_not_raise(tmp_path: Path):
    """A Gemini hiccup in ingest_experiment must not kill the run."""
    ideator_ws = _make_ws(tmp_path, "ideator_0")
    exec_ws = _make_ws(tmp_path, "exec_001")
    memory = _CapturingMemory(raise_on=1)
    exp = _FakeExperiment()

    # Should NOT propagate
    record_run(
        exp, "exec_001",
        result=_FakeResult(workspace=exec_ws),
        info={"valid": True, "best_score": 1.0},
        strategy_meta={"generation": 0},
        iteration=0, run_type="executor", idea="idea",
        memory=memory, ideator_workspace=ideator_ws,
    )
    # The call WAS made (and raised), but record_run kept going.
    assert len(memory.calls) == 1
    # Normal save still happened.
    assert len(exp.saved) == 1
