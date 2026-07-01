"""Experiment settings + loop helpers.

Opt-in helpers that name the common boilerplate without hiding it.
A run.py that doesn't use any of these is still a valid run.py.
"""

from __future__ import annotations

import logging
import argparse
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

import shlex
import yaml

from heuresis._bwrap import run_command as _bwrap_run_command

if TYPE_CHECKING:
    from heuresis.harness import Harness
    from heuresis.memory.protocol import MemoryIngest
    from heuresis.qd.core.base import SearchStrategy
    from heuresis.qd.map_elites.novelty_gated import IdeaReview
    from heuresis.store import Experiment, ResultStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    """Shared experiment configuration."""

    experiment_name: str = "experiment"
    resume_exp_id: str | None = None

    # Agent / model
    agent: str = "opencode"
    model: str = "google/gemini-3.1-pro-preview"

    # Hardware
    gpus: list[int] = field(default_factory=list)

    # Loop
    num_iterations: int = 100
    count_valid: bool = True
    num_ideators: int = 1

    # Timeouts (seconds)
    executor_timeout: int = 35 * 60
    ideator_timeout: int = 10 * 60         # 600s — 240s truncates ideas when Gemini is slow
    reviewer_timeout: int = 300
    judge_timeout: int = 300
    judge_agent: str = "claude"
    judge_model: str = "claude-sonnet-4-6"
    enable_judge: bool = True

    # Novelty
    novelty_threshold: int = 2
    novelty_max_rounds: int = 3

    # Linear
    max_parents: int = 5
    session_reset_every: int = 10

    # Memory primitive (opt-in; see heuresis.memory)
    memory: bool = False

def parse_gpus(raw: str | list[int] | None) -> list[int]:
    """Parse ``--gpus`` values like ``0,1,2`` into integer ids."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    return [int(c.strip()) for c in raw.split(",") if c.strip()]


def str_to_bool(raw: str | bool) -> bool:
    if isinstance(raw, bool):
        return raw
    return raw.lower() in {"1", "true", "yes", "on"}


def add_settings_args(parser: argparse.ArgumentParser, defaults: dict[str, Any]) -> None:
    """Add shared experiment Settings flags to *parser*."""
    base = Settings(**defaults)
    parser.add_argument("--experiment-name", default=base.experiment_name)
    parser.add_argument("--resume-exp-id", default=base.resume_exp_id)
    parser.add_argument("--agent", default=base.agent)
    parser.add_argument("--model", default=base.model)
    parser.add_argument("--gpus", type=parse_gpus, default=base.gpus)
    parser.add_argument("--num-iterations", type=int, default=base.num_iterations)
    count = parser.add_mutually_exclusive_group()
    count.add_argument("--count-valid", dest="count_valid", action="store_true")
    count.add_argument("--count-total", dest="count_valid", action="store_false")
    parser.set_defaults(count_valid=base.count_valid)
    parser.add_argument("--num-ideators", type=int, default=base.num_ideators)
    parser.add_argument("--executor-timeout", type=int, default=base.executor_timeout)
    parser.add_argument("--ideator-timeout", type=int, default=base.ideator_timeout)
    parser.add_argument("--reviewer-timeout", type=int, default=base.reviewer_timeout)
    parser.add_argument("--judge-timeout", type=int, default=base.judge_timeout)
    parser.add_argument("--judge-agent", default=base.judge_agent)
    parser.add_argument("--judge-model", default=base.judge_model)
    judge = parser.add_mutually_exclusive_group()
    judge.add_argument("--enable-judge", dest="enable_judge", action="store_true")
    judge.add_argument("--disable-judge", dest="enable_judge", action="store_false")
    parser.set_defaults(enable_judge=base.enable_judge)
    parser.add_argument("--novelty-threshold", type=int, default=base.novelty_threshold)
    parser.add_argument("--novelty-max-rounds", type=int, default=base.novelty_max_rounds)
    parser.add_argument("--max-parents", type=int, default=base.max_parents)
    parser.add_argument("--session-reset-every", type=int, default=base.session_reset_every)
    memory = parser.add_mutually_exclusive_group()
    memory.add_argument("--memory", dest="memory", action="store_true")
    memory.add_argument("--no-memory", dest="memory", action="store_false")
    parser.set_defaults(memory=base.memory)


# ---------------------------------------------------------------------------
# ExperimentState + resume_or_new
# ---------------------------------------------------------------------------


@dataclass
class ExperimentState:
    exp: "Experiment"
    next_iter_idx: int
    valid_count: int
    is_resume: bool


def resume_or_new(
    store: "ResultStore",
    name: str,
    strategy: "SearchStrategy",
    settings: Settings,
    *,
    root: Path | None = None,
    task: str = "",
) -> ExperimentState:
    """Load an existing experiment (if resume_exp_id set) or create a new one.

    On resume, re-ingests existing runs via strategy.rebuild() and computes
    next_iter_idx and valid_count from persisted state.
    """
    if settings.resume_exp_id:
        exp = store.get_experiment(settings.resume_exp_id)
        if exp is None:
            raise RuntimeError(f"No experiment found with id {settings.resume_exp_id!r}")
        runs = exp.runs(run_type="executor")
        records = [
            (r.run_id, r.score, {**r.metadata,
                                 "parent_ids": r.parent_ids,
                                 "generation": r.generation,
                                 "idea": r.idea})
            for r in runs
        ]
        strategy.rebuild(records)
        next_idx = (max((r.iteration for r in runs if r.iteration is not None),
                        default=-1) + 1)
        valid_count = sum(1 for r in runs if r.valid)
        logger.info("Resumed experiment %s: %d runs, %d valid, next iter %d",
                    exp.id, len(runs), valid_count, next_idx)
        return ExperimentState(exp=exp, next_iter_idx=next_idx,
                               valid_count=valid_count, is_resume=True)

    exp = store.experiment(name, task=task, root=root)
    return ExperimentState(exp=exp, next_iter_idx=0, valid_count=0, is_resume=False)


# ---------------------------------------------------------------------------
# ExperimentLoop (count_valid semantics + shared counter for parallel)
# ---------------------------------------------------------------------------


class ExperimentLoop:
    """Iterator over experiment iterations with count_valid + termination semantics.

    Thread-safe: parallel ideator threads share one ExperimentLoop and call
    mark_done() to update the shared counter.
    """

    def __init__(self, state: "ExperimentState", settings: Settings) -> None:
        self._lock = threading.Lock()
        self._settings = settings
        self._next_idx = state.next_iter_idx
        self._valid_count = state.valid_count
        self._total_started = 0

    def __iter__(self) -> Iterator[int]:
        return self

    def __next__(self) -> int:
        with self._lock:
            if not self._should_continue_locked():
                raise StopIteration
            idx = self._next_idx
            self._next_idx += 1
            self._total_started += 1
            return idx

    def mark_done(self, *, valid: bool) -> None:
        with self._lock:
            if valid:
                self._valid_count += 1

    def _should_continue_locked(self) -> bool:
        target = self._settings.num_iterations
        if self._settings.count_valid:
            return self._valid_count < target
        return self._next_idx < target

    def should_continue(self) -> bool:
        with self._lock:
            return self._should_continue_locked()

    @property
    def valid_count(self) -> int:
        with self._lock:
            return self._valid_count

    @property
    def total_started(self) -> int:
        with self._lock:
            return self._total_started


def iterate_until_valid(state: "ExperimentState", settings: Settings) -> ExperimentLoop:
    """Return an ExperimentLoop honoring count_valid + num_iterations."""
    return ExperimentLoop(state, settings)


# ---------------------------------------------------------------------------
# novelty_retry_loop
# ---------------------------------------------------------------------------


def novelty_retry_loop(
    *,
    ideate: Callable[[Any], str],
    review: Callable[[str, int], "IdeaReview"],
    max_rounds: int,
    persist: Callable[["IdeaReview", int], None] | None = None,
) -> tuple[str, "IdeaReview"] | None:
    """Propose -> review -> retry-on-reject up to max_rounds. Returns (idea, review) or None.

    ideate(feedback) receives None on round 0, then the rejected assessment.
    review(idea, attempt) produces an IdeaReview.
    persist(review, attempt) is called for every attempt (accepted + rejected).
    """
    feedback = None
    for attempt in range(max_rounds):
        idea = ideate(feedback)
        rev = review(idea, attempt)
        if persist is not None:
            persist(rev, attempt)
        if rev.accepted:
            return idea, rev
        feedback = rev.assessment
    return None


# ---------------------------------------------------------------------------
# record_run — parse + strategy update + snapshot + archive_event + save
# ---------------------------------------------------------------------------


SNAPSHOT_WHITELIST = ("train.py", "run.log", "notes.md", "novelty.json")
_MAX_SNAPSHOT_BYTES = 500_000


def record_run(
    exp: "Experiment",
    run_id: str,
    *,
    result: Any,  # RunResult
    info: dict[str, Any],
    strategy_meta: dict[str, Any],
    iteration: int,
    run_type: str = "executor",
    idea: str | None = None,
    parent_ids: list[str] | None = None,
    snapshot_files: tuple[str, ...] | None = None,
    memory: "MemoryIngest | None" = None,
    ideator_workspace: Path | None = None,
) -> None:
    """One-stop persistence: merges info + strategy_meta, snapshots files,
    logs archive events, saves the run row.

    When ``memory`` and ``ideator_workspace`` are both supplied *and*
    ``run_type == "executor"``, also writes an ``experiments`` row into
    the campaign memory DB. Workspace UUIDs are read from
    ``.workspace_id`` markers in both directories — no IDs pass through
    the call site.
    """
    metadata = {**info, **strategy_meta}
    generation = strategy_meta.get("generation", 0)
    valid = info.get("valid")

    exp.save(
        run_id,
        result=result,
        iteration=iteration,
        run_type=run_type,
        valid=valid,
        idea=idea,
        parent_ids=parent_ids,
        generation=generation,
        metadata=metadata,
    )

    # Snapshot files (best-effort)
    for filename in snapshot_files or SNAPSHOT_WHITELIST:
        p = Path(result.workspace) / filename
        if not p.exists() or not p.is_file():
            continue
        try:
            size = p.stat().st_size
            if size > _MAX_SNAPSHOT_BYTES:
                continue
            exp.save_file(run_id, filename, p.read_text(errors="replace"))
        except OSError as e:
            logger.warning("Failed to snapshot %s: %s", p, e)

    # Log archive event if this run was placed into a cell
    status = strategy_meta.get("archive_status")
    if status == "elite":
        cell_key = strategy_meta.get("cell_key", "")
        new_fitness = info.get("best_score")
        exp.log_archive_event(
            cell_key=str(cell_key),
            new_id=run_id,
            new_fitness=new_fitness if new_fitness is not None else 0.0,
            old_id=strategy_meta.get("displaced_id"),
            old_fitness=strategy_meta.get("displaced_fitness"),
        )

    # Memory ingestion — only for real executor rows. Rejected-idea /
    # pseudo rows from OmniEpic (run_type != "executor") fall through.
    if memory is not None and ideator_workspace is not None and run_type == "executor" and idea:
        _ingest_experiment_memory(
            memory=memory,
            ideator_workspace=Path(ideator_workspace),
            executor_workspace=Path(result.workspace),
            valid=bool(valid),
            score=info.get("best_score"),
            features=strategy_meta.get("qd_features"),
            parent_ids=parent_ids,
            generation=generation,
            idea_md=idea,
        )


def _ingest_experiment_memory(
    *,
    memory: "MemoryIngest",
    ideator_workspace: Path,
    executor_workspace: Path,
    valid: bool,
    score: float | None,
    features: dict[str, Any] | None,
    parent_ids: list[str] | None,
    generation: int,
    idea_md: str,
) -> None:
    """Resolve workspace UUIDs + notes.md, then call memory.ingest_experiment.

    Failures are logged, not raised: a memory-side hiccup shouldn't break
    the main experiment loop (the run is already persisted in store.db).
    """
    try:
        ideator_id = (ideator_workspace / ".workspace_id").read_text().strip()
        executor_id = (executor_workspace / ".workspace_id").read_text().strip()
    except OSError as exc:
        logger.warning("memory ingest skipped: missing .workspace_id (%s)", exc)
        return

    notes_path = executor_workspace / "notes.md"
    notes_md: str | None = None
    if notes_path.is_file():
        try:
            notes_md = notes_path.read_text(errors="replace").strip() or None
        except OSError:
            notes_md = None

    try:
        memory.ingest_experiment(
            ideator_id=ideator_id,
            executor_id=executor_id,
            valid=valid,
            score=score,
            features=features,
            parent_ids=parent_ids,
            generation=generation,
            idea_md=idea_md,
            notes_md=notes_md,
        )
    except Exception as exc:
        logger.warning("memory ingest_experiment failed: %s", exc)


# ---------------------------------------------------------------------------
# parallel_ideators
# ---------------------------------------------------------------------------


def parallel_ideators(
    harnesses: "list[Harness]",
    body: Callable[[int, "Harness"], None],
) -> None:
    """Spawn one thread per harness with (tid, harness). Joins at exit."""
    threads = [
        threading.Thread(target=body, args=(i, h), name=f"ideator-{i}")
        for i, h in enumerate(harnesses)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


# ---------------------------------------------------------------------------
# loop setup (harnesses, GPU reservation, executor file sourcing)
# ---------------------------------------------------------------------------


def build_harnesses(settings: Settings, gpu_slice: list[int], *, uses_gpu: bool) -> list:
    """One Harness per ideator. GPU-pinned for GPU tasks, gpu-less for CPU tasks."""
    from heuresis import Harness

    if uses_gpu:
        return [Harness(settings.agent, model=settings.model, gpus=[g])
                for g in gpu_slice]
    return [Harness(settings.agent, model=settings.model, gpus=[])
            for _ in range(settings.num_ideators)]


def reserve_gpus(gpu_slice: list[int]) -> list:
    """Hold ~2 GB per GPU to deter other users during ideation windows.

    No-op when ``RESERVE_GPUS=0`` or ``gpu_slice`` is empty. Returns the
    reservation tensors (keep referenced for the experiment lifetime).
    """
    reservations: list = []
    if not gpu_slice or os.environ.get("RESERVE_GPUS", "1") == "0":
        return reservations
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_slice)
    import torch

    for local_idx in range(len(gpu_slice)):
        reservations.append(
            torch.empty(512 * 1024 * 1024, dtype=torch.float32,
                        device=f"cuda:{local_idx}")
        )
    torch.cuda.synchronize()
    print(f"Reserved ~2 GB × {len(reservations)} GPUs ({gpu_slice}) — "
          f"held for experiment lifetime.")
    return reservations


def executor_files(adapter: Any, parent_runs: list, *, inherit_intent: bool) -> dict[str, Path]:
    """Resolve the files an executor starts from: a parent's evolved files when
    the strategy inherits intent (or the task always does), else task seeds.
    Strategy-policy parent inheritance over task-owned file knowledge."""
    if (inherit_intent or adapter.always_inherits_parent) and parent_runs:
        return adapter.parent_files(parent_runs[0])
    return adapter.seed_files()


def next_run_index(exp: "Experiment") -> int:
    """Next ``exec_NNN`` index = 1 + the highest existing suffix (0 if none).

    Run IDs must NOT be derived from the iteration counter: one iteration can
    emit several executor rows (e.g. omni-epic MoI retries), so seeding the
    counter from ``max(iteration)`` reissues an existing id on resume and
    overwrites a prior run. Derive it from the actual run IDs instead.
    """
    highest = -1
    for r in exp.runs(run_type="executor"):
        rid = r.run_id
        if rid.startswith("exec_"):
            try:
                highest = max(highest, int(rid[len("exec_"):]))
            except ValueError:
                pass
    return highest + 1


# ---------------------------------------------------------------------------
# ideate + execute (thin wrappers — optional)
# ---------------------------------------------------------------------------


def ideate(
    harness: "Harness",
    workspace: Any,
    ideator_dir: Path,
    *,
    prompt_vars: dict[str, Any],
    timeout: int = 120,
    stateful: bool = True,
    reset: bool = False,
    mounts: list[Any] | None = None,
) -> str:
    """Run the ideator; return idea text from <ideator_dir>/idea.md (empty on miss).

    The ``reset`` flag deletes the .session_id file before running, forcing a
    fresh session. This is a shim until Harness.run() gains a native reset param.
    """
    if reset:
        sid_file = ideator_dir / ".session_id"
        if sid_file.exists():
            sid_file.unlink()
    harness.run(
        workspace,
        prompt=prompt_vars,
        mounts=mounts,
        stateful=stateful,
        timeout=timeout,
        path=ideator_dir,
    ).result()
    idea_file = ideator_dir / "idea.md"
    if not idea_file.exists():
        return ""
    return idea_file.read_text().strip()


def execute(
    harness: "Harness",
    workspace: Any,
    exec_dir: Path,
    *,
    prompt_vars: dict[str, Any],
    grader: Any,
    timeout: int,
    mounts: list[Any] | None = None,
    lower_is_better: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Run executor with a grading server context. Returns (RunResult, parsed info).

    ``lower_is_better`` controls how ``best_score`` is chosen across multiple
    executor attempts — pass ``adapter.lower_is_better`` so maximize tasks
    (e.g. ModelUnlearning) don't silently keep the worst attempt.

    When the agent finishes without a score (e.g. it died before calling the
    in-sandbox grade tool), ``execute`` falls back to reading the grader's
    declared ``input_files`` from the workspace and re-invoking ``grade``
    directly. This keeps all task-specific parsing inside the grader class.
    """
    # Local import: defer loading parse_workspace — only needed when execute()
    # is called, not every time heuresis.experiment is imported.
    from heuresis.parsing import parse_workspace

    with grader:
        result = harness.run(
            workspace,
            prompt=prompt_vars,
            mounts=mounts,
            timeout=timeout,
            path=exec_dir,
        ).result()
    info = parse_workspace(result.workspace, lower_is_better=lower_is_better)

    if info.get("best_score") is None and getattr(grader, "input_files", None):
        files = {
            name: (exec_dir / name).read_bytes()
            for name in grader.input_files
            if (exec_dir / name).is_file()
        }
        if files:
            fb = grader.grade(files)
            if fb.get("score") is not None:
                info["best_score"] = fb["score"]
                info["valid"] = fb.get("valid", False)

    return result, info


# ---------------------------------------------------------------------------
# regenerate — re-run task's canonical pipeline without an agent
# ---------------------------------------------------------------------------


def regenerate(
    task_dir: Path,
    exec_workspace: Path,
    *,
    gpu_ids: list[int],
) -> bool:
    """Re-run the task's canonical pipeline without an agent.

    Reads the ``verify:`` section of ``task_dir/task_config.yaml``. Writes
    output under ``exec_workspace/regenerated/<stdout>`` so the original
    (possibly fabricated) evidence at ``exec_workspace/<stdout>`` is preserved
    for post-hoc analysis.

    **Contract**: ``verify.stdout`` should match a filename the task's
    ``GradingServer.input_files`` reads. Otherwise ``judge_and_maybe_regrade``
    will find no regenerated files to re-grade and silently mark the run
    invalid (a warning is logged in that case).

    Returns True if regeneration ran (regardless of the command's exit code);
    False only if the task opts out (no ``verify:`` section).
    """
    cfg_path = task_dir / "task_config.yaml"
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh) or {}
    verify = cfg.get("verify")
    if not verify:
        return False

    stdout_name = verify.get("stdout", "run.log")
    stdout_rel = f"regenerated/{stdout_name}"
    _bwrap_run_command(
        workspace=exec_workspace,
        command=shlex.split(verify["command"]),
        gpu_ids=gpu_ids,
        timeout=verify.get("timeout", 1400),
        stdout_to=stdout_rel,
    )
    return True


# ---------------------------------------------------------------------------
# judge_and_maybe_regrade — adjudicate an executor run, possibly regenerate
# ---------------------------------------------------------------------------


def judge_and_maybe_regrade(
    *,
    judge: Any,     # HackerJudge | None (duck-typed to avoid import cycle)
    task_dir: Path,
    grader: Any,    # GradingServer
    exec_workspace: Path,
    judge_dir: Path,
    idea: str,
    info: dict[str, Any],
    gpu_ids: list[int],
    fail_open: bool = False,
) -> tuple[dict[str, Any], Any]:
    """Run the judge; dispatch on verdict.

    - ``judge=None`` → pass-through, no judging performed.
    - ``info["best_score"] is None`` → short-circuit; nothing to adjudicate.
    - verdict ``errored`` → fail-closed (mark invalid) unless ``fail_open=True``.
    - verdict ``invalid_idea`` → null score, mark invalid, stash reasoning.
    - verdict ``suspicious_evidence`` → ``regenerate()`` + re-grade; if the task
      opts out of regeneration, mark invalid.
    - verdict ``valid`` → info is passed through unchanged except for
      ``judge_verdict`` flag.

    Returns ``(new_info, verdict_or_None)``. The caller is expected to call
    ``exp.save_judge_review(run_id, verdict)`` when verdict is not None.
    """
    if judge is None:
        return info, None
    if info.get("best_score") is None:
        return info, None

    verdict = judge.review(
        exec_workspace=exec_workspace,
        judge_dir=judge_dir,
        idea=idea,
        info=info,
    )

    if verdict.errored:
        if not fail_open:
            info = {**info, "best_score": None, "valid": False,
                    "judge_errored": True}
    elif verdict.decision == "invalid_idea":
        info = {**info, "best_score": None, "valid": False,
                "judge_rejection": verdict.reasoning}
    elif verdict.decision == "suspicious_evidence":
        if regenerate(task_dir, exec_workspace, gpu_ids=gpu_ids):
            regen_dir = exec_workspace / "regenerated"
            files = {
                f: (regen_dir / f).read_bytes()
                for f in getattr(grader, "input_files", [])
                if (regen_dir / f).is_file()
            }
            if not files:
                logger.warning(
                    "judge_and_maybe_regrade: regenerate produced no files matching "
                    "grader.input_files=%r under %s. Check that task_config.yaml's "
                    "verify.stdout matches a name the grader reads.",
                    getattr(grader, "input_files", None), regen_dir,
                )
            new = grader.grade(files) if files else {"score": None, "valid": False}
            info = {**info,
                    "best_score": new.get("score"),
                    "valid": new.get("valid", False),
                    "regenerated": True}
        else:
            info = {**info, "best_score": None, "valid": False,
                    "regenerate_unavailable": True}
    # else: decision == "valid" → leave scores alone.

    info["judge_verdict"] = verdict.decision
    return info, verdict
