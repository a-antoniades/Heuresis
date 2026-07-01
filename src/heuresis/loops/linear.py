from __future__ import annotations

import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from heuresis import (
    ResultStore,
    build_harnesses,
    execute,
    executor_files,
    ideate,
    iterate_until_valid,
    judge_and_maybe_regrade,
    next_run_index,
    parallel_ideators,
    record_run,
    reserve_gpus,
    resume_or_new,
)
from heuresis.experiment_cli import parse_experiment
from heuresis.memory import MemoryStore
from heuresis.qd import LinearSearch
from heuresis.tasks.adapter import get_task_adapter


def annotate_runs(records: list[Any]) -> list[dict[str, Any]]:
    """Attach each run's executor workspace_id (memory DB key) to a dict —
    the ``past_results`` / ``new_since_last_turn`` context the linear ideator
    consumes."""
    out = []
    for r in records:
        wid = ""
        try:
            wid = (Path(r.workspace) / ".workspace_id").read_text().strip()
        except OSError:
            pass
        out.append({"run_id": r.run_id, "executor_id": wid,
                    "score": r.score, "idea": r.idea})
    return out


class DeltaQueue:
    """Tracks which executor runs a thread had already seen, to surface
    "what finished since I last ideated"."""

    def __init__(self, initial_ids: set[str]):
        self._seen = set(initial_ids)

    def new_since_last(self, all_runs: list[Any]) -> list[Any]:
        new = sorted(
            (r for r in all_runs if r.run_id not in self._seen),
            key=lambda r: r.iteration if r.iteration is not None else 0,
        )
        self._seen = {r.run_id for r in all_runs}
        return new


def run_linear(task_name: str, *, argv: list[str] | None = None) -> None:
    """Linear (top-K, no archive) search for any task, via its thin TaskAdapter."""
    parsed = parse_experiment(task_name, "linear", argv=argv,
                              experiment_name=f"{task_name}-linear")
    settings = parsed.settings
    adapter = get_task_adapter(task_name, parsed.task, parsed.settings)
    adapter.normalize_settings(settings)

    errors = adapter.preflight(settings)
    if errors:
        print("Preflight failed:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)
    print("Preflight OK")

    adapter.setup_objective(settings)

    strategy = LinearSearch(
        max_parents=settings.max_parents,
        maximize=not adapter.lower_is_better,
        session_reset_every=settings.session_reset_every,
        memory=settings.memory,
    )
    strategy_lock = threading.Lock()

    store = ResultStore(db_path=adapter.store_path)
    state = resume_or_new(store, settings.experiment_name, strategy, settings,
                          root=adapter.runs_root, task=adapter.store_task)
    exp = state.exp
    adapter.on_experiment(exp, state, settings)

    ideator_prompt = adapter.strategy_prompt("linear")

    gpu_slice = settings.gpus[: settings.num_ideators] if adapter.uses_gpu else []
    print(f"Experiment: {exp.id}  (resume={state.is_resume})")
    print(f"Dir: {exp.dir}")
    _gpu_note = f" | GPUs: {gpu_slice}" if adapter.uses_gpu else ""
    print(f"Target: {settings.num_iterations} "
          f"{'valid runs' if settings.count_valid else 'iterations'} "
          f"| ideators: {settings.num_ideators}{_gpu_note}")

    judge = adapter.make_judge(settings)
    ideator_ws = adapter.ideator_workspace(settings, prompt=ideator_prompt)

    _reservations = reserve_gpus(gpu_slice)            # noqa: F841 (held)
    harnesses = build_harnesses(settings, gpu_slice, uses_gpu=adapter.uses_gpu)

    loop = iterate_until_valid(state, settings)
    exec_counter_lock = threading.Lock()
    _next_exec = [next_run_index(exp)]

    def next_exec_id() -> str:
        with exec_counter_lock:
            idx = _next_exec[0]
            _next_exec[0] += 1
        return f"exec_{idx:03d}"

    def body(tid, harness):
        ideator_dir = exp.dir / f"ideator_{tid}"
        ideator_dir.mkdir(exist_ok=True)
        dq = DeltaQueue({r.run_id for r in exp.runs(run_type="executor")})

        while loop.should_continue():
            try:
                i = next(loop)
            except StopIteration:
                return

            with strategy_lock:
                parent_ids = strategy.select_parents(ideator_id=tid)
                reset = strategy.should_reset_session(i)

            all_runs = exp.runs(run_type="executor")
            parent_runs = [r for r in all_runs if r.run_id in parent_ids]
            new_runs = dq.new_since_last(all_runs)
            visible = {r.run_id: r for r in parent_runs}
            for r in new_runs:
                visible.setdefault(r.run_id, r)

            print(f"  [tid={tid} i={i}] Ideating (reset={reset}, "
                  f"parents={len(parent_runs)}, new={len(new_runs)})", flush=True)
            idea = ideate(
                harness, ideator_ws, ideator_dir,
                prompt_vars={
                    **adapter.ideator_task_vars(
                        timeout_minutes=settings.executor_timeout // 60,
                        memory_on=settings.memory),
                    "past_results": annotate_runs(parent_runs),
                    "new_since_last_turn": annotate_runs(new_runs),
                },
                mounts=[r.workspace for r in visible.values()],
                timeout=settings.ideator_timeout, stateful=True, reset=reset,
            )
            if not idea:
                print(f"  [tid={tid} i={i}] WARN: no idea.md, skipping", flush=True)
                loop.mark_done(valid=False)
                continue

            run_id = next_exec_id()
            exec_dir = exp.dir / run_id
            exec_dir.mkdir(parents=True, exist_ok=True)
            grader = adapter.make_grader(exec_dir)
            files = executor_files(adapter, parent_runs, inherit_intent=False)
            executor_ws = adapter.executor_workspace(files=files, memory_on=settings.memory)
            if memory is not None:
                executor_ws.memory_socket = memory.socket_path

            result, info = execute(
                harness, executor_ws, exec_dir,
                prompt_vars=adapter.executor_task_vars(
                    idea=idea, timeout_minutes=settings.executor_timeout // 60,
                    gpu_count=len(settings.gpus), memory_on=settings.memory),
                grader=grader, timeout=settings.executor_timeout,
                mounts=adapter.mounts() or None,
                lower_is_better=adapter.lower_is_better,
            )
            judge_dir = exp.dir / f"judge_{i}_{tid}"
            info, verdict = judge_and_maybe_regrade(
                judge=judge, task_dir=adapter.task_dir, grader=grader,
                exec_workspace=exec_dir, judge_dir=judge_dir, idea=idea,
                info=info, gpu_ids=harness.gpus)
            score = info.get("best_score")

            with strategy_lock:
                qd_meta = strategy.on_result(run_id, score, features=None,
                                             idea=idea, parent_ids=parent_ids,
                                             ideator_id=tid)
                record_run(exp, run_id, result=result, info=info,
                           strategy_meta=qd_meta, iteration=i,
                           run_type="executor", idea=idea, parent_ids=parent_ids,
                           snapshot_files=adapter.snapshot_files(exec_dir),
                           memory=memory, ideator_workspace=ideator_dir)
                if verdict is not None:
                    exp.save_judge_review(run_id, verdict)
            loop.mark_done(valid=info.get("valid", False))
            print(f"  [tid={tid} i={i}] {run_id}: "
                  f"{adapter.metric_label}={score}", flush=True)

    _mem_cm = (MemoryStore(exp.dir / "memory.db")
               if strategy.memory else nullcontext(None))
    with _mem_cm as memory:
        if memory is not None:
            ideator_ws.memory_socket = memory.socket_path
            print(f"Campaign memory enabled (db={exp.dir / 'memory.db'})")
        parallel_ideators(harnesses, body)

    print()
    print("=" * 60)
    print(strategy.summary())
    adapter.report_best(exp)
    adapter.post_loop(exp, settings)
