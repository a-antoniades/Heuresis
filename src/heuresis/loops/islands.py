from __future__ import annotations

import threading
from contextlib import nullcontext

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
from heuresis.qd import IslandSearch
from heuresis.qd.islands.plotting import plot_islands
from heuresis.tasks.adapter import get_task_adapter


def run_islands(task_name: str, *, argv: list[str] | None = None) -> None:
    """Island-based search (parallel islands, ring migration) for any task."""
    parsed = parse_experiment(task_name, "islands", argv=argv,
                              experiment_name=f"{task_name}-islands")
    settings = parsed.settings
    cfg = parsed.strategy  # IslandConfig
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

    strategy = IslandSearch(
        num_islands=settings.num_ideators,
        topology="ring",
        max_population=30,
        maximize=not adapter.lower_is_better,
        crossover_rate=cfg.crossover_rate,
        parent_k=2,
        tournament_size=2,
        migration_interval=cfg.migration_interval,
        migration_k=1,
        memory=settings.memory,
    )
    strategy_lock = threading.Lock()
    plot_lock = threading.Lock()

    store = ResultStore(db_path=adapter.store_path)
    state = resume_or_new(store, settings.experiment_name, strategy, settings,
                          root=adapter.runs_root, task=adapter.store_task)
    exp = state.exp
    adapter.on_experiment(exp, state, settings)

    ideator_prompt = adapter.strategy_prompt("islands")

    gpu_slice = settings.gpus[: settings.num_ideators] if adapter.uses_gpu else []
    print(f"Experiment: {exp.id}  (resume={state.is_resume})")
    print(f"Dir: {exp.dir}")
    _gpu_note = f" | GPUs: {gpu_slice}" if adapter.uses_gpu else ""
    print(f"Target: {settings.num_iterations} "
          f"{'valid runs' if settings.count_valid else 'iterations'} "
          f"| islands: {settings.num_ideators}{_gpu_note}")

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
        ideator_dir = exp.dir / f"ideator_island_{tid}"
        ideator_dir.mkdir(exist_ok=True)

        while loop.should_continue():
            try:
                i = next(loop)
            except StopIteration:
                return

            with strategy_lock:
                parent_ids = strategy.select_parents(ideator_id=tid)
                ctx = strategy.context(ideator_id=tid)

            parent_runs = [r for r in exp.runs(run_type="executor")
                           if r.run_id in parent_ids]

            print(f"  [island={tid} i={i}] Ideating (parents={len(parent_runs)})",
                  flush=True)
            idea = ideate(
                harness, ideator_ws, ideator_dir,
                prompt_vars={
                    **adapter.ideator_task_vars(
                        timeout_minutes=settings.executor_timeout // 60,
                        memory_on=settings.memory),
                    "past_results": parent_runs,
                    "search_context": ctx,
                },
                mounts=[r.workspace for r in parent_runs],
                timeout=settings.ideator_timeout, stateful=False,
            )
            if not idea:
                print(f"  [island={tid} i={i}] WARN: no idea.md, skipping", flush=True)
                loop.mark_done(valid=False)
                continue

            run_id = next_exec_id()
            exec_dir = exp.dir / run_id
            exec_dir.mkdir(parents=True, exist_ok=True)
            grader = adapter.make_grader(exec_dir)
            files = executor_files(adapter, parent_runs, inherit_intent=True)
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
            op = qd_meta.get("operator", "?")
            rank = qd_meta.get("rank", "?")
            print(f"  [island={tid} i={i}] {run_id}: "
                  f"{adapter.metric_label}={score} [{op}, rank={rank}]", flush=True)

            if info.get("valid") and loop.valid_count % 10 == 0:
                with plot_lock:
                    try:
                        plot_islands(list(exp.runs(run_type="executor")),
                                     num_islands=settings.num_ideators,
                                     path=exp.dir / "progress.png",
                                     title=f"{exp.name} — {exp.id}")
                    except Exception as e:
                        print(f"  [island={tid}] WARN: plot failed: {e}", flush=True)

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
