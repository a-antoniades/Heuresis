from __future__ import annotations

import threading
from contextlib import nullcontext
from typing import Any, Callable

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
from heuresis.qd import CellTargetedMapElitesSearch, feature_namer
from heuresis.qd.map_elites.plotting import plot_archive
from heuresis.tasks.adapter import SupportsCellSearch, get_task_adapter


def _run_cell_search(
    task_name: str,
    strategy_name: str,
    make_strategy: Callable[[Any, Any, Any, Any, Any], Any],
    *,
    argv: list[str] | None = None,
) -> None:
    """Shared cell-targeted loop for map_elites + go_explore (near-identical;
    differ only in the SearchStrategy). Both use the task's cell ideator prompt."""
    parsed = parse_experiment(task_name, strategy_name, argv=argv,
                              experiment_name=f"{task_name}-{strategy_name}")
    settings = parsed.settings
    cfg = parsed.strategy  # CellConfig
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

    if not isinstance(adapter, SupportsCellSearch):
        raise TypeError(
            f"task {adapter.name!r} does not support cell-based strategies "
            f"(no feature classifier / cell ideator prompt)")
    classifier = adapter.make_classifier()
    features = classifier.features
    name_fn = feature_namer(features)
    strategy = make_strategy(features, name_fn, adapter, settings, cfg)
    strategy_lock = threading.Lock()
    plot_lock = threading.Lock()

    store = ResultStore(db_path=adapter.store_path)
    state = resume_or_new(store, settings.experiment_name, strategy, settings,
                          root=adapter.runs_root, task=adapter.store_task)
    exp = state.exp
    adapter.on_experiment(exp, state, settings)

    ideator_prompt = adapter.cell_ideator_prompt

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

            print(f"  [tid={tid} i={i}] Ideating (parents={len(parent_runs)})",
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

            features = classifier.classify(idea, result.workspace) if score is not None else None

            with strategy_lock:
                qd_meta = strategy.on_result(run_id, score, features=features,
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
            print(f"  [tid={tid} i={i}] {run_id}: {adapter.metric_label}={score} "
                  f"[{qd_meta.get('archive_status', '?')}, cell={qd_meta.get('cell_key', '?')}]",
                  flush=True)

            if info.get("valid") and loop.valid_count % 10 == 0:
                with plot_lock:
                    try:
                        plot_archive(strategy.archive, exp.dir / "archive.png",
                                     feature_name_fn=name_fn,
                                     successful_runs=loop.valid_count,
                                     title=f"{exp.name} — {exp.id}")
                    except Exception as e:
                        print(f"  [tid={tid}] WARN: plot failed: {e}", flush=True)

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
    try:
        plot_archive(strategy.archive, exp.dir / "archive.png",
                     feature_name_fn=name_fn,
                     successful_runs=loop.valid_count, title=f"{exp.name} — {exp.id}")
    except Exception as e:
        print(f"WARN: final plot failed: {e}")
    adapter.post_loop(exp, settings)


def run_map_elites(task_name: str, *, argv: list[str] | None = None) -> None:
    """Cell-targeted MAP-Elites search for any task."""
    def _make(features, name_fn, adapter, settings, cfg):
        return CellTargetedMapElitesSearch(
            features,
            maximize=not adapter.lower_is_better,
            empty_weight=cfg.empty_weight,
            crossover_rate=cfg.crossover_rate,
            feature_name_fn=name_fn,
            memory=settings.memory,
        )
    _run_cell_search(task_name, "map_elites", _make, argv=argv)
