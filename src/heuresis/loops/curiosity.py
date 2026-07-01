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
from heuresis.qd import GeminiEmbedder
from heuresis.tasks.adapter import get_task_adapter
from heuresis.qd.curiosity.prediction import predict_outcome
from heuresis.qd.curiosity.seeding import parse_candidates


def _run_curiosity(
    task_name: str,
    strategy_name: str,
    make_strategy: Callable[[Any, Any, Any, Any], Any],
    *,
    argv: list[str] | None = None,
) -> None:
    """Shared prediction-error curiosity loop (curiosity + curiosity_plus)."""
    parsed = parse_experiment(task_name, strategy_name, argv=argv,
                              experiment_name=f"{task_name}-{strategy_name}")
    settings = parsed.settings
    cfg = parsed.strategy  # CuriosityConfig / CuriosityPlusConfig
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

    embedder = GeminiEmbedder()
    strategy = make_strategy(embedder, adapter, settings, cfg)
    strategy_lock = threading.Lock()

    store = ResultStore(db_path=adapter.store_path)
    state = resume_or_new(store, settings.experiment_name, strategy, settings,
                          root=adapter.runs_root, task=adapter.store_task)
    exp = state.exp
    adapter.on_experiment(exp, state, settings)


    gpu_slice = settings.gpus[: settings.num_ideators] if adapter.uses_gpu else []
    print(f"Experiment: {exp.id}  (resume={state.is_resume})")
    print(f"Dir: {exp.dir}")
    _gpu_note = f" | GPUs: {gpu_slice}" if adapter.uses_gpu else ""
    print(f"Target: {settings.num_iterations} "
          f"{'valid runs' if settings.count_valid else 'iterations'} "
          f"| ideators: {settings.num_ideators}{_gpu_note} | n_seed={cfg.n_seed}")

    judge = adapter.make_judge(settings)
    seeding_ws = adapter.ideator_workspace(settings, prompt=adapter.strategy_prompt(strategy_name, "seeding"))
    ideator_ws = adapter.ideator_workspace(settings, prompt=adapter.strategy_prompt(strategy_name, "ideator"))
    prediction_ws = adapter.ideator_workspace(settings, prompt=adapter.strategy_prompt(strategy_name, "prediction"))

    _reservations = reserve_gpus(gpu_slice)            # noqa: F841 (held)
    harnesses = build_harnesses(settings, gpu_slice, uses_gpu=adapter.uses_gpu)

    loop = iterate_until_valid(state, settings)
    exec_counter_lock = threading.Lock()
    _next_exec = [next_run_index(exp)]
    tmin = settings.executor_timeout // 60

    def next_exec_id() -> str:
        with exec_counter_lock:
            idx = _next_exec[0]
            _next_exec[0] += 1
        return f"exec_{idx:03d}"

    def run_seed(tid, harness, ideator_dir, i, run_id):
        with strategy_lock:
            existing = [e.idea[:120].replace("\n", " ")
                        for e in strategy.store.recent_entries(20)]
        ideate(harness, seeding_ws, ideator_dir,
               prompt_vars={
                   **adapter.ideator_task_vars(timeout_minutes=tmin,
                                               memory_on=settings.memory),
                   "num_candidates": cfg.seed_batch,
                   "existing_summaries": existing,
               },
               timeout=settings.ideator_timeout, stateful=False, reset=True)
        cand_file = ideator_dir / "candidates.md"
        raw = cand_file.read_text() if cand_file.exists() else ""
        if not raw:
            idea_file = ideator_dir / "idea.md"
            raw = idea_file.read_text() if idea_file.exists() else ""
        candidates = parse_candidates(raw, expected=cfg.seed_batch) if raw else []
        if not candidates:
            return None
        with strategy_lock:
            _idx, idea = strategy.select_seed_candidate(
                candidates, reserve_run_id=run_id, reserve_iteration=i)
        return idea

    def body(tid, harness):
        ideator_dir = exp.dir / f"ideator_{tid}"
        ideator_dir.mkdir(exist_ok=True)
        prediction_dir = exp.dir / f"prediction_{tid}"
        prediction_dir.mkdir(exist_ok=True)

        while loop.should_continue():
            try:
                i = next(loop)
            except StopIteration:
                return

            with strategy_lock:
                seeding = strategy.is_seeding()

            prediction = None
            if seeding:
                run_id = next_exec_id()
                idea = run_seed(tid, harness, ideator_dir, i, run_id)
                if not idea:
                    loop.mark_done(valid=False)
                    continue
                parent_ids: list = []
            else:
                with strategy_lock:
                    parent_ids = strategy.select_parents(ideator_id=tid)
                    ctx = strategy.context(ideator_id=tid)
                idea = ideate(harness, ideator_ws, ideator_dir,
                              prompt_vars={
                                  **adapter.ideator_task_vars(timeout_minutes=tmin,
                                                              memory_on=settings.memory),
                                  "curiosity_context": ctx,
                              },
                              timeout=settings.ideator_timeout, stateful=True, reset=False)
                if not idea:
                    print(f"  [tid={tid} i={i}] no idea.md, skip", flush=True)
                    loop.mark_done(valid=False)
                    continue
                with strategy_lock:
                    pred_ctx = strategy.prediction_context(max_history=10)
                prediction = predict_outcome(
                    harness, prediction_ws, prediction_dir,
                    prompt_vars={
                        **adapter.ideator_task_vars(timeout_minutes=tmin,
                                                    memory_on=settings.memory),
                        "idea": idea, "prediction_context": pred_ctx,
                    },
                    timeout=cfg.prediction_timeout)
                run_id = next_exec_id()

            exec_dir = exp.dir / run_id
            exec_dir.mkdir(parents=True, exist_ok=True)
            grader = adapter.make_grader(exec_dir)
            parent_runs = [r for r in exp.runs(run_type="executor")
                           if r.run_id in parent_ids]
            files = executor_files(adapter, parent_runs, inherit_intent=False)
            executor_ws = adapter.executor_workspace(files=files, memory_on=settings.memory)
            if memory is not None:
                executor_ws.memory_socket = memory.socket_path

            result, info = execute(
                harness, executor_ws, exec_dir,
                prompt_vars=adapter.executor_task_vars(
                    idea=idea, timeout_minutes=tmin,
                    gpu_count=len(settings.gpus), memory_on=settings.memory),
                grader=grader, timeout=settings.executor_timeout,
                mounts=adapter.mounts() or None,
                lower_is_better=adapter.lower_is_better)
            judge_dir = exp.dir / f"judge_{i}_{tid}"
            info, verdict = judge_and_maybe_regrade(
                judge=judge, task_dir=adapter.task_dir, grader=grader,
                exec_workspace=exec_dir, judge_dir=judge_dir, idea=idea,
                info=info, gpu_ids=harness.gpus)
            score = info.get("best_score")

            with strategy_lock:
                qd_meta = strategy.on_result(run_id, score, idea=idea,
                                             parent_ids=parent_ids, ideator_id=tid,
                                             prediction=prediction,
                                             valid=info.get("valid", False))
                record_run(exp, run_id, result=result, info=info,
                           strategy_meta=qd_meta, iteration=i,
                           run_type="executor", idea=idea, parent_ids=parent_ids,
                           snapshot_files=adapter.snapshot_files(exec_dir),
                           memory=memory, ideator_workspace=ideator_dir)
                if verdict is not None:
                    exp.save_judge_review(run_id, verdict)
            loop.mark_done(valid=info.get("valid", False))
            print(f"  [tid={tid} i={i}] {run_id}: {adapter.metric_label}={score} "
                  f"[{'seed' if seeding else 'steady'}]", flush=True)

    _mem_cm = (MemoryStore(exp.dir / "memory.db")
               if strategy.memory else nullcontext(None))
    with _mem_cm as memory:
        if memory is not None:
            seeding_ws.memory_socket = memory.socket_path
            ideator_ws.memory_socket = memory.socket_path
            print(f"Campaign memory enabled (db={exp.dir / 'memory.db'})")
        parallel_ideators(harnesses, body)

    print()
    print("=" * 60)
    print(strategy.summary())
    adapter.report_best(exp)
    adapter.post_loop(exp, settings)


def run_curiosity(task_name: str, *, argv: list[str] | None = None) -> None:
    """Prediction-error curiosity search."""
    def _make(embedder, adapter, settings, cfg):
        from heuresis.qd import CuriositySearch
        return CuriositySearch(
            embedder, k_neighbors=cfg.k_neighbors, candidate_window=cfg.candidate_window,
            softmax_temperature=cfg.softmax_tau, anchor_history=cfg.anchor_history,
            novelty_threshold=cfg.novelty_threshold, n_seed=cfg.n_seed,
            lower_is_better=adapter.lower_is_better, memory=settings.memory)
    _run_curiosity(task_name, "curiosity", _make, argv=argv)
