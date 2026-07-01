from __future__ import annotations

import threading
from contextlib import nullcontext

from heuresis import (
    ResultStore,
    RunResult,
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
)
from heuresis.experiment import ExperimentState
from heuresis.experiment_cli import parse_experiment
from heuresis.memory import MemoryStore
from heuresis.qd import ArchiveIndex, GeminiEmbedder, MoIReviewer, OmniEpicSearch
from heuresis.tasks.adapter import get_task_adapter


def run_omni_epic(task_name: str, *, argv: list[str] | None = None) -> None:
    """OMNI-EPIC: MoI-gated open-ended search for any task, via its thin adapter."""
    parsed = parse_experiment(task_name, "omni_epic", argv=argv,
                              experiment_name=f"{task_name}-omni_epic")
    settings = parsed.settings
    cfg = parsed.strategy  # OmniEpicConfig
    adapter = get_task_adapter(task_name, parsed.task, parsed.settings)
    adapter.normalize_settings(settings)

    # The omni ideator prompt hard-includes task_prompt_template; fail with a
    # clear message instead of a cryptic Jinja error if a task lacks one.
    if adapter.task_prompt_template is None:
        raise SystemExit(
            f"task {adapter.name!r} has no task_prompt_template; "
            f"it cannot run omni_epic (the ideator prompt requires one).")

    errors = adapter.preflight(settings)
    if errors:
        print("Preflight failed:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)
    print("Preflight OK")

    adapter.setup_objective(settings)

    strategy_lock = threading.Lock()
    store = ResultStore(db_path=adapter.store_path)
    embedder = GeminiEmbedder()

    # omni's strategy needs an ArchiveIndex bound to `exp`, so we create/resume
    # the experiment manually (resume_or_new would require the strategy first).
    if settings.resume_exp_id:
        exp = store.get_experiment(settings.resume_exp_id)
        if exp is None:
            raise RuntimeError(f"No experiment found with id {settings.resume_exp_id!r}")
        _runs = exp.runs(run_type="executor")
        next_idx = max((r.iteration for r in _runs if r.iteration is not None), default=-1) + 1
        valid_count = sum(1 for r in _runs if r.valid)
        is_resume = True
    else:
        exp = store.experiment(settings.experiment_name, task=adapter.store_task,
                               root=adapter.runs_root)
        _runs, next_idx, valid_count, is_resume = [], 0, 0, False
    state = ExperimentState(exp=exp, next_iter_idx=next_idx,
                            valid_count=valid_count, is_resume=is_resume)
    adapter.on_experiment(exp, state, settings)

    archive_index = ArchiveIndex(embedder=embedder, experiment=exp)
    reviewer = MoIReviewer(archive_index, adapter.task_dir,
                           min_archive_size=cfg.min_archive_size,
                           context=adapter.moi_context())
    strategy = OmniEpicSearch(archive_index, reviewer,
                              lower_is_better=adapter.lower_is_better,
                              memory=settings.memory)

    if is_resume:
        archive_index.rebuild_from_experiment(exp)
        strategy.rebuild([
            (r.run_id, r.score, {**r.metadata, "parent_ids": r.parent_ids,
                                 "generation": r.generation, "idea": r.idea})
            for r in _runs
        ])
        print(f"Resumed archive: accepted={archive_index.size('accepted')}, "
              f"failed_moi={archive_index.size('failed_moi')}")
    elif cfg.seed_source:
        seed_rows = store.query(
            "SELECT run_id, idea, score FROM runs WHERE experiment_id=? "
            "AND run_type='executor' AND valid=1 AND idea IS NOT NULL "
            "ORDER BY rowid LIMIT ?", (cfg.seed_source, cfg.seed_count))
        seed_rows = [r for r in seed_rows if r["idea"] and len(r["idea"]) > 50]
        print(f"Pre-seeding from {cfg.seed_source}: {len(seed_rows)} plans")
        for r in seed_rows:
            archive_index.add_accepted(run_id=f"seed_{r['run_id']}",
                                       plan=r["idea"], score=r["score"])

    ideator_prompt = (adapter.project_root / "src" / "heuresis" / "qd"
                      / "omni_epic" / "ideator_prompt.j2")

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

    metric_direction = "min" if adapter.lower_is_better else "max"
    tmin = settings.executor_timeout // 60

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
                archive_context = strategy.context(ideator_id=tid)

            prior_rejection = None
            iter_valid = False
            for attempt in range(settings.novelty_max_rounds):
                run_id = next_exec_id()
                exec_dir = exp.dir / run_id
                exec_dir.mkdir(parents=True, exist_ok=True)

                # Build from the adapter's task vars (correct keys per task:
                # problem/description/idea_schema/metric/baseline/...) then add
                # the omni-specific fields. Never hand-list task vars.
                prompt_vars = {
                    **adapter.ideator_task_vars(timeout_minutes=tmin,
                                                memory_on=settings.memory),
                    "task_name": adapter.name,
                    "domain_description": adapter.description_text,
                    "editable_file": adapter.editable,
                    "task_prompt_template": adapter.task_prompt_template,
                    "metric_direction": metric_direction,
                    "archive_context": archive_context,
                    "training_budget_minutes": tmin,
                }
                # idea_schema must be present (omni prompt references it).
                prompt_vars.setdefault("idea_schema", adapter.idea_schema_text or "")
                prompt_vars.setdefault("metric", adapter.metric)
                prompt_vars.setdefault("baseline", adapter.baseline)
                if prior_rejection is not None:
                    prompt_vars["prior_rejection"] = prior_rejection

                idea = ideate(harness, ideator_ws, ideator_dir,
                              prompt_vars=prompt_vars, mounts=[],
                              timeout=settings.ideator_timeout, stateful=False)
                if not idea:
                    print(f"  [tid={tid} i={i} a={attempt}] no idea.md, abandon", flush=True)
                    break

                with strategy_lock:
                    assessment = strategy.review_idea(idea)
                print(f"  [tid={tid} i={i} a={attempt}] MoI interesting="
                      f"{assessment.interesting}", flush=True)

                if not assessment.interesting:
                    with strategy_lock:
                        qd_meta = strategy.on_moi_rejected(run_id, idea, assessment,
                                                           parent_ids=parent_ids)
                        qd_meta["attempt"] = attempt
                        record_run(exp, run_id,
                                   result=RunResult(workspace=exec_dir, exit_code=0,
                                                    stats={"duration_s": assessment.duration_s}),
                                   info={"valid": False, "best_score": None,
                                         "moi_rejected": True, "attempt": attempt},
                                   strategy_meta=qd_meta, iteration=i,
                                   run_type="executor", idea=idea, parent_ids=parent_ids)
                    prior_rejection = {"summary": idea[:400],
                                       "reasoning": assessment.reasoning}
                    continue

                # accepted → execute
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
                                                 moi_assessment=assessment)
                    record_run(exp, run_id, result=result, info=info,
                               strategy_meta=qd_meta, iteration=i,
                               run_type="executor", idea=idea, parent_ids=parent_ids,
                               snapshot_files=adapter.snapshot_files(exec_dir),
                               memory=memory, ideator_workspace=ideator_dir)
                    if verdict is not None:
                        exp.save_judge_review(run_id, verdict)
                print(f"  [tid={tid} i={i}] {run_id}: {adapter.metric_label}={score}",
                      flush=True)
                iter_valid = bool(info.get("valid", False))
                break

            loop.mark_done(valid=iter_valid)

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
    if not cfg.skip_meta_test:
        adapter.post_loop(exp, settings)
