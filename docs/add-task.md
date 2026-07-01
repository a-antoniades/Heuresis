# Add a Task

A task is a benchmark or problem family that agents can attempt. Adding a task
means defining the task files, prompts, sandbox requirements, grading or
verification behavior, experiment entry points, and CLI registry entries.

## 1. Create the Task Package

Create `src/heuresis/tasks/<task>/` with the task assets:

```text
src/heuresis/tasks/<task>/
  task_config.yaml
  problem.j2
  executor_prompt.j2
  requirements.txt        # only if executor sandbox needs task deps
  grader.py or driver.py  # task-specific evaluation path
  adapter.py              # concrete TaskAdapter subclass (runtime contract)
  features.py             # FEATURES + make_classifier() (cell strategies only)
  seed files...
```

The task's runtime behavior is wired through a `TaskAdapter` subclass in
`adapter.py` (the base ABC and `get_task_adapter()` live in
`src/heuresis/tasks/adapter.py`). Its diversity feature space lives in
`features.py`, which exposes a module-level `FEATURES` list of `Feature` objects
carrying labeled `bin_names` and a `make_classifier(...)` factory. A task with
several domains may use a `features/` package instead of a flat `features.py`
(see `src/heuresis/tasks/discogen/features/`). Cell-based strategies
(`map_elites`, `go_explore`) require the adapter to satisfy the
`SupportsCellSearch` protocol, which depends on `make_classifier`.

`task_config.yaml` should describe:

- `name` and `description`
- prompt templates
- seed files copied into executor workspaces
- editable file or directory
- optional `requirements`
- verification command and output expectations when applicable

## 2. Define the Workspace Contract

Decide what the executor may edit and what the grader expects. Keep the editable
surface small. The executor prompt should explain:

- the problem and score metric
- allowed files
- command to run
- output artifacts expected by the parser or grader
- constraints that invalidate a result

## 3. Add Task CLI Config

If the task needs task-specific flags, add a task config dataclass and parser
logic in `src/heuresis/experiment_cli.py`. DiscoGen is the current example:
it has `--config`, `--domain`, and `--mu-fast-eval`.

Tasks with no extra flags can use an empty config.

## 4. Register Task x Strategy Definitions

Add definitions to the registry in `src/heuresis/experiment_cli.py`. Each
supported strategy should have a task x strategy entry with defaults appropriate
for that task.

The key is `task:strategy`, for example `nanogpt:linear`.

## 5. Wire the Task Into the Launcher

Once the task's `TaskAdapter` is registered in `get_task_adapter`
(`src/heuresis/tasks/adapter.py`) and its launch configs exist, every supported
pairing is reachable through the launcher with no per-pairing entry point:

```bash
uv run heuresis <task> linear --gpus 0 --num-iterations 10
```

The shared loop in `src/heuresis/loops/<strategy>.py` does the real work: it
calls `parse_experiment`, resolves the `TaskAdapter` via `get_task_adapter`,
builds ideator/executor `Workspace` objects and a `Harness`, opens a
`ResultStore` under `runs/<task>/`, runs the loop, and handles resume. A task
implementer only supplies the `TaskAdapter`; the loops consume it uniformly, so
no per-pairing run script is needed.

## 6. Add Launch Configs

Add launch configs under `configs/experiments/<task>/<strategy>.yaml` for each
supported pairing. Keep these separate from task-domain configs such as
`configs/discogen/*.yaml`.

## 7. No Wrapper Script Needed

There is no per-pairing launch script to add. `uv run heuresis <task>
<strategy>` already runs any registered pairing. Put campaign defaults (GPUs,
iterations, model, and any DiscoGen `--config` / `--domain`) in the registry
(`src/heuresis/experiment_cli.py`) and the launch configs from step 6, not in a
script.

## 8. Add Tests

Add focused tests for:

- task schema and prompt rendering
- parser registry construction
- wrapper argument generation
- smoke behavior for the smallest run path
- any grader or parser logic

Do not require live API calls, GPUs, or local run data in default tests.

## 9. Document the Task

Update `docs/experiments.md` with the public launcher and any important setup
steps. If the task needs unusual dependencies, document them in
`docs/dependencies.md` or task-specific notes.
