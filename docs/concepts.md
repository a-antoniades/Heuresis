# Concepts

Heuresis has two layers: reusable execution primitives and experiment
entry points that compose those primitives into task x strategy runs.

## Core Flow

```text
ideator workspace -> idea.md -> executor workspace -> grade/store -> strategy feedback
```

1. An ideator agent receives task context, parent results, and search guidance.
2. It writes an `idea.md` describing the next change to try.
3. An executor agent implements the idea in a sandboxed workspace.
4. A grader or task driver evaluates the result.
5. A `ResultStore` records the run in SQLite.
6. A search strategy selects future parents or context from the accumulated
   results.

## Workspace

`Workspace` describes what an agent sees: prompt template, tools, seed files,
venv, optional mounts, role markers, and sandbox edit restrictions. Calling
`Workspace.setup(path)` materializes that configuration in a run directory.

The default workspace venv is separate from the development environment and is
built from the `sandbox` extra. Task-specific executor venvs use task-local
requirements files.

## Harness

`Harness` launches an agent against a `Workspace`. It handles the agent binary,
model, GPU assignment, sandbox command construction, stateful sessions, mounts,
timeouts, and result collection.

## Tasks

A task defines the problem and evaluation contract. In-tree tasks live under
`src/heuresis/tasks/<task>/` and typically include:

- `task_config.yaml` for templates, seed files, editable surface, requirements,
  and verification metadata.
- Prompt templates such as `problem.j2` and `executor_prompt.j2`.
- Seed files copied into executor workspaces.
- A grader or task driver that reports scores.

Current task families include `nanogpt`, `discogen_onpolicyrl` (On-Policy RL),
`discogen_modelunlearning` (Model Unlearning), and the lightweight CPU benchmark
`bbob`.

## Strategies

A strategy decides which prior results should influence the next idea. Strategies
include linear search, MAP-Elites, cell-targeted MAP-Elites, Go-Explore, island
search, curiosity, curiosity-plus, and OMNI-EPIC.

Experiment entry points combine one task with one strategy. The strict CLI layer
in `heuresis.experiment_cli` parses shared settings plus task-specific and
strategy-specific config.

## Stores and Runs

`ResultStore` stores experiments and runs in SQLite. Generated run directories
and stores live under `runs/` and are gitignored. A run workspace usually
contains logs, `idea.md`, edited files, grading artifacts, and agent output.

## Memory

The memory primitive lets ideators and executors append and retrieve structured
campaign memory through a host-side store and an in-sandbox CLI. It is opt-in
with the shared `--memory` flag for experiments that support it.

## Sandboxing

Sandboxing is built around `bubblewrap`. The sandbox restricts filesystem
visibility, mounts the workspace, exposes selected tools, and prevents agents
from installing packages inside the workspace. Dependencies should be declared
up front through package extras or task requirements.
