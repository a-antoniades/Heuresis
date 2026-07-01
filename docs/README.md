# Documentation

These docs describe the public surface of Heuresis: how to install it,
run experiments, understand the architecture, and extend it with new tasks or
search strategies.

## Reading Order

1. `getting-started.md` - install, credentials, first run, and host checks.
2. `concepts.md` - the core model: tasks, strategies, workspaces, harnesses,
   stores, graders, memory, and sandbox venvs.
3. `experiments.md` - how to run experiments with the `heuresis` CLI.
4. `add-task.md` - how to add a new benchmark/task family.
5. `add-strategy.md` - how to add a new search algorithm.
6. `configuration.md` - credentials, CLI flags, API key precedence, and runtime
   environment variables.
7. `dependencies.md` - package dependencies, sandbox extras, task requirements,
   and `uv.lock`.
8. `troubleshooting.md` - common setup, sandbox, auth, GPU, and smoke-test
   issues.
9. `reference.md` - compact reference for the main public Python APIs.

## What Is Not Public Documentation

Historical plans, private runbooks, paper schedules, bug logs, and maintainer
skills live under `maintainer/`. They are preserved for project operators but
are not required to use or extend the public package.

The `analysis/` tree contains research artifact tooling. It may require
external run data and is not a stable library API.
