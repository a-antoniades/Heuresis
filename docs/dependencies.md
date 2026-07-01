# Dependencies

Heuresis uses different dependency sets for development, the base sandbox,
and task-specific executor environments.

## Development Environment

The development environment is managed by `pyproject.toml` and `uv.lock`:

```bash
uv sync --extra dev
```

The `dev` extra includes tests, linting, type-checking tools, and dependencies
needed by common local workflows.

## Package Dependencies

The main `[project.dependencies]` list is for the importable `heuresis`
package. Keep it focused on runtime library dependencies required by the core
package.

## Optional Extras

Named extras model first-party optional environments:

- `sandbox`: dependencies for the default workspace venv.
- `novelty`: dependencies for the `NoveltyReviewer` sandbox.
- `discogen`: the pinned DiscoGen dependency.
- `dev`: local development tools and test dependencies.

The default workspace venv installs from `.[sandbox]`. The novelty reviewer venv
installs from `.[novelty]`.

## Task Requirements

Task-local `requirements.txt` files under `src/heuresis/tasks/` describe
agent execution sandboxes. They are intentionally separate from the main package
install because executors run task code in isolated venvs.

Examples:

- `src/heuresis/tasks/nanogpt/requirements.txt`
- `src/heuresis/tasks/bbob/requirements.txt`

DiscoGen domains may provide their own upstream domain requirements. Those are
used to build per-domain task venvs.

## Venv Creation

`Workspace.setup(...)` and `ensure_venv(...)` create sandbox venvs lazily. If a
venv already has `bin/python`, creation is skipped.

- Default workspace venv: `venvs/base`, installed from `.[sandbox]`.
- Novelty reviewer venv: `venvs/reviewer`, installed from `.[novelty]`.
- Task venvs: installed from task or domain requirements files.

Agents cannot install packages inside the sandbox. If a task needs a dependency,
declare it before running the experiment.

## Lockfile

`uv.lock` is tracked so public installs are reproducible. Update it when changing
`pyproject.toml` dependencies or extras.
