# Reference

This is a compact map of the main public Python APIs. See docstrings and tests
for detailed behavior.

## `Workspace`

`heuresis.Workspace` describes an agent workspace before it is
materialized. Important fields include:

- `prompt`: inline prompt string or Jinja template path.
- `tools`: additional CLI tools exposed to the agent.
- `files`: seed files copied into the workspace.
- `venv`, `requirements`, `project_extra`: sandbox dependency source.
- `role`, `memory_socket`: optional memory markers.
- `editable`, `lock_down_edits`: edit-surface controls for stricter sandboxes.

## `Mount`

`heuresis.Mount` describes an explicit bind mount with `source`, `target`,
and `readonly`.

## `Harness`

`heuresis.Harness` launches an agent against a workspace. Key parameters
include `agent`, `model`, `gpus`, and `max_workers`. Use `harness.preflight()` to
check host readiness and `harness.run(...)` to launch work.

## `ResultStore`

`heuresis.ResultStore` manages SQLite-backed experiment persistence. Use
`store.experiment(name, root=...)` to create or resume an experiment record, then
save executor results through the returned experiment object.

## `Settings`

`heuresis.Settings` is the shared experiment configuration dataclass. It is
created by the CLI parser rather than environment variables.

## `parse_experiment`

`heuresis.parse_experiment(definition, strategy=None, argv=None,
task_defaults=None, strategy_defaults=None, **settings_defaults)` parses shared
settings plus task and strategy config (the first argument accepts a task or
definition). It returns a `ParsedExperiment` with:

- `definition`: the registered task x strategy definition.
- `settings`: shared `Settings`.
- `task`: task-specific config object.
- `strategy`: strategy-specific config object.

## Search Strategies

Common strategies live under `heuresis.qd`:

- `linear`: top-parent linear search.
- `map_elites`: archive-based quality-diversity search.
- `go_explore`: cell-targeted exploration.
- `islands`: population-based search with migration/crossover.
- `curiosity`: prediction-error curiosity search.
- `curiosity_plus`: curiosity with additional scoring, tag, and memory signals.
- `omni_epic`: novelty-gated MAP-Elites style search.

## Parsing Workspace Results

`heuresis.parse_workspace(path)` extracts score and run metadata from a
workspace after execution. Experiments use it before saving runs to the store.

## API Key Helpers

`heuresis.api_keys.load_api_key(provider)` and
`load_api_keys(provider)` centralize credential loading. Gemini supports multiple
keys for rotation.
