# Add a Search Strategy

A search strategy decides which prior results become context for the next idea.
Adding a strategy usually touches the QD package, prompt fragments, CLI config,
experiment entry points, wrappers, and tests.

## 1. Implement the Strategy

A strategy has two parts:

1. The search algorithm under `src/heuresis/qd/<strategy>/search.py`, subclassing
   `SearchStrategy` (`src/heuresis/qd/core/base.py`).
2. The experiment loop in `src/heuresis/loops/<strategy>.py`, exposing
   `run_<strategy>(task_name)`, which drives the ideate/execute/grade cycle.

Keep the strategy interface explicit:

- select parents or context from prior results
- render concise ideator context
- ingest executor outcomes
- persist any archive or population state needed for resume

A strategy that is a small parameter delta of another can reuse that strategy's
loop (see `loops/go_explore.py` reusing `loops/map_elites.py`, and
`loops/curiosity_plus.py` reusing `loops/curiosity.py`). Prefer small data
objects and deterministic state updates. Avoid hiding task-specific behavior in
generic strategy code.

## 2. Add Prompt Guidance

If ideators need strategy-specific instructions, add or update a prompt fragment
under `src/heuresis/prompts/search/`. The fragment should explain how to
use the context the strategy already selected, not reimplement the selection
policy in prose.

## 3. Add Strategy CLI Config

Add a strategy config dataclass and parser logic in
`src/heuresis/experiment_cli.py` if the strategy needs knobs. Examples:

- curiosity: neighbor counts, softmax temperature, prediction timeout
- OMNI-EPIC: archive size and seed controls
- islands: migration interval and crossover rate
- cell strategies: empty-cell weight and crossover rate

Flags should be scoped to the strategies that support them. Unsupported flags
should fail.

## 4. Register Task x Strategy Defaults

Add registry entries for every task that supports the strategy. Defaults belong
in the registry so wrappers and direct entry points agree on task x strategy
behavior.

If a strategy is not meaningful for a task, do not register that combination.

## 5. Register the Loop

Register the loop in `heuresis.loops.LOOPS` (in `src/heuresis/loops/__init__.py`)
so the `heuresis <task> <strategy>` launcher can dispatch to it. The
`parse_experiment(...)` call and consumption of
`parsed.settings`/`parsed.task`/`parsed.strategy` live inside
`src/heuresis/loops/<strategy>.py`. No per-pairing
`experiments/<task>_<strategy>/run.py` is created for loop-backed tasks.

## 6. Add Launch Configs

Add `configs/experiments/<task>/<strategy>.yaml` entries for each supported
pairing. Each file has top-level `task:`/`strategy:` keys, a `settings:` block
whose keys match `Settings` fields (`num_iterations`, `gpus`, `model`, ...), and
a `strategy_config:` block whose keys match the strategy's config dataclass (for
example `migration_interval`/`crossover_rate` for islands,
`cell_empty_weight`/`cell_crossover_rate` for cell strategies,
`n_seed`/`min_archive_size` for omni_epic). See
`configs/experiments/nanogpt/islands.yaml`.

## 7. Update Wrapper Support

Extend `heuresis.experiment_cli_args` only when wrappers need a new
strategy-specific environment default. The bridge filters those defaults through
the registered task x strategy parser so wrappers only emit supported flags. The
public command is `heuresis`; shell wrappers should call
`uv run heuresis <task> <strategy> "$@"` so explicit flags override environment
defaults. Then add public wrappers where they are useful.

## 8. Add Tests

Add tests for:

- parser accepts valid strategy flags
- parser rejects strategy flags for unsupported strategies
- registry definitions build successfully
- wrappers emit expected CLI flags
- core strategy state transitions

Default tests should not depend on live agents, GPUs, or external APIs.

## 9. Document the Strategy

Update `docs/experiments.md` with launchers and `docs/concepts.md` if the
strategy introduces a new concept users need to understand.
