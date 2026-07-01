# Experiments

A task x strategy run is launched with the `heuresis` CLI, which resolves the
arguments and dispatches to that strategy's loop end-to-end. You can launch
directly or through shell wrappers in `scripts/`.

## The `heuresis` Launcher

`heuresis <task> <strategy> [flags]` runs the experiment. It resolves arguments
(environment-variable defaults from wrappers, an optional launch-config YAML,
then explicit flags), validates them against the registered parser for that
task x strategy, and runs the matching loop in `heuresis.loops.LOOPS`:

```bash
uv run heuresis nanogpt linear --gpus 0 --num-iterations 10 --count-total
MODEL=your-model-id
uv run heuresis nanogpt curiosity --model "$MODEL" --n-seed 10
```

The implementation lives in `src/heuresis/loops/<strategy>.py` (the
`run_<strategy>(task)` functions) plus the per-task adapter in
`src/heuresis/tasks/<task>/adapter.py`. There are no per-pairing
`experiments/<task>_<strategy>/run.py` files for loop-backed tasks.

Loop-backed tasks (reachable via `heuresis <task> <strategy>`): `nanogpt`,
`bbob`, `discogen_onpolicyrl`, and `discogen_modelunlearning`.

Launch YAMLs under `configs/experiments/` provide the same arguments:

```bash
uv run heuresis --launch-config configs/experiments/nanogpt/curiosity.yaml --n-seed 12
uv run heuresis configs/experiments/nanogpt/curiosity.yaml --n-seed 12
uv run heuresis --launch-config configs/experiments/discogen_onpolicyrl/linear.yaml --config configs/discogen/onpolicy_rl_breakout_loss_only.yaml
```

Values in the YAML are validated by the task x strategy parser. CLI flags after
the YAML override matching config values.

Pass `--print-args` to print the resolved arguments instead of running (the mode
shell wrappers and tooling historically used); `--format {nul,lines}` selects
the encoding for that output:

```bash
uv run heuresis nanogpt curiosity --n-seed 10 --print-args --format lines
```

## Common Flags

Common `heuresis <task> <strategy>` flags include:

- `--experiment-name`
- `--resume-exp-id`
- `--agent`
- `--model`
- `--gpus 0,1`
- `--num-iterations`
- `--count-valid` / `--count-total`
- `--num-ideators`
- timeout flags such as `--executor-timeout`
- judge flags such as `--enable-judge`, `--disable-judge`, `--judge-model`
- `--memory` / `--no-memory`

Unsupported flags fail during argument parsing. This catches typoed options and
keeps task/strategy configuration explicit.

## Defaults and Logging

Defaults (GPUs, iterations, model, and the DiscoGen `--config` / `--domain`)
live in the registry (`src/heuresis/experiment_cli.py`) and the launch configs
under `configs/experiments/`; any flag passed to the CLI overrides them. There
is no per-pairing launch script — the CLI is the launcher. Model IDs are
agent/provider-specific; replace `your-model-id` with one of the IDs listed by
your agent CLI. For OpenCode, run `opencode models`.

```bash
uv run heuresis nanogpt linear --gpus 0 --num-iterations 20
MODEL=your-model-id
uv run heuresis nanogpt curiosity --model "$MODEL" --n-seed 12
```

DiscoGen tasks need the optional extra, e.g.
`uv run --extra discogen heuresis discogen_onpolicyrl linear` (or install it
once with `uv sync --extra discogen`). The legacy `discogen` task name remains
as an On-Policy RL compatibility alias. For long campaigns, capture the
orchestrator output with `… 2>&1 | tee logs/run.log` or run under tmux; the run
also writes full per-workspace logs under `runs/<task>/<experiment_id>/`.

## Launch Configs

`configs/experiments/<task>/<strategy>.yaml` contains one launch config for each
registered task x strategy pair. These are public examples and reproducible
defaults for the parser layer. They are separate from task-domain configs such
as `configs/discogen/*.yaml`, which are consumed by DiscoGen tasks via
`--config`.

## Canonical Public Launchers

Each of the three canonical tasks supports the full set of six strategies
(`linear`, `map_elites`, `go_explore`, `islands`, `omni_epic`, `curiosity`),
launched with the CLI:

| Task | Baseline | Other strategies |
| --- | --- | --- |
| NanoGPT | `uv run heuresis nanogpt linear` | `uv run heuresis nanogpt {map_elites,go_explore,islands,omni_epic,curiosity}` |
| On-Policy RL (`discogen_onpolicyrl`) | `uv run --extra discogen heuresis discogen_onpolicyrl linear` | `uv run --extra discogen heuresis discogen_onpolicyrl {map_elites,go_explore,islands,omni_epic,curiosity}` |
| Model Unlearning (`discogen_modelunlearning`) | `uv run --extra discogen heuresis discogen_modelunlearning linear` | `uv run --extra discogen heuresis discogen_modelunlearning {map_elites,go_explore,islands,omni_epic,curiosity}` |

The lightweight CPU benchmark `bbob` is loop-backed and runnable the same way
(`uv run heuresis bbob <strategy>`).

## Task-Specific Flags

DiscoGen entry points accept task flags such as:

```bash
--config configs/discogen/onpolicy_rl_breakout_loss_only.yaml --domain OnPolicyRL --mu-fast-eval
```

Other in-tree tasks currently have no extra task flags.

## Strategy-Specific Flags

Common strategy flags include:

- Curiosity: `--n-seed`, `--k-neighbors`, `--softmax-tau`,
  `--curiosity-novelty-threshold`, `--prediction-timeout`.
- Curiosity-plus: curiosity flags plus `--curiosity-score-weight`,
  `--curiosity-tag-novelty`, and memory weighting flags.
- OMNI-EPIC: `--min-archive-size`, `--seed-source`, `--seed-count`,
  `--skip-meta-test` / `--run-meta-test`.
- MAP-Elites and Go-Explore: `--cell-empty-weight`, `--cell-crossover-rate`,
  and `--go-explore-alpha` for Go-Explore.
- Islands: `--migration-interval`, `--crossover-rate`.

## Optional Live Smokes

`scripts/smoke/` contains opt-in live checks. They may require real API keys,
agent credentials, historical run data, or local stores. They are intentionally
outside the default unit-test path.

## Outputs

Run outputs are written under `runs/`. Wrapper logs are written under `logs/`.
Both are gitignored.
