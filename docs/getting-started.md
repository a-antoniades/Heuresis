# Getting Started

This guide gets you from a fresh checkout to a small working experiment.

## Requirements

- Python 3.12 or 3.13.
- `uv` for dependency management.
- A supported coding-agent CLI for sandboxed experiments, such as OpenCode,
  Claude Code, or Cursor Agent.
- `bubblewrap` (`bwrap`) and `taskset` for the full Linux sandbox path.
- Optional NVIDIA GPUs for NanoGPT and other GPU-backed tasks.

## Install

```bash
uv sync --extra dev
```

This installs the development environment for tests, linting, and local command
execution. Experiment sandboxes use separate venvs; see `dependencies.md`.

## Configure Credentials

```bash
cp .env.example .env
${EDITOR:-vi} .env
```

Credentials belong in `.env` or your shell. Heuresis loads repo-local `.env`
files with `python-dotenv`. Experiment settings such as models, iteration
counts, memory, and strategy knobs are CLI flags.

## Run Your First Experiment

A run pairs one task with one strategy. The simplest entry point is NanoGPT with
the `linear` baseline:

```bash
opencode models   # for OpenCode; use your agent's model-list command otherwise
MODEL=your-model-id
uv run heuresis nanogpt linear --gpus 0 --num-iterations 10 --model "$MODEL" --no-memory --disable-judge
```

Real runs need a configured coding agent, the API keys it uses, and (for
NanoGPT) GPUs and the task data cache (see "Run NanoGPT" below). The run writes
local output under `runs/nanogpt/`, which is gitignored.

The first-run command disables memory and judge calls so the smoke only depends
on the selected executor model. Remove `--no-memory` and `--disable-judge` for
full campaigns after configuring Gemini and judge-agent credentials. Replace
`your-model-id` with one of the model IDs listed by your agent CLI.

## Check Full Sandbox Support

```bash
bash scripts/check.sh
```

This checks host tools, agent binaries, and filesystem assumptions used by the
sandboxed experiment path.

## Run NanoGPT

NanoGPT needs task data and usually a GPU:

```bash
bash scripts/setup.sh nanogpt
MODEL=your-model-id
uv run heuresis nanogpt linear --gpus 0 --model "$MODEL" --no-memory --disable-judge
```

Override the registry defaults by passing additional CLI flags after the wrapper
command. You can also launch directly with the `heuresis` CLI, passing flags
explicitly:

```bash
uv run heuresis nanogpt linear --gpus 0 --num-iterations 10 --model "$MODEL" --count-total
```

## Next Steps

- Read `concepts.md` for the architecture.
- Read `experiments.md` for the task x strategy matrix and CLI behavior.
- Read `add-task.md` or `add-strategy.md` when extending the system.
