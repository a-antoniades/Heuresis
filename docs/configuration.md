# Configuration

Configuration is split into two categories: credentials/runtime environment and
experiment parameters.

## Credentials and Runtime Environment

Credentials are read from environment variables so they never need to be passed
as command-line arguments. Supported variables include:

- `GEMINI_API_KEYS` for comma- or newline-separated Gemini key rotation.
- `GEMINI_API_KEY` and `GOOGLE_GENERATIVE_AI_API_KEY` as single-key Gemini
  fallbacks.
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `CURSOR_API_KEY`.
- `HF_TOKEN` for Hugging Face access.

Use `.env.example` as the template:

```bash
cp .env.example .env
${EDITOR:-vi} .env
```

Heuresis loads repo-local `.env` files with `python-dotenv` and does not require
shell-sourcing credential files.

Do not commit `.env`, credential JSON files, or key files.

Low-level runtime variables such as `CUDA_VISIBLE_DEVICES`, `QD_STORE_PATH`, or
cache path overrides may still be useful for host integration. They are not the
primary experiment configuration mechanism.

If the default uv cache filesystem is small or full, point uv at a larger local
scratch path before installing or running:

```bash
export UV_CACHE_DIR=/tmp/uv-cache-$USER
```

## Experiment Parameters

Experiment parameters are CLI flags passed to the `heuresis` launcher and parsed
by `parse_experiment(...)`. This includes models, agents, iteration counts,
memory, timeouts, task flags, and strategy knobs.

```bash
MODEL=your-model-id
uv run heuresis nanogpt curiosity   --gpus 0   --num-iterations 20   --model "$MODEL"   --n-seed 10   --k-neighbors 10
```

Model names are interpreted by the selected agent CLI. Replace `your-model-id`
with one of the model IDs listed by that CLI. For OpenCode, run
`opencode models`.

The launcher resolves arguments in order: environment-variable defaults (from
wrappers) first, then an optional launch-config YAML, then explicit CLI flags,
which take precedence. The parser is strict: a flag is accepted only if it
belongs to the shared settings, the selected task, or the selected strategy.

## API Key Precedence

Gemini key loading uses this precedence:

1. `GEMINI_API_KEYS`
2. `GEMINI_API_KEY`
3. `GOOGLE_GENERATIVE_AI_API_KEY`

`GEMINI_KEYS_FILE` is intentionally not supported.

## Wrappers

Shell wrappers may expose environment variables such as `GPUS` or
`NUM_ITERATIONS` for convenience. They set those as defaults and call
`uv run heuresis <task> <strategy> "$@"`, where the launcher folds them into the
resolved CLI flags. Explicit flags passed to the wrapper override the
environment defaults.

## Generated Outputs

Generated outputs belong under `runs/`, `logs/`, `venvs/`, `.venv/`, `data/`, or
analysis output directories. These paths are gitignored.
