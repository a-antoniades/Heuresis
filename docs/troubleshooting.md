# Troubleshooting

## `uv` Is Missing

Install `uv` and rerun:

```bash
uv --version
uv sync --extra dev
```

Sandbox venv creation also expects `uv` on `PATH`.

## Preflight Fails

Run:

```bash
bash scripts/check.sh
```

Common failures are missing `bwrap`, missing `taskset`, missing agent binaries,
or GPU visibility problems.

## Credentials Are Not Found

Confirm your repo-local `.env` exists and contains the expected provider keys:

```bash
cp .env.example .env
${EDITOR:-vi} .env
```

Heuresis loads `.env` with `python-dotenv`; do not source it from Bash.
For Gemini rotation, prefer `GEMINI_API_KEYS`. Do not use key files.

## Agent Authentication Fails

Agent CLIs have their own authentication flows. Confirm the agent works outside
Heuresis first, then rerun the experiment. Claude Code OAuth credentials,
OpenCode auth, Cursor auth, and provider API keys are managed by their respective
CLIs or environment variables.

## Agent Model Is Not Found

Model IDs are resolved by the selected agent CLI and provider account. If a run
fails with a provider/model-not-found error, list models in that agent and pass a
known-good model explicitly:

```bash
opencode models
MODEL=your-model-id
uv run heuresis nanogpt linear --gpus 0 --model "$MODEL"
```

For a minimal executor smoke that does not require Gemini memory embeddings or a
judge agent, add `--no-memory --disable-judge`. Replace `your-model-id` with
one of the model IDs listed by your agent CLI.

## `uv` Cache Filesystem Is Full

If `uv sync` or `uv run` fails while initializing its cache, move the uv cache to
a larger scratch filesystem:

```bash
export UV_CACHE_DIR=/tmp/uv-cache-$USER
uv sync --extra dev
```

## Sandbox Cannot See a Dependency

Agents cannot install packages inside a sandbox. Add the dependency to the right
place:

- Core package or default workspace dependency: `pyproject.toml`.
- Novelty reviewer dependency: `pyproject.toml` `novelty` extra.
- Task executor dependency: task-local `requirements.txt`.
- DiscoGen domain dependency: domain requirements from the DiscoGen package.

Then delete the affected venv under `venvs/` and rerun so it is rebuilt.

If `venvs/nanogpt` was created with the wrong Python version, remove it and run
`bash scripts/setup.sh nanogpt` again. The setup script expects Python 3.12 or
3.13 for task venvs.

## GPU Selection Is Wrong

Pass GPUs explicitly:

```bash
uv run heuresis nanogpt linear --gpus 0
```

The `GPUS=0` environment variable also works (the CLI reads it), but the `--gpus`
flag is the preferred form. For low-level host isolation, `CUDA_VISIBLE_DEVICES`
may still be useful, but the experiment should receive `--gpus`.

## Unknown CLI Flag

Experiment parsers are strict. Check whether the flag belongs to the selected
shared settings, task, or strategy. Strategy-specific flags are not accepted by
unrelated strategies.

## Live Smoke Scripts Fail

`scripts/smoke/` scripts are optional and may require local run data, real API
keys, and historical stores. Unit tests do not depend on them.

## Old Results Are Missing

Run outputs live under `runs/` and are gitignored. Public clones do not include
private run data unless it is distributed separately as a research artifact.
