# Maintainer Notes

This directory contains maintainer-facing guidance that is useful for people
developing or operating this repository, but is not required for public users.

## Common Commands

```bash
uv sync --extra dev
cp .env.example .env
uv run python -m pytest tests/ -q
uv run ruff check src/heuresis tests
```

Full sandboxed experiments also need host tools checked by `scripts/check.sh`
(`bwrap`, `taskset`, at least one supported coding agent, and optional GPUs).

## API Key Conventions

Use `heuresis.api_keys` instead of hard-coded key files.

Gemini precedence:

1. `GEMINI_API_KEYS`
2. `GEMINI_API_KEY`
3. `GOOGLE_GENERATIVE_AI_API_KEY`

Do not log or print key values. Only report whether keys were found, or how
many were loaded.

## Development Guidance

- Prefer `uv run ...` for Python commands.
- Keep default tests free of live API calls, agents, GPUs, and network access.
- Put real-key or real-store checks under `scripts/smoke/` and document the
  required environment variables.
- Avoid adding absolute machine paths. Use repo-relative paths, CLI arguments,
  or environment variables such as `QD_STORE_PATH`.
- Keep generated artifacts out of git unless they are intentionally curated
  public outputs.

## Claude Skills

Maintainer-only Claude skill material lives under `maintainer/claude/skills/`.
These skills and helper scripts capture local operating workflows used while
running the research project; they are not part of the public package surface.

## Archive

Historical plans, reports, bug logs, and private runbooks that used to live
under `docs/` are preserved under `maintainer/archive/docs/`. They may mention
old branches, private run data, or local infrastructure and should not be linked
from the public docs.

## Canonical Local Entry Points

- `uv run heuresis <task> <strategy> [flags]` launches any registered pairing.
  Canonical baselines: `uv run heuresis nanogpt linear` (NanoGPT),
  `uv run --extra discogen heuresis discogen linear` (On-Policy RL), and
  `uv run --extra discogen heuresis discogen_modelunlearning linear` (Model
  Unlearning). `uv run heuresis bbob linear` runs the lightweight BBOB benchmark.
- `scripts/smoke/integration.py` and `scripts/smoke/strategy.py` are optional
  live checks that require local store data and Gemini keys.
