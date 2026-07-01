# Heuresis

This repository can be used with Claude Code as one of the supported coding
agents for sandboxed Quality-Diversity experiments.

## Public Setup

```bash
uv sync --extra dev
cp .env.example .env
uv run python -m pytest tests/ -q
```

For full sandboxed experiments, also run:

```bash
bash scripts/check.sh
```

That checks for required host tools such as `bwrap`, `taskset`, configured
coding-agent CLIs, and optional GPU devices.

## Running With Claude Code

Claude Code is selected with CLI flags. For example:

```bash
uv run heuresis nanogpt linear \
  --agent claude \
  --model claude-sonnet-4-6 \
  --num-iterations 3
```

Some workflows use Claude Code via OAuth credentials and intentionally strip
`ANTHROPIC_API_KEY` inside the sandbox. Other provider keys are documented in
`.env.example`.

## Analyzing Runs (Quality / Diversity / Novelty)

The `.agents/skills/heuresis-analyzing-search-runs/` skill is the multi-phase
pipeline (extract -> classify -> verify -> aggregate -> figures) behind the
paper's Q/D/N results. Invoke it the same way in Claude Code or Codex — point at
the skill and the run directory:

```
Follow .agents/skills/heuresis-analyzing-search-runs/SKILL.md to analyze <run_dir>
```

## More Documentation

- `README.md` contains the public quickstart and repository overview.
- `docs/README.md` links public guides.
- `docs/experiments.md` lists canonical experiment entry points.
- `maintainer/` contains maintainer-only operating notes.
