# Heuresis

Heuresis runs coding agents in sandboxed workspaces and uses quality-diversity
search to decide what they try next. This file orients Codex (and other agents);
Claude Code reads `CLAUDE.md`, which mirrors it.

## Setup

```bash
uv sync --extra dev
cp .env.example .env
uv run python -m pytest tests/ -q
bash scripts/check.sh   # host tools for sandboxed experiments
```

## Running Experiments

`uv run heuresis <task> <strategy> [flags]` launches any registered pairing
(tasks: `nanogpt`, `discogen_onpolicyrl`, `discogen_modelunlearning`, `bbob`;
strategies: `linear`, `map_elites`, `go_explore`, `islands`, `omni_epic`,
`curiosity`). The DiscoGen tasks need the optional extra:
`uv run --extra discogen heuresis discogen_onpolicyrl linear`. The legacy
`discogen` task name remains as an On-Policy RL compatibility alias. See
`docs/experiments.md`.

## Skills

Repo skills live under `.agents/skills/`.

- **`heuresis-analyzing-search-runs`** — analyze/compare/evaluate search runs on
  quality, diversity, and novelty. The multi-phase pipeline (extract -> classify
  -> verify -> aggregate -> figures) behind the paper's Q/D/N results. Invoke it
  the same way in Codex or Claude Code — point at the skill and the run dir:

  ```
  Follow .agents/skills/heuresis-analyzing-search-runs/SKILL.md to analyze <run_dir>
  ```

  Classification/verification spawn parallel sub-agents; Codex defaults to
  `agents.max_threads = 6` (set to 10 to match the skill's cap, or expect 6).

## More Documentation

- `README.md`: public quickstart and overview.
- `docs/README.md`: public guides index.
- `maintainer/`: maintainer-only operating notes (not part of the public API).
