# Run Artifacts

Active runs live under `runs/<task>/<experiment_id>/`. SQLite store at `runs/<task>/store.db`.

## Legacy data (pre-foundation-refactor)

Runs from before the foundation refactor (April 2026) live under `runs/_legacy/`:
- `runs/_legacy/store.db` — old schema (has `run_type`, `iteration`, etc. as top-level columns; no `novelty_reviews` table)
- `runs/_legacy/nanogpt_pre_refactor/` — run directory tree in pre-refactor layout
- `runs/_legacy/test*_pre_refactor/` — pre-refactor smoke-test artifacts

Read legacy data via `analysis/libs/legacy_store.py` (returns `RunRecord`-compatible rows).
