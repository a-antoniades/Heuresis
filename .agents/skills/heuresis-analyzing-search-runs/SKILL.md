---
name: heuresis-analyzing-search-runs
description: Use when analyzing, comparing, or evaluating research-agent search runs for quality, diversity, and novelty. Triggers on "analyze run", "compare runs", "run analysis", or any request to evaluate search strategy results.
---

# Analyzing Search Runs

Orchestrate a multi-phase analysis of research-agent search runs, measuring quality, diversity, and novelty. Produces a reusable `analysis_cache.json` per run so future comparisons skip redundant work.

## Rubrics

Novelty scoring is rubric-driven. Each rubric is a markdown file under `rubrics/` with YAML frontmatter declaring its score range, direction, level names, and whether it consumes the task's `novelty_anchor`. The classifier sub-agent loads the rubric body verbatim as its scoring prompt; aggregation and figures read the frontmatter to interpret scores.

| Rubric | Range | Direction | Anchor | Notes |
|---|---|---|---|---|
| **`gupta_pruthi_2025`** *(default)* | 1–5 | higher = more plagiarized | none — web-search only | [Gupta & Pruthi 2025](https://arxiv.org/abs/2502.16487) Table 1. `1=Original`, `5=Direct Copy`. Paper threshold: `score ≥ 4` is plagiarism. |
| `nanogpt_1to4` | 1–4 | higher = more novel | `task_config.yaml:novelty_anchor` | Internal task-anchored rubric. `1=Catalog`, `4=New core`. |

**Direction matters.** The two rubrics score in opposite directions, so downstream metrics derive "novel side" from the rubric's `direction` + `novel_threshold` rather than hardcoding `>= N`. Do not invert scores between rubrics — a score of `1` under G&P (Original) and a score of `1` under nanogpt_1to4 (Catalog) mean opposite things.

**Verification gate.** Both rubrics fire a verification pass on the *rare claim*:
- G&P: rare-novel claims (`score ≤ novel_threshold`, i.e. `≤ 2`) — verify the claim of originality.
- nanogpt_1to4: rare-novel claims (`score ≥ novel_threshold`, i.e. `≥ 3`) — verify the claim of novelty.

**Pareto front.** "Top-10 quality ∩ novel side" — `score ≤ 2` under G&P, `score ≥ 3` under nanogpt_1to4.

To add a new rubric, drop a file into `rubrics/<name>.md` with the frontmatter contract documented in `scripts/_rubric.py`.

## Data Flow

**Store integration:** The extraction script reads from `store.db` when available, falling back to filesystem parsing for older runs. `DB_PATH` is `research-agent/runs/nanogpt/store.db` (not `research-agent/store.db` — moved in the foundation refactor). For pre-refactor runs in `runs/_legacy/`, use `analysis/libs/legacy_store.py` which normalizes the old schema. The aggregation script writes the keys below into the store's `metadata` column, queryable via `Experiment.runs()`.

**Metadata keys written per executor (in `runs.metadata` JSON):**
- `novelty_score`: int — interpret using `rubric` (range + direction depend on it)
- `novelty_label`: str — human-readable level name (e.g. "Combined Borrowing")
- `novelty_mechanism_tag`: str — only set when the active rubric requires it (nanogpt_1to4 only)
- `novelty_explanation`: str — concise grounded explanation
- `novelty_evidence`: list of `{source, relevance}` dicts
- `analysis_classification`: `{components, approaches, technique_tags, primary_mechanism}`
- `rubric`: rubric name (`gupta_pruthi_2025` / `nanogpt_1to4`) — replaces the legacy `rubric_version`

Legacy caches with `rubric_version: "1to4"` and no `rubric` field are still readable; the figures script falls back to `nanogpt_1to4` when it sees the old key.

**`novelty_reviews` table:** Separate from `runs.metadata`. One row per reviewer attempt (including rejected rounds), with columns `attempt`, `accepted`, `novelty_score`, `explanation`, `duration_s`, `input_tokens`, `output_tokens`, `total_cost`.

## Pipeline Overview

```
Phase 1: EXTRACT (deterministic script, takes --rubric)
  store.db (preferred) or filesystem → extracted_ideas.json (stamped with rubric)

Phase 2: CLASSIFY (parallel sub-agents)
  Load rubric prompt body from rubrics/<name>.md → spawn agents → classifications.json

Phase 2.5: VERIFY (dedicated agents for the rubric's "novel-side" claims)
  Rare-novel ideas → thorough web search → confirm or downgrade

Phase 3: AGGREGATE (deterministic script, takes --rubric)
  classifications.json → metrics + analysis_cache.json (stamped with rubric) + store write-back

Phase 4: COMPARE (optional, if multiple runs)
  Load caches → side-by-side summary

Phase 5: FIGURES + REPORT (deterministic script + orchestrator synthesis)
  analysis_cache.json files → figures/ + report.md
```

## Phase 1: Extract

```bash
uv run python .agents/skills/heuresis-analyzing-search-runs/scripts/extract_run.py \
  <run_dir> [--rubric gupta_pruthi_2025]
```

The `--rubric` flag controls one thing in this phase: whether to inject the task's `novelty_anchor` markdown into `task_preamble`. Anchor-using rubrics (`uses_anchor: true`) get the catalog appended; web-search-only rubrics (G&P) skip it. The chosen rubric name is stamped into `extracted_ideas.json` so downstream stages don't have to re-derive it.

**Staleness check:** if an existing `extracted_ideas.json` is found, the script counts currently-valid scored executors via the store (or by grepping `val_bpb` from logs) and compares to the cache's `valid_count`. If the run has grown, it re-extracts, merges previously-classified entries, and deletes `analysis_cache.json` so it is rebuilt. Use `--force` only to re-classify under a new rubric.

`extracted_ideas.json` shape:
```json
{
  "run_id": "2026-04-06_...",
  "run_dir": "/absolute/path",
  "task_preamble": "You are an expert ML engineer... baseline achieves...",
  "metric_name": "val_bpb",
  "metric_direction": "lower_is_better",
  "rubric": "gupta_pruthi_2025",
  "total_executors": 225,
  "ideas": [
    {"executor_id": "executor_092", "score": 0.969891, "idea_text": "## Strategy\n..."}
  ]
}
```

Only ideas with valid scores are included.

## Phase 2: Classify (parallel sub-agents)

### Load the rubric prompt

Read the active rubric from `extracted_ideas.json` (the `rubric` field). Resolve the rubric path:

```
.agents/skills/heuresis-analyzing-search-runs/rubrics/<name>.md
```

The body of that file (everything after the closing `---` of the frontmatter) is the **classification prompt**. Pass it verbatim as the system prompt for each sub-agent — the prompt already covers technique classification, the scoring rubric, the decision procedure, and the output JSON shape. Do **not** edit or paraphrase it.

Augment the prompt at runtime with two things:
1. **Task context block** — `extracted_ideas.json:task_preamble` (which already has the anchor injected if the rubric uses it).
2. **The batch of ideas** to classify, formatted as a JSON list of `{executor_id, score, idea_text}` (truncate `idea_text` to ~3000 chars).

### Incremental classification

An idea is **already classified** if it has both `"classification"` and `"novelty_score"` keys (pulled from store metadata by the extraction script). Only send unclassified ideas to agents. If all ideas are already classified, skip Phase 2 entirely.

### Batching

Filter to unclassified ideas. Batch into groups of **8**. For each batch, spawn a sub-agent. Cap at **10 concurrent agents** (Codex defaults to `agents.max_threads = 6`; set it to 10 to match, or expect 6).

### Collecting results

Each sub-agent returns a JSON array (shape defined in the rubric file). Parse it; merge across batches. Save to `<run_dir>/classifications.json`. The store metadata is the authoritative copy — `classifications.json` is the audit trail.

### Phase 2.5: Verification pass on rare-novel claims

After classification, collect ideas on the rubric's "novel side":
- G&P (`gupta_pruthi_2025`): `score <= novel_threshold` (i.e. `≤ 2` — Original or Minor Similarity).
- nanogpt_1to4: `score >= novel_threshold` (i.e. `≥ 3` — Extension or New core).

These are rare, high-value claims. Spawn a dedicated verification agent per candidate (parallel) with this prompt skeleton:

```
A previous classifier rated this idea on the {rubric_name} rubric as score={original_score}
({label}). The "novel side" of this rubric is the rare claim — your job is to verify or
downgrade it via thorough literature search.

## The idea
{idea_text}

## Previous assessment
Score: {original_score} ({label}; {rubric_direction})
mechanism_tag: {original_tag}     # only present for nanogpt_1to4
Explanation: {previous_explanation}
Evidence: {previous_evidence}

## Your task
1. Run 3-5 targeted web-search queries for the central mechanism (not building blocks).
2. Identify the closest prior art. Quote titles + URLs.
3. Apply the rubric's scoring criteria. Prefer the "less novel" tier when uncertain.
   - Under G&P, that means HIGHER score.
   - Under nanogpt_1to4, that means LOWER score.

## Output
Write a JSON object to {output_path}:
{"executor_id": "...", "verified_score": N, "verified_label": "...",
 "explanation": "...", "evidence": [{"source": "...", "relevance": "..."}]}
```

Update the idea's score in `classifications.json` with the verified value. Log which ideas were verified and whether they were confirmed or downgraded.

## Phase 3: Aggregate

```bash
uv run python .agents/skills/heuresis-analyzing-search-runs/scripts/aggregate.py \
  <run_dir> [--rubric <name>]
```

Reads `extracted_ideas.json` + `classifications.json` and produces `analysis_cache.json` with rubric-aware metrics. The rubric is taken from the explicit flag if provided, else the extraction stamp, else the default.

`analysis_cache.json` shape (G&P example):
```json
{
  "run_id": "...",
  "strategy_type": "island|mapelites|greedy",
  "rubric": "gupta_pruthi_2025",
  "rubric_score_range": [1, 5],
  "rubric_direction": "higher_is_more_plagiarized",
  "analyzed_at": "ISO timestamp",
  "metric_name": "val_bpb",
  "metric_direction": "lower_is_better",
  "ideas": [{"executor_id": "...", "score": 0.969,
             "classification": {...}, "novelty": {...}}],
  "metrics": {
    "quality": {"best": 0.965, "top5_mean": 0.968, "median": 0.980,
                "success_rate": 0.71, "valid_count": 175, "scores": [...]},
    "diversity": {"unique_technique_count": 45, "technique_entropy": 3.21,
                  "component_distribution": {...}, "approach_distribution": {...},
                  "mean_pairwise_jaccard_distance": 0.72},
    "novelty": {
      "mean_score": 3.4,
      "distribution": {"1": 5, "2": 18, "3": 60, "4": 70, "5": 22},
      "novel_count": 23,
      "novel_fraction": 0.131,
      "top5_quality_mean_novelty": 2.4,
      "bottom5_quality_mean_novelty": 4.0
    }
  }
}
```

`distribution` always covers the rubric's full range (zero-fill missing buckets). `novel_count` / `novel_fraction` count ideas on the rubric's novel side, so the field is comparable across rubrics even though the underlying scores aren't.

## Phase 4: Compare (optional)

When multiple runs are passed, load each `analysis_cache.json`. **All caches must use the same rubric** for the comparison to be meaningful — the figures script warns if rubrics differ.

### Balanced Comparison (default)

Truncate each cache to the first N valid ideas (chronological by executor number), where N is the minimum valid count across caches. Quality, diversity, and novelty are recomputed on the truncated set. Disable with `--no-balance`; pin N with `--max-ideas <int>`. `balanced_metrics.json` is written to the figures dir for downstream synthesis.

### Comparison Table

```
| Metric                        | Run A (strategy) | Run B (strategy) |
|-------------------------------|-------------------|-------------------|
| Best score                    | 0.965             | 0.969             |
| Top-5 mean                    | 0.968             | 0.975             |
| Median score                  | 0.980             | 0.992             |
| Success rate                  | 71%               | 36%               |
| Unique techniques             | 45                | 32                |
| Technique entropy             | 3.21              | 2.87              |
| Mean pairwise distance        | 0.72              | 0.65              |
| Mean novelty                  | 3.4               | 3.8               |
| Novelty dist (1..5 G&P)       | 5/18/60/70/22     | 2/8/40/85/40      |
| Novel fraction (≤2)           | 13.1%             | 5.7%              |
| Top-5 ideas mean novelty      | 2.4               | 3.0               |
```

(Distribution/novel-fraction columns adjust to the rubric's range and direction.)

### Cross-Metric Analysis

1. **Quality vs Novelty** — Spearman ρ between score and novelty. Sign interpretation depends on rubric direction.
2. **Diversity leader** — which strategy explored more technique space?
3. **Novelty leader** — which strategy produced more ideas on the novel side of the rubric?
4. **Efficiency** — quality per attempt.
5. **Pareto ideas** — top-10 quality ∩ novel side.

## Phase 5: Figures, Highlights & Synthesis

```bash
uv run python .agents/skills/heuresis-analyzing-search-runs/scripts/figures.py \
  <run_dir1>/analysis_cache.json <run_dir2>/analysis_cache.json \
  --output-dir <output_dir>/figures/
# Add --no-balance to keep the full per-run data (default is balanced)
# Add --max-ideas <int> (alias: --balance-n) to pin the target size
```

The figure legend / y-axis tick labels / Pareto threshold all derive from the cache's `rubric` field, so G&P and nanogpt_1to4 caches render automatically with the right level names.

Six figures:

1. **`score_distribution.png`** — Violin/box plot of score distributions, one per strategy. Baseline reference line.
2. **`novelty_distribution.png`** — Grouped bar chart: x-axis = strategy, bars within = rubric levels (legend names from the rubric's `levels` map). Title shows the rubric name + direction.
3. **`quality_vs_novelty.png`** — Scatter of quality vs novelty (jittered), Pareto rings on `top-10 ∩ novel-side`, verified-idea diamonds with `original→verified` annotations. Spearman ρ per strategy.
4. **`technique_coverage.png`** — Side-by-side component × approach heatmaps.
5. **`diversity_map.png`** — UMAP of idea embeddings + within-strategy pairwise cosine distance violin. Stars mark per-strategy + global best.
6. **`fitness_curve.png`** — Running best vs. valid-solution index, IQR-capped y-axis.

**Dependencies for figures.** `figures.py` requires `scipy`, `umap-learn`, and `google-genai` for the full set. Add missing deps with `uv add <pkg>` at the repo root.

### Highlight ideas (extract from cache)

1. **Top-5 by quality** per strategy — executor_id, score, primary_mechanism, novelty score
2. **Top-5 by novelty** per strategy — defined by rubric direction (lowest under G&P, highest under nanogpt_1to4)
3. **Pareto front** (top-10 quality ∩ novel side) — interesting discoveries

### Synthesis Report

Write `<output_dir>/report.md`. Read numbers from `<output_dir>/balanced_metrics.json`. Call out the rubric explicitly so the reader can interpret directionality.

```markdown
# Search Strategy Comparison: [Strategy A] vs [Strategy B]

## Summary
2-3 sentence overview of the key finding.

## Methodology
- Rubric: <rubric name> (<range>, <direction>)
- Balanced to N=<min> per strategy (originals: <A>, <B>)
- Classification: parallel Claude agents loading `rubrics/<name>.md` verbatim
- Verification pass on rare-novel claims (rubric-defined "novel side")
- Diversity: Shannon entropy + Jaccard distance over technique tags

## Quality
<table>

## Diversity
<table + technique_coverage commentary + diversity_map commentary>

## Novelty
- Distribution + novel-fraction + Spearman ρ
- Best ideas — novel or conventional?

## Highlight Ideas
### Best Performing
### Most Novel (rubric's novel side)
### Pareto Front (Quality + Novelty)
```

Every claim must reference a metric or an idea — no generic observations.

## Cache Reuse

- `extracted_ideas.json` exists → Phase 1 staleness check.
- `classifications.json` exists → Phase 2 runs incrementally.
- `analysis_cache.json` exists → Phase 3 skipped.
- `<output_dir>/report.md` exists → Phase 5 skipped.

When swapping rubrics, delete `classifications.json` and `analysis_cache.json` (the old novelty scores are not portable across rubrics with different directions).

## Invocation

Same for Claude Code and Codex — point the agent at this skill and the run
directory:

```
Follow .agents/skills/heuresis-analyzing-search-runs/SKILL.md to analyze <run_dir>
```

Compare two runs by passing both. The argument shape is:

```
<run_dir1> [run_dir2] [--rubric <name>] [--force] [--no-balance] [--max-ideas N]
```

- `--rubric <name>`: select the rubric (default `gupta_pruthi_2025`). Available: `gupta_pruthi_2025`, `nanogpt_1to4`.
- Single run: Phases 1–3, print summary.
- Multiple runs: Phases 1–5 (full pipeline). Caches balanced to min ideas by default.
- `--force`: delete existing cache files and re-run.
- `--no-balance`: skip balancing.
- `--max-ideas N`: balance to an explicit N.
- Output dir defaults to `analysis/comparisons/<strategy_a>_vs_<strategy_b>/`.

### Pre-flight clarification (multiple runs)

If the user did not specify `--no-balance` or `--max-ideas`, ask one short question:

> "I'll compare these runs. How should I cap idea count?
>   (a) balance to min valid-idea count across runs (default)
>   (b) use a specific max — tell me N
>   (c) no balance — use every valid idea per run"

Skip the question if the user already specified intent.
