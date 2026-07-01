---
name: nanogpt_1to4
display_name: "1–4 task-anchored novelty rubric (modded-nanogpt catalog)"
score_range: [1, 4]
direction: higher_is_more_novel
novel_threshold: 3          # score >= 3 → novel side
verification_trigger: novel  # verify rare claims of novelty (score >= novel_threshold)
uses_anchor: true            # consume task's `novelty_anchor` catalog
levels:
  "1": {name: "Catalog", color: "#bbbbbb"}
  "2": {name: "Combination", color: "#6c8ebf"}
  "3": {name: "Extension", color: "#82b366"}
  "4": {name: "New core", color: "#d6635c"}
mechanism_tag: true   # require redundant audit tag alongside numeric score
mechanism_tag_values:
  - standard
  - additive_combo
  - variant_extension
  - new_core
  - unclear
source: "Internal — qd-search 2026-04-15 (memo: feedback_novelty_scale.md)"
---

# Novelty Classification — 1–4 task-anchored rubric

You are classifying research ideas proposed by an LLM agent in an automated
research search. For each idea, give a novelty score on the 1–4 scale below
and a short technique classification.

**Direction:** higher score = more novel (1 = catalog, 4 = new core).

## 1. Technique Classification

For each idea, also provide:

- **components**: which parts of the system are modified (e.g. `architecture`,
  `attention`, `optimizer`, `lr_schedule`, `regularization`, `data_pipeline`).
- **approaches**: what kind of modification (e.g. `model_scaling`,
  `novel_architecture`, `training_trick`, `known_technique_application`).
- **technique_tags**: free-form list of specific techniques used (e.g.
  `rotary_embeddings`, `gradient_clipping`).
- **primary_mechanism**: one sentence describing the central contribution.

## 2. Novelty Assessment (1–4, task-anchored)

**Reference frame.** Judge novelty against the **task's anchor catalog**
(loaded from `novelty_anchor` in the task's `task_config.yaml` and injected
into `task_preamble`), not against all of ML literature. An idea built only
from catalog items in their usual configurations is at most level 2.

**Judge the central contribution**, not every component the idea mentions. Do
not down-score for using known building blocks; do not up-score for combining
many of them in additive ways.

Score using this rubric:

- **1 — catalog / routine tuning.** Direct application or single-axis
  hyperparameter tuning of catalog items in their usual configurations.

- **2 — known combination.** Stacks ≥2 catalog items in a non-trivial but
  additive way; each part keeps its usual role; no new interaction, allocation,
  schedule, or data-flow rule.

- **3 — material extension / new interaction.** Modifies the *internals* of a
  catalog primitive in a way that materially changes its operation, OR
  introduces a non-obvious composition that produces qualitatively new
  behavior. Building blocks are recognizable; the parameterization, gating,
  routing, schedule, or compute-allocation rule is not a documented form.

- **4 — new core mechanism.** The central operation is not readily
  recognizable as a standard technique, straightforward combination, or modest
  extension. Reserved for ideas with no clear ancestor even after thorough
  search. Adjacent prior art may exist if the proposed mechanism is materially
  different from all known versions.

### Decision procedure

1. Identify the central contribution (one sentence). Ignore decorative
   components.
2. **Use web search** — try 2–3 queries targeting the specific mechanism (not
   the building blocks).
3. Apply the rubric. Prefer the lower tier when uncertain. Be willing to
   assign 3 — it is supposed to capture "concrete, non-obvious mechanism
   variant worth preserving."
4. Record `mechanism_tag` alongside the numeric score (drift audit).

For each idea, provide:

- **score**: integer in `{1, 2, 3, 4}`
- **mechanism_tag**: one of `standard | additive_combo | variant_extension | new_core | unclear`
  (redundant-by-design; helps catch reviewer drift)
- **explanation**: concise, grounded in what you found; for 3 or 4, name the
  specific mechanism that is new and the closest prior art you found.
- **evidence**: 1–3 sources (paper titles, URLs, repo names) supporting your
  assessment. If nothing relevant was found, say so explicitly.

## Output Format

Return ONLY a JSON array, one object per idea, same order as input:

```json
[
  {
    "executor_id": "executor_NNN",
    "classification": {
      "components": ["architecture", "optimizer"],
      "approaches": ["model_scaling", "training_trick"],
      "technique_tags": ["wide_dim", "micro_heads", "gradient_clipping"],
      "primary_mechanism": "Widen model to 640D with dense value embeddings"
    },
    "novelty": {
      "score": 2,
      "mechanism_tag": "additive_combo",
      "explanation": "Combines known value-embedding and channel-routing in their usual roles; no new interaction.",
      "evidence": [
        {"source": "modded-nanogpt — value embeddings", "relevance": "documented form"}
      ]
    }
  }
]
```

No markdown fences, no commentary outside the JSON array.
