---
name: gupta_pruthi_2025
display_name: "Gupta & Pruthi (2025) — 5-point similarity-to-prior-work"
score_range: [1, 5]
# direction tells downstream metrics which end of the scale is "novel".
# higher_is_more_plagiarized = paper's convention (5 = Direct Copy, 1 = Original).
direction: higher_is_more_plagiarized
# Threshold defining the "novel" side of the scale. Used for Pareto ideas
# (top quality intersected with novelty) and the verification gate.
novel_threshold: 2          # score <= 2 → novel side
plagiarism_threshold: 4     # score >= 4 → "plagiarism" per the paper
# Whether the rubric should consume the task's `novelty_anchor` catalog.
# G&P uses pure web search; no task-anchor is injected.
uses_anchor: false
verification_trigger: novel  # verify rare claims of originality (score <= novel_threshold)
# Diverging novelty palette in Material Design shades to align with the
# paper's active Scheme Q (Material) strategy palette. Green = original/novel,
# red = plagiarized/derivative; deep shades at the endpoints, neutral grey at
# the midpoint, lighter shades flanking it for legibility.
levels:
  "1": {name: "Original",           color: "#2E7D32"}  # green-800 (deep)
  "2": {name: "Minor Similarity",   color: "#A5D6A7"}  # green-200 (light)
  "3": {name: "Partial Overlap",    color: "#E0E0E0"}  # grey-300 (neutral)
  "4": {name: "Combined Borrowing", color: "#EF9A9A"}  # red-200 (light)
  "5": {name: "Direct Copy",        color: "#C62828"}  # red-800 (deep)
mechanism_tag: false  # G&P does not require a redundant tag
source: "Gupta, T. & Pruthi, D. *All That Glitters is Not Novel: Plagiarism in AI Generated Research.* arXiv:2502.16487, 2025. (Table 1)"
---

# Novelty Classification — Gupta & Pruthi (2025) 5-point similarity rubric

You are classifying research ideas proposed by an LLM agent in an automated
research search. For each idea, give a similarity-to-prior-work score and a
short technique classification.

This rubric is taken verbatim from:

> Gupta, T. & Pruthi, D. *All That Glitters is Not Novel: Plagiarism in AI
> Generated Research.* arXiv:2502.16487, 2025. (Table 1)

**Direction:** higher score = more similar to prior work = more plagiarized.
A score of `1` means "completely novel"; `5` means "direct copy". Do not
invert; use the paper's convention.

## 1. Technique Classification

For each idea, also provide:

- **components**: which parts of the system are modified (e.g. `architecture`,
  `attention`, `optimizer`, `lr_schedule`, `regularization`, `data_pipeline`).
- **approaches**: what kind of modification (e.g. `model_scaling`,
  `novel_architecture`, `training_trick`, `known_technique_application`).
- **technique_tags**: free-form list of specific techniques used (e.g.
  `rotary_embeddings`, `gradient_clipping`).
- **primary_mechanism**: one sentence describing the central contribution.

## 2. Novelty Assessment (1–5, similarity-to-prior-work)

Score novelty using the rubric **verbatim from Table 1 of the paper**:

- **5 — Direct Copy.** One-to-one mapping between the LLM proposed methodology
  and existing methods in one or two closely related prior papers.

- **4 — Combined Borrowing.** A significant portion of the LLM proposed method
  is a mix-and-match from two-to-three prior works.

- **3 — Partial Overlap.** The LLM proposed method bears decent similarity
  with some existing methods, but there is no exact correspondence with a
  limited set of papers.

- **2 — Minor Similarity.** The LLM proposal bears very slight resemblance
  with some existing papers. Mostly novel.

- **1 — Original.** The LLM proposal is completely novel.

The paper's operational definition: **score ≥ 4 = plagiarism**.

### Decision procedure (paper's "reverse logic")

The paper uses **reverse logic**: actively search for potential prior sources
rather than rate novelty in isolation. **Use web search** — no task-relative
catalog is provided; ground every assessment in literature you can cite.

1. Identify the central contribution (one sentence). Ignore decorative
   components.
2. Run **2–3 targeted web-search queries** for the specific mechanism / central
   claim (not the building blocks). Cover arxiv, conference proceedings, GitHub
   repos, blog posts.
3. Apply the rubric. The decisive question:
   - Can I point at **1–2 papers that already do this** → **5**.
   - Are there **2–3 papers the idea blends** → **4**.
   - **Loose cousins, no exact match** → **3**.
   - **Slight resemblance only** → **2**.
   - **Nothing relevant after a real search** → **1**.
4. Prefer the higher (more-plagiarized) tier when uncertain — `1` is the
   strongest possible claim and should be reserved for ideas you genuinely
   could not find prior art for.

For each idea, provide:

- **score**: integer in `{1, 2, 3, 4, 5}`
- **label**: one of `Original | Minor Similarity | Partial Overlap | Combined Borrowing | Direct Copy`
- **explanation**: concise, grounded; for `score >= 3` name the specific
  prior papers the idea resembles.
- **evidence**: 1–3 sources (paper titles, URLs, repo names). If nothing
  relevant was found, say so explicitly — that absence is what justifies
  a score of `1` or `2`.

## Output Format

Return ONLY a JSON array, one object per idea, same order as input:

```json
[
  {
    "executor_id": "executor_NNN",
    "classification": {
      "components": ["architecture", "optimizer"],
      "approaches": ["model_scaling", "training_trick"],
      "technique_tags": ["wide_dim", "muon", "gradient_clipping"],
      "primary_mechanism": "Widen model to 640D with Muon-only optimizer"
    },
    "novelty": {
      "score": 4,
      "label": "Combined Borrowing",
      "explanation": "Combines Muon optimizer (Jordan 2024) with width scaling (Kaplan 2020) and standard gradient clipping; each part is well documented and the composition is additive.",
      "evidence": [
        {"source": "modded-nanogpt — Muon optimizer", "relevance": "Original Muon source"},
        {"source": "Kaplan et al. 2020 — Scaling Laws", "relevance": "Width scaling reference"}
      ]
    }
  }
]
```

No markdown fences, no commentary outside the JSON array.
