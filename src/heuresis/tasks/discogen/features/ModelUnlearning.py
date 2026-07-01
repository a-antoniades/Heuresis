"""Feature schema + keyword dicts for DiscoGen ModelUnlearning MAP-Elites.

6 mechanism families x 4 approaches = 24 cells. Feeds KeywordClassifier or
LLMClassifier -- strategy doesn't care which.

Narrow-surface task: every idea edits the same file (``discovered/loss.py``).
The "component" axis therefore captures the *mathematical family* of the
loss, not which file/module was touched. The "approach" axis matches the
nanogpt template (Hyperparameter / Architecture-change / Training-technique /
Novel Method) because MUL has a rich literature of mechanism families and we
want an explicit home for genuinely-new ideas.

See heuresis:adding-new-tasks SKILL §11 for the broader narrow-surface
vs broad-surface design rationale.
"""

from __future__ import annotations

from pathlib import Path

from heuresis.qd import (
    Feature,
    FeatureClassifier,
    KeywordClassifier,
    LLMClassifier,
)

FEATURES = [
    Feature("component", min_val=0, max_val=5, num_bins=6,
            bin_names=("Forget-Only", "Retain-Only", "Forget+Retain Contrast",
                       "Reference-Anchored", "Representation-Level",
                       "Output-Distribution / Relabel")),
    Feature("approach", min_val=0, max_val=3, num_bins=4,
            bin_names=("Hyperparameter Tuning", "Architecture Change",
                       "Training Technique", "Novel Method")),
]

KEYWORDS: dict[str, dict[int, list[str]]] = {
    "component": {
        # 0: Forget-Only -- single term on forget batch alone, no retain or ref.
        0: [
            "gradient_ascent", "gradient ascent", "ga_loss", "-forget_loss",
            "negate", "ascent", "margin", "clamp", "threshold",
            "label_smoothing", "forget_only", "unlearn_loss = -",
            "torch.clamp", "neg_forget", "max_loss",
        ],
        # 1: Retain-Only -- forget term dropped/zeroed; SFT-style retain anchoring.
        1: [
            "retain_only", "sft_retain", "no forget term", "retain anchor",
            "alpha = 0", "drop forget", "ignore forget", "retain SFT",
            "fine_tune_retain", "asymmetric",
        ],
        # 2: Forget+Retain Contrast -- two-term additive (GradDiff, GA+R, weighted).
        2: [
            "graddiff", "gradient_difference", "retain - forget",
            "weighted_sum", "alpha *", "ga+retain", "ga_plus", "ga + retain",
            "combined_loss", "additive", "balance forget retain",
            "retain_loss - forget_loss",
        ],
        # 3: Reference-Anchored -- frozen copy of initial model (NPO, KL, DPO).
        3: [
            "npo", "negative_preference", "ref_model", "reference_model",
            "frozen_model", "log_ratio", "logsigmoid", "beta",
            "kl_divergence", "kl_min", "kl divergence", "dpo", "ipo",
            "preference optimization", "anchor model", "ema_reference",
        ],
        # 4: Representation-Level -- signal from intermediate activations.
        4: [
            "rmu", "representation", "hidden_state", "hidden_states",
            "activation", "intermediate", "layer_output", "hidden_layer",
            "output_hidden_states", "projection", "random_direction",
            "forget_steering", "attention_output", "mlp_output",
            "representation misdirection",
        ],
        # 5: Output-Distribution / Relabel -- replace/distill the forget target.
        5: [
            "random_label", "uniform_logits", "random_target",
            "label_replacement", "distillation", "teacher_student",
            "detuned", "who_harry_potter", "whp", "relabel",
            "garbage_target", "uniform target",
        ],
    },
    "approach": {
        # Match nanogpt's approach axis (cf. experiments/nanogpt_map_elites/features.py).
        0: [
            "increase", "decrease", "tune", "scale", "double", "halve",
            "sweep", "grid_search", "reduce", "bump", "lower", "higher",
            "set to", "change to", "alpha", "beta", "gamma",
            "coefficient", "weight factor",
        ],
        1: [
            "replace", "swap", "rewrite", "new module", "add layer",
            "restructure", "modify architecture", "different activation",
            "alternative", "substitute", "redesign",
        ],
        2: [
            "mixed precision", "gradient clipping", "accumulation",
            "curriculum", "distillation", "pruning", "quantization",
            "label smoothing", "dropout", "regularization", "augment",
            "augmentation", "schedule", "anneal", "warmup", "decay",
        ],
        3: [
            "novel", "new approach", "new method", "not in literature",
            "information theoretic", "information-theoretic",
            "mutual information", "infomax", "optimal transport",
            "adversarial", "meta learn", "meta-learn", "meta-learned",
            "learned loss", "differentiable surrogate", "first attempt",
            "experimental", "research", "paper", "inspired by",
            "state of the art", "cutting edge", "custom", "designed",
            "invented", "proposed",
        ],
    },
}

_CLASSIFICATION_PROMPT = """\
Classify the following machine-unlearning loss-function idea along two axes.

component (0-5): the mathematical family of the loss
  0 = Forget-Only -- one term on the forget batch alone (e.g. Gradient Ascent,
      clamped/margin GA, label smoothing on forget; no retain or reference)
  1 = Retain-Only / Asymmetric -- forget term dropped or zeroed; unlearning
      via retain-set SFT or its absence
  2 = Forget+Retain Contrast -- two-term additive of forget and retain
      (GradDiff, GA+Retain, weighted-sum baselines)
  3 = Reference-Model-Anchored -- uses a frozen reference copy of the
      initial model (NPO, KL-Min, DPO-style log-ratios)
  4 = Representation-Level -- signal from intermediate activations / hidden
      states / attention outputs (RMU, hidden-state steering, projection)
  5 = Output-Distribution / Relabeling -- replaces or distills the forget
      target (random labels, uniform logits, distillation, WHP)

When an idea genuinely combines multiple families (e.g. NPO + KL retain),
classify by the *dominant* mechanism -- the one providing the unlearning
gradient. Reference-anchored methods that include a retain anchor still go
to bin 3 because the unlearning signal is reference-derived; the retain is
secondary.

approach (0-3): the kind of change relative to a baseline
  0 = Hyperparameter Tuning (adjust coefficients/weights/scales of an
      existing formulation)
  1 = Architecture Change (swap a sub-module, add/remove a structural piece,
      rewrite the mechanism for a different known formulation)
  2 = Training Technique (curriculum, schedule, regularization, distillation
      delivery mechanism)
  3 = Novel Method (the idea proposes a mechanism that is not adequately
      described by any of the 6 component families above -- e.g. an
      information-theoretic objective, an adversarial training scheme, an
      optimal-transport formulation, or a meta-learned loss surface. Pick
      the *closest* component bin AND use approach=3 to signal "this is
      outside the known taxonomy.")

Return JSON: {"component": <int>, "approach": <int>}
"""


def make_classifier(
    *,
    config: dict | None = None,  # noqa: ARG001 — MUL grid is fixed (same file every idea)
    use_llm: bool = True,
    api_keys_file: Path | None = None,
) -> FeatureClassifier:
    """LLMClassifier with KeywordClassifier fallback (or pure keyword). The
    returned classifier carries the labeled :data:`FEATURES` list."""
    kw = KeywordClassifier(FEATURES, KEYWORDS)
    if not use_llm:
        return kw
    return LLMClassifier(
        FEATURES,
        fallback=kw,
        api_keys_file=api_keys_file,
        classification_prompt=_CLASSIFICATION_PROMPT,
    )
