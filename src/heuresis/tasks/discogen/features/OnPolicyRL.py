"""Feature schema + keyword dicts for DiscoGen OnPolicyRL MAP-Elites.

6 components x 4 approaches = 24 cells. Feeds KeywordClassifier or
LLMClassifier -- strategy doesn't care which.

Component axis: which `discovered/` module was primarily changed.
Approach axis: what kind of change (Replace, Add/Augment, Retune, Dynamic).
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
            bin_names=("Loss", "Optim", "Networks", "Train",
                       "Activation", "Targets")),
    Feature("approach", min_val=0, max_val=3, num_bins=4,
            bin_names=("Replace", "Add/Augment", "Retune/Rescale",
                       "Dynamic/Conditional")),
]

KEYWORDS: dict[str, dict[int, list[str]]] = {
    "component": {
        0: [
            "loss", "actor_critic_loss", "loss_actor", "policy_gradient",
            "ppo_loss", "clipped_surrogate", "value_loss", "entropy_loss",
            "clip_eps", "vf_coef", "ent_coef", "kl_penalty", "surrogate",
        ],
        1: [
            "optim", "optimizer", "scale_by_optimizer", "gradient",
            "learning_rate", "lr", "adam", "sgd", "momentum",
            "gradient_transformation", "init_fn", "update_fn",
            "clip_grad", "max_grad_norm", "grad_norm",
        ],
        2: [
            "network", "actor_critic", "actorcritic", "nn.module",
            "nn.compact", "hidden_layer", "dense", "linear",
            "critic", "policy_network", "value_network", "hsize",
        ],
        3: [
            "train", "make_train", "training_loop", "update_step",
            "env_step", "rollout", "collect", "trajectory",
            "minibatch", "update_epoch", "scan",
        ],
        4: [
            "activation", "get_activation", "relu", "tanh", "gelu",
            "swish", "silu", "elu", "leaky_relu", "mish",
            "softplus", "nonlinearity",
        ],
        5: [
            "targets", "get_targets", "gae", "generalized_advantage",
            "td_target", "bootstrap", "lambda_return", "discount",
            "gamma", "gae_lambda", "value_target", "advantage",
            "advantage_estimation", "compute_advantage", "return_estimate",
        ],
    },
    "approach": {
        0: [
            "replace", "swap", "rewrite", "new implementation",
            "alternative", "substitute", "instead of", "redesign",
        ],
        1: [
            "add", "augment", "combine", "auxiliary", "extra",
            "additional", "extend", "supplement", "alongside",
        ],
        2: [
            "tune", "scale", "adjust", "increase", "decrease",
            "double", "halve", "clip", "normalize", "rescale",
            "coefficient", "weight", "factor",
        ],
        3: [
            "dynamic", "conditional", "adaptive", "schedule",
            "anneal", "curriculum", "warm up", "decay",
            "per-step", "state-dependent", "context",
        ],
    },
}

_CLASSIFICATION_PROMPT = """\
Classify the following RL algorithm discovery idea along two axes:

component (0-5): which discovered/ module is primarily changed
  0=Loss, 1=Optim, 2=Networks, 3=Train, 4=Activation, 5=Targets

approach (0-3): what kind of change
  0=Replace (rewrite from scratch), 1=Add/Augment (extend existing),
  2=Retune/Rescale (adjust parameters), 3=Dynamic/Conditional (adaptive behavior)

Return JSON: {"component": <int>, "approach": <int>}
"""


def make_classifier(
    *,
    config: dict | None = None,  # noqa: ARG001 — OnPolicyRL grid is fixed (all modules editable)
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
