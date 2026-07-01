"""Feature schema + keyword dicts for nanoGPT MAP-Elites.

6 components x 4 approaches = 24 cells. Feeds KeywordClassifier or
LLMClassifier — strategy doesn't care which.
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
            bin_names=("Attention", "FFN", "Normalization",
                       "Positional & Embedding", "Architecture",
                       "Optimizer & Schedule")),
    Feature("approach", min_val=0, max_val=3, num_bins=4,
            bin_names=("Hyperparameter Tuning", "Architecture Change",
                       "Training Technique", "Novel Method")),
]

KEYWORDS: dict[str, dict[int, list[str]]] = {
    "component": {
        0: ["attention", "self_attn", "multihead", "flash_attn", "sdpa",
            "qkv", "query", "key_proj", "value_proj", "head_dim",
            "causal_self_attention", "c_attn", "grouped_query",
            "window_size", "sliding_window", "local_attention"],
        1: ["mlp", "feedforward", "ffn", "swiglu", "reglu", "geglu",
            "gated_mlp", "up_proj", "down_proj", "gate_proj",
            "intermediate_size", "c_fc", "c_proj", "activation", "gelu"],
        2: ["rmsnorm", "layernorm", "prenorm", "postnorm",
            "init_weight", "orthogonal_init", "xavier", "kaiming",
            "zero_init", "residual_scale", "resid_lambda", "x0_lambda",
            "deepnorm", "layer_scale"],
        3: ["rope", "rotary", "alibi", "sinusoidal", "learned_pos",
            "pos_embed", "pos_encoding", "rope_base",
            "embedding", "wte", "wpe", "lm_head", "unembedding",
            "vocab_size", "token_embed", "tie_weight", "n_embd"],
        4: ["n_layer", "depth", "layer_sharing", "recurrent",
            "weight_sharing", "skip_connection", "parallel_layer",
            "seq_len", "context_length", "block_size",
            "adapter", "lora", "aspect_ratio"],
        5: ["learning_rate", "lr", "adam", "muon", "weight_decay",
            "warmup", "warmdown", "schedule", "cosine", "linear_decay",
            "batch_size", "grad_accum", "momentum", "beta1", "beta2",
            "total_batch_size", "embedding_lr", "matrix_lr", "scalar_lr"],
    },
    "approach": {
        0: ["increase", "decrease", "tune", "scale", "double", "halve",
            "sweep", "grid_search", "reduce", "bump", "lower", "higher",
            "set to", "change to", "from", "to"],
        1: ["replace", "swap", "remove", "add layer", "new module",
            "rewrite", "restructure", "modify architecture",
            "different activation", "alternative", "substitute"],
        2: ["mixed precision", "gradient clipping", "accumulation",
            "curriculum", "distillation", "pruning", "quantization",
            "label smoothing", "dropout", "regularization", "augment"],
        3: ["novel", "new approach", "experimental", "research",
            "paper", "inspired by", "state of the art", "cutting edge",
            "custom", "designed", "invented", "proposed"],
    },
}

_CLASSIFICATION_PROMPT = """\
Classify the following nanoGPT training-modification idea along two axes:

component (0-5): which part of the model/training pipeline is primarily changed
  0=Attention, 1=FFN, 2=Normalization, 3=Positional & Embedding,
  4=Architecture, 5=Optimizer & Schedule

approach (0-3): what kind of change
  0=Hyperparameter Tuning, 1=Architecture Change,
  2=Training Technique, 3=Novel Method

Return JSON: {"component": <int>, "approach": <int>}
"""


def make_classifier(
    *,
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
