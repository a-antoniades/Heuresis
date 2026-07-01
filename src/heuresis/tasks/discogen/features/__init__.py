"""DiscoGen feature classifiers, one module per domain.

The domain registry is explicit (not f-string module construction) so an
unknown domain fails early with a clear error. Each domain module exposes a
``make_classifier(*, config, use_llm, api_keys_file) -> FeatureClassifier``
whose returned classifier carries that domain's labeled ``FEATURES`` list.
"""
from __future__ import annotations

from importlib import import_module
from pathlib import Path

from heuresis.qd import FeatureClassifier

_REGISTRY = {
    "OnPolicyRL": "heuresis.tasks.discogen.features.OnPolicyRL",
    "ModelUnlearning": "heuresis.tasks.discogen.features.ModelUnlearning",
}


def make_classifier(
    domain: str,
    *,
    config: dict | None = None,
    use_llm: bool = True,
    api_keys_file: Path | None = None,
) -> FeatureClassifier:
    """Build the feature classifier for a discogen domain."""
    try:
        module_path = _REGISTRY[domain]
    except KeyError as e:
        raise NotImplementedError(
            f"no feature classifier registered for discogen domain {domain!r} "
            f"(known: {sorted(_REGISTRY)})"
        ) from e
    mod = import_module(module_path)
    return mod.make_classifier(
        config=config, use_llm=use_llm, api_keys_file=api_keys_file)
