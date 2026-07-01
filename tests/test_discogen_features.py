"""Tests for the DiscoGen feature classifiers + domain registry."""
import pytest

from heuresis.qd import feature_namer
from heuresis.tasks.discogen import features as discogen_features
from heuresis.tasks.discogen.features import OnPolicyRL


def test_features_defined():
    """FEATURES list has the expected structure + labels."""
    assert len(OnPolicyRL.FEATURES) == 2
    assert OnPolicyRL.FEATURES[0].name == "component"
    assert OnPolicyRL.FEATURES[0].num_bins == 6
    assert len(OnPolicyRL.FEATURES[0].bin_names) == 6
    assert OnPolicyRL.FEATURES[1].name == "approach"
    assert OnPolicyRL.FEATURES[1].num_bins == 4
    assert len(OnPolicyRL.FEATURES[1].bin_names) == 4


def test_bin_names_label_via_feature_namer():
    """Labels resolve generically from Feature.bin_names — no per-task fn."""
    names = feature_namer(OnPolicyRL.FEATURES)({"component": 0.0, "approach": 1.0})
    assert names["component"] == "Loss"
    assert names["approach"] == "Add/Augment"


def test_make_classifier_carries_labeled_features():
    """make_classifier returns a working classifier holding the labeled axes."""
    classifier = OnPolicyRL.make_classifier(use_llm=False)
    assert classifier.features is OnPolicyRL.FEATURES
    result = classifier.classify(
        "Replace the loss function with a novel actor-critic loss"
    )
    assert "component" in result
    assert "approach" in result


def test_keywords_cover_all_bins():
    """Every component and approach bin has at least one keyword."""
    for axis, num_bins in [("component", 6), ("approach", 4)]:
        for i in range(num_bins):
            assert i in OnPolicyRL.KEYWORDS[axis], f"Missing keywords for {axis} bin {i}"
            assert len(OnPolicyRL.KEYWORDS[axis][i]) > 0, f"Empty {axis} bin {i}"


def test_registry_dispatch():
    """The domain registry builds the right classifier per domain."""
    rl = discogen_features.make_classifier("OnPolicyRL", use_llm=False)
    assert rl.features[0].bin_names[0] == "Loss"
    mul = discogen_features.make_classifier("ModelUnlearning", use_llm=False)
    assert mul.features[0].bin_names[0] == "Forget-Only"


def test_registry_unknown_domain():
    """An unknown domain fails early with a clear error."""
    with pytest.raises(NotImplementedError, match="ModelUnlearning"):
        discogen_features.make_classifier("Nope", use_llm=False)
