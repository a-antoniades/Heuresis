"""Feature-space ownership: labeled Feature axes, generic namer, and the
SupportsCellSearch capability that cell loops gate on."""
from heuresis.tasks.nanogpt.adapter import NanoGPTAdapter
from heuresis.tasks.adapter import SupportsCellSearch
from heuresis.qd import Feature, feature_namer
from heuresis.qd.core.archive import GridArchive


def test_feature_namer_uses_bin_names():
    features = [Feature("component", 0, 2, 3, bin_names=("A", "B", "C"))]
    assert feature_namer(features)({"component": 1.0}) == {"component": "B"}


def test_feature_namer_falls_back_without_labels():
    """Unlabeled axis (or out-of-range index) falls back to the raw value."""
    labeled = Feature("x", 0, 1, 2, bin_names=("lo", "hi"))
    unlabeled = Feature("y", 0, 10, 11)
    namer = feature_namer([labeled, unlabeled])
    # out-of-range index (x=5) and unlabeled axis (y) fall back to str(value)
    assert namer({"x": 5.0, "y": 3.0}) == {"x": "5.0", "y": "3.0"}


def test_bin_names_survive_archive_serialization():
    features = [Feature("c", 0, 1, 2, bin_names=("lo", "hi"))]
    restored = GridArchive.from_dict(GridArchive(features).to_dict())
    assert restored.features[0].bin_names == ("lo", "hi")


def test_unlabeled_feature_serializes_as_none():
    features = [Feature("c", 0, 1, 2)]
    assert GridArchive(features).to_dict()["features"][0]["bin_names"] is None


def test_cell_adapter_satisfies_capability():
    """A cell-capable adapter is a structural SupportsCellSearch; a bare object
    is not — this is what map_elites/go_explore gate on."""
    assert isinstance(NanoGPTAdapter(), SupportsCellSearch)
    assert not isinstance(object(), SupportsCellSearch)


def test_nanogpt_classifier_carries_labeled_features():
    clf = NanoGPTAdapter().make_classifier()
    assert clf.features[0].name == "component"
    assert clf.features[0].bin_names[0] == "Attention"
