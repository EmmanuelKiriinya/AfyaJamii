"""Tests for the risk classifier.

The central guarantee here is that a label means what it says. The model is a
three-class classifier whose encoder orders the classes alphabetically —
['high risk', 'low risk', 'mid risk'] — so the middle column is *low risk*, not
a "probability of risk". An earlier revision read that column and thresholded
it at 0.5, which inverted the result for exactly the patients who most needed a
correct one. These tests exist so that cannot come back.

Run with:  pytest tests/ -v
"""

import numpy as np
import pytest

from app.ml_model import (
    FEATURE_NAMES,
    InvalidFeaturesError,
    ModelNotLoadedError,
    RiskPredictionModel,
    initialize_model,
    risk_model,
)

HEALTHY = {
    "Age": 25,
    "SystolicBP": 110,
    "DiastolicBP": 70,
    "BS": 4.5,
    "BodyTemp": 36.8,
    "HeartRate": 72,
}

SEVERE = {
    "Age": 42,
    "SystolicBP": 170,
    "DiastolicBP": 110,
    "BS": 15.0,
    "BodyTemp": 38.5,
    "HeartRate": 95,
}


@pytest.fixture(scope="module", autouse=True)
def loaded_model():
    assert initialize_model(), "the bundled model artefacts failed to load"
    return risk_model


def test_classes_come_from_the_label_encoder():
    # Order matters: predictions are decoded positionally against this list.
    assert risk_model.classes == ["high risk", "low risk", "mid risk"]


def test_severe_vitals_are_not_reported_as_low_risk():
    """The regression that motivated this module.

    BP 170/110 with blood sugar 15.0 and a fever was previously labelled
    "low risk" because the code read the low-risk column and inverted it.
    """
    prediction = risk_model.predict(SEVERE)
    assert prediction.label == "high risk"
    assert prediction.severity == 2


def test_healthy_vitals_are_not_reported_as_high_risk():
    prediction = risk_model.predict(HEALTHY)
    assert prediction.label != "high risk"


def test_label_matches_the_most_likely_class():
    """The reported label must be the argmax of the distribution."""
    for vitals in (HEALTHY, SEVERE):
        prediction = risk_model.predict(vitals)
        most_likely = max(prediction.class_probabilities.items(), key=lambda item: item[1])
        assert prediction.label == most_likely[0]
        assert prediction.probability == pytest.approx(most_likely[1])


def test_probability_is_the_confidence_in_the_reported_class():
    prediction = risk_model.predict(SEVERE)
    assert 0.0 <= prediction.probability <= 1.0
    assert prediction.probability == pytest.approx(
        prediction.class_probabilities[prediction.label]
    )


def test_class_probabilities_cover_every_class_and_sum_to_one():
    prediction = risk_model.predict(HEALTHY)
    assert set(prediction.class_probabilities) == set(risk_model.classes)
    assert sum(prediction.class_probabilities.values()) == pytest.approx(1.0, abs=1e-6)


def test_feature_importances_cover_every_feature():
    prediction = risk_model.predict(HEALTHY)
    assert set(prediction.feature_importances) == set(FEATURE_NAMES)


@pytest.mark.parametrize(
    "field, value",
    [
        ("Age", 99),          # above the trained range
        ("Age", 10),          # below it
        ("SystolicBP", 250),
        ("BS", 1.0),
        ("BodyTemp", 45.0),
        ("HeartRate", 5),
    ],
)
def test_out_of_range_vitals_are_rejected(field, value):
    vitals = {**HEALTHY, field: value}
    with pytest.raises(InvalidFeaturesError):
        risk_model.predict(vitals)


def test_missing_vitals_are_rejected():
    vitals = {key: value for key, value in HEALTHY.items() if key != "BS"}
    with pytest.raises(InvalidFeaturesError, match="BS"):
        risk_model.predict(vitals)


def test_predicting_before_loading_raises():
    with pytest.raises(ModelNotLoadedError):
        RiskPredictionModel().predict(HEALTHY)


def test_artefact_mismatch_is_refused(tmp_path, monkeypatch):
    """A model and encoder that disagree must not load.

    Loading them anyway would silently mislabel every prediction, which is the
    failure mode this guards against.
    """
    model = RiskPredictionModel()

    class TwoClassModel:
        n_classes_ = 2

    class ThreeClassEncoder:
        classes_ = np.array(["high risk", "low risk", "mid risk"], dtype=object)

    monkeypatch.setattr(
        RiskPredictionModel,
        "_read_artifact",
        staticmethod(
            lambda path: ThreeClassEncoder() if "encoder" in str(path) else TwoClassModel()
        ),
    )

    model_file = tmp_path / "model.pkl"
    encoder_file = tmp_path / "encoder.pkl"
    model_file.touch()
    encoder_file.touch()

    assert model.load(model_file, encoder_file) is False
    assert model.is_loaded is False
