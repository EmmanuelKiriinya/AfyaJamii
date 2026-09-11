"""Maternal risk classification.

The bundled artefact is a three-class XGBoost classifier (``multi:softprob``)
trained on six vitals. Its integer outputs are decoded through the companion
``LabelEncoder``, whose classes are ``['high risk', 'low risk', 'mid risk']``.

Note on a past defect: an earlier revision read ``predict_proba(x)[0, 1]`` and
compared it to 0.5, treating the model as binary. Column 1 is *low risk*, so
that logic reported "high risk" precisely when the model was confident the
patient was low risk. Predictions now go through ``argmax`` and the label
encoder, and the encoder's class list is validated against the model's class
count at load time so the two artefacts cannot drift apart unnoticed.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import joblib
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# Column order the model was trained on. Do not reorder.
FEATURE_NAMES: tuple[str, ...] = (
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "BS",
    "BodyTemp",
    "HeartRate",
)

# Accepted input ranges, mirrored by the API schema in app.models.
FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "Age": (15, 50),
    "SystolicBP": (70, 200),
    "DiastolicBP": (40, 130),
    "BS": (3.0, 30.0),
    "BodyTemp": (35.0, 42.0),
    "HeartRate": (40, 150),
}

# Reference values used to explain a prediction when the model itself exposes
# no per-prediction attribution.
REFERENCE_VALUES: dict[str, float] = {
    "Age": 30.0,
    "SystolicBP": 120.0,
    "DiastolicBP": 80.0,
    "BS": 5.5,
    "BodyTemp": 37.0,
    "HeartRate": 70.0,
}

# Ordering used when a risk label needs to be compared or sorted by severity.
RISK_SEVERITY: dict[str, int] = {"low risk": 0, "mid risk": 1, "high risk": 2}


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is attempted before the model is available."""


class InvalidFeaturesError(ValueError):
    """Raised when input vitals are missing or outside the trained ranges."""


@dataclass
class RiskPrediction:
    """The outcome of scoring one set of vitals."""

    label: str
    probability: float
    class_probabilities: dict[str, float]
    feature_importances: dict[str, float]

    @property
    def severity(self) -> int:
        return RISK_SEVERITY.get(self.label.lower(), -1)


@dataclass
class ModelMetadata:
    model_type: str = ""
    classes: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    model_path: str = ""
    encoder_path: str = ""
    loaded_at: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "classes": self.classes,
            "features": self.features,
            "model_path": self.model_path,
            "encoder_path": self.encoder_path,
            "loaded_at": self.loaded_at,
        }


class RiskPredictionModel:
    """Loads the risk classifier and scores vitals against it."""

    def __init__(self) -> None:
        self._model: Any = None
        self._classes: list[str] = []
        self._metadata = ModelMetadata()

    # ── Loading ────────────────────────────────────────────────────────────

    def load(self, model_path: Path, encoder_path: Optional[Path] = None) -> bool:
        """Load the classifier and its label encoder.

        Returns True on success. The caller decides whether a failure is fatal.
        """
        try:
            model = self._read_artifact(model_path)
        except FileNotFoundError:
            logger.error("Risk model not found at %s", model_path)
            return False
        except Exception:  # noqa: BLE001 - unpickling can fail many ways
            logger.exception("Could not load the risk model from %s", model_path)
            return False

        classes = self._load_classes(encoder_path, model)
        if not classes:
            logger.error(
                "Could not determine the model's class labels. Provide a label encoder at %s.",
                encoder_path,
            )
            return False

        expected = self._expected_class_count(model)
        if expected is not None and expected != len(classes):
            # Refusing to start beats mislabelling a clinical prediction.
            logger.error(
                "Artefact mismatch: the model predicts %d classes but the label encoder "
                "defines %d (%s). Risk labels would be wrong; refusing to load.",
                expected,
                len(classes),
                ", ".join(classes),
            )
            return False

        self._model = model
        self._classes = classes
        self._metadata = ModelMetadata(
            model_type=type(model).__name__,
            classes=list(classes),
            model_path=str(model_path),
            encoder_path=str(encoder_path) if encoder_path else "",
            loaded_at=datetime.now(timezone.utc).isoformat(),
        )

        logger.info(
            "Risk model loaded from %s (%s, classes: %s)",
            model_path,
            self._metadata.model_type,
            ", ".join(classes),
        )
        return True

    @staticmethod
    def _read_artifact(path: Path) -> Any:
        if not path.is_file():
            raise FileNotFoundError(path)

        suffix = path.suffix.lower()
        if suffix == ".joblib":
            return joblib.load(path)
        if suffix == ".pkl":
            # joblib reads plain pickles too, and handles the numpy memmap
            # wrappers scikit-learn writes.
            try:
                return joblib.load(path)
            except Exception:  # noqa: BLE001 - fall back to stdlib pickle
                with path.open("rb") as handle:
                    return pickle.load(handle)
        raise ValueError(f"Unsupported model format {suffix!r}; expected .pkl or .joblib")

    def _load_classes(self, encoder_path: Optional[Path], model: Any) -> list[str]:
        """Resolve class labels, preferring the encoder over the raw model."""
        if encoder_path and encoder_path.is_file():
            try:
                encoder = self._read_artifact(encoder_path)
                classes = getattr(encoder, "classes_", None)
                if classes is not None and len(classes) > 0:
                    return [str(label) for label in classes]
                logger.warning("Label encoder at %s exposes no classes_", encoder_path)
            except Exception:  # noqa: BLE001
                logger.exception("Could not load the label encoder from %s", encoder_path)
        elif encoder_path:
            logger.warning("Label encoder not found at %s", encoder_path)

        model_classes = getattr(model, "classes_", None)
        if model_classes is not None and len(model_classes) > 0:
            # Only usable when the model kept string labels; integer codes are
            # meaningless without the encoder that produced them.
            if all(isinstance(label, str) for label in model_classes):
                return [str(label) for label in model_classes]
            logger.error(
                "The model's classes_ are numeric codes (%s). The label encoder is required "
                "to translate them into risk labels.",
                list(model_classes),
            )
        return []

    @staticmethod
    def _expected_class_count(model: Any) -> Optional[int]:
        for attribute in ("n_classes_", "classes_"):
            value = getattr(model, attribute, None)
            if value is None:
                continue
            return int(value) if attribute == "n_classes_" else len(value)
        return None

    # ── Inspection ─────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def classes(self) -> list[str]:
        return list(self._classes)

    def info(self) -> dict[str, Any]:
        return {"loaded": self.is_loaded, **self._metadata.as_dict()}

    # ── Prediction ─────────────────────────────────────────────────────────

    def validate(self, features: Mapping[str, float]) -> None:
        """Check that every feature is present and within its trained range."""
        missing = [name for name in FEATURE_NAMES if name not in features]
        if missing:
            raise InvalidFeaturesError(f"Missing vitals: {', '.join(missing)}")

        out_of_range: list[str] = []
        for name, (low, high) in FEATURE_RANGES.items():
            value = float(features[name])
            if not low <= value <= high:
                out_of_range.append(f"{name}={value:g} (expected {low:g}–{high:g})")

        if out_of_range:
            raise InvalidFeaturesError("Vitals outside the supported range: " + "; ".join(out_of_range))

    def predict(self, features: Mapping[str, float]) -> RiskPrediction:
        """Score one set of vitals.

        Raises:
            ModelNotLoadedError: if called before a successful load.
            InvalidFeaturesError: if the vitals are missing or out of range.
        """
        if not self.is_loaded:
            raise ModelNotLoadedError("The risk model is not loaded")

        self.validate(features)

        row = np.array([[float(features[name]) for name in FEATURE_NAMES]], dtype=np.float64)

        probabilities = self._class_probabilities(row)
        index = int(np.argmax(probabilities))
        label = self._classes[index]
        confidence = float(probabilities[index])

        prediction = RiskPrediction(
            label=label,
            probability=confidence,
            class_probabilities={
                name: float(value) for name, value in zip(self._classes, probabilities)
            },
            feature_importances=self._explain(features),
        )

        logger.debug(
            "Scored vitals: %s (confidence %.3f, distribution %s)",
            label,
            confidence,
            prediction.class_probabilities,
        )
        return prediction

    def _class_probabilities(self, row: np.ndarray) -> np.ndarray:
        """Return a probability per class, in label-encoder order."""
        if hasattr(self._model, "predict_proba"):
            probabilities = np.asarray(self._model.predict_proba(row))[0]
            if len(probabilities) != len(self._classes):
                raise ModelNotLoadedError(
                    f"The model returned {len(probabilities)} probabilities but "
                    f"{len(self._classes)} classes are defined"
                )
            return probabilities

        # Models without predict_proba return a class index; represent that as
        # a one-hot distribution so downstream handling stays uniform.
        index = int(np.asarray(self._model.predict(row)).ravel()[0])
        if not 0 <= index < len(self._classes):
            raise ModelNotLoadedError(f"The model returned an unknown class index: {index}")
        probabilities = np.zeros(len(self._classes), dtype=np.float64)
        probabilities[index] = 1.0
        return probabilities

    def _explain(self, features: Mapping[str, float]) -> dict[str, float]:
        """Attribute the prediction across the input vitals.

        Prefers the model's global feature importances. Where those are absent,
        falls back to each vital's normalised deviation from its reference
        value, which at least tells the user which reading stands out.
        """
        try:
            importances = getattr(self._model, "feature_importances_", None)
            if importances is not None and len(importances) == len(FEATURE_NAMES):
                return {
                    name: float(value)
                    for name, value in zip(FEATURE_NAMES, importances)
                }

            deviations: dict[str, float] = {}
            for name in FEATURE_NAMES:
                reference = REFERENCE_VALUES[name]
                deviations[name] = min(abs(float(features[name]) - reference) / reference, 1.0)

            total = sum(deviations.values())
            if total <= 0:
                return {name: 0.0 for name in FEATURE_NAMES}
            return {name: value / total for name, value in deviations.items()}

        except Exception:  # noqa: BLE001 - explanation must never break scoring
            logger.warning("Could not compute feature importances", exc_info=True)
            return {name: 0.0 for name in FEATURE_NAMES}


risk_model = RiskPredictionModel()


def initialize_model() -> bool:
    """Load the shared model. Called once during application start-up."""
    return risk_model.load(settings.model_file, settings.label_encoder_file)


def describe_features() -> Sequence[str]:
    return FEATURE_NAMES
