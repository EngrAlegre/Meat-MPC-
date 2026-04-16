from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import config


HYBRID_ML_DIR = Path(__file__).resolve().parents[1] / "hybrid_ml"
if str(HYBRID_ML_DIR) not in sys.path:
    sys.path.insert(0, str(HYBRID_ML_DIR))

from meat_classifier_utils import MeatClassifierRuntimeError, load_meat_classifier, predict_meat_class


LOGGER = logging.getLogger(__name__)


class MeatClassifierLoadError(RuntimeError):
    pass


@dataclass
class MeatClassificationResult:
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]
    is_valid_meat: bool
    hybrid_meat_type: str | None


class MeatClassifierService:
    def __init__(self) -> None:
        try:
            self.artifacts = load_meat_classifier(
                model_path=config.MEAT_CLASSIFIER_MODEL_PATH,
                class_names_path=config.MEAT_CLASSIFIER_CLASS_NAMES_PATH,
                metadata_path=config.MEAT_CLASSIFIER_METADATA_PATH,
            )
            LOGGER.info("Meat classifier loaded from %s", config.MEAT_CLASSIFIER_MODEL_PATH)
        except Exception as exc:
            raise MeatClassifierLoadError(f"Failed to load meat classifier artifacts: {exc}") from exc

    def classify(self, image_path: str | Path) -> MeatClassificationResult:
        try:
            result = predict_meat_class(self.artifacts, image_path)
        except MeatClassifierRuntimeError as exc:
            raise MeatClassifierLoadError(str(exc)) from exc

        predicted_class = str(result["predicted_class"])
        confidence = float(result["confidence"])
        class_probabilities = {
            str(label): float(score) for label, score in result["class_probabilities"].items()
        }
        is_valid_meat = predicted_class in config.MEAT_CLASSIFIER_VALID_LABELS
        hybrid_meat_type = config.MEAT_CLASSIFIER_TO_HYBRID_MEAT_TYPE.get(predicted_class)

        prob_str = " | ".join(f"{label}={score:.4f}" for label, score in class_probabilities.items())
        LOGGER.info(
            "Meat classification | predicted=%s (%.4f) valid_meat=%s | %s",
            predicted_class, confidence, is_valid_meat, prob_str,
        )

        return MeatClassificationResult(
            predicted_class=predicted_class,
            confidence=confidence,
            class_probabilities=class_probabilities,
            is_valid_meat=is_valid_meat,
            hybrid_meat_type=hybrid_meat_type,
        )
