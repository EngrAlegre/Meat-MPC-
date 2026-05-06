from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import config


HYBRID_ML_DIR = Path(__file__).resolve().parents[1] / "hybrid_ml"
if str(HYBRID_ML_DIR) not in sys.path:
    sys.path.insert(0, str(HYBRID_ML_DIR))

from meat_classifier_utils import MeatClassifierRuntimeError, load_meat_classifier


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

    def _predict_probabilities(self, image_path: str | Path) -> np.ndarray:
        """Run the network and return softmax probabilities (with optional TTA)."""
        input_size = tuple(self.artifacts.metadata.get("input_size", (224, 224)))
        img = Image.open(image_path).convert("RGB").resize((input_size[0], input_size[1]))
        array = np.asarray(img, dtype=np.float32)

        batch = [array]
        if getattr(config, "MEAT_CLASSIFIER_TTA_ENABLED", False):
            # Horizontal flip is a safe, label-preserving augmentation for meat.
            batch.append(np.fliplr(array))

        stacked = np.stack(batch, axis=0)
        probabilities = self.artifacts.model.predict(stacked, verbose=0)
        return probabilities.mean(axis=0)

    def _build_result(
        self,
        probabilities: np.ndarray,
        *,
        log_prefix: str = "Meat classification",
    ) -> MeatClassificationResult:
        predicted_index = int(np.argmax(probabilities))
        predicted_class = str(self.artifacts.class_names[predicted_index])
        confidence = float(probabilities[predicted_index])
        class_probabilities = {
            str(name): float(score)
            for name, score in zip(self.artifacts.class_names, probabilities)
        }
        min_confidence = getattr(config, "MEAT_CLASSIFIER_MIN_CONFIDENCE", 0.60)
        is_valid_meat = (
            predicted_class in config.MEAT_CLASSIFIER_VALID_LABELS
            and confidence >= min_confidence
        )
        hybrid_meat_type = config.MEAT_CLASSIFIER_TO_HYBRID_MEAT_TYPE.get(predicted_class)

        prob_str = " | ".join(f"{label}={score:.4f}" for label, score in class_probabilities.items())
        LOGGER.info(
            "%s | predicted=%s (%.4f) valid_meat=%s | %s",
            log_prefix, predicted_class, confidence, is_valid_meat, prob_str,
        )

        return MeatClassificationResult(
            predicted_class=predicted_class,
            confidence=confidence,
            class_probabilities=class_probabilities,
            is_valid_meat=is_valid_meat,
            hybrid_meat_type=hybrid_meat_type,
        )

    def classify(self, image_path: str | Path) -> MeatClassificationResult:
        try:
            probabilities = self._predict_probabilities(image_path)
        except Exception as exc:
            raise MeatClassifierLoadError(str(exc)) from exc
        return self._build_result(probabilities)

    def verify(
        self,
        first: MeatClassificationResult,
        second_image_path: str | Path,
    ) -> MeatClassificationResult:
        """Run a second-thought pass on a fresh frame and combine with ``first``.

        If both passes agree on the predicted class, the returned result uses
        the averaged class probabilities (which raises confidence when both
        agree). If they disagree, ``is_valid_meat`` is forced to False and the
        predicted class is set to ``not_meat`` so the scan halts and the
        operator can re-place the sample.
        """
        try:
            second_probabilities = self._predict_probabilities(second_image_path)
        except Exception as exc:
            raise MeatClassifierLoadError(str(exc)) from exc

        second = self._build_result(second_probabilities, log_prefix="Meat classification (2nd pass)")

        if first.predicted_class != second.predicted_class:
            LOGGER.warning(
                "Two-pass meat verification disagreed | first=%s (%.4f) second=%s (%.4f)",
                first.predicted_class, first.confidence,
                second.predicted_class, second.confidence,
            )
            not_meat_label = getattr(config, "MEAT_CLASSIFIER_NOT_MEAT_LABEL", "not_meat")
            class_probabilities = {
                label: (first.class_probabilities.get(label, 0.0) + second.class_probabilities.get(label, 0.0)) / 2.0
                for label in set(first.class_probabilities) | set(second.class_probabilities)
            }
            return MeatClassificationResult(
                predicted_class=not_meat_label,
                confidence=min(first.confidence, second.confidence),
                class_probabilities=class_probabilities,
                is_valid_meat=False,
                hybrid_meat_type=None,
            )

        # Both passes agree -> average the probability vectors and rebuild.
        averaged = (np.asarray(list(self._aligned_probs(first))) + second_probabilities) / 2.0
        combined = self._build_result(averaged, log_prefix="Meat classification (combined)")
        return combined

    def _aligned_probs(self, result: MeatClassificationResult) -> list[float]:
        """Return ``result``'s class probabilities ordered like ``class_names``."""
        return [float(result.class_probabilities.get(name, 0.0)) for name in self.artifacts.class_names]
