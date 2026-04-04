from __future__ import annotations

import csv
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import config
from feature_extractor import extract_live_image_features


HYBRID_ML_DIR = Path(__file__).resolve().parents[1] / "hybrid_ml"
if str(HYBRID_ML_DIR) not in sys.path:
    sys.path.insert(0, str(HYBRID_ML_DIR))

from hybrid_pipeline_utils import DEFAULT_SENSOR_COLUMNS, sensor_values_to_summary_features


LOGGER = logging.getLogger(__name__)


class PredictionLoadError(RuntimeError):
    pass


@dataclass
class LivePredictionResult:
    timestamp_utc: str
    meat_type: str
    image_path: str
    predicted_freshness: str
    confidence: float | None
    confidence_note: str
    class_probabilities: dict[str, float]
    sensor_values: dict[str, float]
    image_feature_preview: dict[str, float]


class HybridFreshnessPredictor:
    def __init__(self, artifacts_dir: Path | None = None) -> None:
        self.artifacts_dir = artifacts_dir or config.MODEL_DIR
        self.model = None
        self.label_encoder = None
        self.metadata: dict[str, Any] = {}
        self.preprocessor = None
        self.feature_columns: list[str] = []
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        try:
            model_path = self.artifacts_dir / "hybrid_freshness_model.joblib"
            encoder_path = self.artifacts_dir / "freshness_label_encoder.joblib"
            metadata_path = self.artifacts_dir / "training_metadata.json"

            self.model = joblib.load(model_path)
            self.label_encoder = joblib.load(encoder_path)
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.preprocessor = self.model.named_steps["preprocessor"]
            self.feature_columns = list(self.preprocessor.feature_names_in_)
            LOGGER.info("Hybrid model artifacts loaded from %s", self.artifacts_dir)
        except Exception as exc:
            raise PredictionLoadError(f"Failed to load model artifacts: {exc}") from exc

    def _normalize_sensor_values(self, sensor_values: dict[str, Any]) -> dict[str, float]:
        normalized = {
            "nh3_ratio": sensor_values.get("nh3_ratio"),
            "h2s_ratio": sensor_values.get("h2s_ratio"),
            "voc_ratio": sensor_values.get("voc_ratio"),
            "nh3_v": sensor_values.get("nh3_v", sensor_values.get("nh3_voltage")),
            "nh3_rs": sensor_values.get("nh3_rs"),
            "h2s_v": sensor_values.get("h2s_v", sensor_values.get("h2s_voltage")),
            "h2s_rs": sensor_values.get("h2s_rs"),
            "voc_v": sensor_values.get("voc_v", sensor_values.get("voc_voltage")),
            "voc_rs": sensor_values.get("voc_rs"),
        }

        for key in ("nh3_ratio", "h2s_ratio", "voc_ratio"):
            if normalized[key] is None:
                raise ValueError(f"Missing required sensor value: {key}")

        return {key: (float(value) if value is not None else np.nan) for key, value in normalized.items()}

    def build_feature_frame(self, image_path: str | Path, meat_type: str, sensor_values: dict[str, Any]) -> pd.DataFrame:
        image_features = extract_live_image_features(image_path)
        normalized_sensor_values = self._normalize_sensor_values(sensor_values)
        sensor_summary = sensor_values_to_summary_features(
            normalized_sensor_values,
            base_sensor_columns=self.metadata.get("sensor_base_columns", DEFAULT_SENSOR_COLUMNS),
        )

        row: dict[str, Any] = {}
        row.update(image_features)
        row.update(sensor_summary)
        row["meat_type"] = meat_type

        return pd.DataFrame([row]).reindex(columns=self.feature_columns, fill_value=np.nan)

    def predict(self, image_path: str | Path, meat_type: str, sensor_values: dict[str, Any]) -> LivePredictionResult:
        feature_frame = self.build_feature_frame(image_path, meat_type, sensor_values)
        normalized_sensor_values = self._normalize_sensor_values(sensor_values)
        image_features = extract_live_image_features(image_path)

        predicted_index = int(self.model.predict(feature_frame)[0])
        predicted_label = str(self.label_encoder.inverse_transform([predicted_index])[0])

        class_probabilities: dict[str, float] = {}
        confidence: float | None = None
        confidence_note = "Probability is not available for the current trained model."

        classifier = self.model.named_steps["classifier"]
        if hasattr(classifier, "predict_proba"):
            probabilities = self.model.predict_proba(feature_frame)[0]
            class_probabilities = {
                str(label): float(probability)
                for label, probability in zip(self.label_encoder.classes_, probabilities)
            }
            confidence = max(class_probabilities.values())
            confidence_note = "Model probability output."
        else:
            decision_function = getattr(self.model, "decision_function", None)
            if callable(decision_function):
                try:
                    decision_values = np.asarray(decision_function(feature_frame)).reshape(1, -1)
                    shifted = decision_values - decision_values.max(axis=1, keepdims=True)
                    softmax_scores = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
                    softmax_scores = softmax_scores[0]
                    class_probabilities = {
                        str(label): float(score)
                        for label, score in zip(self.label_encoder.classes_, softmax_scores)
                    }
                    confidence = max(class_probabilities.values())
                    confidence_note = "Softmax-normalized decision scores (approximate confidence)."
                except Exception:
                    pass

        return LivePredictionResult(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            meat_type=meat_type,
            image_path=str(image_path),
            predicted_freshness=predicted_label,
            confidence=confidence,
            confidence_note=confidence_note,
            class_probabilities=class_probabilities,
            sensor_values=normalized_sensor_values,
            image_feature_preview={
                key: float(image_features[key])
                for key in (
                    "img_rgb_r_mean",
                    "img_rgb_g_mean",
                    "img_rgb_b_mean",
                    "img_gray_mean",
                    "img_edge_density",
                )
                if key in image_features
            },
        )

    def append_prediction_log(self, result: LivePredictionResult) -> None:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_exists = config.PREDICTION_LOG_PATH.exists()
        with config.PREDICTION_LOG_PATH.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "timestamp_utc",
                    "meat_type",
                    "image_path",
                    "nh3_ratio",
                    "h2s_ratio",
                    "voc_ratio",
                    "predicted_freshness",
                    "confidence",
                    "confidence_note",
                ],
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp_utc": result.timestamp_utc,
                    "meat_type": result.meat_type,
                    "image_path": result.image_path,
                    "nh3_ratio": result.sensor_values["nh3_ratio"],
                    "h2s_ratio": result.sensor_values["h2s_ratio"],
                    "voc_ratio": result.sensor_values["voc_ratio"],
                    "predicted_freshness": result.predicted_freshness,
                    "confidence": result.confidence,
                    "confidence_note": result.confidence_note,
                }
            )
        LOGGER.info(
            "Prediction logged | %s | %s | %s",
            result.meat_type,
            result.predicted_freshness,
            config.PREDICTION_LOG_PATH,
        )
