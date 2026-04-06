from __future__ import annotations

import csv
import json
import logging
import math
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
    raw_sensor_values: dict[str, float]
    image_feature_preview: dict[str, float]


class HybridFreshnessPredictor:
    def __init__(self, artifacts_dir: Path | None = None) -> None:
        self.mode = getattr(config, "MODEL_MODE", "hybrid")
        self.artifacts_dir = artifacts_dir or config.MODEL_DIR
        self.target_classes = ["Fresh", "Neutral", "Spoiled"]

        self.image_model = None
        self.image_label_encoder = None
        self.image_preprocessor = None
        self.image_feature_columns: list[str] = []
        self.image_metadata: dict[str, Any] = {}

        self.sensor_metadata: dict[str, Any] = {}
        self.sensor_summary_columns: list[str] = []
        self.sensor_centroids: dict[tuple[str, str], dict[str, float]] = {}
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        try:
            # Image model powers image_only mode and the image branch of hybrid mode.
            if self.mode in {"image_only", "hybrid"}:
                image_dir = config.MODAL_RUNS_DIR / "image_only"
                image_model_path = image_dir / "image_only_freshness_model.joblib"
                image_encoder_path = config.MODAL_RUNS_DIR / "freshness_label_encoder.joblib"
                image_metadata_path = image_dir / "training_metadata.json"

                self.image_model = joblib.load(image_model_path)
                self.image_label_encoder = joblib.load(image_encoder_path)
                self.image_metadata = json.loads(image_metadata_path.read_text(encoding="utf-8"))
                self.image_preprocessor = self.image_model.named_steps["preprocessor"]
                self.image_feature_columns = list(self.image_preprocessor.feature_names_in_)
                self.target_classes = list(self.image_label_encoder.classes_)

            # Sensor metadata / centroids power sensor_only mode and the sensor branch of hybrid mode.
            sensor_dir = config.MODAL_RUNS_DIR / "sensor_only"
            sensor_metadata_path = sensor_dir / "training_metadata.json"
            self.sensor_metadata = json.loads(sensor_metadata_path.read_text(encoding="utf-8"))
            self.sensor_summary_columns = [
                column
                for column in self.sensor_metadata.get("feature_columns", [])
                if column.startswith("sensor_")
            ]
            self.sensor_centroids = self._load_sensor_centroids()

            LOGGER.info("Prediction artifacts loaded | mode=%s", self.mode)
        except Exception as exc:
            raise PredictionLoadError(f"Failed to load model artifacts: {exc}") from exc

    def _resolve_sensor_dataset_path(self) -> Path:
        candidates = [
            config.MODAL_RUNS_DIR / "source_hybrid_dataset.csv",
            config.MODEL_DIR / "hybrid_dataset.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError("Could not find hybrid_dataset.csv for sensor centroid loading.")

    def _load_sensor_centroids(self) -> dict[tuple[str, str], dict[str, float]]:
        dataset_path = self._resolve_sensor_dataset_path()
        frame = pd.read_csv(dataset_path)
        required_columns = ["meat_type", "freshness_label", *self.sensor_summary_columns]
        frame = frame.loc[:, required_columns]
        grouped = frame.groupby(["meat_type", "freshness_label"], dropna=False)[self.sensor_summary_columns].mean()
        centroids: dict[tuple[str, str], dict[str, float]] = {}
        for (meat_type, freshness_label), row in grouped.iterrows():
            centroids[(str(meat_type), str(freshness_label))] = {
                column: float(row[column]) for column in self.sensor_summary_columns
            }
        return centroids

    def _normalize_sensor_values(self, sensor_values: dict[str, Any]) -> dict[str, float]:
        normalized = {
            "nh3_ratio": sensor_values.get("nh3_ratio"),
            "nh3_ratio_raw": sensor_values.get("nh3_ratio_raw", sensor_values.get("nh3_ratio")),
            "h2s_ratio": sensor_values.get("h2s_ratio"),
            "h2s_ratio_raw": sensor_values.get("h2s_ratio_raw", sensor_values.get("h2s_ratio")),
            "voc_ratio": sensor_values.get("voc_ratio"),
            "voc_ratio_raw": sensor_values.get("voc_ratio_raw", sensor_values.get("voc_ratio")),
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

    def _build_sensor_summary(self, sensor_values: dict[str, Any]) -> dict[str, float]:
        normalized_sensor_values = self._normalize_sensor_values(sensor_values)
        direct_sensor_summary = {
            key: float(value)
            for key, value in sensor_values.items()
            if key.startswith("sensor_") and value is not None
        }
        if direct_sensor_summary:
            return direct_sensor_summary
        return sensor_values_to_summary_features(
            normalized_sensor_values,
            base_sensor_columns=self.sensor_metadata.get("sensor_base_columns", DEFAULT_SENSOR_COLUMNS),
        )

    def _predict_image_probabilities(self, image_path: str | Path, meat_type: str) -> tuple[dict[str, float], dict[str, float]]:
        if self.image_model is None or self.image_label_encoder is None:
            raise PredictionLoadError("Image model is not loaded.")

        image_features = extract_live_image_features(image_path)
        row: dict[str, Any] = {column: np.nan for column in self.image_feature_columns}
        row.update({key: value for key, value in image_features.items() if key in row})
        row["meat_type"] = meat_type
        feature_frame = pd.DataFrame([row], columns=self.image_feature_columns)

        classifier = self.image_model.named_steps["classifier"]
        if hasattr(classifier, "predict_proba"):
            probabilities = self.image_model.predict_proba(feature_frame)[0]
            return (
                {
                    str(label): float(probability)
                    for label, probability in zip(self.image_label_encoder.classes_, probabilities)
                },
                image_features,
            )

        decision_function = getattr(self.image_model, "decision_function", None)
        if callable(decision_function):
            decision_values = np.asarray(decision_function(feature_frame)).reshape(1, -1)
            shifted = decision_values - decision_values.max(axis=1, keepdims=True)
            softmax_scores = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
            softmax_scores = softmax_scores[0]
            return (
                {
                    str(label): float(score)
                    for label, score in zip(self.image_label_encoder.classes_, softmax_scores)
                },
                image_features,
            )

        predicted_index = int(self.image_model.predict(feature_frame)[0])
        predicted_label = str(self.image_label_encoder.inverse_transform([predicted_index])[0])
        return ({label: 1.0 if label == predicted_label else 0.0 for label in self.target_classes}, image_features)

    def _predict_sensor_probabilities(self, meat_type: str, sensor_values: dict[str, Any]) -> dict[str, float]:
        sensor_summary = self._build_sensor_summary(sensor_values)
        distances: dict[str, float] = {}
        for label in self.target_classes:
            centroid = self.sensor_centroids.get((meat_type, label))
            if centroid is None:
                distances[label] = float("inf")
                continue
            distance = math.sqrt(
                sum(
                    (float(sensor_summary.get(column, np.nan)) - float(centroid[column])) ** 2
                    for column in self.sensor_summary_columns
                    if not pd.isna(sensor_summary.get(column, np.nan)) and not pd.isna(centroid[column])
                )
            )
            distances[label] = distance

        scores = {label: 0.0 for label in self.target_classes}
        finite_labels = [label for label, distance in distances.items() if math.isfinite(distance)]
        if not finite_labels:
            return scores

        inv_scores = {label: 1.0 / (distances[label] + 1e-6) for label in finite_labels}
        total = sum(inv_scores.values())
        for label in finite_labels:
            scores[label] = inv_scores[label] / total
        return scores

    def _fuse_probabilities(
        self,
        image_probs: dict[str, float] | None,
        sensor_probs: dict[str, float] | None,
    ) -> tuple[dict[str, float], str]:
        if self.mode == "image_only":
            return (image_probs or {label: 0.0 for label in self.target_classes}, "Image-only model probability output.")
        if self.mode == "sensor_only":
            return (sensor_probs or {label: 0.0 for label in self.target_classes}, "Sensor nearest-class scores from dataset centroids.")

        image_probs = image_probs or {label: 0.0 for label in self.target_classes}
        sensor_probs = sensor_probs or {label: 0.0 for label in self.target_classes}
        image_weight = float(getattr(config, "HYBRID_IMAGE_WEIGHT", 0.65))
        sensor_weight = float(getattr(config, "HYBRID_SENSOR_WEIGHT", 0.35))
        total_weight = max(image_weight + sensor_weight, 1e-9)

        fused = {
            label: ((image_weight * image_probs.get(label, 0.0)) + (sensor_weight * sensor_probs.get(label, 0.0))) / total_weight
            for label in self.target_classes
        }
        return (
            fused,
            "Hybrid fusion of image-model probabilities and sensor nearest-class scores.",
        )

    def predict(self, image_path: str | Path, meat_type: str, sensor_values: dict[str, Any]) -> LivePredictionResult:
        normalized_sensor_values = self._normalize_sensor_values(sensor_values)
        image_probabilities: dict[str, float] | None = None
        sensor_probabilities: dict[str, float] | None = None
        image_features: dict[str, float] = {}

        if self.mode in {"image_only", "hybrid"}:
            image_probabilities, image_features = self._predict_image_probabilities(image_path, meat_type)
        if self.mode in {"sensor_only", "hybrid"}:
            sensor_probabilities = self._predict_sensor_probabilities(meat_type, sensor_values)

        class_probabilities, confidence_note = self._fuse_probabilities(image_probabilities, sensor_probabilities)
        predicted_label = max(class_probabilities, key=class_probabilities.get)
        confidence = float(class_probabilities[predicted_label]) if class_probabilities else None

        return LivePredictionResult(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            meat_type=meat_type,
            image_path=str(image_path),
            predicted_freshness=predicted_label,
            confidence=confidence,
            confidence_note=confidence_note,
            class_probabilities=class_probabilities,
            sensor_values=normalized_sensor_values,
            raw_sensor_values={
                "nh3_ratio_raw": normalized_sensor_values["nh3_ratio_raw"],
                "h2s_ratio_raw": normalized_sensor_values["h2s_ratio_raw"],
                "voc_ratio_raw": normalized_sensor_values["voc_ratio_raw"],
            },
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
                    "model_mode",
                    "meat_type",
                    "image_path",
                    "nh3_ratio_raw",
                    "h2s_ratio_raw",
                    "voc_ratio_raw",
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
                    "model_mode": self.mode,
                    "meat_type": result.meat_type,
                    "image_path": result.image_path,
                    "nh3_ratio_raw": result.raw_sensor_values["nh3_ratio_raw"],
                    "h2s_ratio_raw": result.raw_sensor_values["h2s_ratio_raw"],
                    "voc_ratio_raw": result.raw_sensor_values["voc_ratio_raw"],
                    "nh3_ratio": result.sensor_values["nh3_ratio"],
                    "h2s_ratio": result.sensor_values["h2s_ratio"],
                    "voc_ratio": result.sensor_values["voc_ratio"],
                    "predicted_freshness": result.predicted_freshness,
                    "confidence": result.confidence,
                    "confidence_note": result.confidence_note,
                }
            )
        LOGGER.info(
            "Prediction logged | mode=%s | %s | %s",
            self.mode,
            result.predicted_freshness,
            config.PREDICTION_LOG_PATH,
        )
