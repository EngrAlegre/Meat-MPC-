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
        self.image_target_classes: list[str] = []
        self.image_metadata: dict[str, Any] = {}

        self.sensor_model = None
        self.sensor_label_encoder = None
        self.sensor_preprocessor = None
        self.sensor_feature_columns: list[str] = []
        self.sensor_categorical_columns: list[str] = []
        self.sensor_target_classes: list[str] = []
        self.sensor_metadata: dict[str, Any] = {}
        self.sensor_summary_columns: list[str] = []
        self.sensor_centroids: dict[tuple[str, str], dict[str, float]] = {}
        self._load_artifacts()

    def _load_shared_label_encoder(self) -> None:
        shared_encoder_path = config.MODAL_RUNS_DIR / "freshness_label_encoder.joblib"
        if not shared_encoder_path.exists():
            return
        shared_encoder = joblib.load(shared_encoder_path)
        shared_classes = [str(label) for label in shared_encoder.classes_]
        if shared_classes:
            self.target_classes = shared_classes

    def _load_artifacts(self) -> None:
        try:
            self._load_shared_label_encoder()

            # Image model powers image_only mode and the image branch of hybrid mode.
            if self.mode in {"image_only", "hybrid"}:
                image_dir = config.MODAL_RUNS_DIR / "image_only"
                image_model_path = image_dir / "image_only_freshness_model.joblib"
                image_metadata_path = image_dir / "training_metadata.json"
                image_encoder_path = image_dir / "image_only_label_encoder.joblib"
                if not image_encoder_path.exists():
                    image_encoder_path = config.MODAL_RUNS_DIR / "freshness_label_encoder.joblib"

                self.image_model = joblib.load(image_model_path)
                self.image_label_encoder = joblib.load(image_encoder_path)
                self.image_metadata = json.loads(image_metadata_path.read_text(encoding="utf-8"))
                self.image_preprocessor = self.image_model.named_steps["preprocessor"]
                self.image_feature_columns = list(self.image_preprocessor.feature_names_in_)
                self.image_target_classes = [str(label) for label in self.image_label_encoder.classes_]

            # Sensor model powers sensor_only mode and the sensor branch of hybrid mode.
            sensor_dir = config.MODAL_RUNS_DIR / "sensor_only"
            sensor_metadata_path = sensor_dir / "training_metadata.json"
            sensor_model_path = sensor_dir / "sensor_only_freshness_model.joblib"
            sensor_encoder_path = sensor_dir / "sensor_only_label_encoder.joblib"
            self.sensor_metadata = json.loads(sensor_metadata_path.read_text(encoding="utf-8"))

            if sensor_model_path.exists() and sensor_encoder_path.exists():
                self.sensor_model = joblib.load(sensor_model_path)
                self.sensor_label_encoder = joblib.load(sensor_encoder_path)
                self.sensor_preprocessor = self.sensor_model.named_steps["preprocessor"]
                self.sensor_feature_columns = list(self.sensor_preprocessor.feature_names_in_)
                self.sensor_categorical_columns = [
                    str(column)
                    for column in self.sensor_metadata.get("categorical_columns", [])
                ]
                self.sensor_target_classes = [str(label) for label in self.sensor_label_encoder.classes_]
                self.sensor_summary_columns = [
                    column
                    for column in self.sensor_feature_columns
                    if column.startswith("sensor_")
                ]
            else:
                # Fallback to the legacy centroid-based sensor branch when no
                # trained sensor model is available.
                self.sensor_summary_columns = [
                    column
                    for column in self.sensor_metadata.get("feature_columns", [])
                    if column.startswith("sensor_") and not column.startswith("sensor_voc_")
                ]
                self.sensor_centroids = self._load_sensor_centroids()
                self.sensor_target_classes = list(self.target_classes)

            LOGGER.info(
                "Prediction artifacts loaded | mode=%s | image_classes=%s | sensor_classes=%s",
                self.mode,
                self.image_target_classes or "-",
                self.sensor_target_classes or "-",
            )
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
            and not key.startswith("sensor_voc_")
        }
        if direct_sensor_summary:
            return direct_sensor_summary
        return sensor_values_to_summary_features(
            normalized_sensor_values,
            base_sensor_columns=self.sensor_metadata.get("sensor_base_columns", DEFAULT_SENSOR_COLUMNS),
        )

    def _pad_to_target_classes(
        self,
        raw_probs: dict[str, float],
    ) -> dict[str, float]:
        return {label: float(raw_probs.get(label, 0.0)) for label in self.target_classes}

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
            raw = {
                str(label): float(probability)
                for label, probability in zip(self.image_label_encoder.classes_, probabilities)
            }
            return self._pad_to_target_classes(raw), image_features

        decision_function = getattr(self.image_model, "decision_function", None)
        if callable(decision_function):
            decision_values = np.asarray(decision_function(feature_frame)).reshape(1, -1)
            shifted = decision_values - decision_values.max(axis=1, keepdims=True)
            softmax_scores = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
            softmax_scores = softmax_scores[0]
            raw = {
                str(label): float(score)
                for label, score in zip(self.image_label_encoder.classes_, softmax_scores)
            }
            return self._pad_to_target_classes(raw), image_features

        predicted_index = int(self.image_model.predict(feature_frame)[0])
        predicted_label = str(self.image_label_encoder.inverse_transform([predicted_index])[0])
        raw = {label: 1.0 if label == predicted_label else 0.0 for label in self.image_target_classes or self.target_classes}
        return self._pad_to_target_classes(raw), image_features

    def _predict_sensor_probabilities(self, meat_type: str, sensor_values: dict[str, Any]) -> dict[str, float]:
        if self.sensor_model is not None and self.sensor_label_encoder is not None:
            sensor_summary = self._build_sensor_summary(sensor_values)
            row: dict[str, Any] = {column: np.nan for column in self.sensor_feature_columns}
            for column in self.sensor_feature_columns:
                if column in sensor_summary:
                    row[column] = sensor_summary[column]
            if "meat_type" in self.sensor_feature_columns:
                row["meat_type"] = meat_type
            feature_frame = pd.DataFrame([row], columns=self.sensor_feature_columns)

            classifier = self.sensor_model.named_steps["classifier"]
            if hasattr(classifier, "predict_proba"):
                probabilities = self.sensor_model.predict_proba(feature_frame)[0]
                raw = {
                    str(label): float(probability)
                    for label, probability in zip(self.sensor_label_encoder.classes_, probabilities)
                }
            else:
                decision_function = getattr(self.sensor_model, "decision_function", None)
                if callable(decision_function):
                    decision_values = np.asarray(decision_function(feature_frame)).reshape(1, -1)
                    shifted = decision_values - decision_values.max(axis=1, keepdims=True)
                    softmax_scores = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
                    softmax_scores = softmax_scores[0]
                    raw = {
                        str(label): float(score)
                        for label, score in zip(self.sensor_label_encoder.classes_, softmax_scores)
                    }
                else:
                    predicted_index = int(self.sensor_model.predict(feature_frame)[0])
                    predicted_label = str(self.sensor_label_encoder.inverse_transform([predicted_index])[0])
                    raw = {label: 1.0 if label == predicted_label else 0.0 for label in self.sensor_target_classes}

            return self._pad_to_target_classes(raw)

        # Legacy fallback: centroid-based sensor branch.
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
            return (
                image_probs or {label: 0.0 for label in self.target_classes},
                "Image-only model probability output.",
            )
        if self.mode == "sensor_only":
            return (
                sensor_probs or {label: 0.0 for label in self.target_classes},
                "Sensor-only model probability output.",
            )

        image_probs = image_probs or {label: 0.0 for label in self.target_classes}
        sensor_probs = sensor_probs or {label: 0.0 for label in self.target_classes}
        image_weight = float(getattr(config, "HYBRID_IMAGE_WEIGHT", 0.65))
        sensor_weight = float(getattr(config, "HYBRID_SENSOR_WEIGHT", 0.35))
        total_weight = max(image_weight + sensor_weight, 1e-9)

        image_supports = set(self.image_target_classes) if self.image_target_classes else set(self.target_classes)

        fused: dict[str, float] = {}
        for label in self.target_classes:
            if label in image_supports:
                fused[label] = (
                    (image_weight * image_probs.get(label, 0.0))
                    + (sensor_weight * sensor_probs.get(label, 0.0))
                ) / total_weight
            else:
                # Labels that the image model cannot produce (e.g. Neutral when
                # the image head was trained on Fresh/Spoiled only) rely on the
                # sensor branch directly.
                fused[label] = float(sensor_probs.get(label, 0.0))

        total = sum(fused.values())
        if total > 0:
            fused = {label: score / total for label, score in fused.items()}
        return (
            fused,
            "Hybrid fusion: Fresh/Spoiled from image+sensor, Neutral from sensor only.",
        )

    def _apply_spoiled_override(
        self,
        *,
        class_probabilities: dict[str, float],
        normalized_sensor_values: dict[str, float],
        confidence_note: str,
    ) -> tuple[dict[str, float], str, bool]:
        if not getattr(config, "SPOILED_OVERRIDE_ENABLED", False):
            return class_probabilities, confidence_note, False

        legacy_threshold = float(getattr(config, "SPOILED_OVERRIDE_RATIO_THRESHOLD", 0.30))
        nh3_max = float(getattr(config, "SPOILED_OVERRIDE_NH3_MAX", legacy_threshold))
        h2s_max = float(getattr(config, "SPOILED_OVERRIDE_H2S_MAX", legacy_threshold))
        nh3_ratio = float(normalized_sensor_values["nh3_ratio"])
        h2s_ratio = float(normalized_sensor_values["h2s_ratio"])
        if nh3_ratio > nh3_max and h2s_ratio > h2s_max:
            return class_probabilities, confidence_note, False

        overridden = {label: 0.0 for label in self.target_classes}
        overridden["Spoiled"] = 1.0
        override_note = (
            f"Spoiled override applied because NH3<= {nh3_max:.2f} (got {nh3_ratio:.2f}) "
            f"or H2S<= {h2s_max:.2f} (got {h2s_ratio:.2f})."
        )
        return overridden, override_note, True

    def _apply_fresh_override(
        self,
        *,
        class_probabilities: dict[str, float],
        normalized_sensor_values: dict[str, float],
        confidence_note: str,
    ) -> tuple[dict[str, float], str, bool]:
        if not getattr(config, "FRESH_OVERRIDE_ENABLED", False):
            return class_probabilities, confidence_note, False

        nh3_min = float(getattr(config, "FRESH_OVERRIDE_NH3_MIN", 0.85))
        h2s_min = float(getattr(config, "FRESH_OVERRIDE_H2S_MIN", 0.55))
        nh3_ratio = float(normalized_sensor_values["nh3_ratio"])
        h2s_ratio = float(normalized_sensor_values["h2s_ratio"])
        if nh3_ratio < nh3_min or h2s_ratio < h2s_min:
            return class_probabilities, confidence_note, False

        overridden = {label: 0.0 for label in self.target_classes}
        overridden["Fresh"] = 1.0
        override_note = (
            f"Fresh override applied because NH3>= {nh3_min:.2f} (got {nh3_ratio:.2f}) "
            f"and H2S>= {h2s_min:.2f} (got {h2s_ratio:.2f})."
        )
        return overridden, override_note, True

    def _apply_neutral_override(
        self,
        *,
        class_probabilities: dict[str, float],
        normalized_sensor_values: dict[str, float],
        confidence_note: str,
    ) -> tuple[dict[str, float], str, bool]:
        if not getattr(config, "NEUTRAL_OVERRIDE_ENABLED", False):
            return class_probabilities, confidence_note, False

        nh3_min = float(getattr(config, "NEUTRAL_OVERRIDE_NH3_MIN", 0.55))
        nh3_max = float(getattr(config, "NEUTRAL_OVERRIDE_NH3_MAX", 0.85))
        h2s_min = float(getattr(config, "NEUTRAL_OVERRIDE_H2S_MIN", 0.35))
        h2s_max = float(getattr(config, "NEUTRAL_OVERRIDE_H2S_MAX", 0.55))
        nh3_ratio = float(normalized_sensor_values["nh3_ratio"])
        h2s_ratio = float(normalized_sensor_values["h2s_ratio"])
        nh3_in_band = nh3_min <= nh3_ratio <= nh3_max
        h2s_in_band = h2s_min <= h2s_ratio <= h2s_max
        if not (nh3_in_band and h2s_in_band):
            return class_probabilities, confidence_note, False

        overridden = {label: 0.0 for label in self.target_classes}
        overridden["Neutral"] = 1.0
        override_note = (
            f"Neutral override applied because NH3 in [{nh3_min:.2f}, {nh3_max:.2f}] "
            f"(got {nh3_ratio:.2f}) and H2S in [{h2s_min:.2f}, {h2s_max:.2f}] "
            f"(got {h2s_ratio:.2f})."
        )
        return overridden, override_note, True

    def _apply_baseline_delta_override(
        self,
        *,
        class_probabilities: dict[str, float],
        normalized_sensor_values: dict[str, float],
        sensor_baseline: dict[str, Any] | None,
        confidence_note: str,
    ) -> tuple[dict[str, float], str, bool]:
        if not getattr(config, "BASELINE_DELTA_OVERRIDE_ENABLED", False):
            return class_probabilities, confidence_note, False
        if not sensor_baseline:
            return class_probabilities, confidence_note, False

        baseline_nh3 = sensor_baseline.get("nh3_ratio")
        baseline_h2s = sensor_baseline.get("h2s_ratio")
        if baseline_nh3 is None or baseline_h2s is None:
            return class_probabilities, confidence_note, False

        nh3_drop_limit = float(getattr(config, "BASELINE_DELTA_NH3_SPOILED_DROP", 0.08))
        h2s_drop_limit = float(getattr(config, "BASELINE_DELTA_H2S_SPOILED_DROP", 0.04))
        nh3_fresh_tol = float(getattr(config, "BASELINE_DELTA_NH3_FRESH_TOLERANCE", 0.06))
        h2s_fresh_tol = float(getattr(config, "BASELINE_DELTA_H2S_FRESH_TOLERANCE", 0.04))

        nh3_ratio = float(normalized_sensor_values["nh3_ratio"])
        h2s_ratio = float(normalized_sensor_values["h2s_ratio"])
        nh3_delta = nh3_ratio - float(baseline_nh3)
        h2s_delta = h2s_ratio - float(baseline_h2s)

        if nh3_delta <= -nh3_drop_limit or h2s_delta <= -h2s_drop_limit:
            chosen = "Spoiled"
            reason = (
                f"NH3={nh3_ratio:.2f}, H2S={h2s_ratio:.2f} indicate elevated gas activity"
            )
        elif abs(nh3_delta) <= nh3_fresh_tol and abs(h2s_delta) <= h2s_fresh_tol:
            chosen = "Fresh"
            reason = (
                f"NH3={nh3_ratio:.2f}, H2S={h2s_ratio:.2f} remained within the safe range"
            )
        else:
            chosen = "Neutral"
            reason = (
                f"NH3={nh3_ratio:.2f}, H2S={h2s_ratio:.2f} fell within the borderline range"
            )

        overridden = {label: 0.0 for label in self.target_classes}
        if chosen in overridden:
            overridden[chosen] = 1.0
        else:
            overridden[self.target_classes[0]] = 1.0
        override_note = f"Sensor monitor override -> {chosen} because {reason}."
        return overridden, override_note, True

    def _apply_presentation_override(
        self,
        *,
        class_probabilities: dict[str, float],
        meat_type: str,
        confidence_note: str,
    ) -> tuple[dict[str, float], str, bool]:
        if not getattr(config, "DEMO_PRESENTATION_OVERRIDE_ENABLED", False):
            return class_probabilities, confidence_note, False

        forced = getattr(config, "DEMO_PRESENTATION_FORCED_RESULT", None)
        per_meat = getattr(config, "DEMO_PRESENTATION_PER_MEAT", {}) or {}
        chosen: str | None = None
        if isinstance(forced, str) and forced in self.target_classes:
            chosen = forced
            why = "DEMO_PRESENTATION_FORCED_RESULT"
        else:
            mapped = per_meat.get(meat_type) or per_meat.get(str(meat_type).title())
            if isinstance(mapped, str) and mapped in self.target_classes:
                chosen = mapped
                why = f"DEMO_PRESENTATION_PER_MEAT[{meat_type}]"

        if chosen is None:
            return class_probabilities, confidence_note, False

        overridden = {label: 0.0 for label in self.target_classes}
        overridden[chosen] = 1.0
        override_note = f"Presentation override -> {chosen} (source: {why})."
        return overridden, override_note, True

    def _apply_low_confidence_neutral_override(
        self,
        *,
        class_probabilities: dict[str, float],
        confidence_note: str,
    ) -> tuple[dict[str, float], str]:
        if not getattr(config, "LOW_CONFIDENCE_NEUTRAL_OVERRIDE_ENABLED", False):
            return class_probabilities, confidence_note
        if not class_probabilities:
            return class_probabilities, confidence_note

        top_label = max(class_probabilities, key=class_probabilities.get)
        top_score = float(class_probabilities[top_label])
        threshold = float(getattr(config, "LOW_CONFIDENCE_NEUTRAL_THRESHOLD", 0.45))
        if top_score >= threshold:
            return class_probabilities, confidence_note

        overridden = {label: 0.0 for label in self.target_classes}
        overridden["Neutral"] = 1.0
        override_note = (
            f"Low-confidence neutral override applied because top score {top_score:.4f} "
            f"was below {threshold:.2f}. Original top class was {top_label}."
        )
        return overridden, override_note

    def _fmt_probs(self, probs: dict[str, float] | None) -> str:
        if not probs:
            return "N/A"
        return " | ".join(f"{label}={score:.4f}" for label, score in probs.items())

    def predict(
        self,
        image_path: str | Path,
        meat_type: str,
        sensor_values: dict[str, Any],
        sensor_baseline: dict[str, Any] | None = None,
    ) -> LivePredictionResult:
        normalized_sensor_values = self._normalize_sensor_values(sensor_values)
        image_probabilities: dict[str, float] | None = None
        sensor_probabilities: dict[str, float] | None = None
        image_features: dict[str, float] = {}

        LOGGER.info(
            "--- PREDICTION START | meat_type=%s | mode=%s ---",
            meat_type, self.mode,
        )
        LOGGER.info(
            "Sensor input (Rs/Ro) | NH3=%.4f H2S=%.4f VOC=%.4f",
            float(normalized_sensor_values["nh3_ratio"]),
            float(normalized_sensor_values["h2s_ratio"]),
            float(normalized_sensor_values["voc_ratio"]),
        )

        if self.mode in {"image_only", "hybrid"}:
            image_probabilities, image_features = self._predict_image_probabilities(image_path, meat_type)
            LOGGER.info("Image branch  | %s", self._fmt_probs(image_probabilities))
        if self.mode in {"sensor_only", "hybrid"}:
            sensor_probabilities = self._predict_sensor_probabilities(meat_type, sensor_values)
            LOGGER.info("Sensor branch | %s", self._fmt_probs(sensor_probabilities))

        class_probabilities, confidence_note = self._fuse_probabilities(image_probabilities, sensor_probabilities)
        fused_label = max(class_probabilities, key=class_probabilities.get)
        LOGGER.info("Fused result  | %s (top: %s=%.4f)", self._fmt_probs(class_probabilities), fused_label, class_probabilities[fused_label])

        pre_override_label = fused_label

        baseline_applied = False
        class_probabilities, confidence_note, baseline_applied = self._apply_baseline_delta_override(
            class_probabilities=class_probabilities,
            normalized_sensor_values=normalized_sensor_values,
            sensor_baseline=sensor_baseline,
            confidence_note=confidence_note,
        )

        spoiled_applied = False
        fresh_applied = False
        neutral_applied = False
        if not baseline_applied:
            class_probabilities, confidence_note, spoiled_applied = self._apply_spoiled_override(
                class_probabilities=class_probabilities,
                normalized_sensor_values=normalized_sensor_values,
                confidence_note=confidence_note,
            )
            if not spoiled_applied:
                class_probabilities, confidence_note, fresh_applied = self._apply_fresh_override(
                    class_probabilities=class_probabilities,
                    normalized_sensor_values=normalized_sensor_values,
                    confidence_note=confidence_note,
                )
            if not (spoiled_applied or fresh_applied):
                class_probabilities, confidence_note, neutral_applied = self._apply_neutral_override(
                    class_probabilities=class_probabilities,
                    normalized_sensor_values=normalized_sensor_values,
                    confidence_note=confidence_note,
                )
        if not (baseline_applied or spoiled_applied or fresh_applied or neutral_applied):
            class_probabilities, confidence_note = self._apply_low_confidence_neutral_override(
                class_probabilities=class_probabilities,
                confidence_note=confidence_note,
            )

        class_probabilities, confidence_note, presentation_applied = self._apply_presentation_override(
            class_probabilities=class_probabilities,
            meat_type=meat_type,
            confidence_note=confidence_note,
        )
        predicted_label = max(class_probabilities, key=class_probabilities.get)
        confidence = float(class_probabilities[predicted_label]) if class_probabilities else None

        if predicted_label != pre_override_label:
            LOGGER.info("Override applied | %s -> %s | %s", pre_override_label, predicted_label, confidence_note)
        LOGGER.info("--- PREDICTION FINAL | %s (%.4f) ---", predicted_label, confidence or 0.0)

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
