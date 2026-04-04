from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from hybrid_pipeline_utils import extract_image_features, sensor_values_to_summary_features


REQUIRED_RATIO_COLUMNS = ("nh3_ratio", "h2s_ratio", "voc_ratio")
OPTIONAL_RAW_COLUMNS = (
    "nh3_v",
    "nh3_rs",
    "h2s_v",
    "h2s_rs",
    "voc_v",
    "voc_rs",
)
DEFAULT_STABILITY_STD_LIMITS = {
    "nh3_ratio": 0.03,
    "h2s_ratio": 0.03,
    "voc_ratio": 0.05,
}


@dataclass
class DeploymentConfig:
    warmup_seconds: int = 180
    stabilization_samples_min: int = 20
    stabilization_samples_max: int = 50
    stability_std_limits: dict[str, float] | None = None
    include_baseline_in_model_input: bool = False

    def resolved_stability_std_limits(self) -> dict[str, float]:
        return self.stability_std_limits or DEFAULT_STABILITY_STD_LIMITS.copy()


@dataclass
class SensorWindowSummary:
    sample_count: int
    averaged_values: dict[str, float]
    std_values: dict[str, float]
    stable: bool
    stability_reasons: list[str]


@dataclass
class PredictionResult:
    timestamp_utc: str
    state: str
    meat_type: str
    image_path: str
    predicted_freshness: str
    confidence: float | None
    class_probabilities: dict[str, float]
    averaged_sensor_values: dict[str, float]
    sensor_std_values: dict[str, float]
    baseline_snapshot: dict[str, Any] | None


class HybridDeploymentRuntime:
    def __init__(self, artifacts_dir: str | Path, config: DeploymentConfig | None = None) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.config = config or DeploymentConfig()
        self.model = joblib.load(self.artifacts_dir / "hybrid_freshness_model.joblib")
        self.label_encoder = joblib.load(self.artifacts_dir / "freshness_label_encoder.joblib")
        self.metadata = json.loads((self.artifacts_dir / "training_metadata.json").read_text(encoding="utf-8"))
        self.preprocessor = self.model.named_steps["preprocessor"]
        self.feature_columns = list(self.preprocessor.feature_names_in_)
        self.runtime_dir = self.artifacts_dir / "runtime"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.prediction_log_path = self.runtime_dir / "prediction_log.jsonl"
        self.baseline_path = self.runtime_dir / "baseline_debug.json"
        self.baseline_snapshot: dict[str, Any] | None = None

    def warmup_remaining_seconds(self, elapsed_seconds: float) -> float:
        return max(0.0, float(self.config.warmup_seconds) - float(elapsed_seconds))

    def is_warmed_up(self, elapsed_seconds: float) -> bool:
        return self.warmup_remaining_seconds(elapsed_seconds) <= 0.0

    def summarize_sensor_window(self, sensor_rows: list[dict[str, Any]]) -> SensorWindowSummary:
        frame = pd.DataFrame(sensor_rows)
        if frame.empty:
            raise ValueError("No sensor rows were provided.")

        limits = self.config.resolved_stability_std_limits()
        reasons: list[str] = []

        if len(frame) < self.config.stabilization_samples_min:
            reasons.append(
                f"Need at least {self.config.stabilization_samples_min} samples; got {len(frame)}."
            )
        if len(frame) > self.config.stabilization_samples_max:
            frame = frame.iloc[-self.config.stabilization_samples_max :].reset_index(drop=True)

        averaged: dict[str, float] = {}
        std_values: dict[str, float] = {}

        for column in (*REQUIRED_RATIO_COLUMNS, *OPTIONAL_RAW_COLUMNS):
            if column in frame.columns:
                series = pd.to_numeric(frame[column], errors="coerce").dropna()
                if not series.empty:
                    averaged[column] = float(series.mean())
                    std_values[column] = float(series.std(ddof=0))

        for column in REQUIRED_RATIO_COLUMNS:
            if column not in averaged:
                reasons.append(f"Missing required sensor column: {column}.")
                continue
            limit = limits.get(column, 0.03)
            observed_std = std_values.get(column, np.inf)
            if observed_std > limit:
                reasons.append(
                    f"{column} std {observed_std:.4f} exceeds stability limit {limit:.4f}."
                )

        stable = len(reasons) == 0
        return SensorWindowSummary(
            sample_count=int(len(frame)),
            averaged_values=averaged,
            std_values=std_values,
            stable=stable,
            stability_reasons=reasons,
        )

    def capture_baseline(self, sensor_rows: list[dict[str, Any]]) -> SensorWindowSummary:
        summary = self.summarize_sensor_window(sensor_rows)
        self.baseline_snapshot = {
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "sample_count": summary.sample_count,
            "averaged_values": summary.averaged_values,
            "std_values": summary.std_values,
            "stable": summary.stable,
            "stability_reasons": summary.stability_reasons,
        }
        self.baseline_path.write_text(json.dumps(self.baseline_snapshot, indent=2), encoding="utf-8")
        return summary

    def build_feature_frame(
        self,
        *,
        image_path: str | Path,
        meat_type: str,
        averaged_sensor_values: dict[str, float],
    ) -> pd.DataFrame:
        image_features = extract_image_features(image_path)
        sensor_features = sensor_values_to_summary_features(
            {
                "nh3_ratio": averaged_sensor_values.get("nh3_ratio"),
                "h2s_ratio": averaged_sensor_values.get("h2s_ratio"),
                "voc_ratio": averaged_sensor_values.get("voc_ratio"),
                "nh3_v": averaged_sensor_values.get("nh3_v"),
                "nh3_rs": averaged_sensor_values.get("nh3_rs"),
                "h2s_v": averaged_sensor_values.get("h2s_v"),
                "h2s_rs": averaged_sensor_values.get("h2s_rs"),
                "voc_v": averaged_sensor_values.get("voc_v"),
                "voc_rs": averaged_sensor_values.get("voc_rs"),
            },
            base_sensor_columns=self.metadata["sensor_base_columns"],
        )

        row: dict[str, Any] = {}
        row.update(image_features)
        row.update(sensor_features)
        row["meat_type"] = meat_type

        # Exact compatibility with training: reindex to the same feature names that
        # the fitted preprocessor saw during training. This keeps inference stable
        # even as the deployment app becomes safer and more feature-rich internally.
        feature_frame = pd.DataFrame([row]).reindex(columns=self.feature_columns, fill_value=np.nan)
        return feature_frame

    def predict_from_window(
        self,
        *,
        image_path: str | Path,
        meat_type: str,
        sensor_rows: list[dict[str, Any]],
        warmup_elapsed_seconds: float,
    ) -> PredictionResult:
        if not self.is_warmed_up(warmup_elapsed_seconds):
            remaining = self.warmup_remaining_seconds(warmup_elapsed_seconds)
            raise RuntimeError(
                f"System still warming up. {remaining:.1f} seconds remaining before prediction is allowed."
            )

        summary = self.summarize_sensor_window(sensor_rows)
        if not summary.stable:
            raise RuntimeError(
                "Sensors are not stable enough for prediction: " + "; ".join(summary.stability_reasons)
            )

        feature_frame = self.build_feature_frame(
            image_path=image_path,
            meat_type=meat_type,
            averaged_sensor_values=summary.averaged_values,
        )

        predicted_index = int(self.model.predict(feature_frame)[0])
        predicted_label = str(self.label_encoder.inverse_transform([predicted_index])[0])

        class_probabilities: dict[str, float] = {}
        confidence: float | None = None
        classifier = self.model.named_steps["classifier"]
        if hasattr(classifier, "predict_proba"):
            probabilities = self.model.predict_proba(feature_frame)[0]
            class_probabilities = {
                str(label): float(probability)
                for label, probability in zip(self.label_encoder.classes_, probabilities)
            }
            confidence = max(class_probabilities.values())

        result = PredictionResult(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            state="predicted",
            meat_type=meat_type,
            image_path=str(image_path),
            predicted_freshness=predicted_label,
            confidence=confidence,
            class_probabilities=class_probabilities,
            averaged_sensor_values=summary.averaged_values,
            sensor_std_values=summary.std_values,
            baseline_snapshot=self.baseline_snapshot,
        )
        self.append_prediction_log(result)
        return result

    def append_prediction_log(self, result: PredictionResult) -> None:
        with self.prediction_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(result), ensure_ascii=True) + "\n")


def load_sensor_rows_from_csv(
    csv_path: str | Path,
    *,
    limit: int | None = None,
    from_tail: bool = True,
) -> list[dict[str, Any]]:
    csv_path = Path(csv_path)
    frame = pd.read_csv(csv_path)
    if limit is not None and limit > 0:
        frame = frame.tail(limit) if from_tail else frame.head(limit)
    return frame.to_dict(orient="records")


def format_sensor_debug(summary: SensorWindowSummary) -> str:
    lines = [
        f"Sample count: {summary.sample_count}",
        "Averaged sensor values:",
    ]
    for key in (
        "nh3_v",
        "nh3_rs",
        "nh3_ratio",
        "h2s_v",
        "h2s_rs",
        "h2s_ratio",
        "voc_v",
        "voc_rs",
        "voc_ratio",
    ):
        if key in summary.averaged_values:
            avg = summary.averaged_values[key]
            std = summary.std_values.get(key, 0.0)
            lines.append(f"  {key}: avg={avg:.4f}, std={std:.4f}")
    lines.append(f"Stable: {summary.stable}")
    if summary.stability_reasons:
        lines.append("Stability notes:")
        for reason in summary.stability_reasons:
            lines.append(f"  - {reason}")
    return "\n".join(lines)
