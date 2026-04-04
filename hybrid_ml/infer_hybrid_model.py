from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from hybrid_pipeline_utils import (
    DEFAULT_SENSOR_COLUMNS,
    extract_image_features,
    sensor_values_to_summary_features,
)


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    default_artifact_dir = root_dir / "model"

    parser = argparse.ArgumentParser(
        description="Run inference for one image + one MQ sensor reading row."
    )
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--meat-type", required=True, choices=["Chicken", "Beef", "Pork"])
    parser.add_argument("--nh3-ratio", type=float, required=True)
    parser.add_argument("--h2s-ratio", type=float, required=True)
    parser.add_argument("--voc-ratio", type=float, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=default_artifact_dir)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.artifacts_dir / "hybrid_freshness_model.joblib"
    encoder_path = args.artifacts_dir / "freshness_label_encoder.joblib"
    metadata_path = args.artifacts_dir / "training_metadata.json"

    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    image_features = extract_image_features(args.image_path)
    sensor_summary = sensor_values_to_summary_features(
        {
            "nh3_ratio": args.nh3_ratio,
            "h2s_ratio": args.h2s_ratio,
            "voc_ratio": args.voc_ratio,
        },
        base_sensor_columns=metadata.get("sensor_base_columns", DEFAULT_SENSOR_COLUMNS),
    )

    row = {}
    row.update(image_features)
    row.update(sensor_summary)
    row["meat_type"] = args.meat_type

    feature_row = pd.DataFrame([row])
    predicted_index = model.predict(feature_row)[0]
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]

    print(f"Predicted freshness: {predicted_label}")

    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        probabilities = model.predict_proba(feature_row)[0]
        print("Class probabilities:")
        for label, probability in zip(label_encoder.classes_, probabilities):
            print(f"  {label}: {probability:.4f}")


if __name__ == "__main__":
    main()
