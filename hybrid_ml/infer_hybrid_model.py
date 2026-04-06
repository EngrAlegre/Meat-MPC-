from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RASPI_DIR = ROOT_DIR / "raspi"
if str(RASPI_DIR) not in sys.path:
    sys.path.insert(0, str(RASPI_DIR))

import config as raspi_config
from predict_live import HybridFreshnessPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference for one image + one MQ sensor reading row."
    )
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--meat-type", required=True, choices=["Chicken", "Beef", "Pork"])
    parser.add_argument("--nh3-ratio", type=float, required=True)
    parser.add_argument("--h2s-ratio", type=float, required=True)
    parser.add_argument("--voc-ratio", type=float, required=True)
    parser.add_argument("--mode", choices=["sensor_only", "image_only", "hybrid"], default="hybrid")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raspi_config.MODEL_MODE = args.mode
    predictor = HybridFreshnessPredictor()

    result = predictor.predict(
        image_path=args.image_path,
        meat_type=args.meat_type,
        sensor_values={
            "nh3_ratio": args.nh3_ratio,
            "h2s_ratio": args.h2s_ratio,
            "voc_ratio": args.voc_ratio,
        },
    )

    print(f"Mode: {args.mode}")
    print(f"Predicted freshness: {result.predicted_freshness}")
    if result.class_probabilities:
        print("Class probabilities:")
        for label, probability in result.class_probabilities.items():
            print(f"  {label}: {probability:.4f}")


if __name__ == "__main__":
    main()
