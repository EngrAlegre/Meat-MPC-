from __future__ import annotations

import argparse
from pathlib import Path

from meat_classifier_utils import load_meat_classifier, predict_meat_class


def parse_args() -> argparse.Namespace:
    default_model_dir = Path(__file__).resolve().parents[1] / "model" / "meat_classifier"
    parser = argparse.ArgumentParser(description="Run inference with the Option B meat image classifier.")
    parser.add_argument("--image-path", type=Path, required=True, help="Path to the input image.")
    parser.add_argument("--model-dir", type=Path, default=default_model_dir, help="Directory containing meat_classifier.keras and metadata files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = load_meat_classifier(
        model_path=args.model_dir / "meat_classifier.keras",
        class_names_path=args.model_dir / "class_names.json",
        metadata_path=args.model_dir / "metadata.json",
    )
    result = predict_meat_class(artifacts, args.image_path)

    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("Class probabilities:")
    for class_name, score in result["class_probabilities"].items():
        print(f"  {class_name}: {score:.4f}")


if __name__ == "__main__":
    main()
