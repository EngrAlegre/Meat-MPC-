from __future__ import annotations

import argparse
from pathlib import Path

from deployment_runtime import (
    DeploymentConfig,
    HybridDeploymentRuntime,
    format_sensor_debug,
    load_sensor_rows_from_csv,
)


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    default_artifacts_dir = root_dir / "hybrid_ml" / "artifacts"

    parser = argparse.ArgumentParser(
        description="Deployment-safe hybrid inference with warm-up, stabilization, and logging."
    )
    parser.add_argument("--artifacts-dir", type=Path, default=default_artifacts_dir)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--meat-type", required=True, choices=["Chicken", "Beef", "Pork"])
    parser.add_argument("--sensor-window-csv", type=Path, required=True)
    parser.add_argument("--baseline-window-csv", type=Path)
    parser.add_argument("--sample-count", type=int, default=30)
    parser.add_argument("--warmup-seconds", type=int, default=180)
    parser.add_argument("--warmup-elapsed-seconds", type=float, default=180.0)
    parser.add_argument("--stability-min-samples", type=int, default=20)
    parser.add_argument("--stability-max-samples", type=int, default=50)
    parser.add_argument("--head", action="store_true", help="Use the first N rows instead of the last N rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DeploymentConfig(
        warmup_seconds=args.warmup_seconds,
        stabilization_samples_min=args.stability_min_samples,
        stabilization_samples_max=args.stability_max_samples,
    )
    runtime = HybridDeploymentRuntime(args.artifacts_dir, config=config)

    print("[STATE] WARMING_UP")
    remaining = runtime.warmup_remaining_seconds(args.warmup_elapsed_seconds)
    if remaining > 0:
        print(f"Warm-up still in progress. {remaining:.1f} seconds remaining.")
        return

    if args.baseline_window_csv:
        print("[STATE] CAPTURING_BASELINE")
        baseline_rows = load_sensor_rows_from_csv(
            args.baseline_window_csv,
            limit=args.sample_count,
            from_tail=not args.head,
        )
        baseline_summary = runtime.capture_baseline(baseline_rows)
        print(format_sensor_debug(baseline_summary))
        print(f"Baseline saved for debug only: {runtime.baseline_path}")

    print("[STATE] STABILIZING_SENSORS")
    sensor_rows = load_sensor_rows_from_csv(
        args.sensor_window_csv,
        limit=args.sample_count,
        from_tail=not args.head,
    )
    summary = runtime.summarize_sensor_window(sensor_rows)
    print(format_sensor_debug(summary))
    if not summary.stable:
        print("[STATE] NOT_READY")
        print("Prediction blocked until the sensor window is stable.")
        return

    print("[STATE] READY_TO_SCAN")
    print("Using averaged sensor ratios for prediction:")
    print(
        f"  nh3_ratio={summary.averaged_values.get('nh3_ratio', float('nan')):.4f}, "
        f"h2s_ratio={summary.averaged_values.get('h2s_ratio', float('nan')):.4f}, "
        f"voc_ratio={summary.averaged_values.get('voc_ratio', float('nan')):.4f}"
    )

    print("[STATE] PREDICTING")
    result = runtime.predict_from_window(
        image_path=args.image_path,
        meat_type=args.meat_type,
        sensor_rows=sensor_rows,
        warmup_elapsed_seconds=args.warmup_elapsed_seconds,
    )

    print("[STATE] PREDICTED")
    print(f"Predicted freshness: {result.predicted_freshness}")
    if result.confidence is not None:
        print(f"Confidence: {result.confidence:.4f}")
    if result.class_probabilities:
        print("Class probabilities:")
        for label, probability in result.class_probabilities.items():
            print(f"  {label}: {probability:.4f}")
    print(f"Prediction log appended to: {runtime.prediction_log_path}")


if __name__ == "__main__":
    main()
