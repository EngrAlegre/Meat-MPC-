from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from hybrid_pipeline_utils import (
    DEFAULT_SENSOR_COLUMNS,
    DEFAULT_SENSOR_WINDOW_SIZE,
    DEFAULT_SENSOR_WINDOW_STEP,
    build_hybrid_dataset,
    save_json,
)
from train_hybrid_model import (
    build_model_registry,
    build_preprocessor,
    evaluate_cv_mode,
    evaluate_holdout_mode,
)


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    default_sensor_dir = root_dir / "Dataset" / "MQ_Sensors_Dataset"
    default_image_dir = root_dir / "Dataset" / "Image"
    default_output_dir = root_dir / "model" / "modal_runs"

    parser = argparse.ArgumentParser(
        description="Train and compare sensor-only, image-only, and hybrid freshness classifiers."
    )
    parser.add_argument("--sensor-dir", type=Path, default=default_sensor_dir)
    parser.add_argument("--image-dir", type=Path, default=default_image_dir)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--sensor-window-size", type=int, default=DEFAULT_SENSOR_WINDOW_SIZE)
    parser.add_argument("--sensor-window-step", type=int, default=DEFAULT_SENSOR_WINDOW_STEP)
    return parser.parse_args()


def select_feature_frame(hybrid_df: pd.DataFrame, modality: str) -> pd.DataFrame:
    metadata_columns = {
        "freshness_label",
        "image_path",
        "image_folder",
        "sensor_source_file",
        "sensor_source_group",
        "sensor_window_index",
        "sensor_window_start",
        "sensor_window_end",
        "image_index_within_class",
    }
    image_columns = [column for column in hybrid_df.columns if column.startswith("img_")]
    sensor_columns = [
        column
        for column in hybrid_df.columns
        if column.startswith("sensor_")
        and column not in {"sensor_source_file", "sensor_source_group"}
        and not column.startswith("sensor_window_")
    ]

    if modality == "sensor_only":
        selected_columns = sensor_columns + ["meat_type"]
    elif modality == "image_only":
        selected_columns = image_columns + ["meat_type"]
    elif modality == "hybrid":
        selected_columns = [
            column for column in hybrid_df.columns if column not in metadata_columns
        ]
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    return hybrid_df.loc[:, selected_columns].copy()


def train_one_modality(
    *,
    modality: str,
    hybrid_df: pd.DataFrame,
    target: np.ndarray,
    target_names: list[str],
    groups: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    output_dir = args.output_dir / modality
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_frame = select_feature_frame(hybrid_df, modality)
    preprocessor, numeric_columns, categorical_columns = build_preprocessor(feature_frame)

    x_train_random, x_test_random, y_train_random, y_test_random = train_test_split(
        feature_frame,
        target,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=target,
    )

    grouped_splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=args.random_state)
    grouped_train_idx, grouped_test_idx = next(grouped_splitter.split(feature_frame, target, groups))
    x_train_grouped = feature_frame.iloc[grouped_train_idx].reset_index(drop=True)
    x_test_grouped = feature_frame.iloc[grouped_test_idx].reset_index(drop=True)
    y_train_grouped = target[grouped_train_idx]
    y_test_grouped = target[grouped_test_idx]

    models = build_model_registry(args.random_state)
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
    grouped_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=args.random_state)

    results: list[dict[str, object]] = []
    best_pipeline: Pipeline | None = None
    best_model_name = ""
    best_grouped_cv_accuracy = -np.inf

    print("=" * 80)
    print(f"Modality: {modality}")
    print(f"Rows: {len(feature_frame)}")
    print(f"Feature columns: {len(feature_frame.columns)}")
    print()

    for model_name, classifier in models.items():
        pipeline_template = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("classifier", classifier),
            ]
        )

        print("-" * 80)
        print(f"Model: {model_name}")
        random_holdout_result = evaluate_holdout_mode(
            mode_name="random_split",
            model_name=model_name,
            pipeline=clone(pipeline_template),
            x_train=x_train_random,
            x_test=x_test_random,
            y_train=y_train_random,
            y_test=y_test_random,
            target_names=target_names,
            output_dir=output_dir,
        )

        grouped_holdout_result = evaluate_holdout_mode(
            mode_name="grouped_split",
            model_name=model_name,
            pipeline=clone(pipeline_template),
            x_train=x_train_grouped,
            x_test=x_test_grouped,
            y_train=y_train_grouped,
            y_test=y_test_grouped,
            target_names=target_names,
            output_dir=output_dir,
        )

        random_cv_result = evaluate_cv_mode(
            mode_name="random_cv",
            model_name=model_name,
            pipeline=clone(pipeline_template),
            features=feature_frame,
            target=target,
            target_names=target_names,
            splitter=cv,
            output_dir=output_dir,
        )

        grouped_cv_result = evaluate_cv_mode(
            mode_name="grouped_cv",
            model_name=model_name,
            pipeline=clone(pipeline_template),
            features=feature_frame,
            target=target,
            target_names=target_names,
            splitter=grouped_cv,
            output_dir=output_dir,
            groups=groups,
        )

        results.append(
            {
                "modality": modality,
                "model_name": model_name,
                "random_split_accuracy": random_holdout_result["accuracy"],
                "grouped_split_accuracy": grouped_holdout_result["accuracy"],
                "random_cv_accuracy_mean": random_cv_result["accuracy_mean"],
                "random_cv_accuracy_std": random_cv_result["accuracy_std"],
                "grouped_cv_accuracy_mean": grouped_cv_result["accuracy_mean"],
                "grouped_cv_accuracy_std": grouped_cv_result["accuracy_std"],
            }
        )

        if grouped_cv_result["accuracy_mean"] > best_grouped_cv_accuracy:
            best_grouped_cv_accuracy = grouped_cv_result["accuracy_mean"]
            best_model_name = model_name
            best_pipeline = clone(pipeline_template)

    if best_pipeline is None:
        raise RuntimeError(f"No model was trained successfully for modality: {modality}")

    best_pipeline.fit(feature_frame, target)

    results_df = pd.DataFrame(results).sort_values(by="grouped_cv_accuracy_mean", ascending=False)
    results_df.to_csv(output_dir / "model_comparison.csv", index=False)

    model_prefix = f"{modality}_freshness_model"
    joblib.dump(best_pipeline, output_dir / f"{model_prefix}.joblib")
    joblib.dump(best_pipeline.named_steps["preprocessor"], output_dir / f"{modality}_preprocessor.joblib")

    metadata = {
        "modality": modality,
        "best_model_name": best_model_name,
        "best_grouped_cv_accuracy": float(best_grouped_cv_accuracy),
        "feature_columns": list(feature_frame.columns),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "sensor_base_columns": DEFAULT_SENSOR_COLUMNS,
        "target_classes": target_names,
        "group_count": int(len(np.unique(groups))),
        "sensor_window_size": int(args.sensor_window_size),
        "sensor_window_step": int(args.sensor_window_step),
        "recommended_metric": "grouped_cv_accuracy_mean",
        "notes": [
            "Training rows come from the Dataset folder via build_hybrid_dataset().",
            "One row is created per image, matched with a sensor window summary from the corresponding class CSV.",
            "Grouped CV keeps each sensor window group entirely in either train or validation folds.",
            "Use these modality-specific runs to diagnose whether image-side or sensor-side features are causing deployment issues.",
        ],
    }
    save_json(metadata, output_dir / "training_metadata.json")

    print(f"Best {modality} model: {best_model_name}")
    print(f"Best grouped CV accuracy: {best_grouped_cv_accuracy:.4f}")
    print(f"Artifacts saved to: {output_dir}")
    print()

    return {
        "modality": modality,
        "best_model_name": best_model_name,
        "best_grouped_cv_accuracy": float(best_grouped_cv_accuracy),
        "output_dir": str(output_dir),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hybrid_df = build_hybrid_dataset(
        args.sensor_dir,
        args.image_dir,
        base_sensor_columns=DEFAULT_SENSOR_COLUMNS,
        sensor_window_size=args.sensor_window_size,
        sensor_window_step=args.sensor_window_step,
    )
    hybrid_df.to_csv(args.output_dir / "source_hybrid_dataset.csv", index=False)

    target_encoder = LabelEncoder()
    target = target_encoder.fit_transform(hybrid_df["freshness_label"])
    target_names = list(target_encoder.classes_)
    groups = hybrid_df["sensor_source_group"].astype(str).to_numpy()

    modality_summaries = []
    for modality in ("sensor_only", "image_only", "hybrid"):
        modality_summaries.append(
            train_one_modality(
                modality=modality,
                hybrid_df=hybrid_df,
                target=target,
                target_names=target_names,
                groups=groups,
                args=args,
            )
        )

    summary_df = pd.DataFrame(modality_summaries).sort_values(
        by="best_grouped_cv_accuracy", ascending=False
    )
    summary_df.to_csv(args.output_dir / "modality_summary.csv", index=False)
    joblib.dump(target_encoder, args.output_dir / "freshness_label_encoder.joblib")

    print("=" * 80)
    print("Modality summary")
    print(summary_df.to_string(index=False))
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
