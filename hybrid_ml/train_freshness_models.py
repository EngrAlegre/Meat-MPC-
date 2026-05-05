from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from hybrid_pipeline_utils import (
    DEFAULT_SENSOR_COLUMNS,
    DEFAULT_SENSOR_WINDOW_SIZE,
    DEFAULT_SENSOR_WINDOW_STEP,
    aggregate_sensor_frame,
    extract_image_features,
    list_image_files,
    parse_class_key,
    resolve_existing_path,
)


FRESHNESS_CLASSES_3 = ["Fresh", "Neutral", "Spoiled"]
FRESHNESS_CLASSES_2 = ["Fresh", "Spoiled"]
MEAT_TYPES = ["Chicken", "Pork", "Beef"]


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    default_sensor_dir = root_dir / "Dataset" / "MQ_Sensors_Dataset"
    default_image_dir = root_dir / "Dataset" / "NEW IMAGE"
    default_output_dir = root_dir / "model" / "modal_runs"

    parser = argparse.ArgumentParser(
        description=(
            "Train the image-only (2-class Fresh/Spoiled) and sensor-only "
            "(3-class Fresh/Neutral/Spoiled) freshness classifiers."
        )
    )
    parser.add_argument("--sensor-dir", type=Path, default=default_sensor_dir)
    parser.add_argument("--image-dir", type=Path, default=default_image_dir)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--sensor-window-size",
        type=int,
        default=DEFAULT_SENSOR_WINDOW_SIZE,
        help="How many sensor rows per windowed training sample.",
    )
    parser.add_argument(
        "--sensor-window-step",
        type=int,
        default=DEFAULT_SENSOR_WINDOW_STEP,
        help="Step size between windows (set equal to window size for non-overlapping).",
    )
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=None,
        help="Cap the number of images per folder (e.g. 150) to keep classes balanced.",
    )
    return parser.parse_args()


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def build_numeric_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    transformers: list[tuple] = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            )
        )
    if not transformers:
        raise ValueError("No feature columns were detected for preprocessing.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_model_registry(random_state: int) -> dict[str, object]:
    return {
        "svm_rbf": SVC(kernel="rbf", probability=True, random_state=random_state, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }


def infer_image_class(folder_name: str) -> tuple[str, str] | None:
    class_key = parse_class_key(folder_name)
    if class_key is None:
        return None
    if class_key.freshness not in FRESHNESS_CLASSES_2:
        return None
    return class_key.meat_type, class_key.freshness


def build_image_feature_frame(
    image_dir: Path,
    max_per_folder: int | None,
    seed: int,
) -> pd.DataFrame:
    image_dir = resolve_existing_path(image_dir)
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for folder in sorted(path for path in image_dir.iterdir() if path.is_dir()):
        parsed = infer_image_class(folder.name)
        if parsed is None:
            continue
        meat_type, freshness_label = parsed
        image_files = list_image_files(folder)
        if max_per_folder is not None and len(image_files) > max_per_folder:
            image_files = rng.sample(image_files, max_per_folder)
        for image_path in image_files:
            features = extract_image_features(image_path)
            features["meat_type"] = meat_type
            features["freshness_label"] = freshness_label
            features["source_folder"] = folder.name
            features["image_path"] = str(image_path)
            rows.append(features)

    if not rows:
        raise ValueError(
            f"No Fresh/Spoiled images found under {image_dir}. Expected folders such as 'Fresh Chicken', 'Spoiled Beef'."
        )
    return pd.DataFrame(rows)


def build_sensor_feature_frame(
    sensor_dir: Path,
    base_columns: Iterable[str],
    window_size: int,
    window_step: int,
) -> pd.DataFrame:
    sensor_dir = resolve_existing_path(sensor_dir)
    base_columns = list(base_columns)
    rows: list[dict[str, object]] = []

    for csv_path in sorted(sensor_dir.glob("*.csv")):
        frame = pd.read_csv(csv_path)
        frame.columns = [column.strip() for column in frame.columns]
        class_key = parse_class_key(csv_path.stem)
        if class_key is None:
            if {"label", "meat"}.issubset(frame.columns):
                meat_type = str(frame["meat"].dropna().iloc[0]).strip().capitalize()
                freshness = str(frame["label"].dropna().iloc[0]).strip().capitalize()
            else:
                continue
        else:
            meat_type = class_key.meat_type
            freshness = class_key.freshness

        if freshness not in FRESHNESS_CLASSES_3:
            continue
        if meat_type not in MEAT_TYPES:
            continue

        if len(frame) <= window_size:
            windows = [frame]
        else:
            windows = []
            max_start = len(frame) - window_size
            for start in range(0, max_start + 1, max(window_step, 1)):
                windows.append(frame.iloc[start : start + window_size].reset_index(drop=True))

        for window_index, window_frame in enumerate(windows):
            summary = aggregate_sensor_frame(window_frame, base_columns=base_columns)
            summary["meat_type"] = meat_type
            summary["freshness_label"] = freshness
            summary["source_file"] = str(csv_path)
            summary["window_index"] = int(window_index)
            rows.append(summary)

    if not rows:
        raise ValueError(f"No usable sensor CSV files found in {sensor_dir}.")
    return pd.DataFrame(rows)


def select_feature_columns(frame: pd.DataFrame, kind: str) -> tuple[list[str], list[str]]:
    if kind == "image":
        numeric_columns = [column for column in frame.columns if column.startswith("img_")]
    elif kind == "sensor":
        numeric_columns = [column for column in frame.columns if column.startswith("sensor_")]
    else:
        raise ValueError(f"Unknown kind: {kind}")

    categorical_columns = ["meat_type"] if "meat_type" in frame.columns else []
    return numeric_columns, categorical_columns


def train_one_head(
    *,
    head_name: str,
    feature_frame: pd.DataFrame,
    target_frame: pd.Series,
    classes: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    label_encoder = LabelEncoder()
    label_encoder.fit(classes)
    target = label_encoder.transform(target_frame.values)

    preprocessor = build_numeric_preprocessor(numeric_columns, categorical_columns)
    models = build_model_registry(args.random_state)

    x_train, x_test, y_train, y_test = train_test_split(
        feature_frame,
        target,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=target,
    )

    summaries: list[dict] = []
    best_name: str | None = None
    best_pipeline: Pipeline | None = None
    best_score = -np.inf

    for model_name, classifier in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(x_train, y_train)
        holdout_predictions = pipeline.predict(x_test)
        holdout_accuracy = float(accuracy_score(y_test, holdout_predictions))

        cv_splitter = StratifiedKFold(
            n_splits=args.cv_folds,
            shuffle=True,
            random_state=args.random_state,
        )
        cv_scores = cross_val_score(
            clone(pipeline),
            feature_frame,
            target,
            cv=cv_splitter,
            scoring="accuracy",
            error_score="raise",
        )

        report_text = classification_report(
            y_test,
            holdout_predictions,
            target_names=classes,
            digits=4,
            zero_division=0,
        )
        confusion = confusion_matrix(y_test, holdout_predictions, labels=list(range(len(classes))))
        report_path = output_dir / f"classification_report_{model_name}.txt"
        confusion_path = output_dir / f"confusion_matrix_{model_name}.csv"
        report_path.write_text(report_text, encoding="utf-8")
        pd.DataFrame(confusion, index=classes, columns=classes).to_csv(confusion_path, encoding="utf-8")

        summaries.append(
            {
                "head": head_name,
                "model_name": model_name,
                "holdout_accuracy": holdout_accuracy,
                "cv_accuracy_mean": float(cv_scores.mean()),
                "cv_accuracy_std": float(cv_scores.std()),
            }
        )

        print("-" * 80)
        print(f"{head_name} | {model_name}")
        print(f"  holdout acc:   {holdout_accuracy:.4f}")
        print(f"  {args.cv_folds}-fold CV acc:  {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        print(report_text)

        if cv_scores.mean() > best_score:
            best_score = float(cv_scores.mean())
            best_name = model_name
            best_pipeline = clone(pipeline)

    assert best_pipeline is not None and best_name is not None
    best_pipeline.fit(feature_frame, target)

    summary_frame = pd.DataFrame(summaries).sort_values(by="cv_accuracy_mean", ascending=False)
    summary_frame.to_csv(output_dir / "model_comparison.csv", index=False, encoding="utf-8")

    model_file = output_dir / f"{head_name}_freshness_model.joblib"
    encoder_file = output_dir / f"{head_name}_label_encoder.joblib"
    joblib.dump(best_pipeline, model_file)
    joblib.dump(label_encoder, encoder_file)

    metadata = {
        "head": head_name,
        "best_model_name": best_name,
        "best_cv_accuracy": best_score,
        "feature_columns": list(feature_frame.columns),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "target_classes": list(classes),
        "sensor_base_columns": list(DEFAULT_SENSOR_COLUMNS),
        "holdout_summary": summaries,
        "random_state": args.random_state,
    }
    save_json(metadata, output_dir / "training_metadata.json")

    print("=" * 80)
    print(f"Best {head_name} model: {best_name} (CV acc={best_score:.4f})")
    print(f"Artifacts saved under: {output_dir}")

    return {
        "head": head_name,
        "best_model_name": best_name,
        "best_cv_accuracy": best_score,
        "output_dir": str(output_dir),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tf_seed = args.random_state
    np.random.seed(tf_seed)
    random.seed(tf_seed)

    # Image-only head (2-class Fresh / Spoiled, NEW IMAGE dataset).
    image_frame = build_image_feature_frame(
        args.image_dir,
        max_per_folder=args.max_images_per_class,
        seed=args.random_state,
    )
    print("Image feature frame:")
    print(image_frame["source_folder"].value_counts())
    image_numeric, image_categorical = select_feature_columns(image_frame, "image")
    image_features = image_frame[image_numeric + image_categorical]
    image_summary = train_one_head(
        head_name="image_only",
        feature_frame=image_features,
        target_frame=image_frame["freshness_label"],
        classes=FRESHNESS_CLASSES_2,
        numeric_columns=image_numeric,
        categorical_columns=image_categorical,
        output_dir=args.output_dir / "image_only",
        args=args,
    )

    # Sensor-only head (3-class Fresh / Neutral / Spoiled, MQ_Sensors_Dataset).
    sensor_frame = build_sensor_feature_frame(
        args.sensor_dir,
        base_columns=DEFAULT_SENSOR_COLUMNS,
        window_size=args.sensor_window_size,
        window_step=args.sensor_window_step,
    )
    print("Sensor feature frame:")
    print(sensor_frame.groupby(["meat_type", "freshness_label"]).size())
    sensor_numeric, sensor_categorical = select_feature_columns(sensor_frame, "sensor")
    sensor_features = sensor_frame[sensor_numeric + sensor_categorical]
    sensor_summary = train_one_head(
        head_name="sensor_only",
        feature_frame=sensor_features,
        target_frame=sensor_frame["freshness_label"],
        classes=FRESHNESS_CLASSES_3,
        numeric_columns=sensor_numeric,
        categorical_columns=sensor_categorical,
        output_dir=args.output_dir / "sensor_only",
        args=args,
    )

    # Shared 3-class label encoder used by the deployment fusion layer.
    shared_label_encoder = LabelEncoder()
    shared_label_encoder.fit(FRESHNESS_CLASSES_3)
    joblib.dump(shared_label_encoder, args.output_dir / "freshness_label_encoder.joblib")

    modality_summary = pd.DataFrame([image_summary, sensor_summary])
    modality_summary.to_csv(args.output_dir / "modality_summary.csv", index=False, encoding="utf-8")

    combined_source = pd.DataFrame(
        {
            "image_rows": [len(image_frame)],
            "sensor_rows": [len(sensor_frame)],
            "image_classes": [sorted(image_frame["freshness_label"].unique().tolist())],
            "sensor_classes": [sorted(sensor_frame["freshness_label"].unique().tolist())],
        }
    )
    combined_source.to_csv(args.output_dir / "source_summary.csv", index=False, encoding="utf-8")

    print("=" * 80)
    print("Done.")
    print(modality_summary.to_string(index=False))
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
