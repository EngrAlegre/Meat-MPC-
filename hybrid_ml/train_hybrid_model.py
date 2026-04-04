from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from hybrid_pipeline_utils import (
    DEFAULT_SENSOR_COLUMNS,
    build_hybrid_dataset,
    save_json,
)


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    default_sensor_dir = root_dir / "Dataset" / "MQ_Sensors_Dataset"
    default_image_dir = root_dir / "Dataset" / "Image"
    default_output_dir = root_dir / "model"

    parser = argparse.ArgumentParser(
        description="Train a hybrid meat freshness classifier from images + MQ sensor CSVs."
    )
    parser.add_argument("--sensor-dir", type=Path, default=default_sensor_dir)
    parser.add_argument("--image-dir", type=Path, default=default_image_dir)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    return parser.parse_args()


def build_preprocessor(feature_frame: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    categorical_columns = ["meat_type"]
    numeric_columns = [column for column in feature_frame.columns if column not in categorical_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )
    return preprocessor, numeric_columns, categorical_columns


def build_model_registry(random_state: int) -> dict[str, object]:
    registry: dict[str, object] = {
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            C=3.0,
            gamma="scale",
            class_weight="balanced",
        ),
    }

    try:
        from xgboost import XGBClassifier

        registry["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        )
    except Exception:
        print("XGBoost not available; skipping that model.")

    return registry


def save_report_text(text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def evaluate_holdout_mode(
    *,
    mode_name: str,
    model_name: str,
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    target_names: list[str],
    output_dir: Path,
) -> dict[str, object]:
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions)

    print(f"{mode_name} accuracy: {accuracy:.4f}")
    print("Classification report:")
    print(report)
    print("Confusion matrix:")
    print(matrix)
    print()

    matrix_df = pd.DataFrame(matrix, index=target_names, columns=target_names)
    matrix_df.to_csv(output_dir / f"confusion_matrix_{model_name}_{mode_name}.csv")
    save_report_text(report, output_dir / f"classification_report_{model_name}_{mode_name}.txt")

    return {
        "mode_name": mode_name,
        "accuracy": float(accuracy),
        "report": report,
        "confusion_matrix": matrix,
    }


def evaluate_cv_mode(
    *,
    mode_name: str,
    model_name: str,
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: np.ndarray,
    target_names: list[str],
    splitter,
    output_dir: Path,
    groups: np.ndarray | None = None,
) -> dict[str, object]:
    score_kwargs = {"cv": splitter, "scoring": "accuracy", "n_jobs": 1}
    predict_kwargs = {"cv": splitter, "n_jobs": 1}
    if groups is not None:
        score_kwargs["groups"] = groups
        predict_kwargs["groups"] = groups

    scores = cross_val_score(pipeline, features, target, **score_kwargs)
    predictions = cross_val_predict(pipeline, features, target, **predict_kwargs)

    report = classification_report(
        target,
        predictions,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    matrix = confusion_matrix(target, predictions)

    print(f"{mode_name} accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")
    print("Cross-validated classification report:")
    print(report)
    print("Cross-validated confusion matrix:")
    print(matrix)
    print()

    matrix_df = pd.DataFrame(matrix, index=target_names, columns=target_names)
    matrix_df.to_csv(output_dir / f"confusion_matrix_{model_name}_{mode_name}.csv")
    save_report_text(report, output_dir / f"classification_report_{model_name}_{mode_name}.txt")

    return {
        "mode_name": mode_name,
        "accuracy_mean": float(scores.mean()),
        "accuracy_std": float(scores.std()),
        "report": report,
        "confusion_matrix": matrix,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hybrid_df = build_hybrid_dataset(args.sensor_dir, args.image_dir, base_sensor_columns=DEFAULT_SENSOR_COLUMNS)
    hybrid_df.to_csv(args.output_dir / "hybrid_dataset.csv", index=False)

    target_encoder = LabelEncoder()
    target = target_encoder.fit_transform(hybrid_df["freshness_label"])
    target_names = list(target_encoder.classes_)
    groups = hybrid_df["sensor_source_file"].astype(str).to_numpy()

    feature_frame = hybrid_df.drop(columns=["freshness_label", "image_path", "image_folder", "sensor_source_file"])
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

    print(f"Training rows: {len(feature_frame)}")
    print("Class distribution:")
    print(hybrid_df["freshness_label"].value_counts().sort_index())
    print("Group distribution by sensor source:")
    print(hybrid_df["sensor_source_file"].value_counts())
    print()

    for model_name, classifier in models.items():
        pipeline_template = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("classifier", classifier),
            ]
        )

        print("=" * 80)
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
            output_dir=args.output_dir,
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
            output_dir=args.output_dir,
        )

        random_cv_result = evaluate_cv_mode(
            mode_name="random_cv",
            model_name=model_name,
            pipeline=clone(pipeline_template),
            features=feature_frame,
            target=target,
            target_names=target_names,
            splitter=cv,
            output_dir=args.output_dir,
        )

        grouped_cv_result = evaluate_cv_mode(
            mode_name="grouped_cv",
            model_name=model_name,
            pipeline=clone(pipeline_template),
            features=feature_frame,
            target=target,
            target_names=target_names,
            splitter=grouped_cv,
            output_dir=args.output_dir,
            groups=groups,
        )

        results.append(
            {
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
        raise RuntimeError("No model was trained successfully.")

    # Final artifact pipeline is fit on the full dataset after model selection.
    # The best model is selected using grouped CV because that evaluation is less
    # optimistic: the same class-level sensor summary is not allowed to leak across
    # train and validation folds through repeated images.
    best_pipeline.fit(feature_frame, target)

    results_df = pd.DataFrame(results).sort_values(by="grouped_cv_accuracy_mean", ascending=False)
    results_df.to_csv(args.output_dir / "model_comparison.csv", index=False)

    joblib.dump(best_pipeline, args.output_dir / "hybrid_freshness_model.joblib")
    joblib.dump(target_encoder, args.output_dir / "freshness_label_encoder.joblib")
    joblib.dump(best_pipeline.named_steps["preprocessor"], args.output_dir / "hybrid_preprocessor.joblib")

    metadata = {
        "best_model_name": best_model_name,
        "best_grouped_cv_accuracy": float(best_grouped_cv_accuracy),
        "sensor_dir": str(args.sensor_dir),
        "image_dir": str(args.image_dir),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "sensor_base_columns": DEFAULT_SENSOR_COLUMNS,
        "target_classes": list(target_encoder.classes_),
        "random_split_train_rows": int(len(x_train_random)),
        "random_split_test_rows": int(len(x_test_random)),
        "grouped_split_train_rows": int(len(x_train_grouped)),
        "grouped_split_test_rows": int(len(x_test_grouped)),
        "group_count": int(len(np.unique(groups))),
        "aggregation_strategy": "Class-level MQ sensor summary statistics (mean/min/max/std) repeated for each image in the matching class.",
        "recommended_thesis_metric": "grouped_cv_accuracy_mean",
        "notes": [
            "One row is created per image.",
            "MQ sensor CSVs are matched to image folders by meat type and freshness label.",
            "Neutral CSVs may contain only ratio columns; missing voltage/resistance summaries are imputed.",
            "Meat type is included as a known categorical feature because it is available at inference time.",
            "Random split is optimistic because repeated class-level sensor summaries appear in both train and test when images from the same class are mixed.",
            "Grouped split and grouped CV are more realistic because each sensor-source group is kept entirely in either train or validation/test.",
        ],
    }
    save_json(metadata, args.output_dir / "training_metadata.json")

    print("=" * 80)
    print(f"Best model: {best_model_name}")
    print(f"Best grouped CV accuracy: {best_grouped_cv_accuracy:.4f}")
    print("Most thesis-safe metric: grouped_cv_accuracy_mean")
    print(f"Artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
