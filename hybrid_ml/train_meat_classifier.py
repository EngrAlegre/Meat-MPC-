from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from meat_classifier_utils import MeatClassifierRuntimeError, save_json_file

try:
    import tensorflow as tf
    from tensorflow.keras import callbacks, layers, models, optimizers
    from tensorflow.keras.applications import MobileNetV2
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "TensorFlow is required to train the meat classifier. "
        f"Install tensorflow first. Original error: {exc}"
    ) from exc


CLASS_NAMES = ("not_meat", "chicken", "pork", "beef")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    default_dataset_dir = Path(__file__).resolve().parents[1] / "dataset" / "Image"
    default_output_dir = Path(__file__).resolve().parents[1] / "model" / "meat_classifier"

    parser = argparse.ArgumentParser(description="Train the Option B meat image classifier.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=default_dataset_dir,
        help=(
            "Image dataset root. Supports either flat class folders "
            "(not_meat/chicken/pork/beef) or the existing FreshTo image layout "
            "(chicken_fresh, chicken_neutral, chicken_spoiled, etc.)."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir, help="Directory where the trained classifier artifacts will be saved.")
    parser.add_argument("--image-size", type=int, default=224, help="Square image size for MobileNetV2 input.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Validation split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--freeze-base-epochs", type=int, default=8, help="Epochs with the MobileNetV2 base frozen.")
    parser.add_argument("--fine-tune-epochs", type=int, default=6, help="Fine-tuning epochs after unfreezing part of the base.")
    parser.add_argument("--fine-tune-layers", type=int, default=40, help="How many MobileNetV2 layers to unfreeze at the end.")
    parser.add_argument(
        "--max-per-folder",
        type=int,
        default=None,
        help="Optional cap on the number of images taken from each source folder. Useful when one class has many more samples than another.",
    )
    return parser.parse_args()


def infer_meat_label(folder_name: str) -> str | None:
    normalized = folder_name.strip().lower().replace("-", " ").replace("_", " ")
    collapsed = " ".join(normalized.split())
    if collapsed in {"not meat", "non meat", "nonmeat", "notmeat"}:
        return "not_meat"
    tokens = set(collapsed.split())
    if "chicken" in tokens:
        return "chicken"
    if "pork" in tokens:
        return "pork"
    if "beef" in tokens:
        return "beef"
    if collapsed == "not_meat":
        return "not_meat"
    return None


def build_image_index(
    dataset_dir: Path,
    max_per_folder: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for class_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        label = infer_meat_label(class_dir.name)
        if label is None:
            continue
        image_paths: list[Path] = [
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        image_paths.sort()
        if max_per_folder is not None and len(image_paths) > max_per_folder:
            image_paths = rng.sample(image_paths, max_per_folder)
        for image_path in image_paths:
            rows.append(
                {
                    "image_path": str(image_path),
                    "label": label,
                    "source_folder": class_dir.name,
                }
            )

    if not rows:
        raise MeatClassifierRuntimeError(
            "No supported training images were found. Check that the dataset folder contains image files."
        )

    frame = pd.DataFrame(rows)
    missing_labels = [label for label in CLASS_NAMES if label not in set(frame["label"])]
    if missing_labels:
        raise MeatClassifierRuntimeError(
            "The dataset is missing required classes: " + ", ".join(missing_labels)
        )
    return frame


def print_dataset_summary(index_frame: pd.DataFrame) -> None:
    label_counts = index_frame["label"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    print("Image counts by class:")
    for label, count in label_counts.items():
        print(f"  {label}: {int(count)}")

    print("Source folders used:")
    folder_counts = index_frame["source_folder"].value_counts()
    for folder_name, count in folder_counts.items():
        print(f"  {folder_name}: {int(count)}")


def make_tf_dataset(
    frame: pd.DataFrame,
    *,
    class_to_index: dict[str, int],
    image_size: tuple[int, int],
    batch_size: int,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    path_ds = tf.data.Dataset.from_tensor_slices(frame["image_path"].tolist())
    label_ds = tf.data.Dataset.from_tensor_slices(frame["label"].map(class_to_index).astype(np.int32).tolist())
    dataset = tf.data.Dataset.zip((path_ds, label_ds))

    if training:
        dataset = dataset.shuffle(buffer_size=len(frame), seed=seed, reshuffle_each_iteration=True)

    def _load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        bytes_ = tf.io.read_file(path)
        image = tf.io.decode_image(bytes_, channels=3, expand_animations=False)
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32)
        return image, label

    dataset = dataset.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def compute_class_weights(train_labels: list[str], class_names: list[str]) -> dict[int, float]:
    encoded_labels = [class_names.index(label) for label in train_labels]
    counts = Counter(encoded_labels)
    total = float(sum(counts.values()))
    return {
        class_index: total / (len(class_names) * counts[class_index])
        for class_index in counts
        if counts[class_index] > 0
    }


def build_model(input_size: tuple[int, int], num_classes: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    augmentation = models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )

    base_model = MobileNetV2(
        input_shape=(input_size[0], input_size[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(input_size[0], input_size[1], 3), name="image")
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="meat_class")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="meat_classifier_mobilenetv2")
    return model, base_model


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    index_frame = build_image_index(
        args.dataset_dir,
        max_per_folder=args.max_per_folder,
        seed=args.seed,
    )
    print_dataset_summary(index_frame)

    train_frame, validation_frame = train_test_split(
        index_frame,
        test_size=args.validation_split,
        random_state=args.seed,
        stratify=index_frame["label"],
    )
    train_frame = train_frame.reset_index(drop=True)
    validation_frame = validation_frame.reset_index(drop=True)

    image_size = (args.image_size, args.image_size)
    class_names = list(CLASS_NAMES)
    class_to_index = {label: idx for idx, label in enumerate(class_names)}
    train_dataset = make_tf_dataset(
        train_frame,
        class_to_index=class_to_index,
        image_size=image_size,
        batch_size=args.batch_size,
        training=True,
        seed=args.seed,
    )
    validation_dataset = make_tf_dataset(
        validation_frame,
        class_to_index=class_to_index,
        image_size=image_size,
        batch_size=args.batch_size,
        training=False,
        seed=args.seed,
    )

    validation_labels = validation_frame["label"].map(class_to_index).to_numpy(dtype=np.int32)
    class_weights = compute_class_weights(train_frame["label"].tolist(), class_names)

    model, base_model = build_model(image_size, num_classes=len(class_names))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callback_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6),
    ]
    history_rows: list[dict[str, float | int | str]] = []

    frozen_history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.freeze_base_epochs,
        class_weight=class_weights,
        callbacks=callback_list,
        verbose=1,
    )
    for epoch_index in range(len(frozen_history.history["loss"])):
        history_rows.append(
            {
                "stage": "frozen_base",
                "epoch": epoch_index + 1,
                "loss": float(frozen_history.history["loss"][epoch_index]),
                "accuracy": float(frozen_history.history["accuracy"][epoch_index]),
                "val_loss": float(frozen_history.history["val_loss"][epoch_index]),
                "val_accuracy": float(frozen_history.history["val_accuracy"][epoch_index]),
            }
        )

    if args.fine_tune_epochs > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-args.fine_tune_layers]:
            layer.trainable = False

        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        fine_tune_history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=args.fine_tune_epochs,
            class_weight=class_weights,
            callbacks=callback_list,
            verbose=1,
        )
        offset = len(history_rows)
        for epoch_index in range(len(fine_tune_history.history["loss"])):
            history_rows.append(
                {
                    "stage": "fine_tune",
                    "epoch": offset + epoch_index + 1,
                    "loss": float(fine_tune_history.history["loss"][epoch_index]),
                    "accuracy": float(fine_tune_history.history["accuracy"][epoch_index]),
                    "val_loss": float(fine_tune_history.history["val_loss"][epoch_index]),
                    "val_accuracy": float(fine_tune_history.history["val_accuracy"][epoch_index]),
                }
            )

    probabilities = model.predict(validation_dataset, verbose=0)
    predicted_indices = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(validation_labels, predicted_indices)
    report_text = classification_report(validation_labels, predicted_indices, target_names=class_names, digits=4)
    confusion = confusion_matrix(validation_labels, predicted_indices)

    print(f"Validation accuracy: {accuracy:.4f}")
    print("Classification report:")
    print(report_text)
    print("Confusion matrix:")
    print(confusion)

    model_path = args.output_dir / "meat_classifier.keras"
    class_names_path = args.output_dir / "class_names.json"
    metadata_path = args.output_dir / "metadata.json"
    report_path = args.output_dir / "classification_report.txt"
    confusion_path = args.output_dir / "confusion_matrix.csv"
    history_path = args.output_dir / "training_history.csv"
    dataset_index_path = args.output_dir / "dataset_index.csv"

    model.save(model_path)
    save_json_file(class_names_path, class_names)
    save_json_file(
        metadata_path,
        {
            "model_name": "MobileNetV2",
            "class_names": class_names,
            "input_size": list(image_size),
            "validation_split": args.validation_split,
            "batch_size": args.batch_size,
            "freeze_base_epochs": args.freeze_base_epochs,
            "fine_tune_epochs": args.fine_tune_epochs,
            "fine_tune_layers": args.fine_tune_layers,
            "seed": args.seed,
            "dataset_dir": str(args.dataset_dir),
            "dataset_layout": "auto-detected",
            "max_per_folder": args.max_per_folder,
            "label_counts": index_frame["label"].value_counts().reindex(class_names, fill_value=0).to_dict(),
            "source_folders": sorted(index_frame["source_folder"].unique().tolist()),
            "validation_accuracy": accuracy,
        },
    )
    report_path.write_text(report_text, encoding="utf-8")
    pd.DataFrame(confusion, index=class_names, columns=class_names).to_csv(confusion_path, encoding="utf-8")
    pd.DataFrame(history_rows).to_csv(history_path, index=False, encoding="utf-8")
    index_frame.to_csv(dataset_index_path, index=False, encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved class names: {class_names_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
