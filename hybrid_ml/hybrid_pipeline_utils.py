from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Use ratio-based MQ features for training so the deployment app and trained model
# operate on the same sensor representation. Live Raspberry Pi inference currently
# passes aligned Rs/Ro features, not raw voltage/resistance summaries.
DEFAULT_SENSOR_COLUMNS = [
    "nh3_ratio",
    "h2s_ratio",
    "voc_ratio",
]

SENSOR_AGGREGATIONS = ("mean", "min", "max", "std")
DEFAULT_SENSOR_WINDOW_SIZE = 15
DEFAULT_SENSOR_WINDOW_STEP = 15


@dataclass(frozen=True)
class ClassKey:
    meat_type: str
    freshness: str

    @property
    def slug(self) -> str:
        return f"{self.meat_type.lower()}_{self.freshness.lower()}"


def resolve_existing_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path

    raw_text = str(path)
    candidates = []
    if "MQ Sensors_Dataset" in raw_text:
        candidates.append(Path(raw_text.replace("MQ Sensors_Dataset", "MQ_Sensors_Dataset")))
    if "MQ_Sensors_Dataset" in raw_text:
        candidates.append(Path(raw_text.replace("MQ_Sensors_Dataset", "MQ Sensors_Dataset")))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Path does not exist: {path}")


def canonicalize_label(value: str) -> str:
    return value.strip().lower()


def pretty_label(value: str) -> str:
    return canonicalize_label(value).capitalize()


def parse_class_key(text: str) -> ClassKey | None:
    lowered = canonicalize_label(text)
    for meat in ("chicken", "beef", "pork"):
        for freshness in ("fresh", "neutral", "spoiled"):
            if meat in lowered and freshness in lowered:
                return ClassKey(meat_type=pretty_label(meat), freshness=pretty_label(freshness))
    return None


def list_image_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def extract_image_features(image_path: str | Path, resize_to: tuple[int, int] = (224, 224)) -> dict[str, float]:
    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"OpenCV failed to read image: {image_path}")

    resized = cv2.resize(image_bgr, resize_to, interpolation=cv2.INTER_AREA)
    image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    features: dict[str, float] = {}
    for image_name, image in (("rgb", image_rgb), ("hsv", image_hsv)):
        channel_names = ("r", "g", "b") if image_name == "rgb" else ("h", "s", "v")
        for channel_index, channel_name in enumerate(channel_names):
            channel = image[:, :, channel_index]
            features[f"img_{image_name}_{channel_name}_mean"] = float(np.mean(channel))
            features[f"img_{image_name}_{channel_name}_std"] = float(np.std(channel))

    gray_float = gray.astype(np.float32)
    features["img_gray_mean"] = float(np.mean(gray_float))
    features["img_gray_std"] = float(np.std(gray_float))
    features["img_gray_min"] = float(np.min(gray_float))
    features["img_gray_max"] = float(np.max(gray_float))

    laplacian = cv2.Laplacian(gray_float, cv2.CV_32F)
    edges = cv2.Canny(gray, 100, 200)
    features["img_laplacian_var"] = float(laplacian.var())
    features["img_edge_density"] = float(np.count_nonzero(edges) / edges.size)

    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-12)
    entropy = -np.sum(hist * np.log2(hist + 1e-12))
    features["img_gray_entropy"] = float(entropy)

    return features


def _read_sensor_csv(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame.columns = [column.strip() for column in frame.columns]
    return frame


def _infer_class_key_from_sensor_csv(csv_path: Path, frame: pd.DataFrame) -> ClassKey:
    parsed = parse_class_key(csv_path.stem)
    if parsed is not None:
        return parsed

    if {"label", "meat"}.issubset(frame.columns):
        label = pretty_label(str(frame["label"].dropna().iloc[0]))
        meat = pretty_label(str(frame["meat"].dropna().iloc[0]))
        return ClassKey(meat_type=meat, freshness=label)

    raise ValueError(f"Could not infer class key from sensor CSV: {csv_path}")


def aggregate_sensor_frame(frame: pd.DataFrame, base_columns: Iterable[str] | None = None) -> dict[str, float]:
    base_columns = list(base_columns or DEFAULT_SENSOR_COLUMNS)
    summary: dict[str, float] = {}

    # Aggregation strategy:
    # We summarize each class-level CSV using mean/min/max/std so that one sensor
    # signature can be attached to every image in the corresponding folder. This
    # avoids pretending that images and sensor rows were perfectly time-aligned.
    for column in base_columns:
        if column not in frame.columns:
            for agg_name in SENSOR_AGGREGATIONS:
                summary[f"sensor_{column}_{agg_name}"] = np.nan
            continue

        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            for agg_name in SENSOR_AGGREGATIONS:
                summary[f"sensor_{column}_{agg_name}"] = np.nan
            continue

        summary[f"sensor_{column}_mean"] = float(series.mean())
        summary[f"sensor_{column}_min"] = float(series.min())
        summary[f"sensor_{column}_max"] = float(series.max())
        summary[f"sensor_{column}_std"] = float(series.std(ddof=0))

    return summary


def load_sensor_summaries(sensor_dir: str | Path, base_columns: Iterable[str] | None = None) -> dict[ClassKey, dict[str, float]]:
    sensor_dir = resolve_existing_path(sensor_dir)
    summaries: dict[ClassKey, dict[str, float]] = {}

    for csv_path in sorted(sensor_dir.glob("*.csv")):
        frame = _read_sensor_csv(csv_path)
        class_key = _infer_class_key_from_sensor_csv(csv_path, frame)
        summary = aggregate_sensor_frame(frame, base_columns=base_columns)
        summary["sensor_source_file"] = str(csv_path)
        summaries[class_key] = summary

    if not summaries:
        raise FileNotFoundError(f"No sensor CSV files found in {sensor_dir}")

    return summaries


def load_sensor_window_summaries(
    sensor_dir: str | Path,
    base_columns: Iterable[str] | None = None,
    window_size: int = DEFAULT_SENSOR_WINDOW_SIZE,
    window_step: int = DEFAULT_SENSOR_WINDOW_STEP,
) -> dict[ClassKey, list[dict[str, float]]]:
    sensor_dir = resolve_existing_path(sensor_dir)
    summaries: dict[ClassKey, list[dict[str, float]]] = {}
    base_columns = list(base_columns or DEFAULT_SENSOR_COLUMNS)
    window_size = max(1, int(window_size))
    window_step = max(1, int(window_step))

    for csv_path in sorted(sensor_dir.glob("*.csv")):
        frame = _read_sensor_csv(csv_path)
        class_key = _infer_class_key_from_sensor_csv(csv_path, frame)
        class_windows: list[dict[str, float]] = []

        if len(frame) <= window_size:
            summary = aggregate_sensor_frame(frame, base_columns=base_columns)
            summary["sensor_source_file"] = str(csv_path)
            summary["sensor_source_group"] = f"{csv_path}::window_0"
            summary["sensor_window_index"] = 0
            summary["sensor_window_start"] = 0
            summary["sensor_window_end"] = int(len(frame) - 1)
            class_windows.append(summary)
        else:
            window_index = 0
            max_start = len(frame) - window_size
            for start in range(0, max_start + 1, window_step):
                window_frame = frame.iloc[start : start + window_size].reset_index(drop=True)
                summary = aggregate_sensor_frame(window_frame, base_columns=base_columns)
                summary["sensor_source_file"] = str(csv_path)
                summary["sensor_source_group"] = f"{csv_path}::window_{window_index}"
                summary["sensor_window_index"] = int(window_index)
                summary["sensor_window_start"] = int(start)
                summary["sensor_window_end"] = int(start + len(window_frame) - 1)
                class_windows.append(summary)
                window_index += 1

            if not class_windows:
                summary = aggregate_sensor_frame(frame, base_columns=base_columns)
                summary["sensor_source_file"] = str(csv_path)
                summary["sensor_source_group"] = f"{csv_path}::window_0"
                summary["sensor_window_index"] = 0
                summary["sensor_window_start"] = 0
                summary["sensor_window_end"] = int(len(frame) - 1)
                class_windows.append(summary)

        summaries[class_key] = class_windows

    if not summaries:
        raise FileNotFoundError(f"No sensor CSV files found in {sensor_dir}")

    return summaries


def build_hybrid_dataset(
    sensor_dir: str | Path,
    image_dir: str | Path,
    base_sensor_columns: Iterable[str] | None = None,
    sensor_window_size: int = DEFAULT_SENSOR_WINDOW_SIZE,
    sensor_window_step: int = DEFAULT_SENSOR_WINDOW_STEP,
) -> pd.DataFrame:
    image_dir = resolve_existing_path(image_dir)
    sensor_summaries = load_sensor_window_summaries(
        sensor_dir,
        base_columns=base_sensor_columns,
        window_size=sensor_window_size,
        window_step=sensor_window_step,
    )

    rows: list[dict[str, Any]] = []
    for folder in sorted(path for path in image_dir.iterdir() if path.is_dir()):
        class_key = parse_class_key(folder.name)
        if class_key is None:
            continue

        if class_key not in sensor_summaries:
            raise KeyError(
                f"No sensor CSV matched image folder '{folder.name}'. "
                f"Expected a CSV for class key '{class_key.slug}'."
            )

        sensor_feature_windows = sensor_summaries[class_key]
        image_files = list_image_files(folder)
        for image_index, image_path in enumerate(image_files):
            image_features = extract_image_features(image_path)
            sensor_features = sensor_feature_windows[image_index % len(sensor_feature_windows)]
            row: dict[str, Any] = {}
            row.update(image_features)
            row.update(sensor_features)
            row["image_path"] = str(image_path)
            row["image_folder"] = folder.name
            row["image_index_within_class"] = int(image_index)
            row["meat_type"] = class_key.meat_type
            row["freshness_label"] = class_key.freshness
            rows.append(row)

    if not rows:
        raise ValueError("No hybrid training rows were created. Check the dataset paths.")

    return pd.DataFrame(rows)


def sensor_values_to_summary_features(
    sensor_values: dict[str, float],
    base_sensor_columns: Iterable[str] | None = None,
) -> dict[str, float]:
    base_sensor_columns = list(base_sensor_columns or DEFAULT_SENSOR_COLUMNS)
    summary: dict[str, float] = {}
    for column in base_sensor_columns:
        value = sensor_values.get(column, np.nan)
        if value is None:
            value = np.nan
        summary[f"sensor_{column}_mean"] = float(value) if not pd.isna(value) else np.nan
        summary[f"sensor_{column}_min"] = float(value) if not pd.isna(value) else np.nan
        summary[f"sensor_{column}_max"] = float(value) if not pd.isna(value) else np.nan
        summary[f"sensor_{column}_std"] = 0.0 if not pd.isna(value) else np.nan
    return summary


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)
