from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
except Exception as exc:  # pragma: no cover - depends on local runtime
    tf = None
    preprocess_input = None
    TENSORFLOW_IMPORT_ERROR = exc
else:
    TENSORFLOW_IMPORT_ERROR = None


DEFAULT_INPUT_SIZE = (224, 224)


class MeatClassifierRuntimeError(RuntimeError):
    pass


@dataclass
class MeatClassifierArtifacts:
    model: Any
    class_names: list[str]
    metadata: dict[str, Any]


def _require_tensorflow() -> None:
    if tf is None or preprocess_input is None:
        raise MeatClassifierRuntimeError(
            "TensorFlow is required for the meat classifier but is not available. "
            f"Original import error: {TENSORFLOW_IMPORT_ERROR}"
        )


def load_json_file(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json_file(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_meat_classifier(
    model_path: str | Path,
    class_names_path: str | Path,
    metadata_path: str | Path | None = None,
) -> MeatClassifierArtifacts:
    _require_tensorflow()

    model = tf.keras.models.load_model(model_path)
    class_names = list(load_json_file(class_names_path))
    metadata = load_json_file(metadata_path) if metadata_path and Path(metadata_path).exists() else {}
    return MeatClassifierArtifacts(model=model, class_names=class_names, metadata=metadata)


def prepare_image_tensor(image_path: str | Path, input_size: tuple[int, int] | None = None) -> np.ndarray:
    _require_tensorflow()

    width, height = input_size or DEFAULT_INPUT_SIZE
    image = Image.open(image_path).convert("RGB").resize((width, height))
    array = np.asarray(image, dtype=np.float32)
    array = preprocess_input(array)
    return np.expand_dims(array, axis=0)


def predict_meat_class(
    artifacts: MeatClassifierArtifacts,
    image_path: str | Path,
) -> dict[str, Any]:
    input_size = tuple(artifacts.metadata.get("input_size", DEFAULT_INPUT_SIZE))
    image_tensor = prepare_image_tensor(image_path, input_size=input_size)
    probabilities = artifacts.model.predict(image_tensor, verbose=0)[0]

    class_probabilities = {
        class_name: float(score)
        for class_name, score in zip(artifacts.class_names, probabilities)
    }
    predicted_index = int(np.argmax(probabilities))
    predicted_class = artifacts.class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "class_probabilities": class_probabilities,
        "input_size": input_size,
    }
