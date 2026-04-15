from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import config


def prepare_detection_frame(image: Image.Image, size: tuple[int, int] | None = None) -> np.ndarray:
    size = size or config.CHAMBER_DETECTION_IMAGE_SIZE
    grayscale = ImageOps.grayscale(image)
    resized = ImageOps.contain(grayscale, size)
    canvas = Image.new("L", size, color=0)
    offset_x = max((size[0] - resized.width) // 2, 0)
    offset_y = max((size[1] - resized.height) // 2, 0)
    canvas.paste(resized, (offset_x, offset_y))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def difference_score(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    return float(np.mean(np.abs(frame_a - frame_b)))


def load_reference_frame(path: str | Path) -> np.ndarray | None:
    reference_path = Path(path)
    if not reference_path.exists():
        return None
    image = Image.open(reference_path).convert("RGB")
    return prepare_detection_frame(image)


def save_reference_image(image: Image.Image, path: str | Path) -> None:
    reference_path = Path(path)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(reference_path)
