from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


HYBRID_ML_DIR = Path(__file__).resolve().parents[1] / "hybrid_ml"
if str(HYBRID_ML_DIR) not in sys.path:
    sys.path.insert(0, str(HYBRID_ML_DIR))

from hybrid_pipeline_utils import extract_image_features as training_extract_image_features


def extract_live_image_features(image_path: str | Path) -> dict[str, float]:
    """Use the exact same OpenCV image feature extractor used during training."""
    return training_extract_image_features(image_path)


def summarize_for_display(features: dict[str, Any]) -> dict[str, float]:
    keys = (
        "img_rgb_r_mean",
        "img_rgb_g_mean",
        "img_rgb_b_mean",
        "img_hsv_h_mean",
        "img_hsv_s_mean",
        "img_hsv_v_mean",
        "img_gray_mean",
        "img_edge_density",
    )
    return {key: float(features[key]) for key in keys if key in features}
