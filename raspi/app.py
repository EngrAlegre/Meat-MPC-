from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from flask import Flask, jsonify, render_template, send_from_directory

import config
from camera_capture import CameraCaptureError, CameraCaptureService
from predict_live import HybridFreshnessPredictor, PredictionLoadError
from sensor_reader import MQSensorReader, SensorInitializationError, SensorReadError


config.ensure_runtime_dirs()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(config.APP_LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
LOGGER = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder=str(config.TEMPLATE_DIR),
    static_folder=str(config.STATIC_DIR),
)
app.config["SECRET_KEY"] = config.SECRET_KEY

sensor_lock = Lock()
camera_lock = Lock()
state_lock = Lock()

_sensor_reader: MQSensorReader | None = None
_camera_service: CameraCaptureService | None = None
_predictor: HybridFreshnessPredictor | None = None

runtime_state: dict[str, Any] = {
    "boot_time_utc": datetime.now(timezone.utc).isoformat(),
    "last_sensor_snapshot": None,
    "baseline_snapshot": None,
    "latest_image_path": None,
    "latest_prediction": None,
    "last_error": None,
    "sensor_ready": False,
}


def get_sensor_reader() -> MQSensorReader:
    global _sensor_reader
    if _sensor_reader is None:
        _sensor_reader = MQSensorReader()
    return _sensor_reader


def get_camera_service() -> CameraCaptureService:
    global _camera_service
    if _camera_service is None:
        _camera_service = CameraCaptureService()
    return _camera_service


def get_predictor() -> HybridFreshnessPredictor:
    global _predictor
    if _predictor is None:
        _predictor = HybridFreshnessPredictor()
    return _predictor


def set_last_error(message: str | None) -> None:
    with state_lock:
        runtime_state["last_error"] = message


def build_image_url(image_path: str | None) -> str | None:
    if not image_path:
        return None
    filename = Path(image_path).name
    return f"/captures/{filename}"


def get_status_payload() -> dict[str, Any]:
    warmup_remaining = None
    warmed_up = False
    try:
        reader = get_sensor_reader()
        warmup_remaining = round(reader.warmup_remaining_seconds(), 2)
        warmed_up = reader.is_warmed_up()
    except Exception as exc:
        warmup_remaining = None
        set_last_error(str(exc))

    with state_lock:
        latest_image_path = runtime_state["latest_image_path"]
        payload = {
            "boot_time_utc": runtime_state["boot_time_utc"],
            "warmup_remaining_seconds": warmup_remaining,
            "warmed_up": warmed_up,
            "sensor_ready": runtime_state["sensor_ready"],
            "latest_image_path": latest_image_path,
            "latest_image_url": build_image_url(latest_image_path),
            "latest_prediction": runtime_state["latest_prediction"],
            "last_sensor_snapshot": runtime_state["last_sensor_snapshot"],
            "baseline_snapshot": runtime_state["baseline_snapshot"],
            "last_error": runtime_state["last_error"],
            "meat_types": list(config.MEAT_TYPES),
        }
    return payload


def json_error(message: str, status_code: int = 400):
    LOGGER.error(message)
    set_last_error(message)
    return jsonify({"ok": False, "message": message, "status": get_status_payload()}), status_code


@app.route("/")
def index() -> str:
    return render_template("index.html", meat_types=config.MEAT_TYPES)


@app.route("/captures/<path:filename>")
def captured_image(filename: str):
    return send_from_directory(str(config.CAMERA_OUTPUT_DIR), filename)


@app.get("/api/status")
def api_status():
    return jsonify({"ok": True, "status": get_status_payload()})


@app.get("/api/test-sensors")
def api_test_sensors():
    try:
        with sensor_lock:
            snapshot = get_sensor_reader().read_once()
        set_last_error(None)
        return jsonify({"ok": True, "snapshot": snapshot, "status": get_status_payload()})
    except (SensorInitializationError, SensorReadError) as exc:
        return json_error(str(exc), 503)


@app.post("/api/stabilize")
def api_stabilize():
    try:
        with sensor_lock:
            reader = get_sensor_reader()
            if not reader.is_warmed_up():
                remaining = reader.warmup_remaining_seconds()
                return json_error(f"Sensors are still warming up. {remaining:.1f} seconds remaining.", 409)

            snapshot = reader.stabilize()

        with state_lock:
            runtime_state["last_sensor_snapshot"] = snapshot
            runtime_state["sensor_ready"] = bool(snapshot["stable"])
            runtime_state["latest_prediction"] = None
        set_last_error(None)
        return jsonify(
            {
                "ok": True,
                "message": "Sensors stabilized." if snapshot["stable"] else "Sensors are not stable yet.",
                "snapshot": snapshot,
                "ready_to_scan": bool(snapshot["stable"]),
                "status": get_status_payload(),
            }
        )
    except (SensorInitializationError, SensorReadError) as exc:
        return json_error(str(exc), 503)


@app.post("/api/capture-baseline")
def api_capture_baseline():
    try:
        with sensor_lock:
            reader = get_sensor_reader()
            if not reader.is_warmed_up():
                remaining = reader.warmup_remaining_seconds()
                return json_error(f"Sensors are still warming up. {remaining:.1f} seconds remaining.", 409)
            baseline = reader.capture_baseline()

        with state_lock:
            runtime_state["baseline_snapshot"] = baseline
        set_last_error(None)
        return jsonify({"ok": True, "baseline": baseline, "status": get_status_payload()})
    except (SensorInitializationError, SensorReadError) as exc:
        return json_error(str(exc), 503)


@app.post("/api/test-camera")
@app.post("/api/capture-image")
def api_capture_image():
    try:
        with camera_lock:
            image_path = get_camera_service().capture_image()
        with state_lock:
            runtime_state["latest_image_path"] = str(image_path)
            runtime_state["latest_prediction"] = None
        set_last_error(None)
        return jsonify(
            {
                "ok": True,
                "image_path": str(image_path),
                "image_url": build_image_url(str(image_path)),
                "status": get_status_payload(),
            }
        )
    except CameraCaptureError as exc:
        return json_error(str(exc), 503)


def _run_prediction(meat_type: str) -> dict[str, Any]:
    with state_lock:
        sensor_snapshot = runtime_state["last_sensor_snapshot"]
        image_path = runtime_state["latest_image_path"]
        sensor_ready = runtime_state["sensor_ready"]

    if meat_type not in config.MEAT_TYPES:
        raise ValueError(f"Invalid meat type: {meat_type}")
    if not sensor_ready or not sensor_snapshot:
        raise RuntimeError("Stabilize the sensors first before running prediction.")
    if not image_path:
        raise RuntimeError("Capture an image first before running prediction.")

    predictor = get_predictor()
    result = predictor.predict(
        image_path=image_path,
        meat_type=meat_type,
        sensor_values=sensor_snapshot["model_sensor_values"],
    )
    predictor.append_prediction_log(result)

    result_payload = {
        "timestamp_utc": result.timestamp_utc,
        "meat_type": result.meat_type,
        "image_path": result.image_path,
        "image_url": build_image_url(result.image_path),
        "predicted_freshness": result.predicted_freshness,
        "confidence": result.confidence,
        "confidence_note": result.confidence_note,
        "class_probabilities": result.class_probabilities,
        "sensor_values": result.sensor_values,
        "image_feature_preview": result.image_feature_preview,
    }
    with state_lock:
        runtime_state["latest_prediction"] = result_payload
    return result_payload


@app.post("/api/predict/<meat_type>")
def api_predict(meat_type: str):
    try:
        result_payload = _run_prediction(meat_type.capitalize())
        set_last_error(None)
        return jsonify({"ok": True, "prediction": result_payload, "status": get_status_payload()})
    except (RuntimeError, ValueError) as exc:
        return json_error(str(exc), 409)
    except PredictionLoadError as exc:
        return json_error(str(exc), 500)


@app.post("/api/test-inference/<meat_type>")
def api_test_inference(meat_type: str):
    return api_predict(meat_type)


if __name__ == "__main__":
    LOGGER.info("Starting Raspberry Pi Flask app on %s:%s", config.FLASK_HOST, config.FLASK_PORT)
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG)
