from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROGRAM_DIR = BASE_DIR.parent
HYBRID_ML_DIR = PROGRAM_DIR / "hybrid_ml"
MODEL_DIR = PROGRAM_DIR / "model"
MODAL_RUNS_DIR = MODEL_DIR / "modal_runs"

CAPTURE_DIR = BASE_DIR / "captures"
LOG_DIR = BASE_DIR / "logs"
PREDICTION_LOG_PATH = LOG_DIR / "prediction_log.csv"
APP_LOG_PATH = LOG_DIR / "raspi_app.log"

TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = False
SECRET_KEY = "meat-freshness-raspi5"

MODEL_PATH = MODEL_DIR / "hybrid_freshness_model.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "freshness_label_encoder.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "hybrid_preprocessor.joblib"
TRAINING_METADATA_PATH = MODEL_DIR / "training_metadata.json"

# Deployment model mode:
# - "sensor_only" uses MQ ratio summaries + meat type
# - "image_only" uses image features + meat type
# - "hybrid" fuses image-only model scores with sensor nearest-class scores
MODEL_MODE = "hybrid"

HYBRID_IMAGE_WEIGHT = 0.65
HYBRID_SENSOR_WEIGHT = 0.35

ADS_I2C_ADDRESS = 0x48
ADS_GAIN = 1
ADS_DATA_RATE = 128

ADS_CHANNEL_NH3 = 0
ADS_CHANNEL_H2S = 1
ADS_CHANNEL_VOC = 2

VC = 5.0
RL_NH3_KOHM = 10.0
RL_H2S_KOHM = 10.0
RL_VOC_KOHM = 10.0
RO_NH3_KOHM = 35.0
RO_H2S_KOHM = 34.0
RO_VOC_KOHM = 41.7

ADS_AVERAGE_SAMPLES = 20
ADS_SAMPLE_DELAY_SECONDS = 0.005
SENSOR_WARMUP_SECONDS = 30
SENSOR_LIVE_REFRESH_SECONDS = 2.0
SCAN_SUMMARY_READS = 15
STABILIZATION_WINDOW_READS = 30
STABILIZATION_MIN_READS = 20
STABILIZATION_MAX_READS = 50
STABILITY_STD_LIMITS = {
    "nh3_ratio": 0.03,
    "h2s_ratio": 0.03,
    "voc_ratio": 0.05,
}

CAMERA_OUTPUT_DIR = CAPTURE_DIR
CAMERA_STILL_SIZE = (1280, 720)
CAMERA_PREVIEW_SIZE = (640, 360)
CAMERA_PREVIEW_REFRESH_SECONDS = 0.8
CAMERA_FILENAME_PREFIX = "capture"
CAMERA_STARTUP_DELAY_SECONDS = 1.0
ALLOW_OPENCV_CAMERA_FALLBACK = True

MEAT_TYPES = ("Chicken", "Beef", "Pork")
FRESHNESS_CLASSES = ("Fresh", "Neutral", "Spoiled")
MEAT_BUTTON_GPIO_MAP = {
    "Chicken": 17,
    "Pork": 27,
    "Beef": 22,
}
BUTTON_BOUNCE_SECONDS = 0.15

DHT22_ENABLED = True
DHT22_GPIO_PIN = 4
DHT22_READ_RETRIES = 3
DHT22_RETRY_DELAY_SECONDS = 0.8
DHT22_REFRESH_SECONDS = 3.0

# Deployment sensor alignment rescales live Raspberry Pi Rs/Ro values so they
# stay on the same feature scale as the training-time collection setup.
# This is not thresholding or rule-based classification; the ML model remains
# the final decision-maker.
RUNTIME_RATIO_ADJUSTMENT_ENABLED = True
RUNTIME_RATIO_SCALE = {
    "nh3_ratio": 0.7071,
    "h2s_ratio": 0.3769,
    "voc_ratio": 0.1017,
}


def ensure_runtime_dirs() -> None:
    for path in (CAPTURE_DIR, LOG_DIR, TEMPLATE_DIR, STATIC_DIR):
        path.mkdir(parents=True, exist_ok=True)
