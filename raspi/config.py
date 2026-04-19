from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROGRAM_DIR = BASE_DIR.parent
HYBRID_ML_DIR = PROGRAM_DIR / "hybrid_ml"
MODEL_DIR = PROGRAM_DIR / "model"
MODAL_RUNS_DIR = MODEL_DIR / "modal_runs"
MEAT_CLASSIFIER_DIR = MODEL_DIR / "meat_classifier"

CAPTURE_DIR = BASE_DIR / "captures"
LOG_DIR = BASE_DIR / "logs"
PREDICTION_LOG_PATH = LOG_DIR / "prediction_log.csv"
APP_LOG_PATH = LOG_DIR / "raspi_app.log"
EMPTY_CHAMBER_SENSOR_BASELINE_PATH = CAPTURE_DIR / "chamber_calibration.json"

TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = False
SECRET_KEY = "meat-freshness-raspi5"

AUTOMATION_ENABLED = True
EMPTY_CHAMBER_REFERENCE_IMAGE_PATH = CAPTURE_DIR / "empty_chamber_reference.jpg"
ALWAYS_CAPTURE_EMPTY_REFERENCE_ON_STARTUP = False
AUTO_CAPTURE_EMPTY_REFERENCE_IF_MISSING = False
CHAMBER_DETECTION_IMAGE_SIZE = (160, 120)
EMPTY_REFERENCE_CAPTURE_FRAME_COUNT = 5
EMPTY_REFERENCE_CAPTURE_INTERVAL_SECONDS = 0.25
OBJECT_PRESENCE_CONFIRM_SECONDS = 1.5
OBJECT_DETECTION_THRESHOLD = 0.025
OBJECT_STABILITY_DURATION_SECONDS = 3.0
OBJECT_STABLE_FRAME_DIFF_THRESHOLD = 0.045
OBJECT_MONITOR_INTERVAL_SECONDS = 0.8
RESULT_HOLD_MIN_SECONDS = 2.0
REMOVAL_DETECTION_THRESHOLD = 0.012
REMOVAL_STABILITY_SECONDS = 1.5
AUTO_RESET_COOLDOWN_SECONDS = 1.0
AUTO_SENSOR_STABILIZATION_READ_COUNT = 20
SCROLL_UP_GPIO_PIN = 17
SCROLL_DOWN_GPIO_PIN = 27
RESERVED_BUTTON_GPIO_PIN = 22
SCROLL_BUTTON_STEP_UNITS = 3
BUTTON_PULL_UP = True

MODEL_PATH = MODEL_DIR / "hybrid_freshness_model.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "freshness_label_encoder.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "hybrid_preprocessor.joblib"
TRAINING_METADATA_PATH = MODEL_DIR / "training_metadata.json"
MEAT_CLASSIFIER_MODEL_PATH = MEAT_CLASSIFIER_DIR / "meat_classifier.keras"
MEAT_CLASSIFIER_CLASS_NAMES_PATH = MEAT_CLASSIFIER_DIR / "class_names.json"
MEAT_CLASSIFIER_METADATA_PATH = MEAT_CLASSIFIER_DIR / "metadata.json"
MEAT_CLASSIFIER_NOT_MEAT_LABEL = "not_meat"
MEAT_CLASSIFIER_VALID_LABELS = ("chicken", "pork", "beef")
MEAT_CLASSIFIER_MIN_CONFIDENCE = 0.40
MEAT_CLASSIFIER_TO_HYBRID_MEAT_TYPE = {
    "chicken": "Chicken",
    "pork": "Pork",
    "beef": "Beef",
}

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

# Product rule requested for deployment behavior:
# if NH3 or H2S ratio falls below its max threshold, force the final result
# to Spoiled. VOC is not used for this rule.
# SPOILED_OVERRIDE_RATIO_THRESHOLD is kept for backward compatibility and is
# used as the fallback for both NH3 and H2S when the per-gas thresholds are
# not set.
SPOILED_OVERRIDE_ENABLED = True
SPOILED_OVERRIDE_RATIO_THRESHOLD = 0.55
SPOILED_OVERRIDE_NH3_MAX = 0.55
SPOILED_OVERRIDE_H2S_MAX = 0.35

# Product rule for clean-air (fresh) cases: if NH3 and H2S ratios are both
# above their min thresholds, force the final result to Fresh. Acts as the
# upper-band counterpart of the Spoiled override.
FRESH_OVERRIDE_ENABLED = True
FRESH_OVERRIDE_NH3_MIN = 0.85
FRESH_OVERRIDE_H2S_MIN = 0.55

# Product rule for the middle band: if NH3 and H2S both fall inside the
# Neutral window (between the Spoiled and Fresh thresholds), force Neutral.
# Only evaluated when neither Spoiled nor Fresh override fires.
NEUTRAL_OVERRIDE_ENABLED = True
NEUTRAL_OVERRIDE_NH3_MIN = 0.55
NEUTRAL_OVERRIDE_NH3_MAX = 0.85
NEUTRAL_OVERRIDE_H2S_MIN = 0.35
NEUTRAL_OVERRIDE_H2S_MAX = 0.55

# Product rule for borderline cases: if the final fused confidence is too low
# AND none of the band overrides fired, return Neutral instead of allowing a
# weak Fresh/Spoiled decision.
LOW_CONFIDENCE_NEUTRAL_OVERRIDE_ENABLED = True
LOW_CONFIDENCE_NEUTRAL_THRESHOLD = 0.45

# Baseline-relative override:
# We capture an empty-chamber sensor baseline when the user presses the
# "Capture Empty Reference" button (or automatically if missing). At scan
# time we compute delta = current_ratio - baseline_ratio. MQ Rs/Ro normally
# decreases as gas concentration rises, so a negative delta means more gas
# is present. This makes classification stable against environmental drift.
# This override takes priority over the absolute-band overrides whenever a
# baseline is available.
BASELINE_DELTA_OVERRIDE_ENABLED = True
EMPTY_CHAMBER_SENSOR_BASELINE_READS = 6
# Spoiled fires if NH3 OR H2S drops more than this much below baseline.
BASELINE_DELTA_NH3_SPOILED_DROP = 0.08
BASELINE_DELTA_H2S_SPOILED_DROP = 0.04
# Fresh fires if both NH3 and H2S stay within this tolerance of baseline.
BASELINE_DELTA_NH3_FRESH_TOLERANCE = 0.06
BASELINE_DELTA_H2S_FRESH_TOLERANCE = 0.04
# If neither Fresh nor Spoiled bands match, the result is Neutral.

# Presentation override:
# Last-resort manual override for the defense demo. When enabled, the
# predictor forces the result based on the detected meat type. Leave
# DEMO_PRESENTATION_OVERRIDE_ENABLED = False unless you explicitly want
# this for the defense run.
DEMO_PRESENTATION_OVERRIDE_ENABLED = False
DEMO_PRESENTATION_PER_MEAT = {
    # "Chicken": "Fresh",
    # "Pork": "Spoiled",
    # "Beef": "Neutral",
}
# If set (one of "Fresh"/"Neutral"/"Spoiled") this is used regardless of
# meat type. Takes priority over DEMO_PRESENTATION_PER_MEAT.
DEMO_PRESENTATION_FORCED_RESULT = None


def ensure_runtime_dirs() -> None:
    for path in (CAPTURE_DIR, LOG_DIR, TEMPLATE_DIR, STATIC_DIR):
        path.mkdir(parents=True, exist_ok=True)
