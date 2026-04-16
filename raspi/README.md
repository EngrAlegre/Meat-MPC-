# FreshTo

A Raspberry Pi 5-based system that detects meat type and predicts freshness using gas sensors, a camera, and machine learning.

## What It Does

1. A camera watches a sealed chamber continuously.
2. When meat is placed inside, the system automatically detects it.
3. An image classifier identifies whether it's **chicken**, **pork**, or **beef** (rejects non-meat objects).
4. Gas sensors (MQ-137, MQ-136, MQ-135) read ammonia, hydrogen sulfide, and VOC levels.
5. A hybrid ML model fuses the image and sensor data to predict freshness: **Fresh**, **Neutral**, or **Spoiled**.
6. Results are shown on a fullscreen Tkinter GUI on the Pi's display.
7. When the meat is removed, the system resets and waits for the next sample.

No manual interaction is needed — everything runs automatically after startup.

## Hardware

| Component | Purpose |
|---|---|
| Raspberry Pi 5 | Main controller and inference device |
| Camera Module v2 (IMX219) | Captures meat images |
| ADS1115 (I2C ADC) | Reads analog gas sensor outputs |
| MQ-137 | Ammonia (NH3) detection |
| MQ-136 | Hydrogen Sulfide (H2S) detection |
| MQ-135 | VOC detection |
| DHT22 | Ambient temperature and humidity |
| 3x Push buttons | Scroll up, scroll down, capture empty reference |

A full wiring guide with pin mappings is in `docs/Raspberry_Pi_5_Hybrid_Meat_Freshness_Wiring_Guide.pdf`.

## Project Structure

```
raspi/
├── app.py                  # Main GUI app and state machine
├── config.py               # All settings and paths
├── meat_classifier.py      # Meat type classifier (MobileNetV2)
├── predict_live.py         # Hybrid freshness predictor
├── sensor_reader.py        # ADS1115 + MQ + DHT22 sensor reading
├── camera_capture.py       # Picamera2 capture and preview
├── chamber_detector.py     # Object detection via frame differencing
├── button_input.py         # Physical button handler (GPIO)
├── feature_extractor.py    # Image feature extraction for ML
├── requirements.txt        # Python dependencies
├── captures/               # Saved images (empty reference, scans)
├── logs/                   # Runtime and prediction logs
├── docs/                   # Wiring guide PDF
└── templates/ + static/    # Legacy web UI assets
```

## How It Works

The system runs a state machine:

```
INITIALIZING → WARMING_UP → WAITING_FOR_OBJECT → OBJECT_DETECTED
→ STABILITY_CHECK → CLASSIFYING_MEAT → STABILIZING_SENSORS
→ PREDICTING_FRESHNESS → SHOWING_RESULT → WAITING_FOR_REMOVAL → (reset)
```

- **Object detection**: Compares live camera frames against an empty chamber reference image.
- **Stability check**: Waits for the frame to stop changing (no hand/movement) before analyzing.
- **Meat classification**: MobileNetV2-based image classifier trained on chicken, pork, beef, and not_meat.
- **Freshness prediction**: Hybrid model that combines image features with sensor Rs/Ro ratios, weighted 65% image / 35% sensor.
- **Automatic reset**: When the object is removed and the chamber looks empty again, the system returns to idle.

## Setup

### Prerequisites

- Raspberry Pi 5 with Raspberry Pi OS (Bookworm or later)
- Camera Module v2 connected via ribbon cable
- ADS1115, MQ sensors, and DHT22 wired as described in the wiring guide
- I2C enabled on the Pi

### Enable I2C

```bash
sudo raspi-config
# Interface Options → I2C → Enable
sudo reboot
sudo i2cdetect -y 1
# Should show address 0x48
```

### Install Dependencies

```bash
cd raspi
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If Tkinter is missing:

```bash
sudo apt install -y python3-tk
```

If Picamera2 is missing:

```bash
sudo apt install -y python3-picamera2
```

### Camera Config

Make sure `/boot/firmware/config.txt` has the correct overlay for your camera:

```ini
# For Camera Module v2 (IMX219)
dtoverlay=imx219
camera_auto_detect=0
```

Reboot after changing camera config.

## Run

```bash
cd raspi
source .venv/bin/activate
python3 app.py
```

The app launches fullscreen on the Pi's display. Start with an empty chamber — the system captures an empty reference frame on startup, then begins monitoring.

## Key Settings

All configurable in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `MODEL_MODE` | `"hybrid"` | `hybrid`, `image_only`, or `sensor_only` |
| `OBJECT_STABILITY_DURATION_SECONDS` | `3.0` | How long the object must stay still before scanning |
| `OBJECT_DETECTION_THRESHOLD` | `0.025` | Frame difference needed to detect an object |
| `SENSOR_WARMUP_SECONDS` | `30` | Gas sensor warm-up time before first scan |
| `MEAT_CLASSIFIER_MIN_CONFIDENCE` | `0.40` | Minimum confidence to accept a meat classification |
| `HYBRID_IMAGE_WEIGHT` | `0.65` | Image model weight in hybrid fusion |
| `HYBRID_SENSOR_WEIGHT` | `0.35` | Sensor model weight in hybrid fusion |

## Models

Models are stored in the `model/` directory (one level up from `raspi/`):

- **Meat classifier** (`model/meat_classifier/`): MobileNetV2 fine-tuned on chicken, pork, beef, and not_meat images.
- **Freshness model** (`model/`): SVM RBF trained on hybrid image + sensor features, with separate models per modality.

Training scripts are in `hybrid_ml/`.

## License

This project was built as a college capstone/thesis project.
