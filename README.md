# FreshTo

A hybrid meat freshness detection system that combines gas sensor readings, camera images, and machine learning to determine whether chicken, pork, or beef is fresh, neutral, or spoiled.

Built as a capstone/thesis project. Runs on a Raspberry Pi 5.

## How It Works

1. A camera inside a sealed chamber captures an image of the meat.
2. A **MobileNetV2 image classifier** identifies the meat type (chicken, pork, or beef) and rejects non-meat objects.
3. Three **MQ gas sensors** (NH3, H2S, VOC) measure gas levels around the meat as Rs/Ro ratios via an ADS1115 ADC.
4. A **hybrid SVM model** fuses image features and sensor data to predict freshness: **Fresh**, **Neutral**, or **Spoiled**.
5. Everything is displayed on a fullscreen Tkinter GUI — no manual input needed.

The system is fully automatic: it detects when meat is placed in the chamber, runs the analysis, shows the result, and resets when the meat is removed.

## Project Structure

```
Raspi_5_program/
├── raspi/                      # Raspberry Pi deployment app
│   ├── app.py                  # Main GUI and state machine
│   ├── config.py               # All runtime settings
│   ├── meat_classifier.py      # Meat type classification service
│   ├── predict_live.py         # Hybrid freshness predictor
│   ├── sensor_reader.py        # ADS1115 + MQ + DHT22 sensor reading
│   ├── camera_capture.py       # Picamera2 capture and preview
│   ├── chamber_detector.py     # Object detection via frame differencing
│   ├── button_input.py         # Physical button handler (GPIO)
│   ├── feature_extractor.py    # Image feature extraction for ML
│   ├── requirements.txt        # Python dependencies
│   └── docs/                   # Wiring guide PDF
│
├── hybrid_ml/                  # Training and inference scripts
│   ├── train_meat_classifier.py    # Train the MobileNetV2 meat classifier
│   ├── infer_meat_classifier.py    # Run meat classifier on a single image
│   ├── meat_classifier_utils.py    # Shared utilities for meat classifier
│   ├── train_hybrid_model.py       # Train the hybrid freshness model
│   ├── train_modal_models.py       # Train sensor-only / image-only / hybrid variants
│   ├── infer_hybrid_model.py       # Run freshness inference on a single sample
│   ├── hybrid_pipeline_utils.py    # Feature extraction and pipeline utilities
│   ├── deploy_hybrid_app.py        # Deployment helper
│   └── deployment_runtime.py       # Runtime utilities for deployment
│
├── model/                      # Trained models and evaluation results
│   ├── meat_classifier/        # MobileNetV2 meat classifier (.keras)
│   ├── modal_runs/             # Per-modality model comparisons
│   │   ├── hybrid/             # Image + sensor fusion (SVM RBF)
│   │   ├── image_only/         # Image features only
│   │   └── sensor_only/        # Sensor features only
│   ├── hybrid_freshness_model.joblib
│   ├── freshness_label_encoder.joblib
│   ├── hybrid_preprocessor.joblib
│   └── training_metadata.json
│
└── Dataset/                    # Training data (not in repo)
    ├── Image/                  # Meat images by class
    │   ├── chicken_fresh/
    │   ├── chicken_neutral/
    │   ├── chicken_spoiled/
    │   ├── pork_fresh/ ... beef_spoiled/
    │   └── not_meat/
    └── MQ_Sensors_Dataset/     # Sensor CSV files per meat type and freshness
```

## Hardware

| Component | Purpose |
|---|---|
| Raspberry Pi 5 | Controller and inference device |
| Camera Module v2 (IMX219) | Meat image capture |
| ADS1115 (I2C) | Analog-to-digital converter for gas sensors |
| MQ-137 | Ammonia (NH3) sensor |
| MQ-136 | Hydrogen Sulfide (H2S) sensor |
| MQ-135 | VOC sensor |
| DHT22 | Temperature and humidity |
| 3x Push buttons | UI scroll and empty reference capture |

Full wiring diagram with pin mappings: `raspi/docs/Raspberry_Pi_5_Hybrid_Meat_Freshness_Wiring_Guide.pdf`

## Models

### Meat Classifier

- **Architecture**: MobileNetV2 (transfer learning, fine-tuned top 40 layers)
- **Classes**: chicken, pork, beef, not_meat
- **Dataset**: 6,487 images (2,160 chicken, 2,073 pork, 1,905 beef, 349 not_meat)
- **Validation accuracy**: 99.5%

### Freshness Model

- **Architecture**: SVM RBF (hybrid mode), Random Forest (image-only mode)
- **Features**: 19 image features (RGB, HSV, grayscale stats, edge density, entropy) + sensor Rs/Ro ratios + meat type
- **Classes**: Fresh, Neutral, Spoiled
- **Grouped CV accuracy**: 100% across all modalities
- **Deployment mode**: Hybrid (65% image weight, 35% sensor weight)

## Quick Start

### 1. Clone and set up

```bash
git clone <repo-url>
cd Raspi_5_program/raspi
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If Tkinter or Picamera2 is missing:

```bash
sudo apt install -y python3-tk python3-picamera2
```

### 2. Enable I2C

```bash
sudo raspi-config
# Interface Options → I2C → Enable
sudo reboot
```

### 3. Camera config

In `/boot/firmware/config.txt`:

```ini
dtoverlay=imx219
camera_auto_detect=0
```

### 4. Run

```bash
cd raspi
source .venv/bin/activate
python3 app.py
```

Start with an empty chamber. The system captures a reference frame, then monitors automatically.

## Training

### Meat classifier

```bash
cd hybrid_ml
python train_meat_classifier.py --dataset-dir ../Dataset/Image
```

Output goes to `model/meat_classifier/`.

### Freshness model

```bash
cd hybrid_ml
python train_hybrid_model.py
python train_modal_models.py
```

Output goes to `model/` and `model/modal_runs/`.

## Pipeline

```
Camera Image
    │
    ▼
Meat Classifier (MobileNetV2)
    │
    ├── not_meat → "No valid meat detected" (stop)
    │
    └── chicken / pork / beef
            │
            ▼
    Hybrid Freshness Model (SVM RBF)
    ┌───────────────────────────┐
    │ Image features (65%)      │
    │ + Sensor Rs/Ro (35%)      │
    │ + Meat type               │
    └───────────────────────────┘
            │
            ▼
    Fresh / Neutral / Spoiled
```

## Key Configuration

All settings are in `raspi/config.py`:

| Setting | Default | What it does |
|---|---|---|
| `MODEL_MODE` | `"hybrid"` | Fusion mode: `hybrid`, `image_only`, or `sensor_only` |
| `HYBRID_IMAGE_WEIGHT` | `0.65` | Image branch weight in hybrid fusion |
| `HYBRID_SENSOR_WEIGHT` | `0.35` | Sensor branch weight in hybrid fusion |
| `MEAT_CLASSIFIER_MIN_CONFIDENCE` | `0.40` | Minimum confidence to accept a meat classification |
| `OBJECT_STABILITY_DURATION_SECONDS` | `3.0` | Seconds of no movement before scanning |
| `SENSOR_WARMUP_SECONDS` | `30` | Gas sensor warm-up before first scan |
| `RUNTIME_RATIO_ADJUSTMENT_ENABLED` | `True` | Scale live sensor ratios to match training data range |

## License

Capstone/thesis project.
