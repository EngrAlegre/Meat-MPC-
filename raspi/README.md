# Raspberry Pi 5 Hybrid Deployment App

This folder contains the live Raspberry Pi deployment program for your hybrid meat freshness detection system.

The Raspberry Pi app:

- reads MQ-137, MQ-136, and MQ-135 through ADS1115 over I2C
- computes voltage, `Rs`, and `Rs/Ro`
- captures images from the Raspberry Pi camera
- extracts the exact same OpenCV image features used during training
- loads the saved hybrid ML model from `..\model`
- predicts:
  - `Fresh`
  - `Neutral`
  - `Spoiled`
- serves a mobile-friendly Flask web UI

Training still stays in:

- `C:\Users\isaac\Downloads\chickenprok\Raspi_5_program\hybrid_ml`

Deployment stays in:

- `C:\Users\isaac\Downloads\chickenprok\Raspi_5_program\raspi`

## Files

- `config.py`
  - all editable constants in one place
  - ADS1115 config
  - RL and RO values
  - warm-up and stabilization settings
  - artifact paths
  - camera folder
  - Flask host and port
- `sensor_reader.py`
  - reads ADS1115
  - averages multiple readings
  - computes voltage, `Rs`, and `Rs/Ro`
  - provides stabilized sensor windows
- `camera_capture.py`
  - captures images from Picamera2
  - falls back to OpenCV camera capture if enabled
- `feature_extractor.py`
  - reuses the exact training-time OpenCV feature extractor
- `predict_live.py`
  - loads the trained artifacts
  - builds a training-compatible feature row
  - runs live prediction
  - appends prediction logs
- `app.py`
  - Flask UI and live API routes
- `requirements.txt`
  - Python dependencies
- `templates/index.html`
  - responsive browser UI
- `static/app.css`
  - styling

## Model Folder

The saved training results now live in:

- `C:\Users\isaac\Downloads\chickenprok\Raspi_5_program\model`

That folder contains the deployed model files such as:

- `hybrid_freshness_model.joblib`
- `freshness_label_encoder.joblib`
- `hybrid_preprocessor.joblib`
- `training_metadata.json`
- evaluation reports and confusion matrices

## Important Compatibility Note

The deployment code does not retrain the model and does not change the preprocessing.

It keeps inference compatible by:

- reusing the exact training image feature extraction logic from `hybrid_pipeline_utils.py`
- using the same saved artifacts from `..\model`
- rebuilding the sensor summary feature names expected by the trained pipeline
- reindexing the live feature row to the exact feature order stored in the fitted preprocessor

## Install Dependencies

Create or activate your Python environment, then run:

```powershell
cd C:\Users\isaac\Downloads\chickenprok\Raspi_5_program\raspi
pip install -r requirements.txt
```

## Enable I2C On Raspberry Pi

1. Run:

```bash
sudo raspi-config
```

2. Go to:

`Interface Options -> I2C -> Enable`

3. Reboot the Pi:

```bash
sudo reboot
```

4. Verify the ADS1115 appears:

```bash
sudo i2cdetect -y 1
```

You should normally see address `48` for the ADS1115 if wiring is correct.

## Camera Setup

For Raspberry Pi OS with Picamera2:

```bash
sudo apt update
sudo apt install -y python3-picamera2
```

If needed, test the camera first:

```bash
libcamera-still -o test.jpg
```

## Where To Edit RL And RO

Edit these values in:

- `C:\Users\isaac\Downloads\chickenprok\Raspi_5_program\raspi\config.py`

Main constants:

- `RL_NH3_KOHM`
- `RL_H2S_KOHM`
- `RL_VOC_KOHM`
- `RO_NH3_KOHM`
- `RO_H2S_KOHM`
- `RO_VOC_KOHM`

If you recalibrate sensors later, update the `RO_*` values there.

## Run The Flask App

```powershell
cd C:\Users\isaac\Downloads\chickenprok\Raspi_5_program\raspi
python app.py
```

By default the app runs on:

- Host: `0.0.0.0`
- Port: `5000`

Those can also be changed in `config.py`.

## Access The Web UI From Another Device

Find the Raspberry Pi IP address:

```bash
hostname -I
```

Then open this on a phone or laptop connected to the same network:

```text
http://<raspberry-pi-ip>:5000
```

Example:

```text
http://192.168.1.25:5000
```

## App Flow

1. Power on the Raspberry Pi and sensors.
2. Wait for warm-up to finish.
3. Open the Flask UI in a browser.
4. Optionally capture a baseline for debug reference.
5. Click `Stabilize Sensors`.
6. Click `Capture Image`.
7. Select the meat type.
8. Click `Predict Freshness`.

The app blocks prediction until:

- warm-up is complete
- sensors are stabilized
- an image has been captured

## Debug And Test Routes

The app includes these helpful routes:

- `GET /api/test-sensors`
  - reads live sensors once
- `POST /api/capture-baseline`
  - captures a temporary baseline for display/debugging only
- `POST /api/test-camera`
  - captures one test image
- `POST /api/test-inference/<meat_type>`
  - runs full inference using the latest stabilized sensor snapshot and latest captured image

## Prediction Logging

Each prediction is appended to:

- `C:\Users\isaac\Downloads\chickenprok\Raspi_5_program\raspi\logs\prediction_log.csv`

Logged fields:

- timestamp
- meat type
- image path
- `nh3_ratio`
- `h2s_ratio`
- `voc_ratio`
- predicted freshness
- confidence

## Notes

- The old ESP32 code was used only as reference for the sensor math and sampling flow.
- The current trained model is an SVM selected from grouped cross-validation.
- If the UI shows confidence without true `predict_proba`, it is an approximate score derived from the SVM decision function and should be treated as a confidence indicator, not a calibrated probability.
- Baseline capture is for operator awareness and debugging only. It is not injected into the trained model input by default.
