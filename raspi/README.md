# Raspberry Pi 5 Hybrid Deployment App

This folder now contains a native Raspberry Pi touchscreen GUI for your hybrid meat freshness detection system.

The Raspberry Pi desktop app:

- reads MQ-137, MQ-136, and MQ-135 through ADS1115 over I2C
- reads DHT22 for temperature and humidity monitoring
- computes voltage, `Rs`, and `Rs/Ro`
- captures images from the Raspberry Pi camera
- extracts the exact same OpenCV image features used during training
- loads the saved hybrid ML model from `..\model`
- predicts:
  - `Fresh`
  - `Neutral`
  - `Spoiled`
- runs as a local touchscreen-friendly GUI instead of a website
- supports the physical meat-selection buttons connected to Raspberry Pi GPIO

## Main Files

- `app.py`
  - main touchscreen GUI
  - full-screen layout
  - on-screen meat type display and selection
  - physical GPIO meat button support
  - one-tap `Start Scan` flow
  - live debug log
- `config.py`
  - editable constants and paths
- `sensor_reader.py`
  - ADS1115 + MQ logic
  - DHT22 environmental monitoring
- `camera_capture.py`
  - Pi camera capture
- `feature_extractor.py`
  - exact training-compatible image features
- `predict_live.py`
  - live hybrid inference
- `requirements.txt`
  - Python dependencies

## Model Folder

The trained model outputs live here:

- `C:\Users\isaac\Downloads\chickenprok\Raspi_5_program\model`

Important files:

- `hybrid_freshness_model.joblib`
- `freshness_label_encoder.joblib`
- `hybrid_preprocessor.joblib`
- `training_metadata.json`

## Important Compatibility Note

The deployment code does not retrain the model and does not change the preprocessing.

It keeps inference compatible by:

- reusing the exact training image feature extraction logic from `hybrid_pipeline_utils.py`
- using the same saved artifacts from `..\model`
- rebuilding the same sensor summary feature names expected by the trained pipeline
- reindexing the live feature row to the exact feature order stored in the fitted preprocessor

## Install Dependencies

On Raspberry Pi:

```bash
cd ~/Documents/Meat/raspi
python3 -m pip install -r requirements.txt
```

If Tkinter is missing:

```bash
sudo apt update
sudo apt install -y python3-tk
```

If Picamera2 is missing:

```bash
sudo apt install -y python3-picamera2
```

If DHT22 support is missing:

```bash
python3 -m pip install adafruit-circuitpython-dht
```

## Enable I2C On Raspberry Pi

1. Run:

```bash
sudo raspi-config
```

2. Go to:

`Interface Options -> I2C -> Enable`

3. Reboot:

```bash
sudo reboot
```

4. Check the ADS1115:

```bash
sudo i2cdetect -y 1
```

You should usually see `48`.

## Camera Setup

Test the camera first:

```bash
libcamera-still -o test.jpg
```

## Where To Edit RL And RO

Edit these constants in:

- `config.py`

Main values:

- `RL_NH3_KOHM`
- `RL_H2S_KOHM`
- `RL_VOC_KOHM`
- `RO_NH3_KOHM`
- `RO_H2S_KOHM`
- `RO_VOC_KOHM`
- `DHT22_ENABLED`
- `DHT22_GPIO_PIN`

## Run The GUI App

```bash
cd ~/Documents/Meat/raspi
python3 app.py
```

The GUI opens directly on the Raspberry Pi screen.

## GUI Flow

1. Power on the Raspberry Pi and sensors.
2. Wait for warm-up to finish.
3. Select the meat type: Chicken, Pork, or Beef.
   - this can be done with the physical GPIO buttons
   - the on-screen buttons are still available as backup
4. Tap `Start Scan`.
5. The app automatically:
   - stabilizes the sensors
   - reads DHT22 temperature and humidity for environmental context
   - captures the image
   - runs prediction
6. Read the result on screen.

The app does not allow a real scan until warm-up is complete.

## What The GUI Shows

- current app state
- warm-up status
- live NH3, H2S, and VOC ratios
- live temperature and humidity from DHT22
- voltage and `Rs` values for debugging
- captured image preview
- predicted freshness
- confidence indicator
- class score breakdown
- live debug log

## Notes

- The current UI is native, not browser-based.
- Press `Esc` to leave full-screen mode.
- Default physical button mapping is:
  - Chicken = GPIO17
  - Pork = GPIO27
  - Beef = GPIO22
- Default DHT22 pin is GPIO4.
- Those GPIO mappings can be changed in `config.py`.
- The confidence shown for the SVM model may be an approximate confidence derived from decision scores if true probability output is unavailable.
- DHT22 values are for environmental monitoring only and are not currently part of the trained model input.
- Prediction logs are still saved to:
  - `logs/prediction_log.csv`
