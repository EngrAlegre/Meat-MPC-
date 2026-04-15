# FreshTo Raspberry Pi Runtime

This folder contains the native Raspberry Pi 5 deployment app for `FreshTo`.

## Current Architecture

The app keeps the approved **Option B** pipeline:

1. camera image
2. meat classifier
   - `not_meat`
   - `chicken`
   - `pork`
   - `beef`
3. if `not_meat`
   - reject
   - do not run freshness prediction
4. if `chicken`, `pork`, or `beef`
   - pass detected meat type into the existing hybrid freshness model
   - use live MQ sensor ratios and image features
5. output freshness
   - `Fresh`
   - `Neutral`
   - `Spoiled`

The existing hybrid freshness model, MQ math, deployment alignment, and feature order are kept unchanged.

## Fully Automatic Flow

The Raspberry Pi app is now fully automatic.

Flow:

1. wait for sensor warm-up
2. capture a fresh empty chamber reference image at startup
3. monitor the chamber continuously using low-rate preview frames
4. compare each preview frame against the empty chamber reference
5. if a new object appears, run a short stability check
6. after the object stays stable long enough:
   - capture the analysis image
   - run the meat classifier
7. if the classifier says `not_meat`
   - show `No valid meat detected`
   - wait for object removal
8. if the classifier says `chicken`, `pork`, or `beef`
   - stabilize sensors
   - run the existing hybrid freshness model
   - show detected meat type and freshness result
9. keep the result on screen while the object stays inside
10. when the object is removed and the chamber becomes close to the empty reference again, reset automatically to idle monitoring

## State Machine

The automatic runtime uses clear states:

- `INITIALIZING`
- `WARMING_UP`
- `WAITING_FOR_OBJECT`
- `OBJECT_DETECTED`
- `STABILITY_CHECK`
- `CLASSIFYING_MEAT`
- `STABILIZING_SENSORS`
- `PREDICTING_FRESHNESS`
- `SHOWING_RESULT`
- `WAITING_FOR_REMOVAL`
- `RESETTING`

State changes are logged in the debug log panel and in `logs/raspi_app.log`.

## Empty Chamber Reference

The chamber monitor compares preview frames against:

- `captures/empty_chamber_reference.jpg`

The app now captures a fresh empty chamber reference at startup after warm-up and overwrites the saved reference image. That fresh baseline is then used for object detection during the current runtime session.

Important:

- start the system with an empty chamber whenever possible
- the app will capture that empty view first and save it as the new baseline
- that fresh baseline is what later object detection is compared against

## Physical Buttons

The app no longer uses physical buttons to trigger scans.

Current button behavior:

- GPIO17 = scroll up
- GPIO27 = scroll down
- GPIO22 = reserved / unused

The analysis cycle is now started automatically by camera-based chamber detection.

## Main Files

- `app.py`
  - main automatic GUI
  - state machine
  - chamber monitoring
  - result display
- `chamber_detector.py`
  - lightweight empty-chamber difference logic
  - frame preparation and difference scoring
- `meat_classifier.py`
  - loads the new image classifier
  - returns detected meat type and confidence
- `predict_live.py`
  - existing freshness predictor
  - unchanged final classifier for `Fresh / Neutral / Spoiled`
- `sensor_reader.py`
  - ADS1115 + MQ + DHT22 reading
- `camera_capture.py`
  - Raspberry Pi camera capture and preview
- `button_input.py`
  - physical scroll button support
- `config.py`
  - runtime paths and automation settings

## Important Config Settings

These settings can be edited in `config.py`:

- `EMPTY_CHAMBER_REFERENCE_IMAGE_PATH`
- `ALWAYS_CAPTURE_EMPTY_REFERENCE_ON_STARTUP`
- `OBJECT_DETECTION_THRESHOLD`
- `OBJECT_STABILITY_DURATION_SECONDS`
- `OBJECT_STABLE_FRAME_DIFF_THRESHOLD`
- `OBJECT_MONITOR_INTERVAL_SECONDS`
- `REMOVAL_DETECTION_THRESHOLD`
- `REMOVAL_STABILITY_SECONDS`
- `RESULT_HOLD_MIN_SECONDS`
- `AUTO_RESET_COOLDOWN_SECONDS`
- `AUTO_SENSOR_STABILIZATION_READ_COUNT`
- `SCROLL_UP_GPIO_PIN`
- `SCROLL_DOWN_GPIO_PIN`

## Install Dependencies

On Raspberry Pi:

```bash
cd ~/Documents/Meat/raspi
python3 -m venv .venv
source .venv/bin/activate
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

## Enable I2C

```bash
sudo raspi-config
```

Then:

- `Interface Options -> I2C -> Enable`

After that:

```bash
sudo reboot
sudo i2cdetect -y 1
```

You should usually see `48`.

## Run The App

```bash
cd ~/Documents/Meat/raspi
source .venv/bin/activate
python3 app.py
```

## Notes

- the UI is native Raspberry Pi GUI, not a website
- the system is now camera-triggered and fully automatic
- the meat classifier still acts as the first gate before freshness prediction
- the existing hybrid freshness model is still the final freshness classifier
- deployment sensor alignment is still active
- the freshness model feature order was not changed
