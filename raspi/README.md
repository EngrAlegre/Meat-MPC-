# FreshTo Raspberry Pi Runtime

This folder contains the Raspberry Pi 5 deployment app for `FreshTo`.

## Option B Summary

This implementation follows the approved **Option B** plan:

- keep the current hybrid freshness model unchanged
- keep the current MQ sensor math unchanged
- keep the deployment sensor alignment layer unchanged
- add one new image classifier in front of the system

That means the Raspberry Pi app now works in **two stages**:

1. capture image
2. classify image as:
   - `not_meat`
   - `chicken`
   - `pork`
   - `beef`
3. if the result is `not_meat`
   - stop the scan
   - show `No valid meat detected`
   - do not run freshness prediction
4. if the result is `chicken`, `pork`, or `beef`
   - pass that detected meat type into the existing hybrid freshness model
   - run the normal freshness prediction
   - show `Fresh`, `Neutral`, or `Spoiled`

## What Stayed Unchanged

The following parts were intentionally left as they were:

- MQ sensor reading through ADS1115
- voltage, `Rs`, and `Rs/Ro` computation
- deployment sensor alignment
- feature order used by the freshness model
- existing hybrid freshness model artifacts

## Main Files

- `app.py`
  - native Raspberry Pi touchscreen GUI
  - live camera feed
  - auto scan flow
  - hardware button trigger support
  - meat detection stage
  - freshness result stage
- `meat_classifier.py`
  - loads the new image meat classifier
  - returns detected meat type and confidence
- `predict_live.py`
  - existing freshness predictor
  - unchanged final classifier for `Fresh / Neutral / Spoiled`
- `sensor_reader.py`
  - ADS1115 + MQ + DHT22 reading
- `camera_capture.py`
  - Raspberry Pi camera capture
- `config.py`
  - runtime paths and settings

## New Meat Classifier Artifacts

The new image classifier is expected here:

- `..\model\meat_classifier\meat_classifier.keras`
- `..\model\meat_classifier\class_names.json`
- `..\model\meat_classifier\metadata.json`

You can train this classifier on your laptop first, then copy the whole `meat_classifier` folder into the Raspberry Pi `model` folder.

The existing freshness model still lives here:

- `..\model\hybrid_freshness_model.joblib`
- `..\model\freshness_label_encoder.joblib`
- `..\model\hybrid_preprocessor.joblib`
- `..\model\training_metadata.json`

## Meat Classifier Dataset Structure

Train the new image classifier using either:

```text
dataset/
  not_meat/
  chicken/
  pork/
  beef/
```

or the current FreshTo image folder layout:

```text
Dataset/
  Image/
    not_meat/
    chicken_fresh/
    chicken_neutral/
    chicken_spoiled/
    pork_fresh/
    pork_neutral/
    pork_spoiled/
    beef_fresh/
    beef_neutral/
    beef_spoiled/
```

The updated trainer can read that existing layout directly, so you do not need to reorganize all image folders first.

### If You Are Short On Time For `not_meat`

You do not need a huge `not_meat` dataset to get started.

A practical first pass is to collect images such as:

- empty chamber
- chamber lid only
- tissue or paper only
- plastic tray only
- hand in frame
- table surface
- blurred frames
- wrong camera angle

Even a small first set is useful because the only job of `not_meat` is to stop the freshness model when no valid meat is visible.

## How The App Works Now

1. power on the Raspberry Pi and sensors
2. wait for warm-up to complete
3. press any wired hardware button
4. the app captures an image
5. the app runs the new meat classifier
6. if image = `not_meat`
   - scan stops
   - screen shows `No valid meat detected`
7. if image = `chicken`, `pork`, or `beef`
   - app collects the short MQ sensor scan window
   - app runs the existing hybrid freshness predictor
   - screen shows:
     - detected meat type
     - meat classifier confidence
     - freshness result

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
- the three existing hardware buttons are now used as **scan triggers**
- meat type is detected automatically from the image
- the existing hybrid freshness model is still the final freshness classifier
- deployment sensor alignment is still active so live Raspberry Pi ratios stay on the same scale used by the current freshness deployment setup
- the freshness model feature order was not changed
