# Hybrid Meat Freshness Pipeline

This folder contains a complete Python pipeline for hybrid freshness classification using:

- MQ sensor CSV features based on `Rs/Ro`
- image features extracted with OpenCV
- scikit-learn classifiers

The current implementation trains one row per image. Each image inherits the class-level MQ sensor summary for the matching `meat_type + freshness` combination.

## Why The First Results Were Overly Optimistic

The first version used a normal random train/test split only.

That was too easy because:

- MQ sensor features are summarized once per `meat_type + freshness`
- those same sensor summary values are repeated across many images in the same class
- random splitting can place very similar rows from the same source group into both train and test

This leakage can make the reported accuracy unrealistically high.

## What Was Changed

The revised training script now reports multiple evaluation modes:

- `random_split`
  - simple baseline
  - still useful, but optimistic
- `grouped_split`
  - holds out full source groups together
  - prevents the same sensor-source group from appearing in both train and test
- `random_cv`
  - normal stratified cross-validation
- `grouped_cv`
  - stratified grouped cross-validation
  - the safest metric for thesis presentation in the current dataset layout

The grouped evaluations use the sensor source file as the grouping key, which is a practical proxy for one class-level sensor source.

## Files

- `hybrid_pipeline_utils.py`
  - dataset matching
  - sensor aggregation
  - image feature extraction
- `train_hybrid_model.py`
  - training
  - model comparison
  - evaluation
  - artifact saving
- `infer_hybrid_model.py`
  - inference from one image + one MQ reading row

## Raspberry Pi App Folder

For cleaner deployment, the Raspberry Pi runtime app now lives in:

- `C:\Users\isaac\Downloads\chickenprok\Raspi_5_program\raspi`

## Model Folder

For cleaner separation, the saved training outputs now live in:

- `C:\Users\isaac\Downloads\chickenprok\Raspi_5_program\model`

That folder contains:

- `app.py`
  - Flask deployment app for Raspberry Pi 5
- `sensor_reader.py`
  - ADS1115 + MQ sensor reading and stabilization
- `camera_capture.py`
  - Raspberry Pi camera capture
- `feature_extractor.py`
  - exact training-compatible image feature extraction
- `predict_live.py`
  - live hybrid inference using the saved artifacts
- `config.py`
  - Raspberry Pi deployment settings

## Dataset Defaults

- Sensor CSVs:
  - `C:\Users\isaac\Downloads\chickenprok\Raspi_5_program\Dataset\MQ_Sensors_Dataset`
- Images:
  - `C:\Users\isaac\Downloads\chickenprok\Raspi_5_program\Dataset\Image`

## Train

Run from PowerShell:

```powershell
cd C:\Users\isaac\Downloads\chickenprok\Raspi_5_program\hybrid_ml
python train_hybrid_model.py
```

Optional arguments:

```powershell
python train_hybrid_model.py --sensor-dir "C:\path\to\MQ_Sensors_Dataset" --image-dir "C:\path\to\Image" --output-dir "C:\path\to\artifacts"
```

## Inference

```powershell
python infer_hybrid_model.py `
  --image-path "C:\path\to\image.jpg" `
  --meat-type Chicken `
  --nh3-ratio 0.64 `
  --h2s-ratio 0.44 `
  --voc-ratio 0.77
```

## Raspberry Pi Deployment

The Raspberry Pi app keeps the trained model unchanged and adds runtime safeguards:

- configurable warm-up time before prediction
- short sensor stabilization window using multiple samples
- averaged `nh3_ratio`, `h2s_ratio`, and `voc_ratio`
- optional baseline capture for debug/display only
- prediction log entries for traceability

Example:

```powershell
cd C:\Users\isaac\Downloads\chickenprok\Raspi_5_program\raspi
python app.py
```

### Deployment States

The Flask app uses explicit states that the UI mirrors:

- `WARMING_UP`
- `CAPTURING_BASELINE`
- `STABILIZING_SENSORS`
- `READY_TO_SCAN`
- `PREDICTING`
- `PREDICTED`
- `NOT_READY`

### Why These Safeguards Matter

- MQ readings fluctuate, so averaging a short window is more stable than predicting from one instantaneous row.
- Warm-up helps the sensor heater settle before predictions are allowed.
- Baseline capture is useful for operator awareness and debugging, but it is not injected into the trained model by default so inference remains compatible.
- The Raspberry Pi predictor uses the exact same image feature extraction and sensor feature names expected by the fitted preprocessor.

## Saved Artifacts

After training, artifacts are saved in:

- `..\model\hybrid_freshness_model.joblib`
- `..\model\freshness_label_encoder.joblib`
- `..\model\hybrid_preprocessor.joblib`
- `..\model\training_metadata.json`
- `..\model\hybrid_dataset.csv`
- `..\model\model_comparison.csv`
- `..\model\classification_report_*.txt`
- `..\model\confusion_matrix_*.csv`

## Notes

- XGBoost is optional. If it is not installed, the script skips it automatically.
- Neutral CSVs may only contain ratio columns. Missing voltage/resistance summaries are imputed during training.
- `random_split` can still look optimistic because class-level sensor summaries are repeated across many images in the same class.
- `grouped_split` and especially `grouped_cv` are more realistic because they keep full source groups together.
- For thesis or panel presentation, the metric that should be trusted most is:
  - `grouped_cv_accuracy_mean`
- The saved deployment model is selected using grouped cross-validation, then retrained on the full dataset after selection so inference compatibility is preserved.
- The Raspberry Pi deployment app does not retrain the model. It only makes runtime inference safer and more consistent.
