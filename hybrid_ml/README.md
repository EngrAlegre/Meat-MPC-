# FreshTo ML Pipelines

This folder now contains two different ML parts:

1. the existing **hybrid freshness pipeline**
2. the new **Option B meat image classifier**

## 1. Existing Hybrid Freshness Pipeline

This is the already approved freshness pipeline.

It uses:

- MQ sensor features based on `Rs/Ro`
- OpenCV image features
- the existing hybrid freshness model

It predicts:

- `Fresh`
- `Neutral`
- `Spoiled`

Important:

- this model stays unchanged in Option B
- do not retrain or replace it just to add meat detection

Main files:

- `train_hybrid_model.py`
- `train_modal_models.py`
- `infer_hybrid_model.py`
- `hybrid_pipeline_utils.py`

## 2. New Option B Meat Image Classifier

This is the new model added **before** the freshness pipeline.

It predicts:

- `not_meat`
- `chicken`
- `pork`
- `beef`

This model is only for deciding whether the image contains valid meat and, if yes, which meat type should be passed into the existing freshness model.

Main files:

- `train_meat_classifier.py`
- `infer_meat_classifier.py`
- `meat_classifier_utils.py`

## Final Option B Flow

```text
camera image
-> meat classifier
-> if not_meat: stop
-> else detected meat type goes into existing hybrid freshness model
-> freshness result
```

## Meat Classifier Dataset

Supported structures:

```text
dataset/
  not_meat/
  chicken/
  pork/
  beef/
```

or your current FreshTo image layout:

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

The updated training script automatically maps:

- `chicken_*` -> `chicken`
- `pork_*` -> `pork`
- `beef_*` -> `beef`
- `not_meat` -> `not_meat`

If time is tight, start `not_meat` with a small practical set such as empty chamber images, hands, tissue, plastic tray, lid-only frames, and wrong-angle shots. You can improve this class later without replacing the freshness model.

## Train The Meat Classifier

```powershell
cd C:\Users\isaac\Downloads\chickenprok\Raspi_5_program\hybrid_ml
python train_meat_classifier.py --dataset-dir C:\Users\isaac\Downloads\chickenprok\Raspi_5_program\Dataset\Image
```

Default output:

- `..\model\meat_classifier\meat_classifier.keras`
- `..\model\meat_classifier\class_names.json`
- `..\model\meat_classifier\metadata.json`

The script also saves:

- validation classification report
- confusion matrix
- training history

## Run Meat Classifier Inference

```powershell
python infer_meat_classifier.py --image-path C:\path\to\image.jpg
```

## Existing Hybrid Freshness Inference

```powershell
python infer_hybrid_model.py `
  --image-path "C:\path\to\image.jpg" `
  --meat-type Chicken `
  --nh3-ratio 0.64 `
  --h2s-ratio 0.44 `
  --voc-ratio 0.77
```

## Notes

- the hybrid freshness model remains the final classifier for freshness
- the new meat classifier is only a front gate before freshness prediction
- this keeps the working freshness model intact while adding a safer `not_meat` stop condition
