# Object Detection Project

This repository implements a complete object detection pipeline for two models:

* **Faster R-CNN** with MobileNet backbone (`torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn`)
* **YOLOv8n (nano)** via the `ultralytics` package

They are trained and compared on two datasets:

1. **Penn-Fudan Pedestrian** (person detection)
2. **Oxford-IIIT Pet** (subset of 8 breeds, detection + classification)

All code resides under `src/` with modular structure (data, models, training, evaluation, utils).

## Setup

1. Clone the repo and `cd` into `object_detection_project`.
2. Create a Python environment (e.g. `python -m venv venv` and activate).
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Pipeline Overview

The main entrypoint is `main.py`. Running it will:

1. Download datasets automatically if missing
2. Create an 8‑breed subset for the pet dataset
3. Split each dataset into train/val/test (70/15/15)
4. Convert annotations as needed (COCO for Faster R-CNN, YOLO for YOLOv8)
5. Train both models on each dataset (with specified epochs and batch sizes)
6. Evaluate both models on the test split and record metrics and predictions
7. Save metrics to CSV and generate a comparison summary

### Running the full workflow

```bash
python main.py
```

Processed outputs are stored under `outputs/`:

* `outputs/models/` – saved weights and training metrics
* `outputs/predictions/` – example prediction images
* `outputs/metrics/` – evaluation CSVs and summary table

You may also invoke individual components directly (e.g. the training scripts)
for debugging or custom experimentation; see the docstrings in `src/`.

## Additional Notes

* Images are resized to 512×512 during preprocessing for GPU memory safety.
* Augmentations (flip, color jitter, scaling) are applied on-the-fly for Faster R-CNN.
* Mixed precision training is used when a CUDA device is available.
* Batch sizes are kept small (2 for Faster R-CNN, 8 for YOLOv8n) to fit within an 8 GB GPU.

## Report

A template report is available at `report/report_template.md` with placeholders
for dataset descriptions, training details, and results comparison.

Feel free to adapt or extend this project for your own experiments.
