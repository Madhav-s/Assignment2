"""Main pipeline to run the dataset download, preprocessing, training and evaluation.

Usage: python main.py
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from src.utils.config import PROJECT_ROOT
from src.data.download_datasets import download_pennfudan, download_oxford_pets
from src.data.create_pet_subset import create_pet_subset
from src.data.dataset_split import split_dataset, create_coco_from_splits
from src.training.train_fasterrcnn import run_training as train_fasterrcnn
from src.training.train_yolo import train_yolo
from src.evaluation.evaluate import evaluate_model


def main() -> None:
    DATA_ROOT = PROJECT_ROOT / "datasets"
    outputs = PROJECT_ROOT / "outputs"
    outputs.mkdir(exist_ok=True, parents=True)

    print("1) Downloading datasets...")
    penn_dir = download_pennfudan(DATA_ROOT / "penn_fudan")
    pets_dir = download_oxford_pets(DATA_ROOT / "oxford_pets")

    print("2) Creating 8-breed pet subset...")
    pet_subset_dir = create_pet_subset(pets_dir, DATA_ROOT / "oxford_pets_subset", n_breeds=8)

    print("3) Splitting datasets (70/15/15)...")
    penn_split = DATA_ROOT / "penn_fudan_split"
    pet_split = DATA_ROOT / "oxford_pets_subset_split"
    split_dataset(penn_dir, out_dir=penn_split)
    split_dataset(pet_subset_dir, out_dir=pet_split)

    # create COCO annotation files for each dataset
    penn_coco = penn_split / "annotations_all.json"
    create_coco_from_splits(penn_split, penn_coco, class_list=['person'])
    pet_coco = pet_split / "annotations_all.json"
    # Load pet classes from classes.txt
    pet_classes = []
    classes_file = pet_split / 'classes.txt'
    if classes_file.exists():
        with open(classes_file) as f:
            pet_classes = [line.strip() for line in f.readlines()]
    create_coco_from_splits(pet_split, pet_coco, class_list=pet_classes)

    import time
    # Skip Penn-Fudan training to run only 18-epoch pets training
    # print("4) Training Faster R-CNN on Penn-Fudan (12 epochs)...")
    # t0 = time.time()
    # frcnn_metrics = train_fasterrcnn(penn_split, penn_coco, epochs=12)
    # frcnn_time = time.time() - t0

    # print("5) Training YOLOv8n on Penn-Fudan (12 epochs)...")
    # t0 = time.time()
    # yolo_metrics = train_yolo(penn_split, dataset_name="penn_fudan", epochs=12)
    # yolo_time = time.time() - t0

    print("4) Training Faster R-CNN on Penn-Fudan (12 epochs)...")
    t0 = time.time()
    frcnn_metrics = train_fasterrcnn(penn_split, penn_coco, epochs=12)
    frcnn_time = time.time() - t0

    print("5) Training YOLOv8n on Penn-Fudan (12 epochs)...")
    t0 = time.time()
    yolo_metrics = train_yolo(penn_split, dataset_name="penn_fudan", epochs=12)
    yolo_time = time.time() - t0

    # attach training times
    frcnn_metrics['training_time'] = frcnn_time
    yolo_metrics['training_time'] = yolo_time

    print("6) Evaluate both models on penn_fudan test set and save predictions...")
    evaluate_model(fr_model_path=frcnn_metrics.get("model_path"), yolo_model_path=yolo_metrics.get("model_path"),
                   test_dir=DATA_ROOT / "penn_fudan_split" / "test", dataset_name="penn_fudan")

    print("7) Repeat training and evaluation for pets dataset")
    t0 = time.time()
    frcnn_metrics_pets = train_fasterrcnn(pet_split, pet_coco, epochs=18)
    frcnn_time_pets = time.time() - t0
    t0 = time.time()
    yolo_metrics_pets = train_yolo(pet_split, dataset_name="oxford_pets", epochs=18)
    yolo_time_pets = time.time() - t0
    frcnn_metrics_pets['training_time'] = frcnn_time_pets
    yolo_metrics_pets['training_time'] = yolo_time_pets

    evaluate_model(fr_model_path=frcnn_metrics_pets.get("model_path"), yolo_model_path=yolo_metrics_pets.get("model_path"),
                   test_dir=pet_split / "test", dataset_name="oxford_pets")

    # combine metric CSVs into a single summary
    metrics_dir = PROJECT_ROOT / "outputs" / "metrics"
    all_files = list(metrics_dir.glob("*.csv"))
    if all_files:
        import pandas as pd
        df_list = [pd.read_csv(f) for f in all_files]
        summary = pd.concat(df_list, ignore_index=True)
        # merge training times we recorded
        train_info = pd.DataFrame([
            {'dataset': 'penn_fudan', 'model': 'fasterrcnn', 'training_time': frcnn_metrics.get('training_time')},
            {'dataset': 'penn_fudan', 'model': 'yolo', 'training_time': yolo_metrics.get('training_time')},
            {'dataset': 'oxford_pets', 'model': 'fasterrcnn', 'training_time': frcnn_metrics_pets.get('training_time')},
            {'dataset': 'oxford_pets', 'model': 'yolo', 'training_time': yolo_metrics_pets.get('training_time')},
        ])
        summary = summary.merge(train_info, on=['dataset', 'model'], how='left')
        summary.to_csv(metrics_dir / "summary.csv", index=False)
        print("Generated summary metrics at", metrics_dir / "summary.csv")
        # also output markdown table for report
        try:
            md = summary.to_markdown(index=False)
        except ImportError:
            md = summary.to_string(index=False)
        with open(PROJECT_ROOT / "report" / "comparison_table.md", "w") as f:
            f.write(md)
        print("Saved comparison table to report/comparison_table.md")
    print("Done. Metrics and predictions stored in outputs/metrics and outputs/predictions")


if __name__ == "__main__":
    main()
