from pathlib import Path
from typing import Dict, Any
import yaml
from PIL import Image
import time

from src.models.yolo_model import YOLOWrapper
from src.utils.config import MODELS_DIR


def prepare_yolo_data(dataset_dir: Path, output_yaml: Path, num_classes: int, class_names: list) -> None:
    """Create a YOLOv8 data yaml file pointing to train/val/test folders in dataset_dir."""
    # convert existing label files (x1 y1 x2 y2) to YOLO format in place
    for split in ['train', 'val', 'test']:
        img_dir = dataset_dir / split / 'images'
        label_dir = dataset_dir / split / 'labels'
        # Handle both jpg and png files
        for img_path in list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')):
            label_path = label_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            w, h = Image.open(img_path).size
            lines = []
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        x1, y1, x2, y2 = map(float, parts[1:5])
                        cx = (x1 + x2) / 2 / w
                        cy = (y1 + y2) / 2 / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        lines.append(f"{cid} {cx} {cy} {bw} {bh}\n")
            with open(label_path, 'w') as f:
                f.writelines(lines)
    d = {
        'path': str(dataset_dir),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': class_names
    }
    with open(output_yaml, 'w') as f:
        yaml.dump(d, f)


def train_yolo(data_root: Path, dataset_name: str, epochs: int = 10, batch: int = 8) -> Dict[str, Any]:
    # classes list for pets, else person only
    classes = ['person']
    classes_file = data_root / 'classes.txt'
    if classes_file.exists():
        with open(classes_file) as f:
            classes = [l.strip() for l in f.readlines()]
    num_classes = len(classes)

    data_yaml = data_root / 'yolo_data.yaml'
    prepare_yolo_data(data_root, data_yaml, num_classes, classes)

    yolo = YOLOWrapper('yolov8n.pt')
    save_dir = MODELS_DIR / f"yolo_{dataset_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    results = yolo.train(data_yaml=str(data_yaml), epochs=epochs, batch=batch, imgsz=512, save_dir=str(save_dir))
    # results attribute has model path
    model_path = save_dir / 'weights' / 'best.pt'
    return {'model_path': str(model_path), 'results': results}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--name', default='dataset')
    args = parser.parse_args()
    train_yolo(Path(args.data), dataset_name=args.name, epochs=args.epochs, batch=args.batch)
