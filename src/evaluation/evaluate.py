from pathlib import Path
from typing import List, Tuple, Dict, Any
import time
import torch
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import pandas as pd

from src.evaluation.metrics import calculate_metrics
from src.utils.visualization import draw_boxes_on_image
from src.models.faster_rcnn import get_fasterrcnn_model
from src.models.yolo_model import YOLOWrapper


def load_ground_truth(test_dir: Path) -> Tuple[List[Path], List[List[np.ndarray]]]:
    # Look for both jpg and png files
    imgs = sorted((test_dir / 'images').glob('*.jpg')) + sorted((test_dir / 'images').glob('*.png'))
    all_boxes = []
    for img_path in imgs:
        label_path = test_dir / 'labels' / f"{img_path.stem}.txt"
        boxes = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        x1, y1, x2, y2 = map(float, parts[1:5])
                        boxes.append(np.array([x1, y1, x2, y2]))
        all_boxes.append(boxes)
    return imgs, all_boxes


def evaluate_model(fr_model_path: str, yolo_model_path: str, test_dir: Path, dataset_name: str) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    imgs, gt_boxes = load_ground_truth(test_dir)

    # Determine number of classes from classes.txt
    classes_file = test_dir.parent / 'classes.txt'
    if classes_file.exists():
        with open(classes_file) as f:
            class_names = [line.strip() for line in f.readlines()]
        num_classes = len(class_names) + 1  # +1 for background
    else:
        num_classes = 2  # Default for Penn-Fudan (person)

    # FasterRCNN inference
    fr_model = get_fasterrcnn_model(num_classes)
    fr_model.load_state_dict(torch.load(fr_model_path, map_location=device))
    fr_model.to(device)
    fr_model.eval()

    fr_preds = []
    t0 = time.time()
    for img_path in imgs:
        img = Image.open(img_path).convert('RGB')
        img_t = F.to_tensor(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = fr_model(img_t)[0]
        boxes = output['boxes'].cpu().numpy()
        fr_preds.append([np.array(b) for b in boxes])
    fr_infer_time = time.time() - t0

    # YOLO inference
    yolo = YOLOWrapper(yolo_model_path)
    t0 = time.time()
    preds = yolo.predict(source=str(test_dir / 'images'), save=False)
    yolo_infer_time = time.time() - t0
    yolo_preds = []
    # predictions is a list of Results
    for res in preds:
        b = res.boxes.xyxy.cpu().numpy()
        yolo_preds.append([np.array(bb) for bb in b])

    # compute metrics
    fr_map, fr_prec, fr_rec = calculate_metrics(gt_boxes, fr_preds)
    y_map, y_prec, y_rec = calculate_metrics(gt_boxes, yolo_preds)

    # measure inference speed roughly using time per image
    # compute training time not available here

    fr_speed = len(imgs) / fr_infer_time if fr_infer_time > 0 else 0.0
    yolo_speed = len(imgs) / yolo_infer_time if yolo_infer_time > 0 else 0.0

    metrics = pd.DataFrame([{
        'dataset': dataset_name,
        'model': 'fasterrcnn',
        'mAP@0.5': fr_map,
        'precision': fr_prec,
        'recall': fr_rec,
        'speed(imgs/sec)': fr_speed
    }, {
        'dataset': dataset_name,
        'model': 'yolo',
        'mAP@0.5': y_map,
        'precision': y_prec,
        'recall': y_rec,
        'speed(imgs/sec)': yolo_speed
    }])
    out_csv = Path('outputs/metrics') / f"{dataset_name}_metrics.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_csv, index=False)
    print(f"Saved evaluation metrics to {out_csv}")

    # save visualization for first 5 images
    out_pred_dir = Path('outputs/predictions') / dataset_name
    out_pred_dir.mkdir(parents=True, exist_ok=True)
    for i, img_path in enumerate(imgs[:5]):
        img = Image.open(img_path).convert('RGB')
        # draw fr and yolo predictions
        img_fr = draw_boxes_on_image(img.copy(), fr_preds[i])
        img_fr.save(out_pred_dir / f"{img_path.stem}_fr.jpg")
        img_y = draw_boxes_on_image(img.copy(), yolo_preds[i])
        img_y.save(out_pred_dir / f"{img_path.stem}_yolo.jpg")

    return metrics
