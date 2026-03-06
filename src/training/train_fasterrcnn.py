"""
Train Faster R-CNN (MobileNetV3 backbone) with mixed precision and checkpointing.

Usage:
python train_fasterrcnn.py --data ../datasets/oxford_pets_subset --ann pets_subset_coco.json --epochs 18 --batch 2 --device 0
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.models.faster_rcnn import get_fasterrcnn_model


class CocoLikeDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir: Path, ann_json: Path, transforms: Any = None):
        from PIL import Image
        import os
        self.images_dir = Path(images_dir)
        self.ann_json = Path(ann_json)
        with open(self.ann_json) as f:
            coco = json.load(f)
        
        # Build annotations dictionary
        self.anns = {}
        for a in coco["annotations"]:
            self.anns.setdefault(a["image_id"], []).append(a)
        
        # Validate and filter images - only keep images that actually exist
        self.images = []
        missing_count = 0
        for img_meta in coco["images"]:
            img_path = self.images_dir / img_meta["file_name"]
            if os.path.exists(img_path):
                self.images.append(img_meta)
            else:
                missing_count += 1
                # Clean up annotations for this missing image
                if img_meta["id"] in self.anns:
                    del self.anns[img_meta["id"]]
        
        if missing_count > 0:
            print(f"Warning: {missing_count} images missing from dataset, skipping them")
        
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        from PIL import Image
        img_meta = self.images[idx]
        # file_name can be either just filename or relative path like "train/images/img.png"
        img_path = self.images_dir / img_meta["file_name"]
        
        # Extra safety check (shouldn't be needed after validation, but defensive)
        if not img_path.exists():
            print(f"Warning: Image file missing at runtime: {img_path}")
            # Return a blank image with no annotations as fallback
            img = Image.new('RGB', (512, 512), color='black')
            target = {"boxes": torch.zeros((0, 4), dtype=torch.float32), 
                      "labels": torch.zeros((0,), dtype=torch.int64)}
            if self.transforms:
                img = self.transforms(img)
            else:
                tf = T.Compose([T.Resize((512, 512)), T.ToTensor()])
                img = tf(img)
            return img, target
        
        img = Image.open(img_path).convert("RGB")
        boxes = []
        labels = []
        for a in self.anns.get(img_meta["id"], []):
            boxes.append(a["bbox"])  # [x,y,w,h]
            labels.append(a["category_id"])  # COCO already uses 1-indexed categories (0=background reserved, 1+=classes)
        
        # Ensure boxes is always shape [N, 4]
        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # Convert from [x, y, w, h] to [x1, y1, x2, y2]
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        else:
            # No boxes - create empty tensor with proper shape
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        
        target = {"boxes": boxes, "labels": labels}
        if self.transforms:
            img = self.transforms(img)
        else:
            tf = T.Compose([T.Resize((512, 512)), T.ToTensor()])
            img = tf(img)
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, scaler):
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(data_loader, desc="train"):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += float(losses)
    return running_loss / max(1, len(data_loader))


def evaluate_speed(model, data_loader, device, max_batches: int = 100) -> float:
    model.eval()
    n = 0
    t0 = time.time()
    with torch.no_grad():
        for images, _ in data_loader:
            images = list(img.to(device) for img in images)
            _ = model(images)
            n += len(images)
            if n >= max_batches:
                break
    t = time.time() - t0
    return n / t if t > 0 else 0.0


def run_training(data_dir: Path, ann_json: Path, epochs: int = 12, batch: int = 2,
                 device_id: int = 0, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Programmatic entrypoint for Faster R-CNN training."""
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(device_id)}")
    dataset = CocoLikeDataset(data_dir, ann_json)
    n = len(dataset)
    val_size = max(1, int(0.15 * n))
    test_size = max(1, int(0.15 * n))
    train_size = n - val_size - test_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    # Determine number of classes from the dataset
    # Load COCO to get categories
    with open(ann_json) as f:
        coco_data = json.load(f)
    num_classes = len(coco_data.get('categories', [])) + 1  # +1 for background
    
    model = get_fasterrcnn_model(num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler()

    out_dir = Path(output_dir) if output_dir is not None else Path("outputs/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    start_time = time.time()
    metrics = []
    for epoch in range(epochs):
        loss = train_one_epoch(model, optimizer, train_loader, device, scaler)
        scheduler.step()
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} loss={loss:.4f} elapsed={epoch_time:.1f}s")
        metrics.append({"epoch": epoch + 1, "loss": loss, "time": epoch_time})
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), str(out_dir / "best_fasterrcnn.pth"))
    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.1f}s")

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)
    speed = evaluate_speed(model, test_loader, device, max_batches=200)
    print(f"Inference speed: {speed:.2f} imgs/sec")

    import pandas as pd
    df = pd.DataFrame(metrics)
    df.to_csv(out_dir / "fasterrcnn_metrics.csv", index=False)
    return {"metrics": metrics, "speed": speed, "model_path": str(out_dir / "best_fasterrcnn.pth"), "training_time": total_time}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to dataset directory containing images")
    parser.add_argument("--ann", required=True, help="COCO json filename in dataset dir")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", default="outputs/models/", help="checkpoint output dir")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data)
    ann_json = data_dir / args.ann

    dataset = CocoLikeDataset(data_dir, ann_json)
    n = len(dataset)
    val_size = max(1, int(0.15 * n))
    test_size = max(1, int(0.15 * n))
    train_size = n - val_size - test_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    num_classes = 2  # background + 1 class (person or pet class aggregated)
    model = get_fasterrcnn_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    start_time = time.time()
    metrics = []
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, optimizer, train_loader, device, scaler)
        scheduler.step()
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} loss={loss:.4f} elapsed={epoch_time:.1f}s")
        metrics.append({"epoch": epoch + 1, "loss": loss, "time": epoch_time})
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), str(out_dir / "best_fasterrcnn.pth"))
    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.1f}s")

    # Speed test on test set
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)
    speed = evaluate_speed(model, test_loader, device, max_batches=200)
    print(f"Inference speed: {speed:.2f} imgs/sec")

    # Save metrics CSV
    import pandas as pd

    df = pd.DataFrame(metrics)
    df.to_csv(out_dir / "fasterrcnn_metrics.csv", index=False)
    print("Saved metrics to", out_dir / "fasterrcnn_metrics.csv")


if __name__ == "__main__":
    main()
