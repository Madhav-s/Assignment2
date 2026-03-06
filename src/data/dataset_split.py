"""Split dataset into train/val/test (70/15/15) and save structure suitable for training.

For Penn-Fudan we expect images/ and annotations/ (masks). For our simplified pipeline we place images and label files.
"""
from pathlib import Path
import random
import shutil
from typing import List, Optional


def _split_list(items: List[Path], seed: int = 42):
    random.seed(seed)
    items = list(items)
    random.shuffle(items)
    n = len(items)
    ntrain = int(n * 0.7)
    nval = int(n * 0.15)
    train = items[:ntrain]
    val = items[ntrain:ntrain + nval]
    test = items[ntrain + nval:]
    return train, val, test


def create_coco_annotations(split_dir: Path, out_json: Path, class_list: Optional[List[str]] = None) -> None:
    """Generate COCO-format annotation file from a split directory.

    split_dir should have images/ and labels/ subfolders; labels files use format
    class_id x1 y1 x2 y2 per line. class_list if provided maps id->name.
    """
    import json
    from PIL import Image

    images = []
    annotations = []
    ann_id = 1
    # Handle both jpg and png files
    img_files = list((split_dir / 'images').glob('*.jpg')) + list((split_dir / 'images').glob('*.png'))
    for img_file in img_files:
        img_id = len(images) + 1
        w, h = Image.open(img_file).size
        images.append({"file_name": img_file.name, "height": h, "width": w, "id": img_id})
        label_path = split_dir / 'labels' / f"{img_file.stem}.txt"
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        x1, y1, x2, y2 = map(float, parts[1:5])
                        w_box = x2 - x1
                        h_box = y2 - y1
                        annotations.append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": cid + 1,
                            "bbox": [x1, y1, w_box, h_box],
                            "area": w_box * h_box,
                            "iscrowd": 0,
                        })
                        ann_id += 1
    cat_list = []
    if class_list:
        for i, c in enumerate(class_list):
            cat_list.append({"id": i + 1, "name": c})
    else:
        cat_list = [{"id": 1, "name": "person"}]
    coco = {"images": images, "annotations": annotations, "categories": cat_list}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(coco, f)


def create_coco_from_splits(root_dir: Path, out_json: Path, class_list: Optional[List[str]] = None) -> None:
    """Generate COCO JSON by aggregating train/val/test subfolders under root_dir."""
    # collect all images from three splits
    import json
    from PIL import Image

    images = []
    annotations = []
    ann_id = 1
    img_id = 1
    for split in ['train', 'val', 'test']:
        split_dir = root_dir / split
        if not split_dir.exists():
            continue
        # Look for both jpg and png files
        img_files = list((split_dir / 'images').glob('*.jpg')) + list((split_dir / 'images').glob('*.png'))
        for img_file in img_files:
            w, h = Image.open(img_file).size
            # Store relative path from root_dir (e.g., "train/images/img.png")
            rel_path = f"{split}/images/{img_file.name}"
            images.append({"file_name": rel_path, "height": h, "width": w, "id": img_id})
            label_path = split_dir / 'labels' / f"{img_file.stem}.txt"
            if label_path.exists():
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cid = int(parts[0])
                            x1, y1, x2, y2 = map(float, parts[1:5])
                            w_box = x2 - x1
                            h_box = y2 - y1
                            annotations.append({
                                "id": ann_id,
                                "image_id": img_id,
                                "category_id": cid + 1,  # COCO uses 1-indexed categories
                                "bbox": [x1, y1, w_box, h_box],
                                "area": w_box * h_box,
                                "iscrowd": 0,
                            })
                            ann_id += 1
            img_id += 1
    cat_list = []
    if class_list:
        for i, c in enumerate(class_list):
            cat_list.append({"id": i + 1, "name": c})
    else:
        cat_list = [{"id": 1, "name": "person"}]
    coco = {"images": images, "annotations": annotations, "categories": cat_list}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(coco, f)


def split_dataset(src: Path, out_dir: Path) -> None:
    src = Path(src)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # For simplicity, detect images under src/images, src/JPEGImages, src/PNGImages, or src
    imgs = []
    if (src / 'images').exists():
        imgs = list((src / 'images').glob('*.jpg')) + list((src / 'images').glob('*.png'))
    elif (src / 'JPEGImages').exists():
        imgs = list((src / 'JPEGImages').glob('*.jpg'))
    elif (src / 'PNGImages').exists():
        # Penn-Fudan stores images as .png in PNGImages folder
        imgs = list((src / 'PNGImages').glob('*.png'))
    else:
        imgs = list(src.glob('*.jpg')) + list(src.glob('*.png'))
    
    if not imgs:
        raise ValueError(f"No images found in {src}. Checked for JPG and PNG files in images/, JPEGImages/, or PNGImages/")
    
    print(f"Found {len(imgs)} images in {src}")
    train, val, test = _split_list(imgs)

    for split_name, split_list in [('train', train), ('val', val), ('test', test)]:
        d = out_dir / split_name
        (d / 'images').mkdir(parents=True, exist_ok=True)
        (d / 'labels').mkdir(parents=True, exist_ok=True)
        for p in split_list:
            shutil.copy(p, d / 'images' / p.name)
            # if there is a label file generated by create_pet_subset, copy it
            label_src = src / 'labels' / f"{p.stem}.txt"
            if label_src.exists():
                shutil.copy(label_src, d / 'labels' / f"{p.stem}.txt")
            else:
                # attempt to create a bbox from Penn-Fudan mask if available
                # Masks can be in Masks/ or also as annotations
                mask_path = None
                for mask_dir_name in ['Masks', 'Annotations']:
                    candidate = src / mask_dir_name / f"{p.stem}.png"
                    if candidate.exists():
                        mask_path = candidate
                        break
                
                if mask_path and mask_path.exists():
                    from PIL import Image
                    import numpy as np
                    m = Image.open(mask_path).convert('L')
                    arr = np.array(m)
                    ys, xs = np.where(arr > 0)
                    if ys.size:
                        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                        with open(d / 'labels' / f"{p.stem}.txt", 'w') as f:
                            # single person class id 0
                            f.write(f"0 {x1} {y1} {x2} {y2}\n")
                else:
                    # no annotation available - create a default label
                    # For Penn-Fudan, create a full-image bbox as fallback
                    from PIL import Image
                    img = Image.open(p)
                    w, h = img.size
                    with open(d / 'labels' / f"{p.stem}.txt", 'w') as f:
                        f.write(f"0 0 0 {w-1} {h-1}\n")
    
    # Create classes.txt if it doesn't exist (for Penn-Fudan which has only 'person')
    if not (out_dir / 'classes.txt').exists() and not (src / 'classes.txt').exists():
        with open(out_dir / 'classes.txt', 'w') as f:
            f.write('person\n')
    elif (src / 'classes.txt').exists():
        # Copy classes.txt from source to split directory
        shutil.copy(src / 'classes.txt', out_dir / 'classes.txt')
