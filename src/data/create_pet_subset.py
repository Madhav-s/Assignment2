"""Create an 8-breed subset from downloaded Oxford-IIIT Pet dataset and convert annotations to bounding boxes.

This script expects the Oxford dataset extracted into a folder with `images/` and `annotations/`.
"""
from pathlib import Path
import random
import shutil
from typing import List, Tuple
from PIL import Image
import numpy as np


def list_breeds(annotations_dir: Path) -> List[str]:
    # The dataset provides filenames like "Abyssinian_1.jpg" or "Abyssinian_1.png"; breed is prefix before underscore
    images = list((annotations_dir.parent / 'images').glob('*.jpg')) + list((annotations_dir.parent / 'images').glob('*.png'))
    breeds = set(p.stem.split('_')[0] for p in images)
    return sorted(breeds)


def create_pet_subset(src: Path, out_dir: Path, n_breeds: int = 8) -> Path:
    src = Path(src)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = src / 'images'
    if not images_dir.exists():
        raise FileNotFoundError(f"Images not found in {images_dir}")

    # pick breeds - look for jpg and png files
    img_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    breed_names = sorted({p.stem.split('_')[0] for p in img_files})
    chosen = breed_names[:n_breeds]

    (out_dir / 'images').mkdir(exist_ok=True)
    (out_dir / 'labels').mkdir(exist_ok=True)

    # Try to use segmentation masks if present to compute bbox; otherwise use full image bbox
    masks_dir = src / 'annotations' / 'trimaps'

    for img_path in img_files:
        breed = img_path.stem.split('_')[0]
        if breed not in chosen:
            continue
        shutil.copy(img_path, out_dir / 'images' / img_path.name)
        # compute bbox
        bbox = None
        mask_path = masks_dir / f"{img_path.stem}.png"
        if mask_path.exists():
            m = Image.open(mask_path).convert('L')
            arr = np.array(m)
            ys, xs = np.where(arr > 0)
            if ys.size:
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                bbox = (x1, y1, x2, y2)
        if bbox is None:
            im = Image.open(img_path)
            w, h = im.size
            bbox = (0, 0, w - 1, h - 1)
        # save label in a simple CSV-style per-image (class_id x1 y1 x2 y2)
        class_id = chosen.index(breed)
        label_path = out_dir / 'labels' / f"{img_path.stem}.txt"
        with open(label_path, 'w') as f:
            f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

    # save a mapping file
    with open(out_dir / 'classes.txt', 'w') as f:
        for b in chosen:
            f.write(b + '\n')

    return out_dir


if __name__ == '__main__':
    create_pet_subset(Path('./datasets/oxford_pets'), Path('./datasets/oxford_pets_subset'))
