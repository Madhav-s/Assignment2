import time
from typing import Any
import torch

def measure_speed(model: Any, data_loader: Any, device: Any, max_batches: int = 100) -> float:
    model.eval()
    n = 0
    t0 = time.time()
    with torch.no_grad():
        for images, _ in data_loader:
            images = [img.to(device) for img in images]
            _ = model(images)
            n += len(images)
            if n >= max_batches:
                break
    t = time.time() - t0
    return n / t if t > 0 else 0.0
