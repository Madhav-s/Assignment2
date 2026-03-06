from pathlib import Path
from typing import Optional, Dict, Any

from ultralytics import YOLO


class YOLOWrapper:
    def __init__(self, model_name: str = 'yolov8n'):
        try:
            # passing just the model size name lets ultralytics handle fetching
            self.model = YOLO(model_name)
        except Exception as e:
            print(f"Warning: failed to initialize YOLO model {model_name}: {e}")
            # try falling back to a blank model
            self.model = YOLO('yolov8n')

    def train(self, data_yaml: str, epochs: int = 10, batch: int = 8, imgsz: int = 512, save_dir: Optional[str] = None):
        # data_yaml is path to a data configuration file compatible with YOLOv8
        results = self.model.train(data=data_yaml, epochs=epochs, batch=batch, imgsz=imgsz, save_dir=save_dir)
        return results

    def predict(self, source: str, conf: float = 0.25, save: bool = False, save_dir: Optional[str] = None):
        preds = self.model.predict(source=source, conf=conf, save=save, save_dir=save_dir, imgsz=512)
        return preds

    def export(self, format: str = 'onnx'):
        return self.model.export(format=format)
