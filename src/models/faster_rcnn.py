from typing import List
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_fasterrcnn_model(num_classes: int) -> torch.nn.Module:
    # load pre-trained model
    model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    # replace the head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
