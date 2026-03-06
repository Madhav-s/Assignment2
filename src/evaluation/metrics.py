from typing import List, Tuple
import numpy as np


def box_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    # box format [x1,y1,x2,y2]
    xa = max(boxA[0], boxB[0])
    ya = max(boxA[1], boxB[1])
    xb = min(boxA[2], boxB[2])
    yb = min(boxA[3], boxB[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def compute_tp_fp_fn(gt_boxes: List[np.ndarray], pred_boxes: List[np.ndarray], iou_thresh: float=0.5) -> Tuple[int,int,int]:
    gt_matched = [False] * len(gt_boxes)
    tp = 0
    fp = 0
    for pb in pred_boxes:
        matched = False
        for i, gb in enumerate(gt_boxes):
            if not gt_matched[i] and box_iou(pb, gb) >= iou_thresh:
                tp += 1
                gt_matched[i] = True
                matched = True
                break
        if not matched:
            fp += 1
    fn = sum(1 for m in gt_matched if not m)
    return tp, fp, fn


def calculate_metrics(all_gt: List[List[np.ndarray]], all_pred: List[List[np.ndarray]]) -> Tuple[float,float,float]:
    total_tp = total_fp = total_fn = 0
    for gt, pr in zip(all_gt, all_pred):
        tp, fp, fn = compute_tp_fp_fn(gt, pr)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    # mAP approximated as precision here
    mAP = precision
    return mAP, precision, recall
