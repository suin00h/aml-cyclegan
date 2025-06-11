import json
import torch
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy

def evaluate_fcn(pred_list, gt_list, dataset):
    pred_tensor = torch.stack(pred_list)
    gt_tensor = torch.stack(gt_list)

    # assume num_classes from dataset
    num_classes = 19 if dataset == "cityscapes" else 2

    iou = MulticlassJaccardIndex(num_classes=num_classes, average="macro")
    acc = MulticlassAccuracy(num_classes=num_classes, average="macro")

    # Convert RGB to label (dummy)
    pred_labels = (pred_tensor > 0).long()
    gt_labels = (gt_tensor > 0).long()

    return {
        "mIoU": iou(pred_labels, gt_labels).item(),
        "PixelAcc": acc(pred_labels, gt_labels).item()
    }

def save_fcn_result(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)