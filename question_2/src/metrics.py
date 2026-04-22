import torch

def compute_iou(pred, target, num_classes=23):
    ious = []
    pred = pred.argmax(1)

    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            continue

        ious.append(intersection / union)

    return sum(ious) / len(ious)


def dice_score(pred, target):
    pred = pred.argmax(1)
    intersection = (pred * target).sum().float()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-6)
