import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def _hd95(a, b):
    a = a.astype(bool)
    b = b.astype(bool)

    if not a.any() and not b.any():
        return 0.0

    if not a.any() or not b.any():
        return float(max(a.shape))

    surface_a = a ^ binary_erosion(a)
    surface_b = b ^ binary_erosion(b)

    dist_a = distance_transform_edt(~a)
    dist_b = distance_transform_edt(~b)

    distances = np.concatenate([ dist_b[surface_a], dist_a[surface_b], ])

    return float(np.percentile(distances, 95)) if len(distances) else 0.0


def batch_metrics(prob, target, threshold=0.5):
    pred = prob.detach().cpu().numpy() >= threshold
    target = target.detach().cpu().numpy() >= 0.5

    dice = []
    iou = []
    hd95 = []

    for n in range(pred.shape[0]):
        for c in range(pred.shape[1]):
            p = pred[n, c]
            y = target[n, c]

            intersection = np.logical_and(p, y).sum()
            union = np.logical_or(p, y).sum()
            total = p.sum() + y.sum()

            dice.append( (2 * intersection + 1e-6) / (total + 1e-6) )

            iou.append( (intersection + 1e-6) / (union + 1e-6) )

            hd95.append(_hd95(p, y))

    return ( float(np.mean(dice)), float(np.mean(iou)), float(np.mean(hd95)), )