import numpy as np

def get_overlapping_part(damage_mask, part_masks, part_labels):
    max_iou = 0
    matched_part = None
    for mask, label in zip(part_masks, part_labels):
        iou = np.sum(np.logical_and(damage_mask, mask)) / np.sum(np.logical_or(damage_mask, mask))
        if iou > max_iou:
            max_iou = iou
            matched_part = label
    return matched_part, max_iou

def estimate_severity(area, part_label):
    if area < 2000:
        return "minor"
    elif area < 8000:
        return "moderate"
    else:
        return "severe" if part_label in ["front_bumper", "hood", "rear_bumper"] else "moderate"
