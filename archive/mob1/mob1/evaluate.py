# =============================================================================
# evaluate.py — IoU, Precision, Recall, mAP50, mAP50:95
# =============================================================================

import torch
import torch.nn.functional as F
from torchvision.ops import nms
import numpy as np
from mob1.config import (NUM_CLASSES, CLASS_NAMES, CONF_THRESHOLD, NMS_IOU,
                    INPUT_SIZE, REG_MAX, STRIDE)


# ---------------------------------------------------------------------------
# NMS + decode predictions → list of (cls, conf, x1, y1, x2, y2)
# ---------------------------------------------------------------------------

def decode_predictions(cls_pred, decoded_boxes, conf_thresh=CONF_THRESHOLD,
                        nms_thresh=NMS_IOU):
    """
    cls_pred:     (N, num_classes) logits
    decoded_boxes:(N, 4) xyxy pixels
    Returns list of dicts per class or a single tensor (num_det, 6)
      columns: [cls, conf, x1, y1, x2, y2]
    """
    scores = cls_pred.sigmoid()  # (N, C)
    max_scores, max_cls = scores.max(dim=-1)  # (N,)

    mask = max_scores > conf_thresh
    if not mask.any():
        return torch.zeros((0, 6), device=cls_pred.device)

    boxes  = decoded_boxes[mask]
    confs  = max_scores[mask]
    clsids = max_cls[mask].float()

    keep = nms(boxes, confs, nms_thresh)
    result = torch.cat([clsids[keep, None],
                        confs[keep, None],
                        boxes[keep]], dim=-1)
    return result  # (K, 6)


# ---------------------------------------------------------------------------
# IoU between two sets of boxes
# ---------------------------------------------------------------------------

def box_iou_np(b1, b2):
    """b1: (N,4), b2: (M,4) numpy xyxy → (N,M)"""
    ix1 = np.maximum(b1[:, None, 0], b2[None, :, 0])
    iy1 = np.maximum(b1[:, None, 1], b2[None, :, 1])
    ix2 = np.minimum(b1[:, None, 2], b2[None, :, 2])
    iy2 = np.minimum(b1[:, None, 3], b2[None, :, 3])
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    a1 = (b1[:, 2]-b1[:, 0]) * (b1[:, 3]-b1[:, 1])
    a2 = (b2[:, 2]-b2[:, 0]) * (b2[:, 3]-b2[:, 1])
    return inter / (a1[:, None] + a2[None, :] - inter + 1e-7)


# ---------------------------------------------------------------------------
# AP computation (11-point or area under PR curve)
# ---------------------------------------------------------------------------

def compute_ap(recall, precision):
    """Area under precision-recall curve (VOC 2010+ method)."""
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx  = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])


# ---------------------------------------------------------------------------
# Full evaluation over a dataloader
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataloader, device, iou_thresholds=None):
    """
    Returns dict with keys:
      precision, recall, mAP50, mAP50_95, per_class_ap50
      plus per_class breakdown
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)  # 0.5:0.05:0.95

    model.eval()

    # Collect all predictions and GTs per class
    # stats[cls] = list of (conf, tp_iou50, ...) per detection
    all_preds = []   # list of (cls, conf, xyxy) arrays per image
    all_gts   = []   # list of (cls, xyxy) arrays per image

    for imgs, targets in dataloader:
        imgs = imgs.to(device)
        cls_pred, reg_pred, boxes = model(imgs)

        for b in range(len(imgs)):
            # Predictions
            dets = decode_predictions(cls_pred[b], boxes[b])
            if len(dets):
                all_preds.append(dets.cpu().numpy())
            else:
                all_preds.append(np.zeros((0, 6)))

            # Ground truths
            gt = targets[b]
            if len(gt):
                gt = gt.numpy()
                cx = gt[:, 1] * INPUT_SIZE; cy = gt[:, 2] * INPUT_SIZE
                gw = gt[:, 3] * INPUT_SIZE; gh = gt[:, 4] * INPUT_SIZE
                gt_xyxy = np.stack([cx-gw/2, cy-gh/2,
                                    cx+gw/2, cy+gh/2], axis=1)
                all_gts.append(np.hstack([gt[:, 0:1], gt_xyxy]))
            else:
                all_gts.append(np.zeros((0, 5)))

    # ---- compute AP per class per IoU threshold ----
    ap_matrix = np.zeros((NUM_CLASSES, len(iou_thresholds)))

    for ci in range(NUM_CLASSES):
        # Each entry: [score, tp_t0, tp_t1, ..., tp_tN]
        det_records = []
        n_gt = 0

        for img_preds, img_gts in zip(all_preds, all_gts):
            gt_mask  = (img_gts[:, 0] == ci) if len(img_gts) else np.zeros(0, bool)
            gt_boxes = img_gts[gt_mask, 1:] if gt_mask.any() else np.zeros((0, 4))
            n_gt    += len(gt_boxes)

            pred_mask = (img_preds[:, 0] == ci) if len(img_preds) else np.zeros(0, bool)
            preds     = img_preds[pred_mask] if pred_mask.any() else np.zeros((0, 6))

            if len(preds) == 0:
                continue

            # Sort this image's preds by confidence
            order = np.argsort(-preds[:, 1])
            preds = preds[order]

            gt_matched = {t: set() for t in range(len(iou_thresholds))}

            if len(gt_boxes) > 0:
                iou_mat = box_iou_np(preds[:, 2:], gt_boxes)  # (D, G)

            for d in range(len(preds)):
                row = [preds[d, 1]]  # score
                for t, iou_thr in enumerate(iou_thresholds):
                    if len(gt_boxes) == 0:
                        row.append(0)
                        continue
                    best_iou = iou_mat[d].max()
                    best_g   = iou_mat[d].argmax()
                    if best_iou >= iou_thr and best_g not in gt_matched[t]:
                        row.append(1)
                        gt_matched[t].add(best_g)
                    else:
                        row.append(0)
                det_records.append(row)

        if n_gt == 0 or len(det_records) == 0:
            continue

        # Sort ALL detections globally by score descending
        det_arr = np.array(det_records)          # (D, 1+num_thresholds)
        sort_idx = np.argsort(-det_arr[:, 0])
        det_arr  = det_arr[sort_idx]

        for t in range(len(iou_thresholds)):
            tp_arr = det_arr[:, t + 1]
            tp_cum = np.cumsum(tp_arr)
            fp_cum = np.cumsum(1 - tp_arr)
            prec   = tp_cum / (tp_cum + fp_cum + 1e-7)
            rec    = tp_cum / (n_gt + 1e-7)
            ap_matrix[ci, t] = compute_ap(rec, prec)

    ap50       = ap_matrix[:, 0]           # per class @ IoU 0.5
    mAP50      = ap50.mean()
    mAP50_95   = ap_matrix.mean()

    # ---- precision / recall @ conf_thresh, IoU 0.5 ----
    tp_total = fp_total = fn_total = 0
    for img_preds, img_gts in zip(all_preds, all_gts):
        for ci in range(NUM_CLASSES):
            gt_mask   = img_gts[:, 0] == ci if len(img_gts) else np.array([])
            pred_mask = img_preds[:, 0] == ci if len(img_preds) else np.array([])
            gt_boxes  = img_gts[gt_mask, 1:]   if len(img_gts)   else np.zeros((0,4))
            preds     = img_preds[pred_mask, 2:] if len(img_preds) else np.zeros((0,4))

            if len(preds) == 0:
                fn_total += len(gt_boxes)
                continue
            if len(gt_boxes) == 0:
                fp_total += len(preds)
                continue

            iou_mat = box_iou_np(preds, gt_boxes)
            matched_gt = set()
            for d in range(len(preds)):
                best_iou = iou_mat[d].max()
                best_g   = iou_mat[d].argmax()
                if best_iou >= 0.5 and best_g not in matched_gt:
                    tp_total += 1
                    matched_gt.add(best_g)
                else:
                    fp_total += 1
            fn_total += len(gt_boxes) - len(matched_gt)

    precision = tp_total / (tp_total + fp_total + 1e-7)
    recall    = tp_total / (tp_total + fn_total + 1e-7)

    results = {
        "precision":    round(float(precision), 4),
        "recall":       round(float(recall),    4),
        "mAP50":        round(float(mAP50),     4),
        "mAP50_95":     round(float(mAP50_95),  4),
        "per_class_ap50": {CLASS_NAMES[i]: round(float(ap50[i]), 4)
                           for i in range(NUM_CLASSES)},
    }
    return results


def print_metrics(metrics: dict, prefix: str = ""):
    tag = f"[{prefix}] " if prefix else ""
    print(
        f"{tag}P={metrics['precision']:.4f}  "
        f"R={metrics['recall']:.4f}  "
        f"mAP50={metrics['mAP50']:.4f}  "
        f"mAP50:95={metrics['mAP50_95']:.4f}  "
        f"| " +
        "  ".join(f"{k}={v:.4f}"
                  for k, v in metrics["per_class_ap50"].items())
    )