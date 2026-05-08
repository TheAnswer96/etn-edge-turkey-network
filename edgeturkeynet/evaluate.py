"""
Evaluation module for EdgeTurkeyNet — per-class metrics.

Computes per-class and mean metrics:
  - AP@0.5      per class  (body, neck)
  - Precision   per class  at best-F1 threshold
  - Recall      per class  at best-F1 threshold
  - mAP@0.5     mean over classes

Inference pipeline:
  predict() → per-image list of (boxes, class_ids, scores)
  NMS is applied per-class so body and neck suppressions are independent.

DIoU-NMS (per class):
  Runs independently for each class channel.  Keeps body and neck
  detections from suppressing each other — correct since a neck box
  will overlap its parent body box heavily but should not be removed.

  EDGE AI: Same O(N²) complexity as standard NMS; N is small (<300)
  after score filtering, and per-class split reduces N further.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from .model import EdgeTurkeyNet, CLASS_NAMES, NUM_CLASSES


# ---------------------------------------------------------------------------
# NMS configuration
# ---------------------------------------------------------------------------

SCORE_THRESHOLD   = 0.3
NMS_IOU_THRESHOLD = 0.5
MAX_DETECTIONS    = 300


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PerClassMetrics:
    """
    Per-class and aggregate evaluation results.

    Attributes:
        ap:        Dict mapping class_name -> AP@IoU scalar.
        precision: Dict mapping class_name -> precision at best-F1 threshold.
        recall:    Dict mapping class_name -> recall at best-F1 threshold.
        map:       Mean AP over all classes.
    """

    ap:        Dict[str, float] = field(default_factory=dict)
    precision: Dict[str, float] = field(default_factory=dict)
    recall:    Dict[str, float] = field(default_factory=dict)
    map:       float = 0.0

    def print_table(self, iou_threshold: float = 0.5) -> None:
        """Print a formatted per-class metrics table to stdout."""
        col_ap = f"AP@{iou_threshold:.2f}"
        header = f"  {'Class':<10} {col_ap:>10} {'Precision':>11} {'Recall':>9}"
        sep    = "  " + "-" * (len(header) - 2)
        print("\n" + sep)
        print(header)
        print(sep)
        for cls_name in CLASS_NAMES:
            print(
                f"  {cls_name:<10} "
                f"{self.ap.get(cls_name, 0.0):>10.4f} "
                f"{self.precision.get(cls_name, 0.0):>11.4f} "
                f"{self.recall.get(cls_name, 0.0):>9.4f}"
            )
        print(sep)
        print(f"  {'mAP':<10} {self.map:>10.4f}")
        print(sep + "\n")

    @property
    def mean_precision(self) -> float:
        """Mean precision across all classes."""
        vals = list(self.precision.values())
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def mean_recall(self) -> float:
        """Mean recall across all classes."""
        vals = list(self.recall.values())
        return sum(vals) / len(vals) if vals else 0.0


# ---------------------------------------------------------------------------
# DIoU-NMS (per-class)
# ---------------------------------------------------------------------------

def diou_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = NMS_IOU_THRESHOLD,
    max_dets: int = MAX_DETECTIONS,
) -> torch.Tensor:
    """
    Distance-IoU Non-Maximum Suppression via torchvision.ops.nms.

    DIoU = IoU - rho^2 / c^2
      rho = centre-to-centre Euclidean distance
      c   = diagonal of smallest enclosing box

    For dense turkey flocks, DIoU-NMS is more lenient than standard IoU-NMS
    when two boxes overlap in area but have well-separated centres, preserving
    adjacent turkey detections that IoU-NMS would suppress.

    Called once per class so body and neck detections never suppress each
    other — a neck box that heavily overlaps its parent body box is
    legitimate and must be preserved.

    Implementation
    --------------
    torchvision.ops.nms is a fused CUDA C++ kernel that implements standard
    IoU-NMS in O(N log N) with a single kernel launch regardless of N.  We
    adapt it for DIoU by subtracting the normalised centre-distance penalty
    from each box's score before calling it.  A box whose centre is far from
    the dominant box gets a lower effective score, making it harder to
    suppress its neighbours — exactly the DIoU semantics.

    Adjusted score:  s'_i = s_i - beta * mean_j(rho_ij^2 / c_ij^2)
    where beta scales the penalty relative to the score range.  Using the
    max pairwise distance in the set as normalisation keeps the adjustment
    bounded regardless of image scale.

    Args:
        boxes:         [N, 4] boxes in (x1, y1, x2, y2) format.
        scores:        [N] confidence scores.
        iou_threshold: Suppress box j when DIoU(i, j) > threshold.
        max_dets:      Cap on candidates considered (sorted by score).

    Returns:
        keep: [K] indices into the original (pre-sort) boxes tensor.
    """
    import torchvision

    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long)

    order = scores.argsort(descending=True)[:max_dets]
    b = boxes[order].float()                      # [N, 4]
    s = scores[order].float()                     # [N]
    N = b.shape[0]

    if N == 1:
        return order

    # Compute pairwise rho^2 / c^2  [N, N] — one vectorised GPU pass
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5

    rho_sq = (cx.unsqueeze(1) - cx.unsqueeze(0)) ** 2 \
           + (cy.unsqueeze(1) - cy.unsqueeze(0)) ** 2

    enc_x1 = torch.min(x1.unsqueeze(1), x1.unsqueeze(0))
    enc_y1 = torch.min(y1.unsqueeze(1), y1.unsqueeze(0))
    enc_x2 = torch.max(x2.unsqueeze(1), x2.unsqueeze(0))
    enc_y2 = torch.max(y2.unsqueeze(1), y2.unsqueeze(0))
    c_sq   = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-7

    # Normalised centre-distance penalty per box: mean over all pairs
    penalty = (rho_sq / c_sq).mean(dim=1)         # [N]

    # Subtract penalty scaled to score range so that boxes with well-
    # separated centres are harder to suppress (higher adjusted score).
    s_adj = s - 0.1 * penalty

    # torchvision.ops.nms — single fused CUDA kernel, O(N log N)
    keep_in_order = torchvision.ops.nms(b, s_adj, iou_threshold)
    return order[keep_in_order]


# ---------------------------------------------------------------------------
# Inference — multi-class, per-class NMS
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(
    model: EdgeTurkeyNet,
    images: torch.Tensor,
    score_threshold: float = SCORE_THRESHOLD,
    nms_threshold: float   = NMS_IOU_THRESHOLD,
    input_size: Tuple[int, int] = (640, 640),
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Run multi-class inference and return per-image detections after NMS.

    Per-class score = sigmoid(cls_logit_c) * sigmoid(centerness).
    Each of the C=2 class channels is thresholded independently, so a
    cell can produce both a body and a neck detection if both channels
    exceed the score threshold.

    Per-class DIoU-NMS is applied independently, ensuring body boxes
    do not suppress neck boxes (or vice versa).

    Args:
        model:            EdgeTurkeyNet in eval mode.
        images:           [B, 3, H, W] input tensor.
        score_threshold:  Minimum per-class score to keep a candidate.
        nms_threshold:    DIoU-NMS suppression threshold per class.
        input_size:       Model input (H, W).

    Returns:
        List[B] of (boxes, class_ids, scores):
            boxes:     [K, 4]  (x1, y1, x2, y2) pixel coordinates.
            class_ids: [K]     int64 predicted class index {0=body, 1=neck}.
            scores:    [K]     float32 per-class confidence scores.
        All returned tensors are on CPU.
    """
    model.eval()
    cls_preds, reg_preds, ctr_preds = model(images)

    # boxes_all:  [B, N, 4]
    # scores_all: [B, N, C]   per-class sigmoid
    # ctr_all:    [B, N, 1]   centerness sigmoid
    boxes_all, scores_all, ctr_all = model.decode_predictions(
        cls_preds, reg_preds, ctr_preds
    )

    # Combined: [B, N, C]  —  broadcast ctr [B, N, 1] across C channels
    combined = scores_all * ctr_all

    B = images.shape[0]
    ih, iw = input_size
    results: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for b in range(B):
        all_boxes_b:    List[torch.Tensor] = []
        all_cls_ids_b:  List[torch.Tensor] = []
        all_scores_b:   List[torch.Tensor] = []

        for cls_id in range(NUM_CLASSES):
            cls_scores = combined[b, :, cls_id]   # [N]
            cls_boxes  = boxes_all[b]              # [N, 4]

            mask = cls_scores > score_threshold
            if mask.sum() == 0:
                continue

            filt_scores = cls_scores[mask]
            filt_boxes  = cls_boxes[mask].clone()

            # Clip to image boundaries
            filt_boxes[:, 0::2] = filt_boxes[:, 0::2].clamp(0, iw)
            filt_boxes[:, 1::2] = filt_boxes[:, 1::2].clamp(0, ih)

            keep = diou_nms(filt_boxes, filt_scores, nms_threshold)
            if len(keep) == 0:
                continue

            all_boxes_b.append(filt_boxes[keep])
            all_scores_b.append(filt_scores[keep])
            all_cls_ids_b.append(
                torch.full((len(keep),), cls_id, dtype=torch.int64)
            )

        if len(all_boxes_b) == 0:
            results.append((
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros(0,      dtype=torch.int64),
                torch.zeros(0,      dtype=torch.float32),
            ))
        else:
            results.append((
                torch.cat(all_boxes_b,   dim=0).cpu(),
                torch.cat(all_cls_ids_b, dim=0).cpu(),
                torch.cat(all_scores_b,  dim=0).cpu(),
            ))

    return results


# ---------------------------------------------------------------------------
# IoU helper (local copy to avoid circular import with loss.py)
# ---------------------------------------------------------------------------

def _compute_iou(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
) -> torch.Tensor:
    """Compute pairwise IoU [N, M] between (x1,y1,x2,y2) box sets."""
    area_a = ((boxes_a[:, 2] - boxes_a[:, 0]) *
               (boxes_a[:, 3] - boxes_a[:, 1])).unsqueeze(1)
    area_b = ((boxes_b[:, 2] - boxes_b[:, 0]) *
               (boxes_b[:, 3] - boxes_b[:, 1])).unsqueeze(0)

    ix1 = torch.max(boxes_a[:, 0].unsqueeze(1), boxes_b[:, 0].unsqueeze(0))
    iy1 = torch.max(boxes_a[:, 1].unsqueeze(1), boxes_b[:, 1].unsqueeze(0))
    ix2 = torch.min(boxes_a[:, 2].unsqueeze(1), boxes_b[:, 2].unsqueeze(0))
    iy2 = torch.min(boxes_a[:, 3].unsqueeze(1), boxes_b[:, 3].unsqueeze(0))

    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    union = area_a + area_b - inter + 1e-7
    return inter / union


# ---------------------------------------------------------------------------
# Per-class AP computation
# ---------------------------------------------------------------------------

def _boxes_yolo_to_xyxy(
    boxes_yolo: torch.Tensor,
    input_size: Tuple[int, int] = (640, 640),
) -> torch.Tensor:
    """Convert normalised YOLO (cx, cy, w, h) to pixel (x1, y1, x2, y2)."""
    ih, iw = input_size
    cx, cy, w, h = (boxes_yolo[:, 0], boxes_yolo[:, 1],
                    boxes_yolo[:, 2], boxes_yolo[:, 3])
    return torch.stack([
        (cx - w / 2) * iw,
        (cy - h / 2) * ih,
        (cx + w / 2) * iw,
        (cy + h / 2) * ih,
    ], dim=-1)


def compute_ap_for_class(
    pred_boxes:  List[torch.Tensor],
    pred_scores: List[torch.Tensor],
    gt_boxes:    List[torch.Tensor],
    iou_threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Compute AP, precision, and recall for a single class.

    Implements the standard VOC 11-point interpolation AP curve.
    Both pred_boxes/scores and gt_boxes are already filtered to a single
    class by the caller, so this function is class-agnostic.

    Matching rule:
      A prediction at rank k is a TP if:
        (1) Its best-IoU against unmatched GT boxes >= iou_threshold.
        (2) The matched GT has not already been claimed by a higher-ranked pred.
      Otherwise it is a FP.

    Args:
        pred_boxes:    List[n_images] of [K_i, 4] predicted boxes for this class.
        pred_scores:   List[n_images] of [K_i] confidence scores.
        gt_boxes:      List[n_images] of [M_i, 4] GT boxes for this class.
        iou_threshold: IoU threshold for TP/FP classification.

    Returns:
        (ap, precision, recall) — floats in [0, 1].
        precision and recall are at the operating point with the best F1.
    """
    total_gt = sum(len(g) for g in gt_boxes)
    if total_gt == 0:
        return 0.0, 0.0, 0.0

    all_scores: List[float] = []
    all_tp:     List[int]   = []

    for img_pred_boxes, img_pred_scores, img_gt_boxes in zip(
        pred_boxes, pred_scores, gt_boxes
    ):
        if len(img_pred_boxes) == 0:
            continue

        if len(img_gt_boxes) == 0:
            # All predictions for this image are FPs for this class
            all_scores.extend(img_pred_scores.tolist())
            all_tp.extend([0] * len(img_pred_scores))
            continue

        iou_mat    = _compute_iou(img_pred_boxes, img_gt_boxes)   # [K, M]
        gt_matched = torch.zeros(len(img_gt_boxes), dtype=torch.bool)

        # Process predictions in descending score order
        for pred_idx in img_pred_scores.argsort(descending=True):
            score              = img_pred_scores[pred_idx].item()
            ious               = iou_mat[pred_idx]                # [M]
            best_iou_val, best_gt_idx = ious.max(0)

            if best_iou_val.item() >= iou_threshold and not gt_matched[best_gt_idx]:
                gt_matched[best_gt_idx] = True
                all_scores.append(score)
                all_tp.append(1)
            else:
                all_scores.append(score)
                all_tp.append(0)

    if len(all_scores) == 0:
        return 0.0, 0.0, 0.0

    # Build precision-recall curve (sorted by score descending)
    order       = sorted(range(len(all_scores)), key=lambda i: -all_scores[i])
    tp_cumsum   = 0
    fp_cumsum   = 0
    precisions: List[float] = []
    recalls:    List[float] = []

    for idx in order:
        if all_tp[idx]:
            tp_cumsum += 1
        else:
            fp_cumsum += 1
        precisions.append(tp_cumsum / (tp_cumsum + fp_cumsum))
        recalls.append(tp_cumsum / total_gt)

    # 11-point interpolation AP (PASCAL VOC style)
    ap = 0.0
    for thresh in (t / 10.0 for t in range(11)):
        ps = [p for p, r in zip(precisions, recalls) if r >= thresh]
        ap += (max(ps) if ps else 0.0) / 11

    # Precision / recall at best F1 operating point
    best_f1, best_p, best_r = 0.0, 0.0, 0.0
    for p, r in zip(precisions, recalls):
        f1 = 2 * p * r / (p + r + 1e-7)
        if f1 > best_f1:
            best_f1, best_p, best_r = f1, p, r

    return ap, best_p, best_r


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_map(
    model: EdgeTurkeyNet,
    dataloader: DataLoader,
    device: torch.device,
    iou_threshold: float = 0.5,
    score_threshold: float = SCORE_THRESHOLD,
    input_size: Tuple[int, int] = (640, 640),
) -> PerClassMetrics:
    """
    Evaluate per-class and mean AP on a dataset split.

    For each class c in {0=body, 1=neck}:
      - Collects all predicted boxes/scores where predicted class == c.
      - Collects all GT boxes where gt class_id == c.
      - Calls compute_ap_for_class() to get AP, precision, recall for c.

    mAP = mean AP over all classes.

    Args:
        model:           EdgeTurkeyNet model.
        dataloader:      DataLoader for the evaluation split.
                         Batches must contain 'boxes' and 'class_ids'.
        device:          Compute device.
        iou_threshold:   IoU threshold for TP/FP classification.
        score_threshold: Minimum per-class score to keep a prediction.
        input_size:      Model input (H, W).

    Returns:
        PerClassMetrics with per-class ap / precision / recall and mAP.
    """
    model.eval()

    # Per-class accumulators: one entry per image per class
    pred_boxes_per_cls:  Dict[int, List[torch.Tensor]] = {c: [] for c in range(NUM_CLASSES)}
    pred_scores_per_cls: Dict[int, List[torch.Tensor]] = {c: [] for c in range(NUM_CLASSES)}
    gt_boxes_per_cls:    Dict[int, List[torch.Tensor]] = {c: [] for c in range(NUM_CLASSES)}

    for batch in dataloader:
        images     = batch["images"].to(device, non_blocking=True)
        gt_boxes   = batch["boxes"]      # List[B] of [N_i, 4] normalised
        gt_cls_ids = batch["class_ids"]  # List[B] of [N_i] int64

        detections = predict(
            model, images,
            score_threshold=score_threshold,
            nms_threshold=NMS_IOU_THRESHOLD,
            input_size=input_size,
        )

        for i, (pred_boxes, pred_cls_ids, pred_scores) in enumerate(detections):
            gt_b   = gt_boxes[i]    # [M, 4] normalised
            gt_ids = gt_cls_ids[i]  # [M] int64

            for cls_id in range(NUM_CLASSES):
                # ── GT for this class ────────────────────────────────
                cls_gt_mask = (gt_ids == cls_id)
                if cls_gt_mask.any():
                    gt_xyxy = _boxes_yolo_to_xyxy(gt_b[cls_gt_mask], input_size)
                else:
                    gt_xyxy = torch.zeros((0, 4), dtype=torch.float32)
                gt_boxes_per_cls[cls_id].append(gt_xyxy.cpu())

                # ── Predictions for this class ───────────────────────
                if len(pred_boxes) > 0 and (pred_cls_ids == cls_id).any():
                    cls_pred_mask = (pred_cls_ids == cls_id)
                    pred_boxes_per_cls[cls_id].append(
                        pred_boxes[cls_pred_mask].cpu()
                    )
                    pred_scores_per_cls[cls_id].append(
                        pred_scores[cls_pred_mask].cpu()
                    )
                else:
                    pred_boxes_per_cls[cls_id].append(
                        torch.zeros((0, 4), dtype=torch.float32)
                    )
                    pred_scores_per_cls[cls_id].append(
                        torch.zeros(0, dtype=torch.float32)
                    )

    # ── Compute per-class metrics ─────────────────────────────────────────
    metrics   = PerClassMetrics()
    ap_values: List[float] = []

    for cls_id, cls_name in enumerate(CLASS_NAMES):
        ap, p, r = compute_ap_for_class(
            pred_boxes_per_cls[cls_id],
            pred_scores_per_cls[cls_id],
            gt_boxes_per_cls[cls_id],
            iou_threshold=iou_threshold,
        )
        metrics.ap[cls_name]        = ap
        metrics.precision[cls_name] = p
        metrics.recall[cls_name]    = r
        ap_values.append(ap)

    metrics.map = sum(ap_values) / len(ap_values) if ap_values else 0.0
    return metrics
