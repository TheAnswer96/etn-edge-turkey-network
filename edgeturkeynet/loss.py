"""
Loss functions for EdgeTurkeyNet FCOS-style anchor-free detection.

Two detection classes:
  0 — body  (large oval torso)
  1 — neck  (smaller elongated protrusion)

Components:
1. CIoU Loss — Complete IoU for tight bounding box regression.
   Aspect ratio term v is especially valuable for the oval body class.

2. Multi-class Focal Loss — Per-class sigmoid classification.
   Each grid cell has a C=2 target vector; only the GT class channel
   is set to 1.0 for positive cells.  Background cells get all zeros.
   Focal weighting handles the extreme foreground/background imbalance.

3. Centerness Loss — BCE for oval-aware centerness targets.

FCOS Target Assignment (multi-class):
- GT boxes carry class ids; each positive cell's cls_target is a one-hot
  vector of length C rather than a scalar.
- Size ranges per FPN level (pixels in input-image space):
    P3 (stride  8): [  0,  96] — small objects / necks at distance
    P4 (stride 16): [ 96, 192] — medium turkeys (primary body scale)
    P5 (stride 32): [192,  ∞ ] — large / close-up turkeys
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Size range per FPN level (in pixels, input-image space)
FCOS_SIZE_RANGES: List[Tuple[float, float]] = [
    (0,   96),    # P3 — small
    (96,  192),   # P4 — medium (primary for top-down turkeys ~50-150px)
    (192, 1e6),   # P5 — large
]

STRIDES = [8, 16, 32]


# ---------------------------------------------------------------------------
# IoU Utilities
# ---------------------------------------------------------------------------

def ciou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Complete IoU (CIoU) loss for bounding box regression.

    CIoU = 1 - IoU + (ρ²/c²) + α*v
    where:
      ρ = center distance, c = diagonal of enclosing box
      v = aspect ratio consistency term
      α = weight balancing IoU and aspect ratio

    For turkey detection: aspect ratio term v is especially important
    because turkeys have a consistent oval body shape from above.
    CIoU penalizes aspect ratio deviation, improving localization stability.

    Args:
        pred_boxes: [N, 4] predicted boxes (x1, y1, x2, y2).
        target_boxes: [N, 4] target boxes (x1, y1, x2, y2).
        eps: Small epsilon to avoid division by zero.

    Returns:
        loss: Scalar mean CIoU loss.
    """
    # Intersection
    inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    # Union
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(0) * \
                (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(0)
    tgt_area  = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(0) * \
                (target_boxes[:, 3] - target_boxes[:, 1]).clamp(0)
    union_area = pred_area + tgt_area - inter_area + eps

    iou = inter_area / union_area

    # Center distance term ρ²/c²
    pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    tgt_cx  = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    tgt_cy  = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
    rho_sq  = (pred_cx - tgt_cx) ** 2 + (pred_cy - tgt_cy) ** 2

    # Enclosing box diagonal c²
    enc_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    enc_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    enc_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    enc_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    c_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

    # Aspect ratio consistency term v
    pred_w = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(eps)
    pred_h = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(eps)
    tgt_w  = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(eps)
    tgt_h  = (target_boxes[:, 3] - target_boxes[:, 1]).clamp(eps)

    v = (4 / math.pi ** 2) * (
        torch.atan(tgt_w / tgt_h) - torch.atan(pred_w / pred_h)
    ) ** 2

    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - rho_sq / c_sq - alpha * v
    return (1 - ciou).mean()


def compute_iou(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes.

    Args:
        boxes_a: [N, 4] boxes (x1, y1, x2, y2).
        boxes_b: [M, 4] boxes (x1, y1, x2, y2).

    Returns:
        iou: [N, M] IoU matrix.
    """
    area_a = ((boxes_a[:, 2] - boxes_a[:, 0]) *
              (boxes_a[:, 3] - boxes_a[:, 1])).unsqueeze(1)
    area_b = ((boxes_b[:, 2] - boxes_b[:, 0]) *
              (boxes_b[:, 3] - boxes_b[:, 1])).unsqueeze(0)

    inter_x1 = torch.max(boxes_a[:, 0].unsqueeze(1), boxes_b[:, 0].unsqueeze(0))
    inter_y1 = torch.max(boxes_a[:, 1].unsqueeze(1), boxes_b[:, 1].unsqueeze(0))
    inter_x2 = torch.min(boxes_a[:, 2].unsqueeze(1), boxes_b[:, 2].unsqueeze(0))
    inter_y2 = torch.min(boxes_a[:, 3].unsqueeze(1), boxes_b[:, 3].unsqueeze(0))

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area_a + area_b - inter + 1e-7

    return inter / union


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

def focal_loss(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Binary Focal Loss — works for both single-class (1D) and multi-class (2D).

    For multi-class use (C > 1): pred_logits and targets are both [N, C].
    Each channel is treated as an independent binary problem (per-channel
    sigmoid), matching the FCOS one-hot target convention.

    FL = -α(1-p_t)^γ log(p_t)

    For two-class aerial farm detection:
    - Most cells are background (severe imbalance across both channels)
    - γ=2 down-weights easy background predictions
    - α=0.25 balances positive/negative per channel

    EDGE AI: Training-time only — zero inference overhead.

    Args:
        pred_logits: [N] or [N, C] raw logits.
        targets:     [N] or [N, C] binary float targets in {0.0, 1.0}.
        gamma:       Focusing parameter (default 2.0).
        alpha:       Balance parameter (default 0.25).
        reduction:   'mean', 'sum', or 'none'.

    Returns:
        Focal loss scalar (or per-element tensor if reduction='none').
    """
    p = torch.sigmoid(pred_logits)
    ce = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction="none")

    p_t       = p * targets + (1 - p) * (1 - targets)
    alpha_t   = alpha * targets + (1 - alpha) * (1 - targets)
    focal_wt  = alpha_t * (1 - p_t) ** gamma
    loss      = focal_wt * ce

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


# ---------------------------------------------------------------------------
# FCOS Target Assignment — multi-class
# ---------------------------------------------------------------------------

def fcos_assign_targets(
    gt_boxes_batch: List[torch.Tensor],
    gt_class_ids_batch: List[torch.Tensor],
    feature_shapes: List[Tuple[int, int]],
    num_classes: int = 2,
    strides: List[int] = STRIDES,
    size_ranges: List[Tuple[float, float]] = FCOS_SIZE_RANGES,
    input_size: Tuple[int, int] = (640, 640),
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    FCOS target assignment across FPN levels — multi-class variant.

    For each cell at each FPN level, determines:
    - cls_target : one-hot [C] vector (C = num_classes).  Background = all zeros.
    - reg_target : (l, t, r, b) pixel distances to the assigned GT box edges.
    - ctr_target : oval-aware centerness scalar in [0, 1].

    Class id is taken from ``gt_class_ids_batch`` and written into the correct
    channel of cls_target, so the classification head learns to distinguish
    body (0) from neck (1) cells independently through per-channel sigmoid.

    Args:
        gt_boxes_batch:     List[B] of [N_i, 4] normalised (cx, cy, w, h).
        gt_class_ids_batch: List[B] of [N_i] int64 class indices {0, 1}.
        feature_shapes:     List of (H_i, W_i) per FPN level.
        num_classes:        Number of detection classes (default 2).
        strides:            Pixel stride per FPN level.
        size_ranges:        (min_px, max_px) size range per FPN level.
        input_size:         Model input (H, W).

    Returns:
        cls_targets: List[levels] of [B, H*W, C] float32 one-hot tensors.
        reg_targets: List[levels] of [B, H*W, 4] float32 (l, t, r, b).
        ctr_targets: List[levels] of [B, H*W]    float32 centerness.
    """
    B = len(gt_boxes_batch)
    ih, iw = input_size

    cls_targets_all: List[torch.Tensor] = []
    reg_targets_all: List[torch.Tensor] = []
    ctr_targets_all: List[torch.Tensor] = []

    for level_idx, ((fh, fw), stride, (min_sz, max_sz)) in enumerate(
        zip(feature_shapes, strides, size_ranges)
    ):
        N = fh * fw

        # Cell centres in pixel space for this level
        yv, xv = torch.meshgrid(
            torch.arange(fh, dtype=torch.float32),
            torch.arange(fw, dtype=torch.float32),
            indexing='ij'
        )
        cell_cx = (xv.reshape(-1) + 0.5) * stride  # [N]
        cell_cy = (yv.reshape(-1) + 0.5) * stride  # [N]

        batch_cls: List[torch.Tensor] = []
        batch_reg: List[torch.Tensor] = []
        batch_ctr: List[torch.Tensor] = []

        for b in range(B):
            gt       = gt_boxes_batch[b]       # [M, 4] normalised (cx,cy,w,h)
            gt_cls   = gt_class_ids_batch[b]   # [M] int64

            # Derive device from GT tensors so all targets live on the same device.
            device = gt.device

            cls_target = torch.zeros(N, num_classes, dtype=torch.float32, device=device)
            reg_target = torch.zeros(N, 4,           dtype=torch.float32, device=device)
            ctr_target = torch.zeros(N,              dtype=torch.float32, device=device)

            if len(gt) == 0:
                batch_cls.append(cls_target)
                batch_reg.append(reg_target)
                batch_ctr.append(ctr_target)
                continue

            # Convert GT to absolute pixel (x1, y1, x2, y2)
            gt_x1 = (gt[:, 0] - gt[:, 2] / 2) * iw
            gt_y1 = (gt[:, 1] - gt[:, 3] / 2) * ih
            gt_x2 = (gt[:, 0] + gt[:, 2] / 2) * iw
            gt_y2 = (gt[:, 1] + gt[:, 3] / 2) * ih
            gt_abs = torch.stack([gt_x1, gt_y1, gt_x2, gt_y2], dim=-1)  # [M, 4]

            # Cell-centre distances to each GT box edge: [N, M]
            cx = cell_cx.to(device).unsqueeze(1)   # [N, 1]
            cy = cell_cy.to(device).unsqueeze(1)

            l = cx - gt_abs[:, 0].unsqueeze(0)
            t = cy - gt_abs[:, 1].unsqueeze(0)
            r = gt_abs[:, 2].unsqueeze(0) - cx
            b_dist = gt_abs[:, 3].unsqueeze(0) - cy

            # Positive: cell is inside the GT box
            inside_mask = (l > 0) & (t > 0) & (r > 0) & (b_dist > 0)  # [N, M]

            # Size-range filter: the max regression distance must fall in level range
            max_reg   = torch.stack([l, t, r, b_dist], dim=-1).max(dim=-1).values
            size_mask = (max_reg >= min_sz) & (max_reg <= max_sz)

            valid_mask = inside_mask & size_mask  # [N, M]

            # Among valid GT boxes, assign each cell to the smallest one
            gt_areas = (gt_abs[:, 2] - gt_abs[:, 0]) * (gt_abs[:, 3] - gt_abs[:, 1])
            gt_areas_exp = gt_areas.unsqueeze(0).expand(N, -1).clone()  # [N, M]
            gt_areas_exp[~valid_mask] = float('inf')

            best_gt   = gt_areas_exp.argmin(dim=1)   # [N]
            any_valid = valid_mask.any(dim=1)         # [N]
            pos_inds  = any_valid.nonzero(as_tuple=True)[0]

            if len(pos_inds) > 0:
                gt_inds = best_gt[pos_inds]          # [P] indices into GT boxes

                # One-hot classification target: set channel = class_id to 1.0
                assigned_cls = gt_cls[gt_inds]       # [P] int64 {0, 1}
                cls_target[pos_inds, assigned_cls] = 1.0

                # Regression target
                reg_l = l[pos_inds, gt_inds]
                reg_t = t[pos_inds, gt_inds]
                reg_r = r[pos_inds, gt_inds]
                reg_b = b_dist[pos_inds, gt_inds]
                reg_target[pos_inds] = torch.stack(
                    [reg_l, reg_t, reg_r, reg_b], dim=-1
                )

                # Oval-aware centerness: sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b))
                min_lr = torch.min(reg_l, reg_r)
                max_lr = torch.max(reg_l, reg_r).clamp(1e-6)
                min_tb = torch.min(reg_t, reg_b)
                max_tb = torch.max(reg_t, reg_b).clamp(1e-6)
                ctr_target[pos_inds] = torch.sqrt(
                    (min_lr / max_lr) * (min_tb / max_tb)
                )

            batch_cls.append(cls_target)
            batch_reg.append(reg_target)
            batch_ctr.append(ctr_target)

        cls_targets_all.append(torch.stack(batch_cls, dim=0))  # [B, N, C]
        reg_targets_all.append(torch.stack(batch_reg, dim=0))  # [B, N, 4]
        ctr_targets_all.append(torch.stack(batch_ctr, dim=0))  # [B, N]

    return cls_targets_all, reg_targets_all, ctr_targets_all


# ---------------------------------------------------------------------------
# Combined Detection Loss
# ---------------------------------------------------------------------------

class EdgeTurkeyLoss(nn.Module):
    """
    Combined loss function for EdgeTurkeyNet (multi-class).

    total = λ_cls * FocalLoss(cls) + λ_reg * CIoU(boxes) + λ_ctr * BCE(centerness)

    Classification uses per-channel sigmoid (not softmax) so body and neck
    can co-occur in the same cell — e.g. the neck box may be labelled inside
    a body box at lower FPN levels.  Each class is a binary problem:

      body channel: 1 at cells assigned to a body GT box, 0 elsewhere
      neck channel: 1 at cells assigned to a neck GT box, 0 elsewhere

    Loss weights tuned for two-class dense top-down detection:
    - λ_reg = 2.0: prioritise tight localisation of oval bodies
    - λ_cls = 1.0: balanced — both classes need equal classification signal
    - λ_ctr = 0.5: centerness is auxiliary; regression drives accuracy

    Args:
        num_classes:  Number of detection classes (default 2: body + neck).
        lambda_cls:   Classification loss weight.
        lambda_reg:   Regression loss weight.
        lambda_ctr:   Centerness loss weight.
        input_size:   Model input (H, W).
    """

    def __init__(
        self,
        num_classes: int = 2,
        lambda_cls: float = 1.0,
        lambda_reg: float = 2.0,
        lambda_ctr: float = 0.5,
        input_size: Tuple[int, int] = (640, 640),
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.lambda_cls  = lambda_cls
        self.lambda_reg  = lambda_reg
        self.lambda_ctr  = lambda_ctr
        self.input_size  = input_size
        self.strides     = STRIDES

    def forward(
        self,
        cls_preds: List[torch.Tensor],          # [B, C, H, W] per level
        reg_preds: List[torch.Tensor],          # [B, 4, H, W] per level
        ctr_preds: List[torch.Tensor],          # [B, 1, H, W] per level
        gt_boxes_batch: List[torch.Tensor],     # List[B] of [N_i, 4]
        gt_class_ids_batch: List[torch.Tensor], # List[B] of [N_i] int64
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-class detection losses.

        Args:
            cls_preds:          Per-level classification predictions (raw logits).
            reg_preds:          Per-level regression predictions (l, t, r, b pixels).
            ctr_preds:          Per-level centerness predictions (raw logits).
            gt_boxes_batch:     GT boxes per image (normalised cx, cy, w, h).
            gt_class_ids_batch: GT class ids per image {0=body, 1=neck}.

        Returns:
            dict with 'total', 'cls', 'reg', 'ctr' loss scalars.
        """
        device = cls_preds[0].device

        feature_shapes = [(p.shape[2], p.shape[3]) for p in cls_preds]

        # Move all GT tensors to the device where predictions live
        gt_boxes_dev    = [b.to(device) for b in gt_boxes_batch]
        gt_cls_ids_dev  = [c.to(device) for c in gt_class_ids_batch]

        cls_targets, reg_targets, ctr_targets = fcos_assign_targets(
            gt_boxes_dev,
            gt_cls_ids_dev,
            feature_shapes,
            num_classes=self.num_classes,
            strides=self.strides,
            size_ranges=FCOS_SIZE_RANGES,
            input_size=self.input_size,
        )

        total_cls_loss = torch.tensor(0.0, device=device)
        total_reg_loss = torch.tensor(0.0, device=device)
        total_ctr_loss = torch.tensor(0.0, device=device)

        for level_idx in range(len(cls_preds)):
            B, C, H, W = cls_preds[level_idx].shape
            N = H * W

            # Flatten: [B*N, C]
            cls_pred_flat = (cls_preds[level_idx]
                             .reshape(B, C, N).permute(0, 2, 1)
                             .reshape(-1, C))
            reg_pred_flat = (reg_preds[level_idx]
                             .reshape(B, 4, N).permute(0, 2, 1)
                             .reshape(-1, 4))
            ctr_pred_flat = (ctr_preds[level_idx]
                             .reshape(B, 1, N).permute(0, 2, 1)
                             .reshape(-1))

            # cls_targets shape: [B, N, C]  →  [B*N, C]
            cls_tgt_flat = cls_targets[level_idx].reshape(-1, C)
            reg_tgt_flat = reg_targets[level_idx].reshape(-1, 4)
            ctr_tgt_flat = ctr_targets[level_idx].reshape(-1)

            # ── Classification loss (all cells, per-channel sigmoid focal) ──
            # Each channel is an independent binary problem.
            cls_loss = focal_loss(cls_pred_flat, cls_tgt_flat)
            total_cls_loss = total_cls_loss + cls_loss

            # ── Positive mask: any class is positive in this cell ──
            pos_mask = cls_tgt_flat.sum(dim=-1) > 0.5  # [B*N]
            num_pos  = int(pos_mask.sum().item())

            if num_pos > 0:
                reg_pred_pos = reg_pred_flat[pos_mask]
                reg_tgt_pos  = reg_tgt_flat[pos_mask]

                # Encode as x1y1x2y2 relative to cell centre (centre = 0,0)
                pred_boxes_pos = torch.stack([
                    -reg_pred_pos[:, 0], -reg_pred_pos[:, 1],
                     reg_pred_pos[:, 2],  reg_pred_pos[:, 3],
                ], dim=-1)
                tgt_boxes_pos = torch.stack([
                    -reg_tgt_pos[:, 0], -reg_tgt_pos[:, 1],
                     reg_tgt_pos[:, 2],  reg_tgt_pos[:, 3],
                ], dim=-1)

                total_reg_loss = total_reg_loss + ciou_loss(
                    pred_boxes_pos, tgt_boxes_pos
                )
                total_ctr_loss = total_ctr_loss + F.binary_cross_entropy_with_logits(
                    ctr_pred_flat[pos_mask], ctr_tgt_flat[pos_mask]
                )

        num_levels = len(cls_preds)
        total_cls_loss = total_cls_loss / num_levels
        total_reg_loss = total_reg_loss / max(1, num_levels)
        total_ctr_loss = total_ctr_loss / max(1, num_levels)

        total_loss = (
            self.lambda_cls * total_cls_loss +
            self.lambda_reg * total_reg_loss +
            self.lambda_ctr * total_ctr_loss
        )

        return {
            "total": total_loss,
            "cls":   total_cls_loss,
            "reg":   total_reg_loss,
            "ctr":   total_ctr_loss,
        }