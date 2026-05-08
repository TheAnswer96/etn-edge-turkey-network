# =============================================================================
# loss.py — Quality Focal Loss + Distribution Focal Loss + CIoU
#           SimOTA dynamic label assignment (no anchors)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from mob1.config import (NUM_CLASSES, REG_MAX, INPUT_SIZE, STRIDE,
                    LAMBDA_CLS, LAMBDA_REG, LAMBDA_DFL,
                    TOPK_CANDIDATES, CENTER_RADIUS)


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

def box_iou(b1, b2):
    """
    b1: (N,4), b2: (M,4) xyxy pixels
    returns: (N,M) IoU matrix
    """
    inter_x1 = torch.max(b1[:, None, 0], b2[None, :, 0])
    inter_y1 = torch.max(b1[:, None, 1], b2[None, :, 1])
    inter_x2 = torch.min(b1[:, None, 2], b2[None, :, 2])
    inter_y2 = torch.min(b1[:, None, 3], b2[None, :, 3])
    inter    = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    a1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    a2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    union = a1[:, None] + a2[None, :] - inter
    return inter / (union + 1e-7)


def ciou_loss(pred_xyxy, gt_xyxy):
    """
    Element-wise CIoU loss.
    pred_xyxy, gt_xyxy: (N, 4)
    """
    px1, py1, px2, py2 = pred_xyxy.unbind(-1)
    gx1, gy1, gx2, gy2 = gt_xyxy.unbind(-1)

    pw = (px2 - px1).clamp(min=0)
    ph = (py2 - py1).clamp(min=0)
    gw = (gx2 - gx1).clamp(min=0)
    gh = (gy2 - gy1).clamp(min=0)

    inter_x1 = torch.max(px1, gx1); inter_y1 = torch.max(py1, gy1)
    inter_x2 = torch.min(px2, gx2); inter_y2 = torch.min(py2, gy2)
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = pw * ph + gw * gh - inter + 1e-7
    iou   = inter / union

    # Enclosing box
    enclose_x1 = torch.min(px1, gx1); enclose_y1 = torch.min(py1, gy1)
    enclose_x2 = torch.max(px2, gx2); enclose_y2 = torch.max(py2, gy2)
    c2 = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + 1e-7

    # Centre distance
    pcx = (px1 + px2) / 2; pcy = (py1 + py2) / 2
    gcx = (gx1 + gx2) / 2; gcy = (gy1 + gy2) / 2
    rho2 = (pcx - gcx) ** 2 + (pcy - gcy) ** 2

    # Aspect ratio term
    v    = (4 / (torch.pi ** 2)) * (torch.atan(gw / (gh + 1e-7)) - torch.atan(pw / (ph + 1e-7))) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)

    return 1 - iou + rho2 / c2 + alpha * v


# ---------------------------------------------------------------------------
# Quality Focal Loss (QFL) — cls loss with soft target = iou quality
# ---------------------------------------------------------------------------

def quality_focal_loss(pred_logits, target_quality, beta=2.0):
    """
    pred_logits:    (N, num_classes)
    target_quality: (N, num_classes) — soft targets in [0,1]
    """
    pred_sigmoid = pred_logits.sigmoid()
    scale = (pred_sigmoid - target_quality).abs().pow(beta)
    loss  = F.binary_cross_entropy_with_logits(
        pred_logits, target_quality, reduction="none"
    )
    return (loss * scale).sum(-1)


# ---------------------------------------------------------------------------
# Distribution Focal Loss (DFL)
# ---------------------------------------------------------------------------

def distribution_focal_loss(pred_dist, target_dist):
    """
    pred_dist:   (N, 4*(reg_max+1))  — raw logits
    target_dist: (N, 4)              — continuous distances in [0, reg_max]
    """
    reg_max = REG_MAX
    N = pred_dist.shape[0]
    pred_dist = pred_dist.reshape(N, 4, reg_max + 1)  # (N,4,bins)

    # Clamp target to valid range [0, reg_max - epsilon] so floor < reg_max
    target_dist = target_dist.clamp(0, reg_max - 1e-3)
    tl = target_dist.long().clamp(0, reg_max - 1)      # floor bin  [0, reg_max-1]
    tr = (tl + 1).clamp(0, reg_max)                     # ceil  bin  [1, reg_max]
    wl = (tr.float() - target_dist).clamp(0, 1)         # weight for floor
    wr = 1.0 - wl                                        # weight for ceil

    loss = (
        F.cross_entropy(pred_dist.permute(0, 2, 1).reshape(-1, reg_max + 1),
                        tl.reshape(-1), reduction="none") * wl.reshape(-1) +
        F.cross_entropy(pred_dist.permute(0, 2, 1).reshape(-1, reg_max + 1),
                        tr.reshape(-1), reduction="none") * wr.reshape(-1)
    )
    return loss.reshape(N, 4).mean(-1)


# ---------------------------------------------------------------------------
# SimOTA label assignment
# ---------------------------------------------------------------------------

def simota_assign(cls_pred, decoded_boxes, gt_boxes, gt_cls, grid, stride):
    """
    cls_pred:     (N, num_classes) — sigmoid probabilities
    decoded_boxes:(N, 4) xyxy pixels
    gt_boxes:     (G, 4) xyxy pixels
    gt_cls:       (G,)   int class ids
    grid:         (N, 2) anchor centers
    Returns:
        fg_mask:   (N,) bool
        assigned_cls:  (num_fg,)
        assigned_iou:  (num_fg,)
        assigned_boxes:(num_fg, 4)
        assigned_dist: (num_fg, 4)  target distances in [0, reg_max]
    """
    N, G = len(decoded_boxes), len(gt_boxes)
    device = cls_pred.device

    # ---- cost matrix ----
    iou_matrix = box_iou(decoded_boxes, gt_boxes)  # (N, G)

    # cls cost: negative log of predicted prob for GT class
    cls_cost = -cls_pred[:, gt_cls].log().clamp(min=-100)  # (N, G)

    cost_matrix = cls_cost + 3.0 * (1 - iou_matrix)  # (N, G)

    # ---- filter candidates: inside GT box AND within center_radius ----
    # Two-stage gate: (1) anchor must be inside the GT box, AND
    # (2) anchor must be within CENTER_RADIUS strides of GT center.
    # Stage (2) is the key fix: large turkeys pass dozens of anchors through
    # the box filter alone, flooding the regression loss. Radius=2.5 strides
    # means only the 4-9 anchors closest to the GT center are candidates.
    gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2   # (G,)
    gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2   # (G,)
    cx, cy = grid[:, 0:1], grid[:, 1:2]              # (N,1)

    # Inside GT box
    gx1, gy1, gx2, gy2 = gt_boxes[:, 0], gt_boxes[:, 1], \
                          gt_boxes[:, 2], gt_boxes[:, 3]
    in_box = ((cx > gx1) & (cx < gx2) & (cy > gy1) & (cy < gy2))  # (N,G)

    # Within center radius (Chebyshev / L-inf distance in stride units)
    radius_px = CENTER_RADIUS * stride
    in_radius = ((cx - gt_cx).abs() < radius_px) & \
                ((cy - gt_cy).abs() < radius_px)              # (N,G)

    valid = in_box & in_radius
    cost_matrix = torch.where(valid, cost_matrix,
                              torch.full_like(cost_matrix, 1e9))

    # ---- per-GT dynamic k (top-k IoU sum rounded) ----
    topk_ious, _ = iou_matrix.topk(min(TOPK_CANDIDATES, N), dim=0)
    dynamic_k    = topk_ious.sum(0).int().clamp(min=1)  # (G,)

    matching = torch.zeros(N, G, dtype=torch.bool, device=device)
    for g in range(G):
        k = dynamic_k[g].item()
        _, indices = cost_matrix[:, g].topk(k, largest=False)
        matching[indices, g] = True

    # Resolve conflicts: each anchor → lowest-cost GT among its matched GTs
    # Replace unmatched entries with +inf so min() ignores them
    conflict_cost = torch.where(matching, cost_matrix,
                                torch.full_like(cost_matrix, float("inf")))
    _, matched_gt = conflict_cost.min(dim=1)   # (N,) best GT per anchor
    fg_mask = matching.any(dim=1)

    fg_gt_idx   = matched_gt[fg_mask]
    assigned_cls = gt_cls[fg_gt_idx]
    assigned_iou = iou_matrix[fg_mask][torch.arange(fg_mask.sum()), fg_gt_idx]
    assigned_boxes = gt_boxes[fg_gt_idx]

    # Compute target distances for DFL (l, t, r, b in stride units)
    fg_grid = grid[fg_mask]  # (num_fg, 2)
    l = (fg_grid[:, 0] - assigned_boxes[:, 0]) / stride
    t = (fg_grid[:, 1] - assigned_boxes[:, 1]) / stride
    r = (assigned_boxes[:, 2] - fg_grid[:, 0]) / stride
    b = (assigned_boxes[:, 3] - fg_grid[:, 1]) / stride
    assigned_dist = torch.stack([l, t, r, b], dim=-1).clamp(0, REG_MAX)

    return fg_mask, assigned_cls, assigned_iou, assigned_boxes, assigned_dist


# ---------------------------------------------------------------------------
# Main loss function
# ---------------------------------------------------------------------------

class DetectionLoss(nn.Module):
    def __init__(self, grid, stride=STRIDE):
        super().__init__()
        self.grid   = grid    # (N, 2) registered buffer from model
        self.stride = stride

    def forward(self, cls_pred, reg_pred, decoded_boxes, targets):
        """
        cls_pred:     (B, N, num_classes)
        reg_pred:     (B, N, 4*(reg_max+1))
        decoded_boxes:(B, N, 4)
        targets: list of (G_i, 5) tensors [cls, cx, cy, w, h] normalized
        """
        device = cls_pred.device
        B, N   = cls_pred.shape[:2]

        total_cls = cls_pred.sum() * 0.   # graph-connected zero
        total_reg = cls_pred.sum() * 0.
        total_dfl = cls_pred.sum() * 0.
        num_fg    = 0

        cls_sig = cls_pred.sigmoid()  # (B, N, C)

        for b in range(B):
            gt = targets[b].to(device)  # (G, 5)

            if len(gt) == 0:
                # No GTs: all background, cls loss only
                bg_target = torch.zeros(N, NUM_CLASSES, device=device)
                total_cls = total_cls + quality_focal_loss(
                    cls_pred[b], bg_target).sum()
                continue

            gt_cls = gt[:, 0].long()
            # Convert normalized cx,cy,w,h → pixel xyxy
            cx = gt[:, 1] * INPUT_SIZE; cy = gt[:, 2] * INPUT_SIZE
            gw = gt[:, 3] * INPUT_SIZE; gh = gt[:, 4] * INPUT_SIZE
            gt_xyxy = torch.stack([cx - gw/2, cy - gh/2,
                                   cx + gw/2, cy + gh/2], dim=-1)

            fg_mask, a_cls, a_iou, a_boxes, a_dist = simota_assign(
                cls_sig[b].detach(),
                decoded_boxes[b].detach(),
                gt_xyxy, gt_cls,
                self.grid, self.stride,
            )

            nfg = fg_mask.sum().item()
            num_fg += nfg

            # --- cls loss (QFL) ---
            cls_target = torch.zeros(N, NUM_CLASSES, device=device)
            if nfg > 0:
                cls_target[fg_mask, a_cls] = a_iou.float()
            total_cls = total_cls + quality_focal_loss(
                cls_pred[b], cls_target).sum()

            if nfg == 0:
                continue

            # --- reg loss (CIoU) ---
            pred_xyxy_fg = decoded_boxes[b][fg_mask]
            total_reg = total_reg + ciou_loss(pred_xyxy_fg, a_boxes).sum()

            # --- dfl loss ---
            total_dfl = total_dfl + distribution_focal_loss(
                reg_pred[b][fg_mask], a_dist).sum()

        normalizer = max(num_fg, 1)
        loss_cls   = LAMBDA_CLS * total_cls / normalizer
        loss_reg   = LAMBDA_REG * total_reg / normalizer
        loss_dfl   = LAMBDA_DFL * total_dfl / normalizer

        return loss_cls + loss_reg + loss_dfl, {
            "cls": loss_cls.item(),
            "reg": loss_reg.item(),
            "dfl": loss_dfl.item(),
            "num_fg": num_fg,
        }