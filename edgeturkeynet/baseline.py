"""
baseline.py â€” Ablation & Baseline Comparison Module for EdgeTurkeyNet.

Implements three independent experiments against the EdgeTurkeyNet design:

  A. MobileNetV3 + SSD
     Anchor-based SSD detection head attached to the same MobileNetV3
     backbone.  Directly measures the head design contribution (FCOS
     anchor-free vs. SSD anchor-based) while keeping the backbone identical.

  B. Loss Ablation  (Standard BCE + GIoU, flat centerness)
     Disables both novel loss components:
       - Oval-biased centerness target  â†’  standard isotropic centerness
       - CIoU regression loss           â†’  GIoU regression loss
     Same FCOS target assignment, same EdgeTurkeyNet architecture.
     Measures whether the custom loss terms improve turkey AP.

  C. DIoU-NMS vs IoU-NMS â€” Dense Subset Benchmark
     Evaluates both NMS variants on dense-overlap subsets, reporting
     precision / recall / F1 per variant and the retained-detection delta.

Design principles
-----------------
- Zero imports from config.py, logger.py, or train.py.
  All hyper-parameters are plain module-level globals (edit at the top).
- Reuses only low-level primitives from model.py and loss.py.
  No EdgeTurkeyNet, PANLiteNeck, AnchorFreeHead, EdgeTurkeyLoss, or
  ChannelPruner are imported.
- dataset.py and evaluate.py are reused as-is.
- Every experiment writes its own CSV + checkpoints to a flat output dir.
- run_all_baselines() is the single entry-point for a full ablation run.

Usage
-----
from baseline import (
    build_ssd_model,        # A: build MobileNetV3+SSD
    SSDTrainer,             # A: train it
    StandardFCOSLoss,       # B: ablated loss module
    StandardFCOSTrainer,    # B: train EdgeTurkeyNet with ablated loss
    DenseNMSBenchmark,      # C: benchmark both NMS variants
    iou_nms,                # C: standard IoU-NMS implementation
    run_all_baselines,      # run A+B+C and print full comparison table
)
"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# Low-level primitives only â€” no high-level architecture classes
from .model import (
    MobileNetV3Backbone,
    ConvBnAct,
    DepthwiseSeparableConv,
    NUM_CLASSES,
    CLASS_NAMES,
)
from .loss import (
    fcos_assign_targets,
    focal_loss,
    FCOS_SIZE_RANGES,
    STRIDES,
)
from .dataset import get_train_loader, get_val_loader, get_test_loader
from .evaluate import (
    PerClassMetrics,
    evaluate_map,
    diou_nms,
    _compute_iou,
)


# ===========================================================================
# SHARED CONFIGURATION GLOBALS
# Edit these variables to change any aspect of the baseline experiments.
# ===========================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE: Tuple[int, int] = (640, 640)
SEED = 42

# Training schedule (shared across Experiments A and B)
BL_EPOCHS              = 100
BL_BATCH_SIZE          = 8
BL_NUM_WORKERS         = 4
BL_BASE_LR             = 1e-3
BL_MIN_LR              = 1e-5
BL_WEIGHT_DECAY        = 5e-4
BL_WARMUP_EPOCHS       = 5
BL_GRADIENT_CLIP       = 10.0
BL_EARLY_STOP_PATIENCE = 20

# Loss weights (shared across A and B)
BL_LAMBDA_CLS = 1.0
BL_LAMBDA_REG = 2.0
BL_LAMBDA_CTR = 0.5

# Inference thresholds (shared across A, B, C)
BL_SCORE_THRESHOLD = 0.30
BL_IOU_THRESHOLD   = 0.50

# SSD-specific (Experiment A)
SSD_ASPECT_RATIOS  = [1.0, 2.0, 0.5, 3.0]  # anchor aspect ratios per cell
SSD_SCALES         = [0.1, 0.2, 0.37]       # anchor sizes relative to input per FPN level
SSD_POS_IOU_THRESH = 0.50   # anchor â†’ positive if max IoU with GT >= this
SSD_NEG_IOU_THRESH = 0.40   # anchor â†’ hard-negative eligible if max IoU < this
SSD_NEG_POS_RATIO  = 3      # hard-negative : positive ratio for mining

# NMS benchmark (Experiment C)
NMS_DENSE_OVERLAP_THRESH = 0.30   # minimum pairwise IoU for "dense" classification
NMS_BENCHMARK_N_SUBSETS  = 500    # number of subsets to evaluate
NMS_BENCHMARK_SUBSET_SIZE = 20    # maximum boxes per synthetic subset


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def _cosine_lr(epoch: int, epochs: int, base_lr: float,
               min_lr: float, warmup: int) -> float:
    """Linear warmup then cosine decay."""
    if epoch < warmup:
        return base_lr * (epoch + 1) / max(1, warmup)
    t = (epoch - warmup) / max(1, epochs - warmup)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + np.cos(np.pi * t))


# ===========================================================================
# EXPERIMENT A â€” MobileNetV3 + SSD
# ===========================================================================
#
# Architecture vs EdgeTurkeyNet:
#   SAME  backbone: MobileNetV3-Small (pretrained)
#   DIFF  neck:     lateral 1Ã—1 projection only â€” no PAN top-down/bottom-up path
#   DIFF  head:     anchor-based SSD (K anchors/cell, softmax cls, Smooth-L1 reg)
#   DIFF  loss:     multi-box loss with hard-negative mining (no focal, no CIoU)
#   DIFF  targets:  anchor IoU matching, not FCOS cell-centre assignment
# ===========================================================================

class SSDPredictionHead(nn.Module):
    """
    SSD per-level prediction head.

    For each spatial cell predicts K anchors, each with:
      - C+1 classification logits  (+1 for background, softmax convention)
      - 4 box offsets encoded as (Î”cx/wa, Î”cy/ha, log(w/wa), log(h/ha))

    Two DepthwiseSeparableConv layers refine features before the 1Ã—1
    prediction convolutions, matching the parameter budget of the FCOS head.

    Args:
        in_channels: Input feature map channels (PROJ_CHANNELS after lateral).
        num_anchors:  K â€” anchors per cell.
        num_classes:  Foreground class count (background appended internally).
    """

    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.num_anchors = num_anchors
        total_cls = num_classes + 1          # background at index 0

        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, in_channels),
            DepthwiseSeparableConv(in_channels, in_channels),
        )
        self.cls_pred = nn.Conv2d(in_channels, num_anchors * total_cls, 1)
        self.reg_pred = nn.Conv2d(in_channels, num_anchors * 4, 1)

        prior = 0.01
        nn.init.constant_(self.cls_pred.bias, -math.log((1 - prior) / prior))

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            cls_logits:  [B, H*W*K, C+1]
            box_offsets: [B, H*W*K, 4]
        """
        x   = self.conv(x)
        B, _, H, W = x.shape
        K   = self.num_anchors
        Cp1 = self.cls_pred.out_channels // K

        cls = self.cls_pred(x).reshape(B, K, Cp1, H * W)
        cls = cls.permute(0, 3, 1, 2).reshape(B, H * W * K, Cp1)

        reg = self.reg_pred(x).reshape(B, K, 4, H * W)
        reg = reg.permute(0, 3, 1, 2).reshape(B, H * W * K, 4)

        return cls, reg


def _generate_ssd_anchors(
    feature_shapes: List[Tuple[int, int]],
    input_size:     Tuple[int, int],
    strides:        List[int],
    aspect_ratios:  List[float],
    scales:         List[float],
) -> torch.Tensor:
    """
    Generate SSD anchor boxes for all FPN levels.

    For each cell on each level, produces len(aspect_ratios) anchors whose
    dimensions satisfy:
        w = scales[level] * sqrt(ar) * input_w
        h = scales[level] / sqrt(ar) * input_h

    Returns:
        [total_anchors, 4] normalised (cx, cy, w, h).
    """
    ih, iw = input_size
    all_anchors: List[torch.Tensor] = []

    for (fh, fw), stride, scale in zip(feature_shapes, strides, scales):
        xs = (torch.arange(fw, dtype=torch.float32) + 0.5) * stride / iw
        ys = (torch.arange(fh, dtype=torch.float32) + 0.5) * stride / ih
        yv, xv = torch.meshgrid(ys, xs, indexing='ij')

        for ar in aspect_ratios:
            aw = scale * math.sqrt(ar)
            ah = scale / math.sqrt(ar)
            cx = xv.reshape(-1)
            cy = yv.reshape(-1)
            all_anchors.append(torch.stack([
                cx,
                cy,
                torch.full_like(cx, aw),
                torch.full_like(cy, ah),
            ], dim=-1))

    return torch.cat(all_anchors, dim=0)    # [total, 4]


class MobileNetV3SSD(nn.Module):
    """
    MobileNetV3 + SSD Baseline (Experiment A).

    Attaches a classic SSD multi-scale prediction head to the MobileNetV3-Small
    backbone, providing a directly comparable anchor-based alternative to the
    FCOS anchor-free head in EdgeTurkeyNet.

    Neck design (intentionally simplified):
        Each backbone level is projected to PROJ_CHANNELS = 256 via a 1Ã—1
        ConvBnAct.  No top-down / bottom-up feature fusion is performed.
        This isolates the head architecture contribution by keeping the backbone
        and channel width identical while removing the PAN-Lite neck.

    Anchor specification:
        Levels:        P3 (stride 8), P4 (stride 16), P5 (stride 32)
        Aspect ratios: SSD_ASPECT_RATIOS  (default [1.0, 2.0, 0.5, 3.0])
        Scales:        SSD_SCALES per level (default [0.10, 0.20, 0.37])
        Anchors/cell:  4   (= len(aspect_ratios))

    Predictions per anchor:
        cls: C+1 logits  (softmax with background class at index 0)
        reg: 4 SSD-encoded offsets (Î”cx/wa, Î”cy/ha, log(w/wa), log(h/ha))

    Args:
        num_classes:    Foreground class count (default 2: body + neck).
        pretrained:     Load pretrained MobileNetV3-Small backbone weights.
        input_size:     Expected input resolution (H, W).
        aspect_ratios:  Per-cell anchor aspect ratios.
        scales:         Per-level anchor scale (relative to input size).
    """

    PROJ_CHANNELS = 256
    STRIDES       = [8, 16, 32]

    def __init__(
        self,
        num_classes:    int               = NUM_CLASSES,
        pretrained:     bool              = True,
        input_size:     Tuple[int, int]   = INPUT_SIZE,
        aspect_ratios:  List[float]       = None,
        scales:         List[float]       = None,
    ) -> None:
        super().__init__()
        self.num_classes   = num_classes
        self.input_size    = input_size
        self.aspect_ratios = aspect_ratios or SSD_ASPECT_RATIOS
        self.scales        = scales        or SSD_SCALES
        self.num_anchors   = len(self.aspect_ratios)

        self.backbone = MobileNetV3Backbone(pretrained=pretrained)
        in_chs = self.backbone.out_channels          # [24, 48, 576]

        # Lateral projections to common width
        self.lat_p3 = ConvBnAct(in_chs[0], self.PROJ_CHANNELS, 1)
        self.lat_p4 = ConvBnAct(in_chs[1], self.PROJ_CHANNELS, 1)
        self.lat_p5 = ConvBnAct(in_chs[2], self.PROJ_CHANNELS, 1)

        # Independent SSD prediction heads per level
        self.heads = nn.ModuleList([
            SSDPredictionHead(self.PROJ_CHANNELS, self.num_anchors, num_classes)
            for _ in self.STRIDES
        ])

        # Pre-compute anchor boxes once (not updated during training)
        h, w = input_size
        self._feature_shapes: List[Tuple[int, int]] = [
            (h // s, w // s) for s in self.STRIDES
        ]
        self._anchors: torch.Tensor = _generate_ssd_anchors(
            self._feature_shapes, input_size,
            self.STRIDES, self.aspect_ratios, self.scales,
        )  # [total_anchors, 4] normalised (cx, cy, w, h)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            cls_logits:  [B, total_anchors, num_classes+1]
            box_offsets: [B, total_anchors, 4]
        """
        p3, p4, p5 = self.backbone(x)
        f3 = self.lat_p3(p3)
        f4 = self.lat_p4(p4)
        f5 = self.lat_p5(p5)

        cls_list, reg_list = [], []
        for feat, head in zip([f3, f4, f5], self.heads):
            c, r = head(feat)
            cls_list.append(c)
            reg_list.append(r)

        return torch.cat(cls_list, dim=1), torch.cat(reg_list, dim=1)

    def decode_boxes(
        self,
        box_offsets: torch.Tensor,
        anchors:     Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode SSD offsets to (x1, y1, x2, y2) pixel coordinates.

        Inverse of the SSD encoding:
            pred_cx = Î”cx * a_w + a_cx
            pred_cy = Î”cy * a_h + a_cy
            pred_w  = exp(Î”w)  * a_w
            pred_h  = exp(Î”h)  * a_h

        Args:
            box_offsets: [B, A, 4]
            anchors:     [A, 4] normalised (cx, cy, w, h); defaults to self._anchors.

        Returns:
            [B, A, 4] clipped (x1, y1, x2, y2) pixel boxes.
        """
        if anchors is None:
            anchors = self._anchors
        anchors = anchors.to(box_offsets.device)
        ih, iw  = self.input_size

        a_cx = anchors[:, 0] * iw
        a_cy = anchors[:, 1] * ih
        a_w  = anchors[:, 2] * iw
        a_h  = anchors[:, 3] * ih

        pred_cx = box_offsets[..., 0] * a_w  + a_cx
        pred_cy = box_offsets[..., 1] * a_h  + a_cy
        pred_w  = torch.exp(box_offsets[..., 2].clamp(-6, 6)) * a_w
        pred_h  = torch.exp(box_offsets[..., 3].clamp(-6, 6)) * a_h

        x1 = (pred_cx - pred_w / 2).clamp(0, iw)
        y1 = (pred_cy - pred_h / 2).clamp(0, ih)
        x2 = (pred_cx + pred_w / 2).clamp(0, iw)
        y2 = (pred_cy + pred_h / 2).clamp(0, ih)

        return torch.stack([x1, y1, x2, y2], dim=-1)

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_ssd_model(
    pretrained:    bool            = True,
    input_size:    Tuple[int, int] = INPUT_SIZE,
    aspect_ratios: List[float]     = None,
    scales:        List[float]     = None,
) -> MobileNetV3SSD:
    """
    Build a MobileNetV3 + SSD baseline model ready for training.

    Args:
        pretrained:    Load pretrained MobileNetV3 weights.
        input_size:    Expected input resolution (H, W).
        aspect_ratios: Per-cell anchor aspect ratios (default: SSD_ASPECT_RATIOS).
        scales:        Per-level anchor scales (default: SSD_SCALES).

    Returns:
        MobileNetV3SSD instance.
    """
    return MobileNetV3SSD(
        num_classes   = NUM_CLASSES,
        pretrained    = pretrained,
        input_size    = input_size,
        aspect_ratios = aspect_ratios,
        scales        = scales,
    )


# ---------------------------------------------------------------------------
# SSD Loss
# ---------------------------------------------------------------------------

class SSDLoss(nn.Module):
    """
    SSD Multi-box Loss.

    Classification: cross-entropy over foreground + hard-mined negatives.
    Regression:     Smooth-L1 on positive anchors only.

    Anchor assignment per image:
        positive  â€” max IoU with any GT box >= SSD_POS_IOU_THRESH
        ignored   â€” IoU in [NEG_THRESH, POS_THRESH)
        negative  â€” max IoU < SSD_NEG_IOU_THRESH  (candidates for hard mining)

    Hard negative mining:
        Sort background losses descending, keep top k where
        k = SSD_NEG_POS_RATIO * num_positives.

    Args:
        num_classes:       Foreground class count.
        pos_iou_thresh:    Positive IoU threshold.
        neg_iou_thresh:    Hard-negative IoU ceiling.
        neg_pos_ratio:     Negative-to-positive sampling ratio.
        lambda_reg:        Smooth-L1 loss weight.
    """

    def __init__(
        self,
        num_classes:    int   = NUM_CLASSES,
        pos_iou_thresh: float = SSD_POS_IOU_THRESH,
        neg_iou_thresh: float = SSD_NEG_IOU_THRESH,
        neg_pos_ratio:  int   = SSD_NEG_POS_RATIO,
        lambda_reg:     float = BL_LAMBDA_REG,
    ) -> None:
        super().__init__()
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.neg_pos_ratio  = neg_pos_ratio
        self.lambda_reg     = lambda_reg

    @staticmethod
    def _encode_offsets(
        gt_boxes:   torch.Tensor,      # [P, 4] pixel (x1,y1,x2,y2)
        anchors:    torch.Tensor,      # [P, 4] normalised (cx,cy,w,h)
        input_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Encode GT boxes as SSD delta offsets relative to matched anchors."""
        ih, iw = input_size
        a_cx = anchors[:, 0] * iw
        a_cy = anchors[:, 1] * ih
        a_w  = (anchors[:, 2] * iw).clamp(1e-4)
        a_h  = (anchors[:, 3] * ih).clamp(1e-4)

        gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
        gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
        gt_w  = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(1e-4)
        gt_h  = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(1e-4)

        return torch.stack([
            (gt_cx - a_cx) / a_w,
            (gt_cy - a_cy) / a_h,
            torch.log(gt_w / a_w),
            torch.log(gt_h / a_h),
        ], dim=-1)

    def forward(
        self,
        cls_logits:         torch.Tensor,           # [B, A, C+1]
        box_offsets:        torch.Tensor,           # [B, A, 4]
        anchors:            torch.Tensor,           # [A, 4] normalised
        gt_boxes_batch:     List[torch.Tensor],     # List[B] of [M, 4] norm cx,cy,w,h
        gt_class_ids_batch: List[torch.Tensor],     # List[B] of [M] int64
        input_size:         Tuple[int, int]         = INPUT_SIZE,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with 'total', 'cls', 'reg' loss scalars.
        """
        device = cls_logits.device
        B = cls_logits.shape[0]
        ih, iw = input_size

        # Move both anchor representations to device once â€” no per-iteration transfers.
        anchors_dev = anchors.to(device)          # [A, 4] normalised (cx,cy,w,h)
        a_cx = anchors_dev[:, 0] * iw
        a_cy = anchors_dev[:, 1] * ih
        a_w  = anchors_dev[:, 2] * iw
        a_h  = anchors_dev[:, 3] * ih
        anc_xyxy = torch.stack(
            [a_cx - a_w/2, a_cy - a_h/2, a_cx + a_w/2, a_cy + a_h/2], dim=-1
        )                                         # [A, 4] pixel xyxy, already on device

        total_cls = torch.tensor(0.0, device=device)
        total_reg = torch.tensor(0.0, device=device)

        for b in range(B):
            gt_norm = gt_boxes_batch[b].to(device)
            gt_cls  = gt_class_ids_batch[b].to(device)

            if len(gt_norm) == 0:
                bg = torch.zeros(anc_xyxy.shape[0], dtype=torch.long, device=device)
                total_cls = total_cls + F.cross_entropy(cls_logits[b], bg)
                continue

            # Convert GT normalised cx,cy,w,h â†’ pixel x1,y1,x2,y2
            gt_x1 = (gt_norm[:, 0] - gt_norm[:, 2] / 2) * iw
            gt_y1 = (gt_norm[:, 1] - gt_norm[:, 3] / 2) * ih
            gt_x2 = (gt_norm[:, 0] + gt_norm[:, 2] / 2) * iw
            gt_y2 = (gt_norm[:, 1] + gt_norm[:, 3] / 2) * ih
            gt_xyxy = torch.stack([gt_x1, gt_y1, gt_x2, gt_y2], dim=-1)

            iou_mat = _compute_iou(anc_xyxy, gt_xyxy)           # [A, M]
            best_gt_iou, best_gt_idx = iou_mat.max(dim=1)       # [A]

            # Assignment: background=0, foreground=class_id+1
            cls_tgt   = torch.zeros(anc_xyxy.shape[0], dtype=torch.long, device=device)
            pos_mask  = best_gt_iou >= self.pos_iou_thresh
            if pos_mask.any():
                cls_tgt[pos_mask] = gt_cls[best_gt_idx[pos_mask]] + 1

            # Cross-entropy per anchor
            cls_loss_raw = F.cross_entropy(cls_logits[b], cls_tgt, reduction='none')

            n_pos = int(pos_mask.sum().item())
            if n_pos == 0:
                total_cls = total_cls + cls_loss_raw.mean()
                continue

            # Hard-negative mining
            neg_mask = best_gt_iou < self.neg_iou_thresh
            n_neg    = min(n_pos * self.neg_pos_ratio, int(neg_mask.sum().item()))

            if n_neg > 0:
                _, top_neg = cls_loss_raw[neg_mask].topk(n_neg)
                neg_idx    = neg_mask.nonzero(as_tuple=True)[0][top_neg]
                sel        = torch.cat([pos_mask.nonzero(as_tuple=True)[0], neg_idx])
                total_cls  = total_cls + cls_loss_raw[sel].mean()
            else:
                total_cls  = total_cls + cls_loss_raw[pos_mask].mean()

            # Smooth-L1 regression on positives
            # anchors_dev is already on device â€” direct boolean indexing, no transfer.
            matched_gt     = gt_xyxy[best_gt_idx[pos_mask]]
            matched_anc    = anchors_dev[pos_mask]
            offset_targets = self._encode_offsets(matched_gt, matched_anc, input_size)
            total_reg      = total_reg + F.smooth_l1_loss(
                box_offsets[b][pos_mask], offset_targets
            )

        cls_l = total_cls / max(1, B)
        reg_l = total_reg / max(1, B)
        return {
            "total": cls_l + self.lambda_reg * reg_l,
            "cls":   cls_l,
            "reg":   reg_l,
        }


# ---------------------------------------------------------------------------
# SSD inference helper  (returns same format as evaluate.predict)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _predict_ssd(
    model:           MobileNetV3SSD,
    images:          torch.Tensor,
    score_threshold: float = BL_SCORE_THRESHOLD,
    nms_threshold:   float = BL_IOU_THRESHOLD,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    SSD inference: decode boxes, per-class DIoU-NMS.

    Returns List[B] of (boxes [K,4], class_ids [K], scores [K]).
    Same format as evaluate.predict() for compatibility with
    _compute_per_class_metrics.
    """
    model.eval()
    cls_logits, box_offsets = model(images)
    boxes_all = model.decode_boxes(box_offsets)          # [B, A, 4]
    probs     = torch.softmax(cls_logits, dim=-1)
    fg_probs  = probs[..., 1:]                           # [B, A, C]

    B = images.shape[0]
    results = []

    for b in range(B):
        all_b, all_c, all_s = [], [], []
        for cls_id in range(NUM_CLASSES):
            sc   = fg_probs[b, :, cls_id]
            mask = sc > score_threshold
            if not mask.any():
                continue
            bx   = boxes_all[b][mask]
            s    = sc[mask]
            keep = diou_nms(bx, s, nms_threshold)
            if not len(keep):
                continue
            all_b.append(bx[keep])
            all_s.append(s[keep])
            all_c.append(torch.full((len(keep),), cls_id, dtype=torch.int64))

        if not all_b:
            results.append((
                torch.zeros((0, 4)), torch.zeros(0, dtype=torch.int64), torch.zeros(0),
            ))
        else:
            results.append((
                torch.cat(all_b).cpu(), torch.cat(all_c).cpu(), torch.cat(all_s).cpu(),
            ))

    return results


# ---------------------------------------------------------------------------
# SSD Trainer
# ---------------------------------------------------------------------------

class SSDTrainer:
    """
    Self-contained training loop for MobileNetV3SSD (Experiment A).

    No RunConfig or RunLogger dependencies.  All outputs go to output_dir.

    Args:
        output_dir: Where to write checkpoints and train_metrics.csv.
        pretrained: Load pretrained MobileNetV3 backbone.
        epochs:     Max training epochs.
        patience:   Early-stopping patience on val mAP.
    """

    def __init__(
        self,
        output_dir: Path = Path("baseline_runs/ssd"),
        pretrained: bool = True,
        epochs:     int  = BL_EPOCHS,
        patience:   int  = BL_EARLY_STOP_PATIENCE,
    ) -> None:
        _set_seed(SEED)
        self.epochs     = epochs
        self.patience   = patience
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = build_ssd_model(pretrained=pretrained).to(DEVICE)
        print(
            f"[SSD] MobileNetV3+SSD  params={self.model.get_parameter_count():,}  "
            f"anchors={len(self.model._anchors)}"
        )

        self.criterion    = SSDLoss()
        self.optimizer    = optim.AdamW(
            self.model.parameters(), lr=BL_BASE_LR, weight_decay=BL_WEIGHT_DECAY,
        )
        self.scaler       = GradScaler('cuda',enabled=torch.cuda.is_available())
        self.train_loader = get_train_loader(BL_BATCH_SIZE, BL_NUM_WORKERS)
        self.val_loader   = get_val_loader(BL_BATCH_SIZE,   BL_NUM_WORKERS)
        self.best_map     = 0.0
        self.no_improve   = 0

        self._csv = self.output_dir / "train_metrics.csv"
        with open(self._csv, "w") as f:
            f.write("epoch,lr,train_total,train_cls,train_reg,val_map,"
                    "val_ap_body,val_ap_neck\n")

    def _train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        self.model.train()
        lr = _cosine_lr(epoch, self.epochs, BL_BASE_LR, BL_MIN_LR, BL_WARMUP_EPOCHS)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        t_tot = t_cls = t_reg = 0.0
        n = 0
        anchors = self.model._anchors.to(DEVICE)

        for batch in self.train_loader:
            imgs  = batch["images"].to(DEVICE, non_blocking=True)
            boxes = batch["boxes"]
            cids  = batch["class_ids"]

            self.optimizer.zero_grad(set_to_none=True)
            with autocast('cuda',enabled=torch.cuda.is_available()):
                cls_l, reg_l = self.model(imgs)
                losses = self.criterion(cls_l, reg_l, anchors, boxes, cids)

            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), BL_GRADIENT_CLIP)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            t_tot += losses["total"].item()
            t_cls += losses["cls"].item()
            t_reg += losses["reg"].item()
            n     += 1

        return t_tot/max(1,n), t_cls/max(1,n), t_reg/max(1,n)

    def train(self) -> MobileNetV3SSD:
        """
        Run training loop.

        Returns:
            Best MobileNetV3SSD (eval mode, CPU).
        """
        best_ckpt = self.output_dir / "ssd_best.pth"
        last_ckpt = self.output_dir / "ssd_last.pth"
        print(f"\n{'='*60}\n  Experiment A â€” MobileNetV3 + SSD\n{'='*60}\n")

        for epoch in range(self.epochs):
            t_tot, t_cls, t_reg = self._train_epoch(epoch)
            lr = _cosine_lr(epoch, self.epochs, BL_BASE_LR, BL_MIN_LR, BL_WARMUP_EPOCHS)

            val = _evaluate_ssd(self.model, self.val_loader)
            print(
                f"  [SSD E{epoch:03d}]  total={t_tot:.4f}  cls={t_cls:.4f}  "
                f"reg={t_reg:.4f}  val_mAP={val.map:.4f}"
            )
            val.print_table(BL_IOU_THRESHOLD)

            with open(self._csv, "a") as f:
                f.write(f"{epoch},{lr:.6f},{t_tot:.6f},{t_cls:.6f},{t_reg:.6f},"
                        f"{val.map:.6f},{val.ap.get('body',0):.6f},"
                        f"{val.ap.get('neck',0):.6f}\n")

            torch.save({"epoch": epoch, "model_state": self.model.state_dict(),
                        "best_map": self.best_map}, last_ckpt)

            if val.map > self.best_map:
                self.best_map = val.map
                self.no_improve = 0
                torch.save({"epoch": epoch, "model_state": self.model.state_dict(),
                            "best_map": self.best_map}, best_ckpt)
                print(f"  âœ“ New best SSD mAP: {self.best_map:.4f}")
            else:
                self.no_improve += 1
                if self.no_improve >= self.patience:
                    print(f"[SSD] Early stopping at epoch {epoch}.")
                    break

        ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(DEVICE).eval()
        print(f"[SSD] Done.  Best mAP = {self.best_map:.4f}")
        return self.model


def _evaluate_ssd(
    model:           MobileNetV3SSD,
    loader:          DataLoader,
    score_threshold: float = BL_SCORE_THRESHOLD,
    iou_threshold:   float = BL_IOU_THRESHOLD,
) -> PerClassMetrics:
    """
    Evaluate MobileNetV3SSD on a DataLoader.

    Runs _predict_ssd per batch and accumulates predictions for
    _compute_per_class_metrics.

    Returns:
        PerClassMetrics with per-class AP, precision, recall, mAP.
    """
    ih, iw = INPUT_SIZE
    pred_b_all, pred_s_all, pred_c_all = [], [], []
    gt_b_all,   gt_c_all               = [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            imgs  = batch["images"].to(DEVICE, non_blocking=True)
            preds = _predict_ssd(model, imgs, score_threshold, iou_threshold)

            for pb, pc, ps in preds:
                pred_b_all.append(pb)
                pred_c_all.append(pc)
                pred_s_all.append(ps)

            for bi in range(len(batch["images"])):
                gt_n = batch["boxes"][bi]
                gt_c = batch["class_ids"][bi]
                if len(gt_n):
                    x1 = (gt_n[:, 0] - gt_n[:, 2] / 2) * iw
                    y1 = (gt_n[:, 1] - gt_n[:, 3] / 2) * ih
                    x2 = (gt_n[:, 0] + gt_n[:, 2] / 2) * iw
                    y2 = (gt_n[:, 1] + gt_n[:, 3] / 2) * ih
                    gt_n = torch.stack([x1, y1, x2, y2], dim=-1)
                gt_b_all.append(gt_n)
                gt_c_all.append(gt_c)

    return _compute_per_class_metrics(
        pred_b_all, pred_s_all, pred_c_all,
        gt_b_all,   gt_c_all,
        iou_threshold=iou_threshold,
    )


# ===========================================================================
# EXPERIMENT B â€” Standard BCE + GIoU Loss Ablation
# ===========================================================================
#
# Disables both custom loss components from EdgeTurkeyLoss:
#
#   1. CIoU â†’ GIoU
#      CIoU = IoU - ÏÂ²/cÂ² - Î±Â·v   includes aspect-ratio penalty v
#      GIoU = IoU - |C\(AâˆªB)|/|C| penalises enclosing waste only, no AR term
#      Hypothesis: without v, oval body boxes are localised less tightly.
#
#   2. Oval-biased centerness target â†’ standard isotropic centerness target
#      EdgeTurkeyNet applies an aspect-ratio prior in the centerness formula
#      so vertically elongated neck boxes do not get penalised.
#      Ablation: standard sqrt(min_lr/max_lr * min_tb/max_tb) with no prior.
#      Hypothesis: neck AP drops more than body AP.
#
# Architecture: unchanged (EdgeTurkeyNet with MobileNetV3 backbone).
# ===========================================================================

def _giou_loss(
    pred_boxes:   torch.Tensor,
    target_boxes: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Generalised Intersection over Union (GIoU) loss.

    GIoU = IoU - (|C minus (A union B)| / |C|)
    where C is the smallest enclosing box.

    Differs from CIoU: no centre-distance term (ÏÂ²/cÂ²) and no
    aspect-ratio consistency term (Î±Â·v).

    Args:
        pred_boxes:   [N, 4] (x1, y1, x2, y2).
        target_boxes: [N, 4] (x1, y1, x2, y2).

    Returns:
        Scalar mean GIoU loss.
    """
    # Intersection
    ix1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    iy1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    ix2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    iy2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)

    pw = (pred_boxes[:, 2]   - pred_boxes[:, 0]).clamp(0)
    ph = (pred_boxes[:, 3]   - pred_boxes[:, 1]).clamp(0)
    tw = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(0)
    th = (target_boxes[:, 3] - target_boxes[:, 1]).clamp(0)

    union = pw * ph + tw * th - inter + eps
    iou   = inter / union

    # Enclosing box
    ex1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    ey1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    ex2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    ey2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    enc = (ex2 - ex1).clamp(0) * (ey2 - ey1).clamp(0) + eps

    giou = iou - (enc - union) / enc
    return (1 - giou).mean()


def _std_centerness(
    l: torch.Tensor, t: torch.Tensor,
    r: torch.Tensor, b: torch.Tensor,
) -> torch.Tensor:
    """
    Standard isotropic centerness target.

    FCOS formula (no aspect-ratio prior):
        ctr = sqrt( min(l,r)/max(l,r) * min(t,b)/max(t,b) )

    EdgeTurkeyNet applies a 1.2:1 aspect-ratio prior in the denominator
    to prevent the vertically elongated neck class from scoring near zero.
    This function deliberately omits that correction, matching the ablation.
    """
    min_lr = torch.min(l, r)
    max_lr = torch.max(l, r).clamp(1e-6)
    min_tb = torch.min(t, b)
    max_tb = torch.max(t, b).clamp(1e-6)
    return torch.sqrt((min_lr / max_lr) * (min_tb / max_tb))


class StandardFCOSLoss(nn.Module):
    """
    Ablated FCOS loss: GIoU regression + standard isotropic centerness.

    Drop-in replacement for EdgeTurkeyLoss â€” identical forward() signature.
    Removes both novel loss components:
      - CIoU  â†’  GIoU  (drops aspect-ratio consistency term v)
      - Oval-biased centerness target  â†’  standard sqrt formula

    Classification (per-channel sigmoid focal) and centerness BCE remain
    unchanged.  FCOS target assignment is identical.

    Args:
        num_classes:  Detection class count (default 2).
        lambda_cls:   Focal loss weight.
        lambda_reg:   GIoU loss weight.
        lambda_ctr:   Centerness BCE weight.
        input_size:   Model input (H, W).
    """

    def __init__(
        self,
        num_classes: int   = NUM_CLASSES,
        lambda_cls:  float = BL_LAMBDA_CLS,
        lambda_reg:  float = BL_LAMBDA_REG,
        lambda_ctr:  float = BL_LAMBDA_CTR,
        input_size:  Tuple[int, int] = INPUT_SIZE,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.lambda_cls  = lambda_cls
        self.lambda_reg  = lambda_reg
        self.lambda_ctr  = lambda_ctr
        self.input_size  = input_size

    def forward(
        self,
        cls_preds:          List[torch.Tensor],
        reg_preds:          List[torch.Tensor],
        ctr_preds:          List[torch.Tensor],
        gt_boxes_batch:     List[torch.Tensor],
        gt_class_ids_batch: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute ablated FCOS loss.

        Identical signature to EdgeTurkeyLoss.forward() â€” can be substituted
        in any trainer that accepts a loss module with this interface.

        Returns:
            Dict with 'total', 'cls', 'reg', 'ctr' scalar losses.
        """
        device = cls_preds[0].device
        feature_shapes = [(p.shape[2], p.shape[3]) for p in cls_preds]
        gt_boxes_dev   = [b.to(device) for b in gt_boxes_batch]
        gt_cls_dev     = [c.to(device) for c in gt_class_ids_batch]

        # Standard FCOS target assignment (oval-biased centerness is in the
        # *targets*, not the architecture; we override them below per-positive)
        cls_tgts, reg_tgts, _ = fcos_assign_targets(
            gt_boxes_dev, gt_cls_dev, feature_shapes,
            num_classes=self.num_classes,
            strides=STRIDES,
            size_ranges=FCOS_SIZE_RANGES,
            input_size=self.input_size,
        )

        total_cls = torch.tensor(0.0, device=device)
        total_reg = torch.tensor(0.0, device=device)
        total_ctr = torch.tensor(0.0, device=device)

        for li in range(len(cls_preds)):
            B, C, H, W = cls_preds[li].shape
            N = H * W

            cls_flat = cls_preds[li].reshape(B, C, N).permute(0, 2, 1).reshape(-1, C)
            reg_flat = reg_preds[li].reshape(B, 4, N).permute(0, 2, 1).reshape(-1, 4)
            ctr_flat = ctr_preds[li].reshape(B, 1, N).permute(0, 2, 1).reshape(-1)

            cls_tgt  = cls_tgts[li].reshape(-1, C)
            reg_tgt  = reg_tgts[li].reshape(-1, 4)

            # â”€â”€ Classification â€” focal loss (unchanged) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            total_cls = total_cls + focal_loss(cls_flat, cls_tgt)

            pos_mask = cls_tgt.sum(dim=-1) > 0.5
            if pos_mask.any():
                rp = reg_flat[pos_mask]
                rt = reg_tgt[pos_mask]

                # â”€â”€ Regression: GIoU instead of CIoU â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                pred_boxes = torch.stack([-rp[:,0], -rp[:,1],  rp[:,2],  rp[:,3]], dim=-1)
                tgt_boxes  = torch.stack([-rt[:,0], -rt[:,1],  rt[:,2],  rt[:,3]], dim=-1)
                total_reg  = total_reg + _giou_loss(pred_boxes, tgt_boxes)

                # â”€â”€ Centerness: standard (override oval-biased targets) â”€
                ctr_tgt_std = _std_centerness(rt[:,0], rt[:,1], rt[:,2], rt[:,3])
                total_ctr   = total_ctr + F.binary_cross_entropy_with_logits(
                    ctr_flat[pos_mask], ctr_tgt_std
                )

        n = len(cls_preds)
        cls_l  = total_cls / n
        reg_l  = total_reg / max(1, n)
        ctr_l  = total_ctr / max(1, n)
        return {
            "total": self.lambda_cls * cls_l + self.lambda_reg * reg_l + self.lambda_ctr * ctr_l,
            "cls":   cls_l, "reg": reg_l, "ctr": ctr_l,
        }


class StandardFCOSTrainer:
    """
    Training loop for Experiment B â€” EdgeTurkeyNet with ablated loss.

    Architecture is identical to the main pipeline (MobileNetV3 backbone,
    PAN-Lite neck, FCOS head).  Only the loss module is replaced.

    Args:
        output_dir:  Where to write checkpoints and train_metrics.csv.
        pretrained:  Load pretrained MobileNetV3 backbone.
        epochs:      Max training epochs.
        patience:    Early-stopping patience on val mAP.
    """

    def __init__(
        self,
        output_dir: Path = Path("baseline_runs/standard_loss"),
        pretrained: bool = True,
        epochs:     int  = BL_EPOCHS,
        patience:   int  = BL_EARLY_STOP_PATIENCE,
    ) -> None:
        _set_seed(SEED)
        self.epochs     = epochs
        self.patience   = patience
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        from model import EdgeTurkeyNet
        self.model = EdgeTurkeyNet(
            num_classes=NUM_CLASSES,
            pretrained_backbone=pretrained,
            input_size=INPUT_SIZE,
            backbone="mobilenetv3",
        ).to(DEVICE)
        print(
            f"[StdLoss] EdgeTurkeyNet(MobileNetV3) + GIoU + std-centerness  "
            f"params={self.model.get_parameter_count():,}"
        )

        # The only structural change vs the main pipeline
        self.criterion = StandardFCOSLoss(
            num_classes=NUM_CLASSES,
            lambda_cls=BL_LAMBDA_CLS,
            lambda_reg=BL_LAMBDA_REG,
            lambda_ctr=BL_LAMBDA_CTR,
            input_size=INPUT_SIZE,
        )
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=BL_BASE_LR, weight_decay=BL_WEIGHT_DECAY,
        )
        self.scaler       = GradScaler('cuda',enabled=torch.cuda.is_available())
        self.train_loader = get_train_loader(BL_BATCH_SIZE, BL_NUM_WORKERS)
        self.val_loader   = get_val_loader(BL_BATCH_SIZE,   BL_NUM_WORKERS)
        self.best_map     = 0.0
        self.no_improve   = 0

        self._csv = self.output_dir / "train_metrics.csv"
        with open(self._csv, "w") as f:
            f.write("epoch,lr,train_total,train_cls,train_reg,train_ctr,"
                    "val_map,val_ap_body,val_ap_neck\n")

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        lr = _cosine_lr(epoch, self.epochs, BL_BASE_LR, BL_MIN_LR, BL_WARMUP_EPOCHS)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        tots = {"total": 0.0, "cls": 0.0, "reg": 0.0, "ctr": 0.0}
        n = 0

        for batch in self.train_loader:
            imgs  = batch["images"].to(DEVICE, non_blocking=True)
            boxes = batch["boxes"]
            cids  = batch["class_ids"]

            self.optimizer.zero_grad(set_to_none=True)
            with autocast('cuda',enabled=torch.cuda.is_available()):
                c, r, ctr = self.model(imgs)
                losses    = self.criterion(c, r, ctr, boxes, cids)

            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), BL_GRADIENT_CLIP)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for k in tots:
                tots[k] += losses[k].item()
            n += 1

        return {k: v/max(1,n) for k,v in tots.items()}

    def train(self):
        """
        Run the ablation training loop.

        Returns:
            Best EdgeTurkeyNet (eval mode, CPU).
        """
        from evaluate import evaluate_map
        best_ckpt = self.output_dir / "stdloss_best.pth"
        last_ckpt = self.output_dir / "stdloss_last.pth"
        print(f"\n{'='*60}\n  Experiment B â€” Standard BCE + GIoU Loss Ablation\n{'='*60}\n")

        for epoch in range(self.epochs):
            tl = self._train_epoch(epoch)
            lr = _cosine_lr(epoch, self.epochs, BL_BASE_LR, BL_MIN_LR, BL_WARMUP_EPOCHS)

            val = evaluate_map(
                self.model, self.val_loader, DEVICE,
                iou_threshold=BL_IOU_THRESHOLD,
                score_threshold=BL_SCORE_THRESHOLD,
            )
            print(
                f"  [Std E{epoch:03d}]  total={tl['total']:.4f}  "
                f"cls={tl['cls']:.4f}  reg={tl['reg']:.4f}  "
                f"ctr={tl['ctr']:.4f}  val_mAP={val.map:.4f}"
            )
            val.print_table(BL_IOU_THRESHOLD)

            with open(self._csv, "a") as f:
                f.write(
                    f"{epoch},{lr:.6f},{tl['total']:.6f},{tl['cls']:.6f},"
                    f"{tl['reg']:.6f},{tl['ctr']:.6f},{val.map:.6f},"
                    f"{val.ap.get('body',0):.6f},{val.ap.get('neck',0):.6f}\n"
                )

            torch.save({"epoch": epoch, "model_state": self.model.state_dict(),
                        "best_map": self.best_map}, last_ckpt)

            if val.map > self.best_map:
                self.best_map   = val.map
                self.no_improve = 0
                torch.save({"epoch": epoch, "model_state": self.model.state_dict(),
                            "best_map": self.best_map}, best_ckpt)
                print(f"  âœ“ New best mAP (standard loss): {self.best_map:.4f}")
            else:
                self.no_improve += 1
                if self.no_improve >= self.patience:
                    print(f"[StdLoss] Early stopping at epoch {epoch}.")
                    break

        ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(DEVICE).eval()
        print(f"[StdLoss] Done.  Best mAP = {self.best_map:.4f}")
        return self.model


# ===========================================================================
# EXPERIMENT C â€” DIoU-NMS vs IoU-NMS  Dense Subset Benchmark
# ===========================================================================

def iou_nms(
    boxes:         torch.Tensor,
    scores:        torch.Tensor,
    iou_threshold: float = BL_IOU_THRESHOLD,
    max_dets:      int   = 300,
) -> torch.Tensor:
    """
    Standard IoU-based Non-Maximum Suppression.

    Thin wrapper around torchvision.ops.nms, which is a fused CUDA kernel
    written in C++.  Replaces the previous pure-Python i/j nested loop that
    issued one CUDA kernel launch per scalar comparison.

    torchvision.ops.nms runs in a single kernel call regardless of N,
    reducing per-image NMS time from ~100 ms (Python loop, GPU tensors)
    to < 1 ms on a 4090.

    Suppresses box j whenever IoU(i, j) > iou_threshold for any higher-
    scoring box i.  Does not account for centre separation â€” adjacent turkey
    boxes with significant area overlap will be suppressed even when their
    centres are clearly distinct, which is the expected failure mode in dense
    flock scenarios and the basis for the Experiment C comparison.

    Args:
        boxes:         [N, 4] (x1, y1, x2, y2).
        scores:        [N] confidence scores.
        iou_threshold: Suppress box j when IoU(i, j) > threshold.
        max_dets:      Cap on candidates before calling NMS (score-sorted).

    Returns:
        keep: [K] indices into the original (pre-sort) boxes tensor.
    """
    import torchvision
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long)

    order = scores.argsort(descending=True)[:max_dets]
    keep_in_order = torchvision.ops.nms(
        boxes[order].float(),
        scores[order].float(),
        iou_threshold,
    )
    return order[keep_in_order]


@dataclass
class NMSBenchmarkResult:
    """
    Per-metric results from DenseNMSBenchmark.

    Attributes:
        n_subsets:           Dense subsets evaluated.
        diou_avg_kept:       Mean detections kept by DIoU-NMS.
        iou_avg_kept:        Mean detections kept by IoU-NMS.
        diou_avg_precision:  Mean precision of DIoU-NMS output vs GT.
        iou_avg_precision:   Mean precision of IoU-NMS output vs GT.
        diou_avg_recall:     Mean recall of DIoU-NMS output vs GT.
        iou_avg_recall:      Mean recall of IoU-NMS output vs GT.
        diou_avg_f1:         Mean F1 of DIoU-NMS.
        iou_avg_f1:          Mean F1 of IoU-NMS.
        delta_kept:          diou_avg_kept - iou_avg_kept.
        delta_f1:            diou_avg_f1  - iou_avg_f1.
    """

    n_subsets:          int   = 0
    diou_avg_kept:      float = 0.0
    iou_avg_kept:       float = 0.0
    diou_avg_precision: float = 0.0
    iou_avg_precision:  float = 0.0
    diou_avg_recall:    float = 0.0
    iou_avg_recall:     float = 0.0
    diou_avg_f1:        float = 0.0
    iou_avg_f1:         float = 0.0
    delta_kept:         float = 0.0
    delta_f1:           float = 0.0

    def print_report(self) -> None:
        sep = "-" * 54
        print(f"\n{sep}")
        print(f"  Experiment C â€” DIoU-NMS vs IoU-NMS (n={self.n_subsets} dense subsets)")
        print(sep)
        print(f"  {'Metric':<26} {'DIoU-NMS':>10} {'IoU-NMS':>10} {'Delta':>8}")
        print(sep)
        for name, d, i in [
            ("Avg detections kept", self.diou_avg_kept,      self.iou_avg_kept),
            ("Avg precision",       self.diou_avg_precision,  self.iou_avg_precision),
            ("Avg recall",          self.diou_avg_recall,     self.iou_avg_recall),
            ("Avg F1",              self.diou_avg_f1,         self.iou_avg_f1),
        ]:
            delta = d - i
            sign = "+" if delta >= 0 else ""
            print(f"  {name:<26} {d:>10.4f} {i:>10.4f} {sign}{delta:>7.4f}")
        print(sep)
        verdict = "DIoU-NMS BETTER" if self.delta_f1 > 0 else "IoU-NMS BETTER"
        print(f"  Verdict: {verdict} on dense subsets (Î”F1={self.delta_f1:+.4f})")
        print(f"{sep}\n")


class DenseNMSBenchmark:
    """
    Benchmarks DIoU-NMS vs IoU-NMS on dense-overlap detection subsets.

    A subset is "dense" if at least one pair of boxes has IoU
    >= NMS_DENSE_OVERLAP_THRESH.  This mimics the tight-flock scenario
    where adjacent turkeys partially overlap in the camera view.

    Two operating modes:

    Model mode (model + test_loader provided):
        Runs the model on the test set in no-grad mode.
        Collects all per-image pre-NMS candidates (score > 0.5 * threshold)
        and retains only images where the candidate boxes are dense.
        Up to n_subsets images are collected.

    Synthetic mode (no model):
        Generates n_subsets random box collections with a controlled dense
        cluster (3-5 overlapping boxes) plus random background boxes.
        Ground-truth = one box centred on the cluster centroid.
        Useful for unit-testing the benchmark without a trained model.

    In both modes, both NMS variants receive the same candidate boxes and
    scores; their outputs are evaluated against the GT boxes.

    Args:
        iou_threshold:        NMS suppression threshold (same for both).
        dense_overlap_thresh: Minimum pairwise IoU to classify subset as dense.
        n_subsets:            Target number of dense subsets.
        subset_size:          Maximum boxes per synthetic subset.
    """

    def __init__(
        self,
        iou_threshold:        float = BL_IOU_THRESHOLD,
        dense_overlap_thresh: float = NMS_DENSE_OVERLAP_THRESH,
        n_subsets:            int   = NMS_BENCHMARK_N_SUBSETS,
        subset_size:          int   = NMS_BENCHMARK_SUBSET_SIZE,
    ) -> None:
        self.iou_threshold        = iou_threshold
        self.dense_overlap_thresh = dense_overlap_thresh
        self.n_subsets            = n_subsets
        self.subset_size          = subset_size

    def _is_dense(self, boxes: torch.Tensor) -> bool:
        if len(boxes) < 2:
            return False
        m = _compute_iou(boxes, boxes)
        m.fill_diagonal_(0.0)
        return bool((m >= self.dense_overlap_thresh).any().item())

    @staticmethod
    def _pr(
        kept: torch.Tensor,   # [K, 4]
        gt:   torch.Tensor,   # [G, 4]
        thr:  float,
    ) -> Tuple[float, float]:
        """Compute (precision, recall) of kept boxes vs GT at IoU thr."""
        if len(gt) == 0:
            return (1.0, 1.0) if len(kept) == 0 else (0.0, 1.0)
        if len(kept) == 0:
            return (1.0, 0.0)
        mat = _compute_iou(kept, gt)     # [K, G]
        matched = set()
        tp = 0
        for k in range(len(kept)):
            best_v, best_g = mat[k].max(dim=0)
            g = best_g.item()
            if best_v.item() >= thr and g not in matched:
                tp += 1
                matched.add(g)
        return tp / len(kept), tp / len(gt)

    def _synthetic_subsets(
        self,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Generate n_subsets synthetic dense box collections."""
        _set_seed(SEED)
        ih, iw = INPUT_SIZE
        out = []
        for _ in range(self.n_subsets):
            boxes, scores, gt = [], [], []
            n_cl = random.randint(3, 5)
            cx0  = random.uniform(0.3, 0.7) * iw
            cy0  = random.uniform(0.3, 0.7) * ih
            bw   = random.uniform(40, 80)
            bh   = random.uniform(40, 80)
            for _ in range(n_cl):
                cx = cx0 + random.gauss(0, bw * 0.3)
                cy = cy0 + random.gauss(0, bh * 0.3)
                w  = bw * random.uniform(0.8, 1.2)
                h  = bh * random.uniform(0.8, 1.2)
                boxes.append([cx-w/2, cy-h/2, cx+w/2, cy+h/2])
                scores.append(random.uniform(0.5, 1.0))
            gt.append([cx0-bw/2, cy0-bh/2, cx0+bw/2, cy0+bh/2])
            for _ in range(self.subset_size - n_cl):
                cx = random.uniform(0.1, 0.9) * iw
                cy = random.uniform(0.1, 0.9) * ih
                w  = random.uniform(30, 120)
                h  = random.uniform(30, 120)
                boxes.append([cx-w/2, cy-h/2, cx+w/2, cy+h/2])
                scores.append(random.uniform(0.3, 0.7))
            out.append((
                torch.tensor(boxes,  dtype=torch.float32),
                torch.tensor(scores, dtype=torch.float32),
                torch.tensor(gt,     dtype=torch.float32),
            ))
        return out

    def _model_subsets(
        self, model, loader: DataLoader,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Collect dense pre-NMS subsets from real model predictions."""
        ih, iw = INPUT_SIZE
        out = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                if len(out) >= self.n_subsets:
                    break
                imgs = batch["images"].to(DEVICE, non_blocking=True)

                if isinstance(model, MobileNetV3SSD):
                    cls_l, reg_l  = model(imgs)
                    boxes_all     = model.decode_boxes(reg_l)
                    probs         = torch.softmax(cls_l, dim=-1)
                    scores_all    = probs[..., 1:].max(dim=-1).values
                else:
                    cls_p, reg_p, ctr_p = model(imgs)
                    boxes_all, sc_cls, ctr_sc = model.decode_predictions(
                        cls_p, reg_p, ctr_p
                    )
                    scores_all = (sc_cls * ctr_sc).max(dim=-1).values

                for bi in range(len(imgs)):
                    if len(out) >= self.n_subsets:
                        break
                    bx = boxes_all[bi]
                    sc = scores_all[bi]
                    mask = sc > BL_SCORE_THRESHOLD * 0.5
                    cb, cs = bx[mask], sc[mask]
                    if not self._is_dense(cb):
                        continue
                    gt_n = batch["boxes"][bi]
                    if len(gt_n):
                        x1 = (gt_n[:, 0] - gt_n[:, 2]/2) * iw
                        y1 = (gt_n[:, 1] - gt_n[:, 3]/2) * ih
                        x2 = (gt_n[:, 0] + gt_n[:, 2]/2) * iw
                        y2 = (gt_n[:, 1] + gt_n[:, 3]/2) * ih
                        gt_xy = torch.stack([x1, y1, x2, y2], dim=-1)
                    else:
                        gt_xy = torch.zeros((0, 4))
                    out.append((cb.cpu(), cs.cpu(), gt_xy.cpu()))
        return out

    def run(
        self,
        model=None,
        test_loader: Optional[DataLoader] = None,
    ) -> NMSBenchmarkResult:
        """
        Run the benchmark.

        Args:
            model:       Optional trained detection model for real-data mode.
            test_loader: Optional test DataLoader (required with model).

        Returns:
            NMSBenchmarkResult with all comparison metrics.
        """
        print(f"\n{'='*60}\n  Experiment C â€” DIoU-NMS vs IoU-NMS\n{'='*60}")

        if model is not None and test_loader is not None:
            print("  Collecting dense subsets from real model predictions â€¦")
            subsets = self._model_subsets(model, test_loader)
        else:
            print("  Using synthetic dense subsets â€¦")
            subsets = self._synthetic_subsets()

        if not subsets:
            print("  WARNING: zero dense subsets found.")
            return NMSBenchmarkResult()

        dk, ik = [], []
        dp, ip = [], []
        dr, ir = [], []

        for boxes, scores, gt in subsets:
            d_keep = diou_nms(boxes, scores, self.iou_threshold)
            i_keep = iou_nms( boxes, scores, self.iou_threshold)

            dk.append(len(d_keep))
            ik.append(len(i_keep))

            d_prec, d_rec = self._pr(boxes[d_keep], gt, self.iou_threshold)
            i_prec, i_rec = self._pr(boxes[i_keep], gt, self.iou_threshold)

            dp.append(d_prec); ip.append(i_prec)
            dr.append(d_rec);  ir.append(i_rec)

        def _m(lst): return sum(lst)/len(lst) if lst else 0.0
        def _f1(p,r): return 2*p*r/(p+r) if p+r>0 else 0.0

        d_p, d_r = _m(dp), _m(dr)
        i_p, i_r = _m(ip), _m(ir)
        d_f1 = _f1(d_p, d_r)
        i_f1 = _f1(i_p, i_r)
        d_avg = _m(dk)
        i_avg = _m(ik)

        result = NMSBenchmarkResult(
            n_subsets=len(subsets),
            diou_avg_kept=d_avg,      iou_avg_kept=i_avg,
            diou_avg_precision=d_p,   iou_avg_precision=i_p,
            diou_avg_recall=d_r,      iou_avg_recall=i_r,
            diou_avg_f1=d_f1,         iou_avg_f1=i_f1,
            delta_kept=d_avg-i_avg,   delta_f1=d_f1-i_f1,
        )
        result.print_report()
        return result


# ===========================================================================
# SHARED EVALUATION HELPER
# ===========================================================================

def _compute_per_class_metrics(
    pred_boxes_list:  List[torch.Tensor],
    pred_scores_list: List[torch.Tensor],
    pred_cls_list:    List[torch.Tensor],
    gt_boxes_list:    List[torch.Tensor],
    gt_cls_list:      List[torch.Tensor],
    iou_threshold:    float = BL_IOU_THRESHOLD,
) -> PerClassMetrics:
    """
    Compute per-class AP, precision, recall, mAP from accumulated predictions.

    Uses the same 11-point interpolated AP as evaluate.py.  Standalone
    implementation so models with non-standard predict() interfaces can
    use it without coupling to evaluate_map().

    Matching: greedy, score-sorted, single assignment per GT box.

    Args:
        pred_boxes_list:  List[images] of [K, 4] (x1,y1,x2,y2).
        pred_scores_list: List[images] of [K] confidence.
        pred_cls_list:    List[images] of [K] int64 class ids.
        gt_boxes_list:    List[images] of [G, 4] (x1,y1,x2,y2).
        gt_cls_list:      List[images] of [G] int64 class ids.
        iou_threshold:    TP matching IoU threshold.

    Returns:
        PerClassMetrics.
    """
    per_scores: Dict[int, List[float]] = {c: [] for c in range(NUM_CLASSES)}
    per_tp:     Dict[int, List[int]]   = {c: [] for c in range(NUM_CLASSES)}
    per_n_gt:   Dict[int, int]         = {c: 0  for c in range(NUM_CLASSES)}

    for pb, ps, pc, gb, gc in zip(
        pred_boxes_list, pred_scores_list, pred_cls_list,
        gt_boxes_list,   gt_cls_list,
    ):
        for cls_id in range(NUM_CLASSES):
            gt_mask = (gc == cls_id)
            per_n_gt[cls_id] += int(gt_mask.sum().item())
            gt_b = gb[gt_mask] if len(gb) else torch.zeros((0, 4))

            pred_mask = (pc == cls_id)
            if not pred_mask.any():
                continue
            p_b = pb[pred_mask]
            p_s = ps[pred_mask]

            order = p_s.argsort(descending=True)
            p_b   = p_b[order]
            p_s   = p_s[order]

            matched = set()
            for k in range(len(p_b)):
                per_scores[cls_id].append(p_s[k].item())
                if len(gt_b) == 0:
                    per_tp[cls_id].append(0)
                    continue
                row = _compute_iou(p_b[k:k+1], gt_b)[0]
                bv, bg = row.max(dim=0)
                if bv.item() >= iou_threshold and bg.item() not in matched:
                    per_tp[cls_id].append(1)
                    matched.add(bg.item())
                else:
                    per_tp[cls_id].append(0)

    ap_d, pr_d, re_d = {}, {}, {}

    for cls_id in range(NUM_CLASSES):
        cname = CLASS_NAMES[cls_id]
        n_gt  = per_n_gt[cls_id]
        sc    = torch.tensor(per_scores[cls_id])
        tp    = torch.tensor(per_tp[cls_id], dtype=torch.float32)

        if n_gt == 0 or len(sc) == 0:
            ap_d[cname] = pr_d[cname] = re_d[cname] = 0.0
            continue

        order = sc.argsort(descending=True)
        tp    = tp[order]
        cum_tp = tp.cumsum(0)
        cum_fp = (1 - tp).cumsum(0)
        prec   = cum_tp / (cum_tp + cum_fp + 1e-7)
        rec    = cum_tp / (n_gt + 1e-7)

        ap = sum(
            float(prec[rec >= r].max().item()) if (rec >= r).any() else 0.0
            for r in np.linspace(0, 1, 11)
        ) / 11.0

        f1   = 2 * prec * rec / (prec + rec + 1e-7)
        bi   = int(f1.argmax().item())
        ap_d[cname] = ap
        pr_d[cname] = float(prec[bi].item())
        re_d[cname] = float(rec[bi].item())

    map_v = float(np.mean(list(ap_d.values()))) if ap_d else 0.0
    return PerClassMetrics(ap=ap_d, precision=pr_d, recall=re_d, map=map_v)


# ===========================================================================
# COMBINED REPORT
# ===========================================================================

@dataclass
class BaselineReport:
    """
    Aggregated results from all three baseline experiments.

    Attributes:
        ssd_metrics:      Experiment A â€” MobileNetV3+SSD test metrics.
        stdloss_metrics:  Experiment B â€” standard loss ablation test metrics.
        nms_result:       Experiment C â€” NMS benchmark result.
        edge_metrics:     Reference EdgeTurkeyNet metrics (passed in).
    """

    ssd_metrics:     Optional[PerClassMetrics]    = None
    stdloss_metrics: Optional[PerClassMetrics]    = None
    nms_result:      Optional[NMSBenchmarkResult] = None
    edge_metrics:    Optional[PerClassMetrics]    = None

    def print_summary(self) -> None:
        """Print a single ablation comparison table then the NMS report."""
        sep = "=" * 72
        print(f"\n{sep}")
        print(f"  BASELINE COMPARISON SUMMARY")
        print(sep)
        print(f"  {'Variant':<32} {'mAP':>7} {'AP-body':>8} {'AP-neck':>8}  Note")
        print(f"  {'-'*68}")

        rows = []
        if self.edge_metrics:
            rows.append(("EdgeTurkeyNet (proposed)",
                         self.edge_metrics,
                         "CIoU + oval-ctr + DIoU-NMS"))
        if self.stdloss_metrics:
            rows.append(("Exp B: standard BCE+GIoU",
                         self.stdloss_metrics,
                         "GIoU + std-ctr, same arch"))
        if self.ssd_metrics:
            rows.append(("Exp A: MobileNetV3+SSD",
                         self.ssd_metrics,
                         "anchor-based, no PAN neck"))

        for name, m, note in rows:
            print(
                f"  {name:<32} {m.map:>7.4f} "
                f"{m.ap.get('body',0.0):>8.4f} "
                f"{m.ap.get('neck',0.0):>8.4f}  {note}"
            )
        print(sep)

        if self.nms_result:
            self.nms_result.print_report()


def run_all_baselines(
    test_loader:        Optional[DataLoader]    = None,
    edge_metrics:       Optional[PerClassMetrics] = None,
    run_ssd:            bool = True,
    run_stdloss:        bool = True,
    run_nms_bench:      bool = True,
    ssd_output_dir:     Path = Path("baseline_runs/ssd"),
    stdloss_output_dir: Path = Path("baseline_runs/standard_loss"),
    ssd_pretrained:     bool = True,
    stdloss_pretrained: bool = True,
    nms_model               = None,
) -> BaselineReport:
    """
    Run all three baseline experiments and return a combined BaselineReport.

    Each experiment is independent and can be disabled with its flag.

    Args:
        test_loader:          Test DataLoader (used by A for final eval, C for real mode).
        edge_metrics:         Pre-computed EdgeTurkeyNet PerClassMetrics for reference.
        run_ssd:              Run Experiment A (MobileNetV3+SSD training + eval).
        run_stdloss:          Run Experiment B (standard loss ablation).
        run_nms_bench:        Run Experiment C (DIoU vs IoU NMS benchmark).
        ssd_output_dir:       Output dir for Exp A checkpoints and CSV.
        stdloss_output_dir:   Output dir for Exp B checkpoints and CSV.
        ssd_pretrained:       Pretrained backbone for Exp A.
        stdloss_pretrained:   Pretrained backbone for Exp B.
        nms_model:            Optional trained model for Exp C real-data mode.

    Returns:
        BaselineReport.
    """
    report = BaselineReport(edge_metrics=edge_metrics)

    if run_ssd:
        ssd_model = SSDTrainer(
            output_dir=ssd_output_dir, pretrained=ssd_pretrained,
        ).train()
        if test_loader is not None:
            report.ssd_metrics = _evaluate_ssd(
                ssd_model, test_loader,
                score_threshold=BL_SCORE_THRESHOLD,
                iou_threshold=BL_IOU_THRESHOLD,
            )

    if run_stdloss:
        from evaluate import evaluate_map
        std_model = StandardFCOSTrainer(
            output_dir=stdloss_output_dir, pretrained=stdloss_pretrained,
        ).train()
        if test_loader is not None:
            report.stdloss_metrics = evaluate_map(
                std_model, test_loader, DEVICE,
                iou_threshold=BL_IOU_THRESHOLD,
                score_threshold=BL_SCORE_THRESHOLD,
            )

    if run_nms_bench:
        report.nms_result = DenseNMSBenchmark().run(
            model=nms_model, test_loader=test_loader,
        )

    report.print_summary()
    return report


# ===========================================================================
# EXPERIMENT D â€” NanoDet (GFL anchor-free, ShuffleNetV2 backbone)
# ===========================================================================
#
# NanoDet uses:
#   Backbone  : ShuffleNetV2-0.5x  (same as EdgeTurkeyNet's alt backbone)
#   Neck      : PAN (lightweight, 96 ch) with depthwise convolutions
#   Head      : GFL (Generalised Focal Loss) â€” anchor-free, single conv tower,
#               predicts a discrete distribution over box sides instead of
#               direct regression. No centerness branch.
#   Loss      : QFL (quality focal) on class + distribution focal on reg
#
# Design differences vs EdgeTurkeyNet:
#   - No centerness branch (quality is folded into the cls score via QFL)
#   - Regression uses integral representation (DFL) not direct l,t,r,b
#   - Neck is shallower (1 DSConv per merge vs 2 + SE in EdgeTurkeyNet)
#   - Backbone is ShuffleNetV2 (channel split+shuffle vs depthwise on MV3)
# ===========================================================================

class _NanoDetPAN(nn.Module):
    """Lightweight PAN neck for NanoDet â€” single DSConv per merge, no SE."""

    CHANNELS = 96   # fixed neck width matching NanoDet-Plus reference

    def __init__(self, backbone_channels: List[int]) -> None:
        super().__init__()
        C  = self.CHANNELS
        p3, p4, p5 = backbone_channels

        self.lat3 = ConvBnAct(p3, C, 1)
        self.lat4 = ConvBnAct(p4, C, 1)
        self.lat5 = ConvBnAct(p5, C, 1)

        self.td4  = DepthwiseSeparableConv(C + C, C)
        self.td3  = DepthwiseSeparableConv(C + C, C)
        self.bu4  = DepthwiseSeparableConv(C + C, C)
        self.bu5  = DepthwiseSeparableConv(C + C, C)

        self.up   = nn.Upsample(scale_factor=2, mode='nearest')
        self.dn   = nn.MaxPool2d(2, 2)

    def forward(
        self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p3 = self.lat3(p3);  p4 = self.lat4(p4);  p5 = self.lat5(p5)
        t4 = self.td4(torch.cat([self.up(p5), p4], 1))
        t3 = self.td3(torch.cat([self.up(t4), p3], 1))
        b4 = self.bu4(torch.cat([self.dn(t3), t4], 1))
        b5 = self.bu5(torch.cat([self.dn(b4), p5], 1))
        return t3, b4, b5


class _GFLHead(nn.Module):
    """
    Generalised Focal Loss detection head (NanoDet style).

    Per FPN level, a single shared conv tower outputs:
      - cls_pred : [B, num_classes, H, W]  â€” class logits
      - reg_pred : [B, 4*(reg_max+1), H, W] â€” discrete distribution per side

    The distribution over [0, reg_max] is decoded to a scalar distance via
    the integral (expected value): d = sum_{i=0}^{reg_max} i * softmax(p_i)

    Args:
        in_channels_list: Per-scale input channel counts.
        num_classes:      Detection class count.
        reg_max:          Maximum regression distance in cells (default 7).
    """

    def __init__(
        self,
        in_channels_list: List[int],
        num_classes: int = NUM_CLASSES,
        reg_max:     int = 7,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reg_max     = reg_max

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        for ch in in_channels_list:
            self.cls_convs.append(nn.Sequential(
                DepthwiseSeparableConv(ch, ch),
                DepthwiseSeparableConv(ch, ch),
            ))
            self.reg_convs.append(nn.Sequential(
                DepthwiseSeparableConv(ch, ch),
                DepthwiseSeparableConv(ch, ch),
            ))
            self.cls_preds.append(nn.Conv2d(ch, num_classes, 1))
            self.reg_preds.append(nn.Conv2d(ch, 4 * (reg_max + 1), 1))

        prior = 0.01
        bv = -math.log((1 - prior) / prior)
        for p in self.cls_preds:
            nn.init.constant_(p.bias, bv)

        # Pre-compute the project vector for DFL integral decoding
        self.register_buffer(
            'project',
            torch.linspace(0, reg_max, reg_max + 1),
        )

    def forward(
        self, features: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        cls_outs, reg_outs = [], []
        for i, feat in enumerate(features):
            cls_outs.append(self.cls_preds[i](self.cls_convs[i](feat)))
            reg_outs.append(self.reg_preds[i](self.reg_convs[i](feat)))
        return cls_outs, reg_outs

    def decode_reg(
        self, reg_pred: torch.Tensor, stride: int
    ) -> torch.Tensor:
        """
        Decode one level's reg_pred [B, 4*(reg_max+1), H, W]
        â†’ [B, H*W, 4] pixel (l,t,r,b) distances.
        """
        B, _, H, W = reg_pred.shape
        R = self.reg_max + 1
        # [B, 4, R, H*W]
        x = reg_pred.reshape(B, 4, R, H * W)
        dist = torch.softmax(x, dim=2)           # distribution
        dist = (dist * self.project.to(reg_pred.device).view(1, 1, R, 1)).sum(dim=2)  # integral
        return dist.permute(0, 2, 1) * stride    # [B, H*W, 4] pixels


class NanoDet(nn.Module):
    """
    NanoDet-style detector.

    Backbone : ShuffleNetV2-0.5x (pretrained, same as EdgeTurkeyNet alt)
    Neck     : Lightweight PAN, 96 channels
    Head     : GFL â€” anchor-free, QFL cls + DFL reg, no centerness

    Args:
        num_classes: Detection class count (default 2).
        pretrained:  Load pretrained ShuffleNetV2-0.5x backbone weights.
        input_size:  Expected input (H, W).
        reg_max:     GFL regression discretisation max (default 7).
    """

    STRIDES = [8, 16, 32]

    def __init__(
        self,
        num_classes: int             = NUM_CLASSES,
        pretrained:  bool            = True,
        input_size:  Tuple[int, int] = INPUT_SIZE,
        reg_max:     int             = 7,
    ) -> None:
        super().__init__()
        import torchvision
        self.num_classes = num_classes
        self.input_size  = input_size
        self.reg_max     = reg_max

        weights = (
            torchvision.models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1
            if pretrained else None
        )
        net = torchvision.models.shufflenet_v2_x0_5(weights=weights)
        self.stage1 = nn.Sequential(net.conv1, net.maxpool)
        self.stage2 = net.stage2   # /8,  48ch
        self.stage3 = net.stage3   # /16, 96ch
        self.stage4 = net.stage4   # /32, 192ch
        backbone_chs = [48, 96, 192]

        self.neck = _NanoDetPAN(backbone_chs)
        neck_chs  = [_NanoDetPAN.CHANNELS] * 3
        self.head = _GFLHead(neck_chs, num_classes, reg_max)

        h, w = input_size
        self._grids: List[torch.Tensor] = []
        for stride in self.STRIDES:
            fh, fw = h // stride, w // stride
            yv, xv = torch.meshgrid(
                torch.arange(fh, dtype=torch.float32),
                torch.arange(fw, dtype=torch.float32),
                indexing='ij',
            )
            grid = torch.stack([xv, yv], -1).reshape(-1, 2) * stride + stride / 2
            self._grids.append(grid)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        s = self.stage1(x)
        p3 = self.stage2(s); p4 = self.stage3(p3); p5 = self.stage4(p4)
        p3, p4, p5 = self.neck(p3, p4, p5)
        return self.head([p3, p4, p5])

    @torch.no_grad()
    def decode_predictions(
        self,
        cls_preds: List[torch.Tensor],
        reg_preds: List[torch.Tensor],
        score_threshold: float = BL_SCORE_THRESHOLD,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Decode raw outputs â†’ list[B] of (boxes[K,4], class_ids[K], scores[K]).
        """
        B = cls_preds[0].shape[0]
        device = cls_preds[0].device
        all_boxes_b:  List[List] = [[] for _ in range(B)]
        all_scores_b: List[List] = [[] for _ in range(B)]
        all_cls_b:    List[List] = [[] for _ in range(B)]

        for i, (cls_p, reg_p) in enumerate(zip(cls_preds, reg_preds)):
            stride = self.STRIDES[i]
            grid   = self._grids[i].to(device)  # [HW, 2]
            B2, C, H, W = cls_p.shape
            N = H * W

            scores = torch.sigmoid(cls_p).reshape(B2, C, N).permute(0, 2, 1)  # [B, N, C]
            dist   = self.head.decode_reg(reg_p, stride)  # [B, N, 4]
            cx     = grid[:, 0]; cy = grid[:, 1]
            x1 = (cx - dist[..., 0]).clamp(0)
            y1 = (cy - dist[..., 1]).clamp(0)
            x2 = (cx + dist[..., 2]).clamp(0, self.input_size[1])
            y2 = (cy + dist[..., 3]).clamp(0, self.input_size[0])
            boxes = torch.stack([x1, y1, x2, y2], -1)  # [B, N, 4]

            for b in range(B2):
                max_scores, cls_ids = scores[b].max(dim=-1)  # [N]
                mask = max_scores > score_threshold
                if not mask.any():
                    continue
                all_boxes_b[b].append(boxes[b][mask])
                all_scores_b[b].append(max_scores[mask])
                all_cls_b[b].append(cls_ids[mask])

        results = []
        for b in range(B):
            if not all_boxes_b[b]:
                results.append((
                    torch.zeros((0, 4)), torch.zeros(0, dtype=torch.int64),
                    torch.zeros(0),
                ))
                continue
            bx = torch.cat(all_boxes_b[b])
            sc = torch.cat(all_scores_b[b])
            ci = torch.cat(all_cls_b[b])
            # per-class DIoU-NMS
            kept_b, kept_s, kept_c = [], [], []
            for cls_id in range(self.num_classes):
                m = ci == cls_id
                if not m.any():
                    continue
                k = diou_nms(bx[m], sc[m], BL_IOU_THRESHOLD)
                kept_b.append(bx[m][k]); kept_s.append(sc[m][k])
                kept_c.append(torch.full((len(k),), cls_id, dtype=torch.int64))
            if not kept_b:
                results.append((
                    torch.zeros((0, 4)), torch.zeros(0, dtype=torch.int64),
                    torch.zeros(0),
                ))
            else:
                results.append((
                    torch.cat(kept_b).cpu(),
                    torch.cat(kept_c).cpu(),
                    torch.cat(kept_s).cpu(),
                ))
        return results

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# NanoDet Loss â€” QFL + DFL
# ---------------------------------------------------------------------------

class NanoDetLoss(nn.Module):
    """
    NanoDet loss: Quality Focal Loss (QFL) + Distribution Focal Loss (DFL).

    QFL:
        Combines sigmoid focal loss with the IoU quality score as the
        continuous target for classification.  No separate centerness branch.
        target = IoU(pred_box, gt_box) for positive locations, 0 for negatives.
        loss   = |target - sigmoid(logit)|^beta * BCE(logit, target)

    DFL:
        Cross-entropy loss over the discretised distribution for each of the
        four box sides.  Targets are computed from the ground-truth l,t,r,b
        distances at each positive location, spread to the two adjacent bins.

    Target assignment:
        FCOS-style cell-centre assignment reused from loss.py.

    Args:
        num_classes: Detection class count.
        reg_max:     GFL max distance in cells.
        beta:        QFL modulating exponent (default 2).
        lambda_cls:  QFL weight.
        lambda_reg:  DFL weight.
        input_size:  Model input (H, W).
    """

    def __init__(
        self,
        num_classes: int             = NUM_CLASSES,
        reg_max:     int             = 7,
        beta:        float           = 2.0,
        lambda_cls:  float           = 1.0,
        lambda_reg:  float           = 2.0,
        input_size:  Tuple[int, int] = INPUT_SIZE,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reg_max     = reg_max
        self.beta        = beta
        self.lambda_cls  = lambda_cls
        self.lambda_reg  = lambda_reg
        self.input_size  = input_size
        self.register_buffer('project', torch.linspace(0, reg_max, reg_max + 1))

    def _dfl_loss(
        self,
        pred_dist: torch.Tensor,   # [N, 4*(reg_max+1)]
        target_ltrb: torch.Tensor, # [N, 4] pixel distances
        stride: float,
    ) -> torch.Tensor:
        """Cross-entropy loss on discretised regression distribution."""
        N   = pred_dist.shape[0]
        R   = self.reg_max + 1
        # Convert pixel distances to cell-relative â†’ clamp to [0, reg_max]
        tgt = (target_ltrb / stride).clamp(0, self.reg_max)

        pred_dist = pred_dist.reshape(N, 4, R)
        lo  = tgt.long().clamp(0, self.reg_max - 1)       # lower bin [N, 4]
        hi  = (lo + 1).clamp(0, self.reg_max)              # upper bin
        wlo = hi.float() - tgt                             # weight for lo
        whi = tgt - lo.float()                             # weight for hi

        loss = torch.zeros(1, device=pred_dist.device)
        for side in range(4):
            p = pred_dist[:, side, :]              # [N, R] logits
            l = F.cross_entropy(p, lo[:, side], reduction='none')
            h = F.cross_entropy(p, hi[:, side], reduction='none')
            loss = loss + (wlo[:, side] * l + whi[:, side] * h).mean()
        return loss / 4.0

    def forward(
        self,
        cls_preds: List[torch.Tensor],
        reg_preds: List[torch.Tensor],
        gt_boxes_batch:     List[torch.Tensor],
        gt_class_ids_batch: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        device = cls_preds[0].device
        feature_shapes = [(p.shape[2], p.shape[3]) for p in cls_preds]
        gt_boxes_dev = [b.to(device) for b in gt_boxes_batch]
        gt_cls_dev   = [c.to(device) for c in gt_class_ids_batch]

        # Reuse FCOS target assignment for positive mask + ltrb targets
        cls_tgts, reg_tgts, _ = fcos_assign_targets(
            gt_boxes_dev, gt_cls_dev, feature_shapes,
            num_classes=self.num_classes,
            strides=STRIDES,
            size_ranges=FCOS_SIZE_RANGES,
            input_size=self.input_size,
        )

        total_cls = torch.zeros(1, device=device)
        total_reg = torch.zeros(1, device=device)

        for li, (cls_p, reg_p) in enumerate(zip(cls_preds, reg_preds)):
            B, C, H, W = cls_p.shape
            N = H * W
            stride = float(STRIDES[li])

            cls_flat = cls_p.reshape(B, C, N).permute(0, 2, 1).reshape(-1, C)
            reg_flat = reg_p.reshape(B, 4 * (self.reg_max + 1), N).permute(0, 2, 1)
            reg_flat = reg_flat.reshape(-1, 4 * (self.reg_max + 1))

            cls_tgt = cls_tgts[li].reshape(-1, C)      # [B*N, C]
            reg_tgt = reg_tgts[li].reshape(-1, 4)      # [B*N, 4] pixel distances

            pos_mask = cls_tgt.sum(dim=-1) > 0.5

            # â”€â”€ QFL on all locations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # Quality target: for positives use per-class GT indicator (0/1
            # because we don't have IoU here without decoding); for negatives 0.
            # Simplified QFL: just focal BCE on the binary target.
            cls_prob = torch.sigmoid(cls_flat)
            qfl = F.binary_cross_entropy_with_logits(
                cls_flat, cls_tgt, reduction='none'
            )
            # Focal modulation: weight by |target - prob|^beta
            weight = (cls_tgt - cls_prob).abs().pow(self.beta).detach()
            total_cls = total_cls + (qfl * weight).mean()

            # â”€â”€ DFL on positives only â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if pos_mask.any():
                total_reg = total_reg + self._dfl_loss(
                    reg_flat[pos_mask], reg_tgt[pos_mask], stride
                )

        n = max(1, len(cls_preds))
        cls_l = total_cls / n
        reg_l = total_reg / n
        return {
            "total": self.lambda_cls * cls_l + self.lambda_reg * reg_l,
            "cls":   cls_l,
            "reg":   reg_l,
        }


def _evaluate_nanodet(
    model:           NanoDet,
    loader:          DataLoader,
    score_threshold: float = BL_SCORE_THRESHOLD,
    iou_threshold:   float = BL_IOU_THRESHOLD,
) -> PerClassMetrics:
    """Evaluate NanoDet on a DataLoader using _compute_per_class_metrics."""
    ih, iw = INPUT_SIZE
    pred_b, pred_s, pred_c = [], [], []
    gt_b, gt_c             = [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            imgs  = batch["images"].to(DEVICE, non_blocking=True)
            cls_p, reg_p = model(imgs)
            preds = model.decode_predictions(cls_p, reg_p, score_threshold)
            for pb, pc, ps in preds:
                pred_b.append(pb); pred_c.append(pc); pred_s.append(ps)
            for bi in range(len(batch["images"])):
                gt_n = batch["boxes"][bi]
                gt_ci = batch["class_ids"][bi]
                if len(gt_n):
                    x1 = (gt_n[:,0] - gt_n[:,2]/2) * iw
                    y1 = (gt_n[:,1] - gt_n[:,3]/2) * ih
                    x2 = (gt_n[:,0] + gt_n[:,2]/2) * iw
                    y2 = (gt_n[:,1] + gt_n[:,3]/2) * ih
                    gt_n = torch.stack([x1, y1, x2, y2], -1)
                gt_b.append(gt_n); gt_c.append(gt_ci)

    return _compute_per_class_metrics(
        pred_b, pred_s, pred_c, gt_b, gt_c, iou_threshold=iou_threshold,
    )


class NanoDetTrainer:
    """
    Self-contained training loop for NanoDet (Experiment D).

    Args:
        output_dir: Checkpoint and CSV output directory.
        pretrained: Pretrained ShuffleNetV2-0.5x backbone.
        epochs:     Maximum training epochs.
        patience:   Early-stopping patience on val mAP.
    """

    def __init__(
        self,
        output_dir: Path = Path("baseline_runs/nanodet"),
        pretrained: bool = True,
        epochs:     int  = BL_EPOCHS,
        patience:   int  = BL_EARLY_STOP_PATIENCE,
    ) -> None:
        _set_seed(SEED)
        self.epochs     = epochs
        self.patience   = patience
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = NanoDet(
            num_classes=NUM_CLASSES, pretrained=pretrained, input_size=INPUT_SIZE,
        ).to(DEVICE)
        print(
            f"[NanoDet] ShuffleNetV2-0.5x + PAN + GFL head  "
            f"params={self.model.get_parameter_count():,}"
        )

        self.criterion    = NanoDetLoss(num_classes=NUM_CLASSES, input_size=INPUT_SIZE)
        self.optimizer    = optim.AdamW(
            self.model.parameters(), lr=BL_BASE_LR, weight_decay=BL_WEIGHT_DECAY,
        )
        self.scaler       = GradScaler('cuda',enabled=torch.cuda.is_available())
        self.train_loader = get_train_loader(BL_BATCH_SIZE, BL_NUM_WORKERS)
        self.val_loader   = get_val_loader(BL_BATCH_SIZE,   BL_NUM_WORKERS)
        self.best_map     = 0.0
        self.no_improve   = 0

        self._csv = self.output_dir / "train_metrics.csv"
        with open(self._csv, "w") as f:
            f.write("epoch,lr,train_total,train_cls,train_reg,val_map,"
                    "val_ap_body,val_ap_neck\n")

    def _train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        self.model.train()
        lr = _cosine_lr(epoch, self.epochs, BL_BASE_LR, BL_MIN_LR, BL_WARMUP_EPOCHS)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        t_tot = t_cls = t_reg = 0.0
        n = 0
        for batch in self.train_loader:
            imgs  = batch["images"].to(DEVICE, non_blocking=True)
            boxes = batch["boxes"]
            cids  = batch["class_ids"]

            self.optimizer.zero_grad(set_to_none=True)
            with autocast('cuda',enabled=torch.cuda.is_available()):
                cls_p, reg_p = self.model(imgs)
                losses = self.criterion(cls_p, reg_p, boxes, cids)

            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), BL_GRADIENT_CLIP)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            t_tot += losses["total"].item()
            t_cls += losses["cls"].item()
            t_reg += losses["reg"].item()
            n += 1

        return t_tot/max(1,n), t_cls/max(1,n), t_reg/max(1,n)

    def train(self) -> NanoDet:
        """Run training.  Returns best NanoDet (eval mode, on DEVICE)."""
        best_ckpt = self.output_dir / "nanodet_best.pth"
        last_ckpt = self.output_dir / "nanodet_last.pth"
        print(f"\n{'='*60}\n  Experiment D â€” NanoDet\n{'='*60}\n")

        for epoch in range(self.epochs):
            t_tot, t_cls, t_reg = self._train_epoch(epoch)
            lr  = _cosine_lr(epoch, self.epochs, BL_BASE_LR, BL_MIN_LR, BL_WARMUP_EPOCHS)
            val = _evaluate_nanodet(self.model, self.val_loader)

            print(
                f"  [NanoDet E{epoch:03d}]  total={t_tot:.4f}  "
                f"cls={t_cls:.4f}  reg={t_reg:.4f}  val_mAP={val.map:.4f}"
            )
            val.print_table(BL_IOU_THRESHOLD)

            with open(self._csv, "a") as f:
                f.write(f"{epoch},{lr:.6f},{t_tot:.6f},{t_cls:.6f},{t_reg:.6f},"
                        f"{val.map:.6f},{val.ap.get('body',0):.6f},"
                        f"{val.ap.get('neck',0):.6f}\n")

            torch.save({"epoch": epoch, "model_state": self.model.state_dict(),
                        "best_map": self.best_map}, last_ckpt)
            if val.map > self.best_map:
                self.best_map = val.map; self.no_improve = 0
                torch.save({"epoch": epoch, "model_state": self.model.state_dict(),
                            "best_map": self.best_map}, best_ckpt)
                print(f"  âœ“ New best NanoDet mAP: {self.best_map:.4f}")
            else:
                self.no_improve += 1
                if self.no_improve >= self.patience:
                    print(f"[NanoDet] Early stopping at epoch {epoch}.")
                    break

        ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(DEVICE).eval()
        print(f"[NanoDet] Done.  Best mAP = {self.best_map:.4f}")
        return self.model


# ===========================================================================
# EXPERIMENT E â€” YOLOv5-nano (CSP-Tiny backbone, anchor-based YOLO head)
# ===========================================================================
#
# YOLOv5-nano uses:
#   Backbone  : CSP-Tiny â€” Focus stem + 3 CSP stages (C3-lite)
#   Neck      : PANNet â€” FPN top-down + PAN bottom-up, CSP merges
#   Head      : YOLO-style anchor-based, 3 scales Ã— 3 anchors per cell
#               Predicts (tx, ty, tw, th, obj, cls...) per anchor
#   Loss      : BCE objectness + BCE cls + CIoU box regression
#               (matching Ultralytics YOLOv5 loss structure)
#
# Key difference from SSD:
#   - Uses sigmoid objectness score separate from class probabilities
#   - Anchor sizes tuned to turkey body scale (medium objects, top-down)
#   - CSP blocks are more compute-efficient than standard residuals
# ===========================================================================

class _Focus(nn.Module):
    """YOLOv5 Focus stem: space-to-depth then 1 conv. Avoids strided conv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = ConvBnAct(in_ch * 4, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Slice 4 sub-images from alternating pixels â†’ concat â†’ conv
        return self.conv(torch.cat([
            x[..., ::2, ::2], x[..., 1::2, ::2],
            x[..., ::2, 1::2], x[..., 1::2, 1::2],
        ], dim=1))


class _BottleneckLite(nn.Module):
    """Single depthwise-separable residual block for CSP-lite stages."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.cv1 = ConvBnAct(ch, ch, 1)
        self.cv2 = DepthwiseSeparableConv(ch, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x))


class _C3Lite(nn.Module):
    """
    CSP Bottleneck with 3 convolutions (C3), lite variant.

    Splits channels, runs n bottleneck blocks on one branch,
    concatenates with the other branch, projects back.

    Args:
        in_ch:  Input channel count.
        out_ch: Output channel count.
        n:      Number of bottleneck blocks (default 1).
    """

    def __init__(self, in_ch: int, out_ch: int, n: int = 1) -> None:
        super().__init__()
        mid = out_ch // 2
        self.cv1 = ConvBnAct(in_ch,  mid, 1)
        self.cv2 = ConvBnAct(in_ch,  mid, 1)
        self.cv3 = ConvBnAct(mid * 2, out_ch, 1)
        self.m   = nn.Sequential(*[_BottleneckLite(mid) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], dim=1))


class _YOLOv5NanoBackbone(nn.Module):
    """
    YOLOv5-nano CSP-Tiny backbone.

    Output channels and strides:
        P3: 64 ch, stride 8
        P4: 128 ch, stride 16
        P5: 256 ch, stride 32
    """

    def __init__(self) -> None:
        super().__init__()
        # stem: Focus /2 â†’ /2
        self.stem = nn.Sequential(
            _Focus(3, 16),                           # /2 â†’ /2, 16ch
            ConvBnAct(16, 32, 3, stride=2, padding=1),  # /4,  32ch
        )
        # stage2 â†’ /8
        self.stage2 = nn.Sequential(
            ConvBnAct(32, 64, 3, stride=2, padding=1),
            _C3Lite(64, 64, n=1),
        )
        # stage3 â†’ /16
        self.stage3 = nn.Sequential(
            ConvBnAct(64, 128, 3, stride=2, padding=1),
            _C3Lite(128, 128, n=2),
        )
        # stage4 â†’ /32
        self.stage4 = nn.Sequential(
            ConvBnAct(128, 256, 3, stride=2, padding=1),
            _C3Lite(256, 256, n=1),
        )
        self.out_channels = [64, 128, 256]
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s   = self.stem(x)
        p3  = self.stage2(s)
        p4  = self.stage3(p3)
        p5  = self.stage4(p4)
        return p3, p4, p5


class _YOLOv5NanoPAN(nn.Module):
    """YOLOv5-nano PANNet neck â€” FPN top-down + PAN bottom-up with C3 merges."""

    def __init__(self, in_channels: List[int]) -> None:
        super().__init__()
        p3, p4, p5 = in_channels
        C = p3  # use P3 channels (64) as the common width

        # FPN top-down: p5â†’p4, p4â†’p3
        self.lat5    = ConvBnAct(p5, C, 1)
        self.lat4    = ConvBnAct(p4, C, 1)
        self.merge4  = _C3Lite(C + C, C, n=1)
        self.merge3  = _C3Lite(C + C, C, n=1)

        # PAN bottom-up: p3â†’p4, p4â†’p5
        self.down3   = ConvBnAct(C, C, 3, stride=2, padding=1)
        self.pan4    = _C3Lite(C + C, C, n=1)
        self.down4   = ConvBnAct(C, C, 3, stride=2, padding=1)
        self.pan5    = _C3Lite(C + C, C, n=1)

        self.up      = nn.Upsample(scale_factor=2, mode='nearest')
        self.out_channels = [C, C, C]

    def forward(
        self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p5_lat = self.lat5(p5)
        p4_lat = self.lat4(p4)

        # Top-down
        t4 = self.merge4(torch.cat([self.up(p5_lat), p4_lat], 1))
        t3 = self.merge3(torch.cat([self.up(t4),     p3],     1))

        # Bottom-up
        b4 = self.pan4(torch.cat([self.down3(t3), t4],     1))
        b5 = self.pan5(torch.cat([self.down4(b4), p5_lat], 1))

        return t3, b4, b5


class _YOLOv5Head(nn.Module):
    """
    YOLOv5-style anchor-based detection head.

    Per scale, a single 1Ã—1 conv predicts for each of na anchors:
        (tx, ty, tw, th, obj, cls_0, cls_1, ..., cls_C-1)

    Args:
        in_channels_list: Channel count per FPN level.
        num_anchors_per_scale: Number of anchor templates per scale.
        num_classes:      Foreground class count.
    """

    def __init__(
        self,
        in_channels_list:      List[int],
        num_anchors_per_scale: int = 3,
        num_classes:           int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.na  = num_anchors_per_scale
        self.nc  = num_classes
        self.no  = num_classes + 5  # outputs per anchor

        self.preds = nn.ModuleList([
            nn.Conv2d(ch, self.na * self.no, 1) for ch in in_channels_list
        ])

        prior = 0.01
        bv = -math.log((1 - prior) / prior)
        for p in self.preds:
            if p.bias is not None:
                nn.init.zeros_(p.bias)
                # objectness and class scores init
                p.bias.data.view(self.na, self.no)[..., 4:].fill_(bv)

    def forward(
        self, features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Returns list of raw pred maps [B, na*(nc+5), H, W] per scale."""
        return [pred(feat) for feat, pred in zip(features, self.preds)]


# Default YOLOv5-nano anchors tuned for medium top-down turkey bodies on 640px
# Three scales Ã— three anchor templates each (w, h in pixels at input res)
YOLOV5_ANCHORS = [
    [[10, 13], [16, 30], [33, 23]],    # P3 / stride 8  â€” small
    [[30, 61], [62, 45], [59, 119]],   # P4 / stride 16 â€” medium
    [[116, 90], [156, 198], [373, 326]], # P5 / stride 32 â€” large
]


class YOLOv5Nano(nn.Module):
    """
    YOLOv5-nano detector.

    Backbone : CSP-Tiny (Focus stem + C3 stages), trained from scratch
    Neck     : PANNet with C3 merges
    Head     : YOLO anchor-based (3 anchors/cell Ã— 3 scales)

    Args:
        num_classes: Detection class count (default 2).
        input_size:  Expected input (H, W).
        anchors:     [[3Ã—(w,h)] Ã— 3 scales] in input pixels.
    """

    STRIDES = [8, 16, 32]

    def __init__(
        self,
        num_classes: int             = NUM_CLASSES,
        input_size:  Tuple[int, int] = INPUT_SIZE,
        anchors:     List           = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_size  = input_size
        self.anchors_cfg = anchors or YOLOV5_ANCHORS
        self.na          = len(self.anchors_cfg[0])   # 3
        self.no          = num_classes + 5

        self.backbone    = _YOLOv5NanoBackbone()
        self.neck        = _YOLOv5NanoPAN(self.backbone.out_channels)
        self.head        = _YOLOv5Head(self.neck.out_channels, self.na, num_classes)

        # Register anchor tensors as buffers
        for i, (anch, stride) in enumerate(
            zip(self.anchors_cfg, self.STRIDES)
        ):
            t = torch.tensor(anch, dtype=torch.float32) / stride
            self.register_buffer(f'anchor_grid_{i}', t)

    def _get_anchor_grid(self, level: int) -> torch.Tensor:
        return getattr(self, f'anchor_grid_{level}')   # [na, 2] in cell units

    def forward(
        self, x: torch.Tensor
    ) -> List[torch.Tensor]:
        p3, p4, p5 = self.backbone(x)
        p3, p4, p5 = self.neck(p3, p4, p5)
        return self.head([p3, p4, p5])  # list of [B, na*no, H, W]

    @torch.no_grad()
    def decode_predictions(
        self,
        raw_preds: List[torch.Tensor],
        score_threshold: float = BL_SCORE_THRESHOLD,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Decode YOLO raw outputs â†’ list[B] of (boxes[K,4], class_ids[K], scores[K]).
        """
        ih, iw = self.input_size
        B = raw_preds[0].shape[0]
        device = raw_preds[0].device

        all_boxes_b:  List[List] = [[] for _ in range(B)]
        all_scores_b: List[List] = [[] for _ in range(B)]
        all_cls_b:    List[List] = [[] for _ in range(B)]

        for li, (pred, stride) in enumerate(zip(raw_preds, self.STRIDES)):
            B2, _, H, W = pred.shape
            anch = self._get_anchor_grid(li).to(device)   # [na, 2] cell units

            # [B, na, no, H, W] â†’ [B, H, W, na, no]
            p = pred.reshape(B2, self.na, self.no, H, W)
            p = p.permute(0, 3, 4, 1, 2).contiguous()    # [B, H, W, na, no]
            p = torch.sigmoid(p)

            # Grid of cell centres
            yv, xv = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij',
            )
            grid = torch.stack([xv, yv], -1).float()     # [H, W, 2]

            # Decode xy (relative to cell centre â†’ absolute cell coords)
            xy = (p[..., :2] * 2.0 - 0.5 + grid.unsqueeze(2)) * stride
            # Decode wh
            wh = (p[..., 2:4] * 2.0) ** 2 * anch.unsqueeze(0).unsqueeze(0) * stride

            x1 = (xy[..., 0] - wh[..., 0] / 2).clamp(0, iw)
            y1 = (xy[..., 1] - wh[..., 1] / 2).clamp(0, ih)
            x2 = (xy[..., 0] + wh[..., 0] / 2).clamp(0, iw)
            y2 = (xy[..., 1] + wh[..., 1] / 2).clamp(0, ih)
            boxes = torch.stack([x1, y1, x2, y2], -1)   # [B, H, W, na, 4]

            obj   = p[..., 4]                             # [B, H, W, na]
            cls_p = p[..., 5:]                            # [B, H, W, na, C]
            scores_all = obj.unsqueeze(-1) * cls_p        # [B, H, W, na, C]

            # Flatten spatial
            boxes      = boxes.reshape(B2, -1, 4)
            scores_all = scores_all.reshape(B2, -1, self.num_classes)

            max_sc, cls_ids = scores_all.max(dim=-1)      # [B, N]

            for b in range(B2):
                mask = max_sc[b] > score_threshold
                if not mask.any():
                    continue
                all_boxes_b[b].append(boxes[b][mask])
                all_scores_b[b].append(max_sc[b][mask])
                all_cls_b[b].append(cls_ids[b][mask])

        results = []
        for b in range(B):
            if not all_boxes_b[b]:
                results.append((
                    torch.zeros((0, 4)), torch.zeros(0, dtype=torch.int64),
                    torch.zeros(0),
                ))
                continue
            bx = torch.cat(all_boxes_b[b])
            sc = torch.cat(all_scores_b[b])
            ci = torch.cat(all_cls_b[b])
            kept_b, kept_s, kept_c = [], [], []
            for cls_id in range(self.num_classes):
                m = ci == cls_id
                if not m.any():
                    continue
                k = diou_nms(bx[m], sc[m], BL_IOU_THRESHOLD)
                kept_b.append(bx[m][k]); kept_s.append(sc[m][k])
                kept_c.append(torch.full((len(k),), cls_id, dtype=torch.int64))
            if not kept_b:
                results.append((
                    torch.zeros((0, 4)), torch.zeros(0, dtype=torch.int64),
                    torch.zeros(0),
                ))
            else:
                results.append((
                    torch.cat(kept_b).cpu(),
                    torch.cat(kept_c).cpu(),
                    torch.cat(kept_s).cpu(),
                ))
        return results

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# YOLOv5 Loss
# ---------------------------------------------------------------------------

class YOLOv5Loss(nn.Module):
    """
    YOLOv5 loss: BCE objectness + BCE classification + CIoU box regression.

    Positive assignment: per anchor per scale, assign GT box to the anchor
    whose aspect ratio best matches (smallest log ratio).  Centre of the GT
    box must fall within the cell Â± 0.5 cells offset tolerance.

    Args:
        num_classes:    Foreground class count.
        anchors_cfg:    [[naÃ—(w,h)] Ã— scales] in input pixels.
        strides:        FPN strides.
        input_size:     Model input (H, W).
        lambda_obj:     Objectness loss weight.
        lambda_cls:     Classification loss weight.
        lambda_box:     Box regression (CIoU) weight.
        iou_threshold:  Anchor-GT IoU threshold for positive assignment.
    """

    def __init__(
        self,
        num_classes:   int             = NUM_CLASSES,
        anchors_cfg:   List            = None,
        strides:       List[int]       = None,
        input_size:    Tuple[int, int] = INPUT_SIZE,
        lambda_obj:    float           = 1.0,
        lambda_cls:    float           = 0.5,
        lambda_box:    float           = 0.05,
        iou_threshold: float           = 0.20,
    ) -> None:
        super().__init__()
        self.num_classes   = num_classes
        self.anchors_cfg   = anchors_cfg or YOLOV5_ANCHORS
        self.strides       = strides or [8, 16, 32]
        self.input_size    = input_size
        self.lambda_obj    = lambda_obj
        self.lambda_cls    = lambda_cls
        self.lambda_box    = lambda_box
        self.iou_threshold = iou_threshold

    @staticmethod
    def _ciou_loss(
        pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-7
    ) -> torch.Tensor:
        """CIoU loss â€” reuses the same formula as loss.py for consistency."""
        pw  = (pred[:, 2] - pred[:, 0]).clamp(0)
        ph  = (pred[:, 3] - pred[:, 1]).clamp(0)
        tw  = (tgt[:, 2]  - tgt[:, 0]).clamp(0)
        th  = (tgt[:, 3]  - tgt[:, 1]).clamp(0)
        pcx = pred[:, 0] + pw / 2;  pcy = pred[:, 1] + ph / 2
        tcx = tgt[:, 0]  + tw / 2;  tcy = tgt[:, 1]  + th / 2

        ix1 = torch.max(pred[:, 0], tgt[:, 0])
        iy1 = torch.max(pred[:, 1], tgt[:, 1])
        ix2 = torch.min(pred[:, 2], tgt[:, 2])
        iy2 = torch.min(pred[:, 3], tgt[:, 3])
        inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
        union = pw * ph + tw * th - inter + eps
        iou   = inter / union

        ex1 = torch.min(pred[:, 0], tgt[:, 0])
        ey1 = torch.min(pred[:, 1], tgt[:, 1])
        ex2 = torch.max(pred[:, 2], tgt[:, 2])
        ey2 = torch.max(pred[:, 3], tgt[:, 3])
        c2  = (ex2 - ex1).pow(2) + (ey2 - ey1).pow(2) + eps
        rho2 = (pcx - tcx).pow(2) + (pcy - tcy).pow(2)

        v    = (4 / math.pi ** 2) * (
            torch.atan(tw / (th + eps)) - torch.atan(pw / (ph + eps))
        ).pow(2)
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)
        ciou = iou - rho2 / c2 - alpha * v
        return (1 - ciou).mean()

    def _assign_targets(
        self,
        gt_boxes_batch:     List[torch.Tensor],
        gt_class_ids_batch: List[torch.Tensor],
        device:             torch.device,
    ) -> List[List[Optional[Tuple]]]:
        """
        Per-scale, per-image anchor assignment.

        Returns a list[scales] of list[batch] of optional (pos_pred_indices,
        box_targets, cls_targets, obj_targets) tuples.
        """
        ih, iw = self.input_size
        results = [[None] * len(gt_boxes_batch) for _ in self.strides]

        for bi, (gt_n, gt_cls) in enumerate(
            zip(gt_boxes_batch, gt_class_ids_batch)
        ):
            gt_n   = gt_n.to(device)
            gt_cls = gt_cls.to(device)
            if len(gt_n) == 0:
                continue

            # Convert YOLO norm cx,cy,w,h â†’ pixel x1,y1,x2,y2
            gx1 = (gt_n[:, 0] - gt_n[:, 2] / 2) * iw
            gy1 = (gt_n[:, 1] - gt_n[:, 3] / 2) * ih
            gx2 = (gt_n[:, 0] + gt_n[:, 2] / 2) * iw
            gy2 = (gt_n[:, 1] + gt_n[:, 3] / 2) * ih
            gt_wh = torch.stack([gt_n[:, 2] * iw, gt_n[:, 3] * ih], -1)  # [M, 2]

            for si, (anch, stride) in enumerate(
                zip(self.anchors_cfg, self.strides)
            ):
                anch_t = torch.tensor(anch, dtype=torch.float32, device=device)
                # Anchor-GT ratio matching
                r = gt_wh.unsqueeze(1) / anch_t.unsqueeze(0)  # [M, na, 2]
                match = torch.max(r, 1.0 / r).max(dim=2).values  # [M, na]
                pos   = match < 4.0   # YOLOv5 default threshold

                pos_gt, pos_an = pos.nonzero(as_tuple=True)
                if len(pos_gt) == 0:
                    continue

                H, W  = ih // stride, iw // stride
                gcx   = gt_n[pos_gt, 0] * W   # fractional cell col
                gcy   = gt_n[pos_gt, 1] * H   # fractional cell row
                ci    = gcx.long().clamp(0, W - 1)
                ri    = gcy.long().clamp(0, H - 1)

                flat_idx = ri * W * len(anch) + ci * len(anch) + pos_an  # [P]
                n_pred   = H * W * len(anch)
                obj_tgt  = torch.zeros(n_pred, device=device)
                obj_tgt[flat_idx] = 1.0

                box_tgt = torch.stack([
                    gx1[pos_gt], gy1[pos_gt], gx2[pos_gt], gy2[pos_gt]
                ], dim=-1)
                cls_tgt = gt_cls[pos_gt].long()

                results[si][bi] = (flat_idx, box_tgt, cls_tgt, obj_tgt)

        return results

    def forward(
        self,
        raw_preds:          List[torch.Tensor],
        gt_boxes_batch:     List[torch.Tensor],
        gt_class_ids_batch: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        device = raw_preds[0].device
        B      = raw_preds[0].shape[0]
        ih, iw = self.input_size
        na     = len(self.anchors_cfg[0])
        nc     = self.num_classes
        no     = nc + 5

        assignments = self._assign_targets(gt_boxes_batch, gt_class_ids_batch, device)

        total_obj = torch.zeros(1, device=device)
        total_cls = torch.zeros(1, device=device)
        total_box = torch.zeros(1, device=device)
        n_scales  = len(raw_preds)

        for si, (pred, stride) in enumerate(zip(raw_preds, self.strides)):
            B2, _, H, W = pred.shape
            # Raw logits reshaped â€” [B, H, W, na, no]
            p_raw = pred.reshape(B2, na, no, H, W).permute(0, 3, 4, 1, 2).contiguous()

            # Sigmoided copy â€” used ONLY for box geometry decoding, not for loss.
            # Keeping this separate is the key fix: BCE losses use raw logits
            # via binary_cross_entropy_with_logits, which is AMP-safe.
            p_sig = torch.sigmoid(p_raw)

            # Decode boxes from sigmoided predictions (geometry only)
            anch_t = torch.tensor(
                self.anchors_cfg[si], dtype=torch.float32, device=device
            ) / stride   # [na, 2] cell-relative

            yv, xv = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij',
            )
            grid = torch.stack([xv, yv], -1).float()  # [H, W, 2]

            xy  = (p_sig[..., :2] * 2.0 - 0.5 + grid.unsqueeze(2)) * stride
            wh  = (p_sig[..., 2:4] * 2.0) ** 2 * anch_t.unsqueeze(0).unsqueeze(0) * stride
            x1  = xy[..., 0] - wh[..., 0] / 2
            y1  = xy[..., 1] - wh[..., 1] / 2
            x2  = xy[..., 0] + wh[..., 0] / 2
            y2  = xy[..., 1] + wh[..., 1] / 2
            pred_boxes      = torch.stack([x1, y1, x2, y2], -1)  # [B, H, W, na, 4]
            pred_boxes_flat = pred_boxes.reshape(B2, -1, 4)

            # Raw logits for BCE losses â€” AMP-safe with BCEWithLogitsLoss
            obj_logits = p_raw[..., 4].reshape(B2, -1)        # [B, N]
            cls_logits = p_raw[..., 5:].reshape(B2, -1, nc)   # [B, N, C]

            for b in range(B2):
                assign  = assignments[si][b]
                N_anchors = H * W * na
                obj_tgt = torch.zeros(N_anchors, device=device)

                if assign is not None:
                    flat_idx, box_tgt, cls_tgt, _ = assign
                    obj_tgt[flat_idx] = 1.0

                # Objectness â€” BCEWithLogitsLoss is AMP-safe
                total_obj = total_obj + F.binary_cross_entropy_with_logits(
                    obj_logits[b], obj_tgt, reduction='mean'
                )

                if assign is None or len(flat_idx) == 0:
                    continue

                # Box regression (CIoU on positives, uses decoded geometry)
                pb = pred_boxes_flat[b][flat_idx]
                total_box = total_box + self._ciou_loss(pb, box_tgt)

                # Classification â€” BCEWithLogitsLoss is AMP-safe
                cp = cls_logits[b][flat_idx]           # [P, C] raw logits
                ct = torch.zeros_like(cp)
                ct[torch.arange(len(cls_tgt)), cls_tgt] = 1.0
                total_cls = total_cls + F.binary_cross_entropy_with_logits(
                    cp, ct, reduction='mean'
                )

        d = max(1, B * n_scales)
        obj_l = total_obj / d
        cls_l = total_cls / d
        box_l = total_box / d

        return {
            "total": self.lambda_obj * obj_l + self.lambda_cls * cls_l
                     + self.lambda_box * box_l,
            "obj":   obj_l,
            "cls":   cls_l,
            "box":   box_l,
        }


def _evaluate_yolov5(
    model:           YOLOv5Nano,
    loader:          DataLoader,
    score_threshold: float = BL_SCORE_THRESHOLD,
    iou_threshold:   float = BL_IOU_THRESHOLD,
) -> PerClassMetrics:
    """Evaluate YOLOv5-nano on a DataLoader."""
    ih, iw = INPUT_SIZE
    pred_b, pred_s, pred_c = [], [], []
    gt_b, gt_c             = [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            imgs   = batch["images"].to(DEVICE, non_blocking=True)
            raw    = model(imgs)
            preds  = model.decode_predictions(raw, score_threshold)
            for pb, pc, ps in preds:
                pred_b.append(pb); pred_c.append(pc); pred_s.append(ps)
            for bi in range(len(batch["images"])):
                gt_n  = batch["boxes"][bi]
                gt_ci = batch["class_ids"][bi]
                if len(gt_n):
                    x1 = (gt_n[:,0] - gt_n[:,2]/2) * iw
                    y1 = (gt_n[:,1] - gt_n[:,3]/2) * ih
                    x2 = (gt_n[:,0] + gt_n[:,2]/2) * iw
                    y2 = (gt_n[:,1] + gt_n[:,3]/2) * ih
                    gt_n = torch.stack([x1, y1, x2, y2], -1)
                gt_b.append(gt_n); gt_c.append(gt_ci)

    return _compute_per_class_metrics(
        pred_b, pred_s, pred_c, gt_b, gt_c, iou_threshold=iou_threshold,
    )


class YOLOv5NanoTrainer:
    """
    Self-contained training loop for YOLOv5-nano (Experiment E).

    Args:
        output_dir: Checkpoint and CSV output directory.
        epochs:     Maximum training epochs.
        patience:   Early-stopping patience on val mAP.
    """

    def __init__(
        self,
        output_dir: Path = Path("baseline_runs/yolov5n"),
        epochs:     int  = BL_EPOCHS,
        patience:   int  = BL_EARLY_STOP_PATIENCE,
    ) -> None:
        _set_seed(SEED)
        self.epochs     = epochs
        self.patience   = patience
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = YOLOv5Nano(
            num_classes=NUM_CLASSES, input_size=INPUT_SIZE,
        ).to(DEVICE)
        print(
            f"[YOLOv5n] CSP-Tiny + PANNet + YOLO head  "
            f"params={self.model.get_parameter_count():,}"
        )

        self.criterion    = YOLOv5Loss(num_classes=NUM_CLASSES, input_size=INPUT_SIZE)
        self.optimizer    = optim.AdamW(
            self.model.parameters(), lr=BL_BASE_LR, weight_decay=BL_WEIGHT_DECAY,
        )
        self.scaler       = GradScaler('cuda',enabled=torch.cuda.is_available())
        self.train_loader = get_train_loader(BL_BATCH_SIZE, BL_NUM_WORKERS)
        self.val_loader   = get_val_loader(BL_BATCH_SIZE,   BL_NUM_WORKERS)
        self.best_map     = 0.0
        self.no_improve   = 0

        self._csv = self.output_dir / "train_metrics.csv"
        with open(self._csv, "w") as f:
            f.write("epoch,lr,train_total,train_obj,train_cls,train_box,"
                    "val_map,val_ap_body,val_ap_neck\n")

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        lr = _cosine_lr(epoch, self.epochs, BL_BASE_LR, BL_MIN_LR, BL_WARMUP_EPOCHS)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        tots = {"total": 0.0, "obj": 0.0, "cls": 0.0, "box": 0.0}
        n = 0
        for batch in self.train_loader:
            imgs  = batch["images"].to(DEVICE, non_blocking=True)
            boxes = batch["boxes"]
            cids  = batch["class_ids"]

            self.optimizer.zero_grad(set_to_none=True)
            with autocast('cuda',enabled=torch.cuda.is_available()):
                raw    = self.model(imgs)
                losses = self.criterion(raw, boxes, cids)

            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), BL_GRADIENT_CLIP)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for k in tots:
                tots[k] += losses[k].item()
            n += 1

        return {k: v/max(1,n) for k,v in tots.items()}

    def train(self) -> YOLOv5Nano:
        """Run training.  Returns best YOLOv5Nano (eval mode, on DEVICE)."""
        best_ckpt = self.output_dir / "yolov5n_best.pth"
        last_ckpt = self.output_dir / "yolov5n_last.pth"
        print(f"\n{'='*60}\n  Experiment E â€” YOLOv5-nano\n{'='*60}\n")

        for epoch in range(self.epochs):
            tl  = self._train_epoch(epoch)
            lr  = _cosine_lr(epoch, self.epochs, BL_BASE_LR, BL_MIN_LR, BL_WARMUP_EPOCHS)
            val = _evaluate_yolov5(self.model, self.val_loader)

            print(
                f"  [YOLOv5n E{epoch:03d}]  total={tl['total']:.4f}  "
                f"obj={tl['obj']:.4f}  cls={tl['cls']:.4f}  "
                f"box={tl['box']:.4f}  val_mAP={val.map:.4f}"
            )
            val.print_table(BL_IOU_THRESHOLD)

            with open(self._csv, "a") as f:
                f.write(
                    f"{epoch},{lr:.6f},{tl['total']:.6f},{tl['obj']:.6f},"
                    f"{tl['cls']:.6f},{tl['box']:.6f},{val.map:.6f},"
                    f"{val.ap.get('body',0):.6f},{val.ap.get('neck',0):.6f}\n"
                )

            torch.save({"epoch": epoch, "model_state": self.model.state_dict(),
                        "best_map": self.best_map}, last_ckpt)
            if val.map > self.best_map:
                self.best_map = val.map; self.no_improve = 0
                torch.save({"epoch": epoch, "model_state": self.model.state_dict(),
                            "best_map": self.best_map}, best_ckpt)
                print(f"  âœ“ New best YOLOv5n mAP: {self.best_map:.4f}")
            else:
                self.no_improve += 1
                if self.no_improve >= self.patience:
                    print(f"[YOLOv5n] Early stopping at epoch {epoch}.")
                    break

        ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(DEVICE).eval()
        print(f"[YOLOv5n] Done.  Best mAP = {self.best_map:.4f}")
        return self.model


# ===========================================================================
# EXPERIMENT F â€” YOLOv8-nano (C2f backbone, anchor-free decoupled head)
# ===========================================================================
#
# YOLOv8-nano uses:
#   Backbone  : C2f-Tiny â€” no Focus stem, uses stride-2 conv; C2f blocks
#               (cross-stage with 2 bottlenecks per stage, fewer params than C3)
#   Neck      : C2f PANNet (same topology as v5 but C2f merge blocks)
#   Head      : Decoupled anchor-free head â€” separate cls + reg branches
#               per scale.  DFL regression (same as NanoDet).
#               No objectness score â€” score = max class probability directly.
#   Loss      : VFL (varifocal) cls + DFL reg + CIoU box
#
# Key differences vs YOLOv5-nano:
#   - No anchor boxes, no objectness branch
#   - Decoupled cls/reg towers (not shared)
#   - C2f blocks (more gradient paths than C3)
#   - Closer to NanoDet in spirit but with CIoU instead of DFL-only reg
# ===========================================================================

class _C2f(nn.Module):
    """
    YOLOv8 C2f block â€” cross-stage partial with 2 internal bottlenecks.

    Splits channels, runs n sequential bottleneck-lite blocks on one branch,
    concatenates all intermediate outputs, projects to out_ch.

    Args:
        in_ch:  Input channels.
        out_ch: Output channels.
        n:      Number of bottleneck blocks.
    """

    def __init__(self, in_ch: int, out_ch: int, n: int = 2) -> None:
        super().__init__()
        mid = out_ch // 2
        self.cv1 = ConvBnAct(in_ch, mid * 2, 1)
        self.cv2 = ConvBnAct(mid * (2 + n), out_ch, 1)
        self.m   = nn.ModuleList([_BottleneckLite(mid) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        for blk in self.m:
            y.append(blk(y[-1]))
        return self.cv2(torch.cat(y, dim=1))


class _YOLOv8NanoBackbone(nn.Module):
    """
    YOLOv8-nano C2f-Tiny backbone.

    No Focus stem â€” uses plain stride-2 conv for simplicity.
    Output channels and strides:
        P3: 64 ch, stride 8
        P4: 128 ch, stride 16
        P5: 256 ch, stride 32
    """

    def __init__(self) -> None:
        super().__init__()
        self.stem   = nn.Sequential(
            ConvBnAct(3,  32, 3, stride=2, padding=1),   # /2
            ConvBnAct(32, 64, 3, stride=2, padding=1),   # /4
            _C2f(64, 64, n=1),
        )
        self.stage2 = nn.Sequential(
            ConvBnAct(64, 128, 3, stride=2, padding=1),
            _C2f(128, 128, n=2),
        )
        self.stage3 = nn.Sequential(
            ConvBnAct(128, 256, 3, stride=2, padding=1),
            _C2f(256, 256, n=2),
        )
        self.stage4 = nn.Sequential(
            ConvBnAct(256, 512, 3, stride=2, padding=1),
            _C2f(512, 256, n=1),
        )
        self.out_channels = [128, 256, 256]
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s  = self.stem(x)      # /4
        p3 = self.stage2(s)    # /8,  128ch
        p4 = self.stage3(p3)   # /16, 256ch
        p5 = self.stage4(p4)   # /32, 256ch
        return p3, p4, p5


class _YOLOv8NanoPAN(nn.Module):
    """YOLOv8-nano PANNet neck using C2f merge blocks instead of C3."""

    def __init__(self, in_channels: List[int]) -> None:
        super().__init__()
        p3, p4, p5 = in_channels
        C = p3   # 128

        self.lat5  = ConvBnAct(p5, C, 1)
        self.lat4  = ConvBnAct(p4, C, 1)
        self.td4   = _C2f(C + C, C, n=1)
        self.td3   = _C2f(C + C, C, n=1)
        self.dn3   = ConvBnAct(C, C, 3, stride=2, padding=1)
        self.pan4  = _C2f(C + C, C, n=1)
        self.dn4   = ConvBnAct(C, C, 3, stride=2, padding=1)
        self.pan5  = _C2f(C + C, C, n=1)
        self.up    = nn.Upsample(scale_factor=2, mode='nearest')
        self.out_channels = [C, C, C]

    def forward(
        self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p5l = self.lat5(p5);  p4l = self.lat4(p4)
        t4  = self.td4(torch.cat([self.up(p5l), p4l], 1))
        t3  = self.td3(torch.cat([self.up(t4),  p3],  1))
        b4  = self.pan4(torch.cat([self.dn3(t3), t4],  1))
        b5  = self.pan5(torch.cat([self.dn4(b4), p5l], 1))
        return t3, b4, b5


class _YOLOv8DecoupledHead(nn.Module):
    """
    YOLOv8 decoupled anchor-free head.

    Per scale:
      cls branch : 2Ã— DSConv â†’ cls_pred [B, C, H, W]
      reg branch : 2Ã— DSConv â†’ reg_pred [B, 4*(reg_max+1), H, W]  (DFL)

    Args:
        in_channels_list: Channel count per FPN level.
        num_classes:      Detection class count.
        reg_max:          DFL discretisation max.
    """

    def __init__(
        self,
        in_channels_list: List[int],
        num_classes: int = NUM_CLASSES,
        reg_max:     int = 16,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reg_max     = reg_max

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        for ch in in_channels_list:
            self.cls_convs.append(nn.Sequential(
                DepthwiseSeparableConv(ch, ch),
                DepthwiseSeparableConv(ch, ch),
            ))
            self.reg_convs.append(nn.Sequential(
                DepthwiseSeparableConv(ch, ch),
                DepthwiseSeparableConv(ch, ch),
            ))
            self.cls_preds.append(nn.Conv2d(ch, num_classes, 1))
            self.reg_preds.append(nn.Conv2d(ch, 4 * (reg_max + 1), 1))

        prior = 0.01
        bv = -math.log((1 - prior) / prior)
        for p in self.cls_preds:
            nn.init.constant_(p.bias, bv)

        self.register_buffer('project', torch.linspace(0, reg_max, reg_max + 1))

    def forward(
        self, features: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        cls_outs, reg_outs = [], []
        for i, feat in enumerate(features):
            cls_outs.append(self.cls_preds[i](self.cls_convs[i](feat)))
            reg_outs.append(self.reg_preds[i](self.reg_convs[i](feat)))
        return cls_outs, reg_outs

    def decode_reg(self, reg_pred: torch.Tensor, stride: int) -> torch.Tensor:
        """[B, 4*(reg_max+1), H, W] â†’ [B, H*W, 4] pixel distances (DFL)."""
        B, _, H, W = reg_pred.shape
        R = self.reg_max + 1
        x = reg_pred.reshape(B, 4, R, H * W)
        d = torch.softmax(x, dim=2)
        d = (d * self.project.to(reg_pred.device).view(1, 1, R, 1)).sum(dim=2)
        return d.permute(0, 2, 1) * stride   # [B, H*W, 4]


class YOLOv8Nano(nn.Module):
    """
    YOLOv8-nano detector.

    Backbone : C2f-Tiny (no Focus, plain stride-2 stems), trained from scratch
    Neck     : C2f PANNet
    Head     : Decoupled anchor-free (cls + DFL reg), no objectness

    Args:
        num_classes: Detection class count (default 2).
        input_size:  Expected input (H, W).
        reg_max:     DFL discretisation max (default 16).
    """

    STRIDES = [8, 16, 32]

    def __init__(
        self,
        num_classes: int             = NUM_CLASSES,
        input_size:  Tuple[int, int] = INPUT_SIZE,
        reg_max:     int             = 16,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_size  = input_size
        self.reg_max     = reg_max

        self.backbone    = _YOLOv8NanoBackbone()
        self.neck        = _YOLOv8NanoPAN(self.backbone.out_channels)
        self.head        = _YOLOv8DecoupledHead(
            self.neck.out_channels, num_classes, reg_max
        )

        h, w = input_size
        self._grids: List[torch.Tensor] = []
        for stride in self.STRIDES:
            fh, fw = h // stride, w // stride
            yv, xv = torch.meshgrid(
                torch.arange(fh, dtype=torch.float32),
                torch.arange(fw, dtype=torch.float32),
                indexing='ij',
            )
            grid = torch.stack([xv, yv], -1).reshape(-1, 2) * stride + stride / 2
            self._grids.append(grid)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        p3, p4, p5 = self.backbone(x)
        p3, p4, p5 = self.neck(p3, p4, p5)
        return self.head([p3, p4, p5])

    @torch.no_grad()
    def decode_predictions(
        self,
        cls_preds: List[torch.Tensor],
        reg_preds: List[torch.Tensor],
        score_threshold: float = BL_SCORE_THRESHOLD,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Decode raw outputs â†’ list[B] of (boxes[K,4], class_ids[K], scores[K]).
        No objectness â€” score = sigmoid(max class logit).
        """
        ih, iw = self.input_size
        B = cls_preds[0].shape[0]
        device = cls_preds[0].device

        all_boxes_b:  List[List] = [[] for _ in range(B)]
        all_scores_b: List[List] = [[] for _ in range(B)]
        all_cls_b:    List[List] = [[] for _ in range(B)]

        for i, (cls_p, reg_p) in enumerate(zip(cls_preds, reg_preds)):
            stride = self.STRIDES[i]
            grid   = self._grids[i].to(device)
            B2, C, H, W = cls_p.shape
            N = H * W

            scores = torch.sigmoid(cls_p).reshape(B2, C, N).permute(0, 2, 1)  # [B, N, C]
            dist   = self.head.decode_reg(reg_p, stride)                        # [B, N, 4]

            cx = grid[:, 0];  cy = grid[:, 1]
            x1 = (cx - dist[..., 0]).clamp(0, iw)
            y1 = (cy - dist[..., 1]).clamp(0, ih)
            x2 = (cx + dist[..., 2]).clamp(0, iw)
            y2 = (cy + dist[..., 3]).clamp(0, ih)
            boxes = torch.stack([x1, y1, x2, y2], -1)   # [B, N, 4]

            for b in range(B2):
                max_scores, cls_ids = scores[b].max(dim=-1)   # [N]
                mask = max_scores > score_threshold
                if not mask.any():
                    continue
                all_boxes_b[b].append(boxes[b][mask])
                all_scores_b[b].append(max_scores[mask])
                all_cls_b[b].append(cls_ids[mask])

        results = []
        for b in range(B):
            if not all_boxes_b[b]:
                results.append((
                    torch.zeros((0, 4)), torch.zeros(0, dtype=torch.int64),
                    torch.zeros(0),
                ))
                continue
            bx = torch.cat(all_boxes_b[b])
            sc = torch.cat(all_scores_b[b])
            ci = torch.cat(all_cls_b[b])
            kept_b, kept_s, kept_c = [], [], []
            for cls_id in range(self.num_classes):
                m = ci == cls_id
                if not m.any():
                    continue
                k = diou_nms(bx[m], sc[m], BL_IOU_THRESHOLD)
                kept_b.append(bx[m][k]); kept_s.append(sc[m][k])
                kept_c.append(torch.full((len(k),), cls_id, dtype=torch.int64))
            if not kept_b:
                results.append((
                    torch.zeros((0, 4)), torch.zeros(0, dtype=torch.int64),
                    torch.zeros(0),
                ))
            else:
                results.append((
                    torch.cat(kept_b).cpu(),
                    torch.cat(kept_c).cpu(),
                    torch.cat(kept_s).cpu(),
                ))
        return results

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# YOLOv8 Loss â€” VFL + DFL + CIoU
# ---------------------------------------------------------------------------

class YOLOv8Loss(nn.Module):
    """
    YOLOv8 loss: Varifocal Loss (VFL) + Distribution Focal Loss (DFL) + CIoU.

    VFL (classification):
        Binary focal loss with the IoU quality score as the positive weight.
        For negatives the target is 0; for positives target = max(iou_score, 0).
        Simplified to QFL (focal BCE) here since we don't have TAL assignment.

    DFL (regression):
        Same distribution focal loss as NanoDetLoss â€” cross-entropy on
        discretised [l,t,r,b] distributions.

    CIoU (box):
        Complete IoU loss on decoded boxes at positive locations.

    Target assignment:
        FCOS cell-centre assignment reused from loss.py (same as NanoDet).

    Args:
        num_classes: Detection class count.
        reg_max:     DFL discretisation max (must match head).
        lambda_cls:  VFL weight.
        lambda_reg:  DFL weight.
        lambda_box:  CIoU weight.
        input_size:  Model input (H, W).
    """

    def __init__(
        self,
        num_classes: int             = NUM_CLASSES,
        reg_max:     int             = 16,
        lambda_cls:  float           = 1.0,
        lambda_reg:  float           = 1.5,
        lambda_box:  float           = 7.5,
        input_size:  Tuple[int, int] = INPUT_SIZE,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reg_max     = reg_max
        self.lambda_cls  = lambda_cls
        self.lambda_reg  = lambda_reg
        self.lambda_box  = lambda_box
        self.input_size  = input_size
        self.register_buffer('project', torch.linspace(0, reg_max, reg_max + 1))

    @staticmethod
    def _ciou(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """Scalar CIoU loss â€” shared implementation with YOLOv5Loss."""
        pw  = (pred[:, 2] - pred[:, 0]).clamp(0)
        ph  = (pred[:, 3] - pred[:, 1]).clamp(0)
        tw  = (tgt[:,  2] - tgt[:,  0]).clamp(0)
        th  = (tgt[:,  3] - tgt[:,  1]).clamp(0)
        pcx = pred[:, 0] + pw / 2;  pcy = pred[:, 1] + ph / 2
        tcx = tgt[:,  0] + tw / 2;  tcy = tgt[:,  1] + th / 2

        ix1 = torch.max(pred[:, 0], tgt[:, 0])
        iy1 = torch.max(pred[:, 1], tgt[:, 1])
        ix2 = torch.min(pred[:, 2], tgt[:, 2])
        iy2 = torch.min(pred[:, 3], tgt[:, 3])
        inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
        union = pw * ph + tw * th - inter + eps
        iou   = inter / union

        ex1 = torch.min(pred[:, 0], tgt[:, 0]); ex2 = torch.max(pred[:, 2], tgt[:, 2])
        ey1 = torch.min(pred[:, 1], tgt[:, 1]); ey2 = torch.max(pred[:, 3], tgt[:, 3])
        c2  = (ex2 - ex1).pow(2) + (ey2 - ey1).pow(2) + eps
        rho2 = (pcx - tcx).pow(2) + (pcy - tcy).pow(2)
        v    = (4 / math.pi ** 2) * (
            torch.atan(tw / (th + eps)) - torch.atan(pw / (ph + eps))
        ).pow(2)
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)
        return (1 - iou + rho2 / c2 + alpha * v).mean()

    def _dfl_loss(
        self,
        pred_dist:   torch.Tensor,   # [N, 4*(reg_max+1)]
        target_ltrb: torch.Tensor,   # [N, 4] pixel distances
        stride:      float,
    ) -> torch.Tensor:
        """DFL â€” identical to NanoDetLoss._dfl_loss."""
        N   = pred_dist.shape[0]
        R   = self.reg_max + 1
        tgt = (target_ltrb / stride).clamp(0, self.reg_max)
        pred_dist = pred_dist.reshape(N, 4, R)
        lo  = tgt.long().clamp(0, self.reg_max - 1)
        hi  = (lo + 1).clamp(0, self.reg_max)
        wlo = hi.float() - tgt
        whi = tgt - lo.float()
        loss = torch.zeros(1, device=pred_dist.device)
        for side in range(4):
            p = pred_dist[:, side, :]
            l = F.cross_entropy(p, lo[:, side], reduction='none')
            h = F.cross_entropy(p, hi[:, side], reduction='none')
            loss = loss + (wlo[:, side] * l + whi[:, side] * h).mean()
        return loss / 4.0

    def _decode_boxes(
        self,
        reg_pred: torch.Tensor,   # [B, 4*(reg_max+1), H, W]
        grid:     torch.Tensor,   # [H*W, 2]
        stride:   int,
    ) -> torch.Tensor:
        """DFL integral decode â†’ [B, H*W, 4] pixel (x1,y1,x2,y2)."""
        B, _, H, W = reg_pred.shape
        R = self.reg_max + 1
        x = reg_pred.reshape(B, 4, R, H * W)
        d = torch.softmax(x, dim=2)
        d = (d * self.project.to(reg_pred.device).view(1, 1, R, 1)).sum(dim=2)  # [B, 4, H*W]
        d = d.permute(0, 2, 1) * stride                      # [B, H*W, 4]
        cx = grid[:, 0]; cy = grid[:, 1]
        x1 = cx - d[..., 0]; y1 = cy - d[..., 1]
        x2 = cx + d[..., 2]; y2 = cy + d[..., 3]
        return torch.stack([x1, y1, x2, y2], -1)             # [B, H*W, 4]

    def forward(
        self,
        cls_preds: List[torch.Tensor],
        reg_preds: List[torch.Tensor],
        gt_boxes_batch:     List[torch.Tensor],
        gt_class_ids_batch: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        device = cls_preds[0].device
        feature_shapes = [(p.shape[2], p.shape[3]) for p in cls_preds]
        gt_boxes_dev = [b.to(device) for b in gt_boxes_batch]
        gt_cls_dev   = [c.to(device) for c in gt_class_ids_batch]

        cls_tgts, reg_tgts, _ = fcos_assign_targets(
            gt_boxes_dev, gt_cls_dev, feature_shapes,
            num_classes=self.num_classes,
            strides=STRIDES,
            size_ranges=FCOS_SIZE_RANGES,
            input_size=self.input_size,
        )

        total_cls = torch.zeros(1, device=device)
        total_reg = torch.zeros(1, device=device)
        total_box = torch.zeros(1, device=device)

        h, w = self.input_size
        for li, (cls_p, reg_p) in enumerate(zip(cls_preds, reg_preds)):
            B, C, H, W_feat = cls_p.shape
            N      = H * W_feat
            stride = float(STRIDES[li])

            cls_flat = cls_p.reshape(B, C, N).permute(0, 2, 1).reshape(-1, C)
            reg_flat = reg_p.reshape(B, 4 * (self.reg_max + 1), N).permute(0, 2, 1)
            reg_flat = reg_flat.reshape(-1, 4 * (self.reg_max + 1))

            cls_tgt = cls_tgts[li].reshape(-1, C)
            reg_tgt = reg_tgts[li].reshape(-1, 4)
            pos_mask = cls_tgt.sum(-1) > 0.5

            # â”€â”€ VFL (simplified quality focal BCE) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            cls_prob = torch.sigmoid(cls_flat)
            bce = F.binary_cross_entropy_with_logits(cls_flat, cls_tgt, reduction='none')
            weight = (cls_tgt - cls_prob).abs().pow(2).detach()
            total_cls = total_cls + (bce * weight).mean()

            if pos_mask.any():
                # â”€â”€ DFL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                total_reg = total_reg + self._dfl_loss(
                    reg_flat[pos_mask], reg_tgt[pos_mask], stride
                )

                # â”€â”€ CIoU â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                # Decode positive reg predictions â†’ pixel boxes
                yv, xv = torch.meshgrid(
                    torch.arange(H, device=device, dtype=torch.float32),
                    torch.arange(W_feat, device=device, dtype=torch.float32),
                    indexing='ij',
                )
                grid = torch.stack([xv, yv], -1).reshape(-1, 2)
                grid = grid * stride + stride / 2

                # Reshape reg_p for batch decode
                R = self.reg_max + 1
                rp_all = reg_p.reshape(B, 4, R, N).permute(0, 3, 1, 2)  # [B, N, 4, R]
                d_all  = torch.softmax(rp_all, dim=-1)
                d_all  = (d_all * self.project.to(reg_p.device).view(1, 1, 1, R)).sum(-1) * stride  # [B, N, 4]
                cx_g   = grid[:, 0]; cy_g = grid[:, 1]
                px1 = cx_g - d_all[..., 0]; py1 = cy_g - d_all[..., 1]
                px2 = cx_g + d_all[..., 2]; py2 = cy_g + d_all[..., 3]
                pred_boxes_all = torch.stack([px1, py1, px2, py2], -1)   # [B, N, 4]
                pred_boxes_flat = pred_boxes_all.reshape(-1, 4)

                gt_xyxy = reg_tgt[pos_mask]
                gt_cx = (gt_xyxy[:, 0] + gt_xyxy[:, 2]) / 2
                gt_cy = (gt_xyxy[:, 1] + gt_xyxy[:, 3]) / 2
                # reg_tgt is [l,t,r,b] from cell centre â†’ convert to xyxy
                # (l = cx - x1, r = x2 - cx, t = cy - y1, b = y2 - cy)
                # We need absolute pixel xyxy for CIoU
                # Build grid-relative centres for positives
                pos_idx = pos_mask.nonzero(as_tuple=True)[0]
                bi_idx  = pos_idx // N
                n_idx   = pos_idx %  N
                pcx     = grid[n_idx, 0]
                pcy     = grid[n_idx, 1]
                l, t, r, b_side = (reg_tgt[pos_mask][:, i] for i in range(4))
                box_tgt_xyxy = torch.stack([
                    pcx - l, pcy - t, pcx + r, pcy + b_side,
                ], dim=-1)

                total_box = total_box + self._ciou(
                    pred_boxes_flat[pos_mask], box_tgt_xyxy,
                )

        n = max(1, len(cls_preds))
        cls_l = total_cls / n
        reg_l = total_reg / n
        box_l = total_box / n
        return {
            "total": self.lambda_cls * cls_l + self.lambda_reg * reg_l
                     + self.lambda_box * box_l,
            "cls":   cls_l,
            "reg":   reg_l,
            "box":   box_l,
        }


def _evaluate_yolov8(
    model:           YOLOv8Nano,
    loader:          DataLoader,
    score_threshold: float = BL_SCORE_THRESHOLD,
    iou_threshold:   float = BL_IOU_THRESHOLD,
) -> PerClassMetrics:
    """Evaluate YOLOv8-nano on a DataLoader."""
    ih, iw = INPUT_SIZE
    pred_b, pred_s, pred_c = [], [], []
    gt_b, gt_c             = [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            imgs  = batch["images"].to(DEVICE, non_blocking=True)
            cls_p, reg_p = model(imgs)
            preds = model.decode_predictions(cls_p, reg_p, score_threshold)
            for pb, pc, ps in preds:
                pred_b.append(pb); pred_c.append(pc); pred_s.append(ps)
            for bi in range(len(batch["images"])):
                gt_n  = batch["boxes"][bi]
                gt_ci = batch["class_ids"][bi]
                if len(gt_n):
                    x1 = (gt_n[:,0] - gt_n[:,2]/2) * iw
                    y1 = (gt_n[:,1] - gt_n[:,3]/2) * ih
                    x2 = (gt_n[:,0] + gt_n[:,2]/2) * iw
                    y2 = (gt_n[:,1] + gt_n[:,3]/2) * ih
                    gt_n = torch.stack([x1, y1, x2, y2], -1)
                gt_b.append(gt_n); gt_c.append(gt_ci)

    return _compute_per_class_metrics(
        pred_b, pred_s, pred_c, gt_b, gt_c, iou_threshold=iou_threshold,
    )


class YOLOv8NanoTrainer:
    """
    Self-contained training loop for YOLOv8-nano (Experiment F).

    Args:
        output_dir: Checkpoint and CSV output directory.
        epochs:     Maximum training epochs.
        patience:   Early-stopping patience on val mAP.
    """

    def __init__(
        self,
        output_dir: Path = Path("baseline_runs/yolov8n"),
        epochs:     int  = BL_EPOCHS,
        patience:   int  = BL_EARLY_STOP_PATIENCE,
    ) -> None:
        _set_seed(SEED)
        self.epochs     = epochs
        self.patience   = patience
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = YOLOv8Nano(
            num_classes=NUM_CLASSES, input_size=INPUT_SIZE,
        ).to(DEVICE)
        print(
            f"[YOLOv8n] C2f-Tiny + C2f-PAN + decoupled anchor-free head  "
            f"params={self.model.get_parameter_count():,}"
        )

        self.criterion    = YOLOv8Loss(num_classes=NUM_CLASSES, input_size=INPUT_SIZE)
        self.optimizer    = optim.AdamW(
            self.model.parameters(), lr=BL_BASE_LR, weight_decay=BL_WEIGHT_DECAY,
        )
        self.scaler       = GradScaler('cuda',enabled=torch.cuda.is_available())
        self.train_loader = get_train_loader(BL_BATCH_SIZE, BL_NUM_WORKERS)
        self.val_loader   = get_val_loader(BL_BATCH_SIZE,   BL_NUM_WORKERS)
        self.best_map     = 0.0
        self.no_improve   = 0

        self._csv = self.output_dir / "train_metrics.csv"
        with open(self._csv, "w") as f:
            f.write("epoch,lr,train_total,train_cls,train_reg,train_box,"
                    "val_map,val_ap_body,val_ap_neck\n")

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        lr = _cosine_lr(epoch, self.epochs, BL_BASE_LR, BL_MIN_LR, BL_WARMUP_EPOCHS)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        tots = {"total": 0.0, "cls": 0.0, "reg": 0.0, "box": 0.0}
        n = 0
        for batch in self.train_loader:
            imgs  = batch["images"].to(DEVICE, non_blocking=True)
            boxes = batch["boxes"]
            cids  = batch["class_ids"]

            self.optimizer.zero_grad(set_to_none=True)
            with autocast('cuda',enabled=torch.cuda.is_available()):
                cls_p, reg_p = self.model(imgs)
                losses = self.criterion(cls_p, reg_p, boxes, cids)

            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), BL_GRADIENT_CLIP)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for k in tots:
                tots[k] += losses[k].item()
            n += 1

        return {k: v/max(1,n) for k,v in tots.items()}

    def train(self) -> YOLOv8Nano:
        """Run training.  Returns best YOLOv8Nano (eval mode, on DEVICE)."""
        best_ckpt = self.output_dir / "yolov8n_best.pth"
        last_ckpt = self.output_dir / "yolov8n_last.pth"
        print(f"\n{'='*60}\n  Experiment F â€” YOLOv8-nano\n{'='*60}\n")

        for epoch in range(self.epochs):
            tl  = self._train_epoch(epoch)
            lr  = _cosine_lr(epoch, self.epochs, BL_BASE_LR, BL_MIN_LR, BL_WARMUP_EPOCHS)
            val = _evaluate_yolov8(self.model, self.val_loader)

            print(
                f"  [YOLOv8n E{epoch:03d}]  total={tl['total']:.4f}  "
                f"cls={tl['cls']:.4f}  reg={tl['reg']:.4f}  "
                f"box={tl['box']:.4f}  val_mAP={val.map:.4f}"
            )
            val.print_table(BL_IOU_THRESHOLD)

            with open(self._csv, "a") as f:
                f.write(
                    f"{epoch},{lr:.6f},{tl['total']:.6f},{tl['cls']:.6f},"
                    f"{tl['reg']:.6f},{tl['box']:.6f},{val.map:.6f},"
                    f"{val.ap.get('body',0):.6f},{val.ap.get('neck',0):.6f}\n"
                )

            torch.save({"epoch": epoch, "model_state": self.model.state_dict(),
                        "best_map": self.best_map}, last_ckpt)
            if val.map > self.best_map:
                self.best_map = val.map; self.no_improve = 0
                torch.save({"epoch": epoch, "model_state": self.model.state_dict(),
                            "best_map": self.best_map}, best_ckpt)
                print(f"  âœ“ New best YOLOv8n mAP: {self.best_map:.4f}")
            else:
                self.no_improve += 1
                if self.no_improve >= self.patience:
                    print(f"[YOLOv8n] Early stopping at epoch {epoch}.")
                    break

        ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(DEVICE).eval()
        print(f"[YOLOv8n] Done.  Best mAP = {self.best_map:.4f}")
        return self.model


# ===========================================================================
# EXTENDED BaselineReport â€” add D / E / F metrics
# ===========================================================================

@dataclass
class ExtendedBaselineReport(BaselineReport):
    """
    Extends BaselineReport with the three new detector results.

    Inherits:
        ssd_metrics, stdloss_metrics, nms_result, edge_metrics

    Adds:
        nanodet_metrics:  Experiment D â€” NanoDet test metrics.
        yolov5n_metrics:  Experiment E â€” YOLOv5-nano test metrics.
        yolov8n_metrics:  Experiment F â€” YOLOv8-nano test metrics.
    """

    nanodet_metrics: Optional[PerClassMetrics] = None
    yolov5n_metrics: Optional[PerClassMetrics] = None
    yolov8n_metrics: Optional[PerClassMetrics] = None

    def print_summary(self) -> None:
        """Print full comparison table across all six experiments."""
        sep = "=" * 78
        print(f"\n{sep}")
        print(f"  BASELINE COMPARISON SUMMARY  (all experiments)")
        print(sep)
        print(f"  {'Variant':<36} {'mAP':>7} {'AP-body':>8} {'AP-neck':>8}  Note")
        print(f"  {'-'*74}")

        rows = []
        if self.edge_metrics:
            rows.append(("EdgeTurkeyNet (proposed)",
                         self.edge_metrics,
                         "CIoU+oval-ctr+DIoU-NMS"))
        if self.stdloss_metrics:
            rows.append(("Exp B: standard BCE+GIoU",
                         self.stdloss_metrics,
                         "GIoU+std-ctr, same arch"))
        if self.ssd_metrics:
            rows.append(("Exp A: MobileNetV3+SSD",
                         self.ssd_metrics,
                         "anchor-based, no PAN neck"))
        if self.nanodet_metrics:
            rows.append(("Exp D: NanoDet",
                         self.nanodet_metrics,
                         "GFL+DFL, ShuffleNetV2 PAN"))
        if self.yolov5n_metrics:
            rows.append(("Exp E: YOLOv5-nano",
                         self.yolov5n_metrics,
                         "anchor-based, CSP-Tiny PAN"))
        if self.yolov8n_metrics:
            rows.append(("Exp F: YOLOv8-nano",
                         self.yolov8n_metrics,
                         "anchor-free, C2f decoupled"))

        for name, m, note in rows:
            print(
                f"  {name:<36} {m.map:>7.4f} "
                f"{m.ap.get('body', 0.0):>8.4f} "
                f"{m.ap.get('neck', 0.0):>8.4f}  {note}"
            )
        print(sep)

        if self.nms_result:
            self.nms_result.print_report()
