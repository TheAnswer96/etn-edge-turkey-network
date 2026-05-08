"""
run_distillation.py â€” Independent Knowledge Distillation Runner.

A fully self-contained module that trains a compressed student model using a
frozen full-size teacher, then evaluates it on the test set.  No dependency
on config.py, logger.py, train.py, or main.py.

Architecture overview
---------------------
  Teacher  :  Full-size EdgeTurkeyNet  â€” frozen, loaded from a checkpoint.
  Student  :  StudentEdgeTurkeyNet     â€” not frozen, trained from scratch
              via three distillation signals + ground-truth detection loss.

Student backbone options
------------------------
Two backbones are supported and selected by the STUDENT_BACKBONE global:

  'shufflenetv2'
      Custom ScaledShuffleNetV2Backbone â€” parametric width and depth applied
      to the x0.5 reference architecture.  BACKBONE_WIDTH_MULT and
      BACKBONE_DEPTH_MULT shrink stage channels and block counts independently.

  'mobilenetv3'
      MobileNetV3-Small, always from scratch.  Backbone scaling parameters
      are ignored since torchvision does not expose a width API for this model.

Backbone scaling (ShuffleNetV2 only)
-------------------------------------
  BACKBONE_WIDTH_MULT â€” scales stage output channels vs x0.5 reference [48,96,192].
    1.0  â†’  [48, 96, 192]  (reference)
    0.5  â†’  [24, 48,  96]
    0.25 â†’  [12, 24,  48]

  BACKBONE_DEPTH_MULT â€” scales InvertedResidual blocks per stage vs reference [4,8,4].
    1.0  â†’  [4, 8, 4]  (reference)
    0.5  â†’  [2, 4, 2]
    0.25 â†’  [1, 2, 1]  (minimum 1 per stage)

Distillation losses (FD + RD + GT)
-----------------------------------
  FD  Feature distillation  â€” MSE between projected student/teacher neck outputs.
  RD  Response distillation â€” soft-target KL on cls logits + MSE on reg/ctr.
  GT  Ground-truth loss     â€” standard EdgeTurkeyLoss (focal + CIoU + BCE-ctr).

  L_total = w_gt * L_gt + w_fd * L_fd + w_rd * (L_cls_kl + L_reg_mse + L_ctr_mse)

Folder hierarchy
----------------
  KD_OUTPUT_ROOT/
    <backbone>_bw<bwm>_bd<bdm>_kd/        (USE_DISTILLATION=True)
    <backbone>_bw<bwm>_bd<bdm>_scratch/   (USE_DISTILLATION=False)
      checkpoints/
        student_best.pth   â€” best val mAP checkpoint
        student_last.pth   â€” last epoch checkpoint
      train_metrics.csv    â€” per-epoch training losses
      val_metrics.csv      â€” per-epoch val mAP
      test_results.json    â€” final test-set metrics
      test_report.txt      â€” human-readable test table

Usage
-----
  # Run with defaults
  python run_distillation.py

  # Shrink backbone by half in both axes, compare with/without distillation
  BACKBONE_WIDTH_MULT = 0.5
  BACKBONE_DEPTH_MULT = 0.5
  USE_DISTILLATION    = True   # then rerun with False

  # Programmatic
  from run_distillation import run_distillation
  student = run_distillation(backbone_width_mult=0.5, backbone_depth_mult=0.5)
"""

from __future__ import annotations

import io
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast

# ---------------------------------------------------------------------------
# Project root on sys.path so this file runs from any working directory
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reuse all low-level building blocks from existing modules
from edgeturkeynet.dataset import get_train_loader, get_val_loader, get_test_loader
from edgeturkeynet.distill import (
    DistillationLoss,
    FeatureAdapters,
    StudentEdgeTurkeyNet,
    _KDCfgProxy,
    _StudentPrunerShim,
    _teacher_neck_features,
    build_student,
    load_frozen_teacher,
)
from edgeturkeynet.evaluate import evaluate_map, PerClassMetrics
from edgeturkeynet.model import (
    ChannelPruner,
    EdgeTurkeyNet,
    NUM_CLASSES,
    PANLiteNeck,
)
from edgeturkeynet.train import get_lr, set_seed


# ===========================================================================
# CONFIGURATION â€” edit these globals to change any aspect of the run.
# No argparse is used; all behaviour is controlled from here.
# ===========================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Teacher
# ---------------------------------------------------------------------------
# Path to a trained EdgeTurkeyNet checkpoint (output of main.py or train.py).
TEACHER_CHECKPOINT: Path = Path("outputs/runs/20260309_154046_shufflenetv2/checkpoints/best.pth")

# The teacher's backbone name â€” must match what the checkpoint was trained with.
TEACHER_BACKBONE: str = "shufflenetv2"

# ---------------------------------------------------------------------------
# Student backbone
# ---------------------------------------------------------------------------
# One of: 'shufflenetv2', 'mobilenetv3'
# 'shufflenetv2' â†’ ShuffleNetV2-0.5x backbone  (pretrained=False, fixed arch)
# 'mobilenetv3'  â†’ MobileNetV3-Small backbone   (pretrained=False)
STUDENT_BACKBONE: str = "shufflenetv2"

# Backbone scaling â€” ShuffleNetV2 only
# These two multipliers shrink the ShuffleNetV2 backbone itself.
# They are ignored when STUDENT_BACKBONE='mobilenetv3' or 'mobilenetv1'.
#
# BACKBONE_WIDTH_MULT â€” scales stage output channel counts.
#   Applied to the x0.5 reference channels [48, 96, 192].
#   1.0  â†’  [48, 96, 192]  (x0.5 reference, default)
#   0.5  â†’  [24, 48,  96]  (half-width backbone)
#   0.25 â†’  [12, 24,  48]  (quarter-width backbone)
#
# BACKBONE_DEPTH_MULT â€” scales the number of InvertedResidual blocks per stage.
#   Applied to the x0.5 reference depths [4, 8, 4] per stage.
#   1.0  â†’  [4, 8, 4]  (reference)
#   0.5  â†’  [2, 4, 2]  (half depth)
#   0.25 â†’  [1, 2, 1]  (quarter depth, minimum 1 per stage)
BACKBONE_WIDTH_MULT: float = 0.1
BACKBONE_DEPTH_MULT: float = 0.1

# ---------------------------------------------------------------------------
# Distillation loss weights
# ---------------------------------------------------------------------------
KD_W_GT: float = 1.0   # Ground-truth EdgeTurkeyLoss weight
KD_W_FD: float = 0.5   # Feature distillation MSE weight
KD_W_RD: float = 1.0   # Response distillation (KL + MSE) weight

# Softmax temperature for classification KL divergence (higher = softer targets)
KD_TEMPERATURE: float = 4.0

# EdgeTurkeyLoss component weights (GT branch only)
KD_LAMBDA_CLS: float = 1.0
KD_LAMBDA_REG: float = 2.0
KD_LAMBDA_CTR: float = 0.5

# ---------------------------------------------------------------------------
# Training schedule
# ---------------------------------------------------------------------------
KD_EPOCHS:              int   = 1000
KD_EARLY_STOP_PATIENCE: int   = 300
KD_BASE_LR:             float = 1e-3
KD_MIN_LR:              float = 1e-5
KD_WEIGHT_DECAY:        float = 5e-4
KD_WARMUP_EPOCHS:       int   = 5
KD_GRADIENT_CLIP:       float = 10.0

# ---------------------------------------------------------------------------
# DataLoader settings
# ---------------------------------------------------------------------------
KD_BATCH_SIZE:    int = 16
KD_NUM_WORKERS:   int = 8
TEST_BATCH_SIZE:  int = 16
TEST_NUM_WORKERS: int = 8

# ---------------------------------------------------------------------------
# Evaluation thresholds
# ---------------------------------------------------------------------------
KD_IOU_THRESHOLD:   float = 0.50
KD_SCORE_THRESHOLD: float = 0.30
KD_INPUT_SIZE: Tuple[int, int] = (640, 640)

# ---------------------------------------------------------------------------
# Student pruning during KD training
# ---------------------------------------------------------------------------
KD_PRUNE_STUDENT:      bool  = False
KD_PRUNE_START_EPOCH:  int   = 20
KD_PRUNE_INTERVAL:     int   = 10
KD_PRUNE_PER_CALL:     float = 0.15
KD_PRUNE_MAX_SPARSITY: float = 0.50

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
KD_SEED: int = 42

# ---------------------------------------------------------------------------
# Output root
# ---------------------------------------------------------------------------
# All results are written under:  KD_OUTPUT_ROOT / <backbone>_w<width_mult> /
KD_OUTPUT_ROOT: Path = Path("outputs/distillation")

# ---------------------------------------------------------------------------
# Mode flag
# ---------------------------------------------------------------------------
# True  â†’ full knowledge distillation (teacher + FD + RD + GT losses).
#          Requires TEACHER_CHECKPOINT to point to a valid .pth file.
# False â†’ train the scaled student from scratch with GT loss only.
#          Teacher checkpoint is not loaded; no teacher GPU memory used.
#          Use this to measure the baseline improvement from distillation.
USE_DISTILLATION: bool = True


# ===========================================================================
# FOLDER HIERARCHY
# ===========================================================================

def _make_run_dir(
    root: Path,
    backbone: str,
    backbone_width_mult: float,
    backbone_depth_mult: float,
    distillation: bool,
) -> Path:
    """
    Create and return the run directory for this configuration.

    Layout
    ------
    KD_OUTPUT_ROOT/
      <backbone>_bw<bwm>_bd<bdm>_kd/        (distillation=True)
      <backbone>_bw<bwm>_bd<bdm>_scratch/   (distillation=False)
        checkpoints/

    bw = backbone_width_mult, bd = backbone_depth_mult.
    """
    mode = "kd" if distillation else "scratch"
    tag  = (
        f"{backbone}"
        f"_bw{backbone_width_mult:.2f}"
        f"_bd{backbone_depth_mult:.2f}"
        f"_{mode}"
    ).replace(".", "p")
    run_dir = root / tag
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    print(f"[KD] Run directory: {run_dir}")
    return run_dir


# ===========================================================================
# CSV LOGGING
# ===========================================================================

def _init_csv(run_dir: Path, distillation: bool = True) -> Tuple[Path, Path]:
    """Initialise train_metrics.csv and val_metrics.csv with mode-correct headers."""
    train_csv = run_dir / "train_metrics.csv"
    val_csv   = run_dir / "val_metrics.csv"

    with open(train_csv, "w") as f:
        if distillation:
            f.write("epoch,lr,total,gt,fd,rd_cls,rd_reg,rd_ctr\n")
        else:
            f.write("epoch,lr,total,cls,reg,ctr\n")

    with open(val_csv, "w") as f:
        f.write("epoch,val_mAP,val_ap_body,val_ap_neck\n")

    return train_csv, val_csv


def _log_train(
    csv_path: Path,
    epoch: int,
    lr: float,
    losses: Dict[str, float],
) -> None:
    with open(csv_path, "a") as f:
        f.write(
            f"{epoch},{lr:.8f},"
            f"{losses['total']:.6f},{losses['gt']:.6f},{losses['fd']:.6f},"
            f"{losses['rd_cls']:.6f},{losses['rd_reg']:.6f},{losses['rd_ctr']:.6f}\n"
        )


def _log_val(
    csv_path: Path,
    epoch: int,
    metrics: PerClassMetrics,
) -> None:
    with open(csv_path, "a") as f:
        f.write(
            f"{epoch},{metrics.map:.6f},"
            f"{metrics.ap.get('body', 0.0):.6f},"
            f"{metrics.ap.get('neck', 0.0):.6f}\n"
        )


# ===========================================================================
# LR SCHEDULE (standalone â€” no RunConfig dependency)
# ===========================================================================

def _cosine_lr(epoch: int, total_epochs: int, base_lr: float, min_lr: float,
               warmup_epochs: int) -> float:
    """Linear warmup then cosine decay."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    cosine   = 0.5 * (1.0 + np.cos(np.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


# ===========================================================================
# KNOWLEDGE DISTILLATION TRAINER (standalone)
# ===========================================================================

class StandaloneKDTrainer:
    """
    Self-contained knowledge distillation trainer.

    Wraps everything needed for a complete KD run with zero dependency
    on RunConfig or RunLogger.  All settings come from module-level globals
    and constructor arguments.

    Args:
        teacher_checkpoint: Path to frozen teacher .pth.
        teacher_backbone:   Backbone name stored in that checkpoint.
        student_backbone:   Backbone for the student ('shufflenetv2' | 'mobilenetv3').
        width_mult:         Student neck + head channel multiplier.
        run_dir:            Directory for checkpoints and CSV logs.
    """

    def __init__(
        self,
        teacher_checkpoint:  Path,
        teacher_backbone:    str,
        student_backbone:    str,
        backbone_width_mult: float,
        backbone_depth_mult: float,
        run_dir:             Path,
    ) -> None:
        set_seed(KD_SEED)

        self.run_dir             = run_dir
        self.best_map            = 0.0
        self.no_improve          = 0
        self.backbone_width_mult = backbone_width_mult
        self.backbone_depth_mult = backbone_depth_mult
        self.student_backbone    = student_backbone

        # â”€â”€ Teacher (frozen) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # load_frozen_teacher from distill.py requires a RunConfig-like object
        # only for its backbone fallback â€” use a tiny proxy.
        class _CfgProxy:
            backbone = teacher_backbone

        self.teacher = load_frozen_teacher(
            Path(teacher_checkpoint), _CfgProxy()
        )

        # â”€â”€ Student â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.student = build_student(
            backbone_name       = student_backbone,
            width_mult          = 0.5,   # neck + head fixed at half
            depth_mult          = 0.5,   # head towers fixed at half depth
            backbone_width_mult = backbone_width_mult,
            backbone_depth_mult = backbone_depth_mult,
            num_classes         = NUM_CLASSES,
            input_size          = KD_INPUT_SIZE,
        ).to(DEVICE)

        print(
            f"[KD] Student backbone={student_backbone}  "
            f"backbone_width={backbone_width_mult}  "
            f"backbone_depth={backbone_depth_mult}  "
            f"params={self.student.get_parameter_count():,}"
        )
        print(
            f"[KD] Teacher params={sum(p.numel() for p in self.teacher.parameters()):,}"
            f"  [FROZEN]"
        )

        # â”€â”€ Feature adapters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        teacher_neck_chs = list(PANLiteNeck.NECK_CHANNELS.values())  # [128, 96, 64]
        student_neck_chs = self.student.neck.out_channels
        self.adapters = FeatureAdapters(
            student_neck_chs, teacher_neck_chs
        ).to(DEVICE)

        # â”€â”€ Loss â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.criterion = DistillationLoss(
            num_classes = NUM_CLASSES,
            temperature = KD_TEMPERATURE,
            w_gt        = KD_W_GT,
            w_fd        = KD_W_FD,
            w_rd        = KD_W_RD,
            lambda_cls  = KD_LAMBDA_CLS,
            lambda_reg  = KD_LAMBDA_REG,
            lambda_ctr  = KD_LAMBDA_CTR,
            input_size  = KD_INPUT_SIZE,
        )

        # â”€â”€ Optimiser â€” backbone at 0.1Ã— LR, rest at full LR â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        backbone_params = list(self.student.backbone.parameters())
        backbone_ids    = {id(p) for p in backbone_params}
        other_params    = (
            [p for p in self.student.parameters() if id(p) not in backbone_ids]
            + list(self.adapters.parameters())
        )

        self.optimizer = optim.AdamW(
            [
                {"params": backbone_params, "lr": KD_BASE_LR * 0.1},
                {"params": other_params,    "lr": KD_BASE_LR},
            ],
            weight_decay=KD_WEIGHT_DECAY,
        )

        self.scaler = GradScaler('cuda',enabled=torch.cuda.is_available())

        # â”€â”€ DataLoaders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.train_loader = get_train_loader(KD_BATCH_SIZE, KD_NUM_WORKERS)
        self.val_loader   = get_val_loader(KD_BATCH_SIZE,   KD_NUM_WORKERS)

        # â”€â”€ Pruner (optional) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if KD_PRUNE_STUDENT:
            self.pruner = ChannelPruner(
                _StudentPrunerShim(self.student),
                per_call_ratio = KD_PRUNE_PER_CALL,
                max_sparsity   = KD_PRUNE_MAX_SPARSITY,
            )
        else:
            self.pruner = None

        # â”€â”€ CSV paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.train_csv, self.val_csv = _init_csv(run_dir, distillation=True)

    # -----------------------------------------------------------------------
    # Training epoch
    # -----------------------------------------------------------------------

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """
        One full distillation epoch.

        Teacher forward runs inside torch.no_grad() â€” no activations stored.
        Student + adapters forward runs with autocast for AMP.
        """
        self.student.train()
        self.adapters.train()

        lr = _cosine_lr(epoch, KD_EPOCHS, KD_BASE_LR, KD_MIN_LR, KD_WARMUP_EPOCHS)
        self.optimizer.param_groups[0]["lr"] = lr * 0.1   # backbone
        self.optimizer.param_groups[1]["lr"] = lr

        totals: Dict[str, float] = {
            "total": 0.0, "gt": 0.0, "fd": 0.0,
            "rd_cls": 0.0, "rd_reg": 0.0, "rd_ctr": 0.0,
        }
        n   = 0
        t0  = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            images     = batch["images"].to(DEVICE, non_blocking=True)
            gt_boxes   = batch["boxes"]
            gt_cls_ids = batch["class_ids"]

            # â”€â”€ Teacher (no grad, no stored activations) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            with torch.no_grad():
                t_feats, t_cls, t_reg, t_ctr = _teacher_neck_features(
                    self.teacher, images
                )

            # â”€â”€ Student forward + loss â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            self.optimizer.zero_grad(set_to_none=True)

            with autocast('cuda',enabled=torch.cuda.is_available()):
                s_feats, s_cls, s_reg, s_ctr = (
                    self.student.forward_with_features(images)
                )
                s_feats_adapted = self.adapters(s_feats)

                losses = self.criterion(
                    student_feats_adapted = s_feats_adapted,
                    student_cls  = s_cls,
                    student_reg  = s_reg,
                    student_ctr  = s_ctr,
                    teacher_feats = t_feats,
                    teacher_cls  = t_cls,
                    teacher_reg  = t_reg,
                    teacher_ctr  = t_ctr,
                    gt_boxes_batch     = gt_boxes,
                    gt_class_ids_batch = gt_cls_ids,
                )

            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                list(self.student.parameters()) + list(self.adapters.parameters()),
                KD_GRADIENT_CLIP,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for k in totals:
                totals[k] += losses[k].item()
            n += 1

            if (batch_idx + 1) % 20 == 0:
                print(
                    f"  [KD E{epoch:03d}] Batch {batch_idx+1}/"
                    f"{len(self.train_loader)} | "
                    f"loss={losses['total'].item():.4f}  "
                    f"gt={losses['gt'].item():.4f}  "
                    f"fd={losses['fd'].item():.4f}  "
                    f"rd_cls={losses['rd_cls'].item():.4f} | "
                    f"LR={lr:.6f} | {time.time()-t0:.1f}s"
                )

        return {k: v / max(1, n) for k, v in totals.items()}

    # -----------------------------------------------------------------------
    # Pruning
    # -----------------------------------------------------------------------

    def _maybe_prune(self, epoch: int) -> None:
        if self.pruner is None:
            return
        if epoch < KD_PRUNE_START_EPOCH:
            return
        if (epoch - KD_PRUNE_START_EPOCH) % KD_PRUNE_INTERVAL != 0:
            return
        print(f"\n[KD] â”€â”€ Periodic student pruning at epoch {epoch} â”€â”€")
        self.pruner.prune()

    # -----------------------------------------------------------------------
    # Main training loop
    # -----------------------------------------------------------------------

    def train(self) -> StudentEdgeTurkeyNet:
        """
        Run the full knowledge distillation training loop.

        Per epoch:
          1. Train one distillation epoch (FD + RD + GT).
          2. Optional periodic student pruning.
          3. Validate student mAP on val set.
          4. Append row to train_metrics.csv and val_metrics.csv.
          5. Save best/last checkpoints.
          6. Early stopping check.

        Returns:
            StudentEdgeTurkeyNet with best checkpoint loaded, on DEVICE,
            in eval mode.
        """
        best_ckpt = self.run_dir / "checkpoints" / "student_best.pth"
        last_ckpt = self.run_dir / "checkpoints" / "student_last.pth"

        print(f"\n{'='*65}")
        print(
            f"  Knowledge Distillation Training\n"
            f"  Teacher backbone     : {TEACHER_BACKBONE}\n"
            f"  Student backbone     : {self.student_backbone}  "
            f"bw={self.backbone_width_mult}  bd={self.backbone_depth_mult}\n"
            f"  Epochs               : {KD_EPOCHS}  "
            f"patience={KD_EARLY_STOP_PATIENCE}\n"
            f"  Device               : {DEVICE}\n"
            f"  Run dir              : {self.run_dir}"
        )
        print(f"{'='*65}\n")

        for epoch in range(KD_EPOCHS):

            # â”€â”€ 1. Train â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            train_losses = self._train_one_epoch(epoch)
            lr_now = _cosine_lr(
                epoch, KD_EPOCHS, KD_BASE_LR, KD_MIN_LR, KD_WARMUP_EPOCHS
            )

            print(
                f"\n[KD E{epoch:03d}] Train | "
                f"total={train_losses['total']:.4f}  "
                f"gt={train_losses['gt']:.4f}  "
                f"fd={train_losses['fd']:.4f}  "
                f"rd_cls={train_losses['rd_cls']:.4f}  "
                f"rd_reg={train_losses['rd_reg']:.4f}  "
                f"rd_ctr={train_losses['rd_ctr']:.4f}"
            )

            # â”€â”€ 2. Optional pruning â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            self._maybe_prune(epoch)

            # â”€â”€ 3. Validate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            val_metrics = evaluate_map(
                self.student, self.val_loader, DEVICE,
                iou_threshold   = KD_IOU_THRESHOLD,
                score_threshold = KD_SCORE_THRESHOLD,
            )
            val_metrics.print_table(iou_threshold=KD_IOU_THRESHOLD)

            # â”€â”€ 4. CSV logs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            _log_train(self.train_csv, epoch, lr_now, train_losses)
            _log_val(self.val_csv, epoch, val_metrics)

            # â”€â”€ 5. Checkpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            self._save(last_ckpt, epoch)

            if val_metrics.map > self.best_map:
                self.best_map   = val_metrics.map
                self.no_improve = 0
                self._save(best_ckpt, epoch)
                print(
                    f"  âœ“ New best student mAP@{KD_IOU_THRESHOLD:.2f}: "
                    f"{self.best_map:.4f}"
                )
            else:
                self.no_improve += 1
                print(
                    f"  No improvement "
                    f"{self.no_improve}/{KD_EARLY_STOP_PATIENCE}"
                )
                if self.no_improve >= KD_EARLY_STOP_PATIENCE:
                    print(f"\n[KD] Early stopping at epoch {epoch}.")
                    break

        if self.pruner is not None:
            self.pruner.report_sparsity()

        print(
            f"\n[KD] Training complete.  "
            f"Best student mAP@{KD_IOU_THRESHOLD:.2f} = {self.best_map:.4f}"
        )

        # Load best checkpoint back onto DEVICE for immediate test use
        return self._load_student(best_ckpt, device=DEVICE)

    # -----------------------------------------------------------------------
    # Test evaluation
    # -----------------------------------------------------------------------

    def test(self, student: StudentEdgeTurkeyNet) -> PerClassMetrics:
        """
        Evaluate the student on the held-out test set.

        Creates a fresh test DataLoader, runs evaluate_map, and writes
        test_results.json + test_report.txt into the run directory.

        Args:
            student: StudentEdgeTurkeyNet already on DEVICE in eval mode.

        Returns:
            PerClassMetrics with test-set AP, precision, recall, mAP.
        """
        print(
            f"\n[KD] â”€â”€ Test Evaluation â”€â”€"
            f"  (batch={TEST_BATCH_SIZE}, workers={TEST_NUM_WORKERS})"
        )
        test_loader = get_test_loader(TEST_BATCH_SIZE, TEST_NUM_WORKERS)

        test_metrics = evaluate_map(
            student, test_loader, DEVICE,
            iou_threshold   = KD_IOU_THRESHOLD,
            score_threshold = KD_SCORE_THRESHOLD,
        )

        print("[KD] Test results:")
        test_metrics.print_table(iou_threshold=KD_IOU_THRESHOLD)

        self._save_test_results(test_metrics)
        return test_metrics

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _save(self, path: Path, epoch: int) -> None:
        torch.save({
            "epoch":                epoch,
            "model_state":          self.student.state_dict(),
            "adapter_state":        self.adapters.state_dict(),
            "optimizer_state":      self.optimizer.state_dict(),
            "best_map":             self.best_map,
            "backbone":             self.student_backbone,
            "backbone_width_mult":  self.backbone_width_mult,
            "backbone_depth_mult":  self.backbone_depth_mult,
        }, path)

    def _load_student(
        self,
        path: Path,
        device: torch.device = DEVICE,
    ) -> StudentEdgeTurkeyNet:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        s = StudentEdgeTurkeyNet(
            backbone_name       = ckpt.get("backbone",            self.student_backbone),
            num_classes         = NUM_CLASSES,
            width_mult          = 0.5,
            depth_mult          = 0.5,
            backbone_width_mult = ckpt.get("backbone_width_mult", self.backbone_width_mult),
            backbone_depth_mult = ckpt.get("backbone_depth_mult", self.backbone_depth_mult),
            input_size          = KD_INPUT_SIZE,
        )
        s.load_state_dict(ckpt["model_state"])
        s.to(device).eval()
        print(
            f"[KD] Loaded student: {path.name}  "
            f"backbone={ckpt.get('backbone', self.student_backbone)}  "
            f"backbone_width={ckpt.get('backbone_width_mult', self.backbone_width_mult):.2f}  "
            f"backbone_depth={ckpt.get('backbone_depth_mult', self.backbone_depth_mult):.2f}  "
            f"mAP={ckpt.get('best_map', 0.0):.4f}"
        )
        return s

    def _save_test_results(self, metrics: PerClassMetrics) -> None:
        """Write test_results.json and test_report.txt to run_dir."""
        data = {
            "student_backbone":    self.student_backbone,
            "teacher_backbone":    TEACHER_BACKBONE,
            "backbone_width_mult": self.backbone_width_mult,
            "backbone_depth_mult": self.backbone_depth_mult,
            "iou_threshold":       KD_IOU_THRESHOLD,
            "score_threshold":  KD_SCORE_THRESHOLD,
            "map":              round(metrics.map, 6),
            "ap":               {k: round(v, 6) for k, v in metrics.ap.items()},
            "precision":        {k: round(v, 6) for k, v in metrics.precision.items()},
            "recall":           {k: round(v, 6) for k, v in metrics.recall.items()},
        }

        json_path = self.run_dir / "test_results.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[KD] test_results.json â†’ {json_path}")

        buf = io.StringIO()
        with redirect_stdout(buf):
            metrics.print_table(iou_threshold=KD_IOU_THRESHOLD)

        txt_path = self.run_dir / "test_report.txt"
        with open(txt_path, "w") as f:
            f.write(f"Student backbone     : {self.student_backbone}\n")
            f.write(f"Teacher backbone     : {TEACHER_BACKBONE}\n")
            f.write(f"Backbone width mult  : {self.backbone_width_mult}\n")
            f.write(f"Backbone depth mult  : {self.backbone_depth_mult}\n")
            f.write(f"IoU threshold        : {KD_IOU_THRESHOLD}\n\n")
            f.write(buf.getvalue())
        print(f"[KD] test_report.txt  â†’ {txt_path}")


def _log_train_scratch(
    csv_path: Path,
    epoch: int,
    lr: float,
    losses: Dict[str, float],
) -> None:
    with open(csv_path, "a") as f:
        f.write(
            f"{epoch},{lr:.8f},"
            f"{losses['total']:.6f},{losses['cls']:.6f},"
            f"{losses['reg']:.6f},{losses['ctr']:.6f}\n"
        )


# ===========================================================================
# SCRATCH TRAINER â€” scaled student, GT loss only, no teacher
# ===========================================================================

class StandaloneScratchTrainer:
    """
    Trains the scaled StudentEdgeTurkeyNet from scratch using only the
    ground-truth EdgeTurkeyLoss (focal + CIoU + BCE-centerness).

    No teacher is loaded.  No feature adapters are created.  The training
    loop is structurally identical to StandaloneKDTrainer but uses the
    same EdgeTurkeyLoss that the full teacher was trained with, applied
    directly to the student's predictions.

    This produces the fair ablation baseline: the same scaled architecture
    trained without any distillation signal.  Comparing its test mAP
    against StandaloneKDTrainer's result isolates the distillation benefit.

    Args:
        student_backbone:    Backbone ('shufflenetv2' | 'mobilenetv3').
        backbone_width_mult: ShuffleNetV2 stage channel multiplier.
        backbone_depth_mult: ShuffleNetV2 stage block-depth multiplier.
        run_dir:             Directory for checkpoints and CSV logs.
    """

    def __init__(
        self,
        student_backbone:    str,
        backbone_width_mult: float,
        backbone_depth_mult: float,
        run_dir:             Path,
    ) -> None:
        from edgeturkeynet.loss import EdgeTurkeyLoss

        set_seed(KD_SEED)

        self.run_dir             = run_dir
        self.best_map            = 0.0
        self.no_improve          = 0
        self.backbone_width_mult = backbone_width_mult
        self.backbone_depth_mult = backbone_depth_mult
        self.student_backbone    = student_backbone

        # â”€â”€ Student â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.student = build_student(
            backbone_name       = student_backbone,
            width_mult          = 0.5,   # neck + head fixed at half
            depth_mult          = 0.5,   # head towers fixed at half depth
            backbone_width_mult = backbone_width_mult,
            backbone_depth_mult = backbone_depth_mult,
            num_classes         = NUM_CLASSES,
            input_size          = KD_INPUT_SIZE,
        ).to(DEVICE)

        print(
            f"[Scratch] Student backbone={student_backbone}  "
            f"backbone_width={backbone_width_mult}  "
            f"backbone_depth={backbone_depth_mult}  "
            f"params={self.student.get_parameter_count():,}  "
            f"[NO TEACHER]"
        )

        # â”€â”€ Loss â€” GT only â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.criterion = EdgeTurkeyLoss(
            num_classes = NUM_CLASSES,
            lambda_cls  = KD_LAMBDA_CLS,
            lambda_reg  = KD_LAMBDA_REG,
            lambda_ctr  = KD_LAMBDA_CTR,
            input_size  = KD_INPUT_SIZE,
        )

        # â”€â”€ Optimiser â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        backbone_params = list(self.student.backbone.parameters())
        backbone_ids    = {id(p) for p in backbone_params}
        head_params     = [p for p in self.student.parameters()
                           if id(p) not in backbone_ids]

        self.optimizer = optim.AdamW(
            [
                {"params": backbone_params, "lr": KD_BASE_LR * 0.1},
                {"params": head_params,     "lr": KD_BASE_LR},
            ],
            weight_decay=KD_WEIGHT_DECAY,
        )

        self.scaler = GradScaler('cuda', enabled=torch.cuda.is_available())

        # â”€â”€ DataLoaders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.train_loader = get_train_loader(KD_BATCH_SIZE, KD_NUM_WORKERS)
        self.val_loader   = get_val_loader(KD_BATCH_SIZE,   KD_NUM_WORKERS)

        # â”€â”€ Pruner (optional) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if KD_PRUNE_STUDENT:
            self.pruner = ChannelPruner(
                _StudentPrunerShim(self.student),
                per_call_ratio = KD_PRUNE_PER_CALL,
                max_sparsity   = KD_PRUNE_MAX_SPARSITY,
            )
        else:
            self.pruner = None

        # â”€â”€ CSV paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.train_csv, self.val_csv = _init_csv(run_dir, distillation=False)

    # -----------------------------------------------------------------------
    # Training epoch
    # -----------------------------------------------------------------------

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """One full GT-only training epoch â€” no teacher forward pass."""
        self.student.train()

        lr = _cosine_lr(epoch, KD_EPOCHS, KD_BASE_LR, KD_MIN_LR, KD_WARMUP_EPOCHS)
        self.optimizer.param_groups[0]["lr"] = lr * 0.1
        self.optimizer.param_groups[1]["lr"] = lr

        totals: Dict[str, float] = {
            "total": 0.0, "cls": 0.0, "reg": 0.0, "ctr": 0.0,
        }
        n  = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            images     = batch["images"].to(DEVICE, non_blocking=True)
            gt_boxes   = batch["boxes"]
            gt_cls_ids = batch["class_ids"]

            self.optimizer.zero_grad(set_to_none=True)

            with autocast('cuda',enabled=torch.cuda.is_available()):
                cls_p, reg_p, ctr_p = self.student(images)
                losses = self.criterion(
                    cls_p, reg_p, ctr_p, gt_boxes, gt_cls_ids,
                )

            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.student.parameters(), KD_GRADIENT_CLIP,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for k in totals:
                totals[k] += losses[k].item()
            n += 1

            if (batch_idx + 1) % 20 == 0:
                print(
                    f"  [Scratch E{epoch:03d}] Batch {batch_idx+1}/"
                    f"{len(self.train_loader)} | "
                    f"loss={losses['total'].item():.4f}  "
                    f"cls={losses['cls'].item():.4f}  "
                    f"reg={losses['reg'].item():.4f}  "
                    f"ctr={losses['ctr'].item():.4f} | "
                    f"LR={lr:.6f} | {time.time()-t0:.1f}s"
                )

        return {k: v / max(1, n) for k, v in totals.items()}

    # -----------------------------------------------------------------------
    # Pruning
    # -----------------------------------------------------------------------

    def _maybe_prune(self, epoch: int) -> None:
        if self.pruner is None:
            return
        if epoch < KD_PRUNE_START_EPOCH:
            return
        if (epoch - KD_PRUNE_START_EPOCH) % KD_PRUNE_INTERVAL != 0:
            return
        print(f"\n[Scratch] â”€â”€ Periodic student pruning at epoch {epoch} â”€â”€")
        self.pruner.prune()

    # -----------------------------------------------------------------------
    # Main training loop
    # -----------------------------------------------------------------------

    def train(self) -> StudentEdgeTurkeyNet:
        """
        Run the full scratch training loop.

        Same structure as StandaloneKDTrainer.train() â€” per-epoch train,
        prune, validate, log, checkpoint, early-stop â€” but with GT loss only.

        Returns:
            StudentEdgeTurkeyNet with best checkpoint loaded, on DEVICE,
            in eval mode.
        """
        best_ckpt = self.run_dir / "checkpoints" / "student_best.pth"
        last_ckpt = self.run_dir / "checkpoints" / "student_last.pth"

        print(f"\n{'='*65}")
        print(
            f"  Scaled Student â€” Scratch Training (no distillation)\n"
            f"  Student backbone : {self.student_backbone}  "
            f"backbone_width={self.backbone_width_mult}  "
            f"backbone_depth={self.backbone_depth_mult}\n"
            f"  Epochs           : {KD_EPOCHS}  "
            f"patience={KD_EARLY_STOP_PATIENCE}\n"
            f"  Device           : {DEVICE}\n"
            f"  Run dir          : {self.run_dir}"
        )
        print(f"{'='*65}\n")

        for epoch in range(KD_EPOCHS):

            # â”€â”€ 1. Train â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            train_losses = self._train_one_epoch(epoch)
            lr_now = _cosine_lr(
                epoch, KD_EPOCHS, KD_BASE_LR, KD_MIN_LR, KD_WARMUP_EPOCHS
            )

            print(
                f"\n[Scratch E{epoch:03d}] Train | "
                f"total={train_losses['total']:.4f}  "
                f"cls={train_losses['cls']:.4f}  "
                f"reg={train_losses['reg']:.4f}  "
                f"ctr={train_losses['ctr']:.4f}"
            )

            # â”€â”€ 2. Optional pruning â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            self._maybe_prune(epoch)

            # â”€â”€ 3. Validate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            val_metrics = evaluate_map(
                self.student, self.val_loader, DEVICE,
                iou_threshold   = KD_IOU_THRESHOLD,
                score_threshold = KD_SCORE_THRESHOLD,
            )
            val_metrics.print_table(iou_threshold=KD_IOU_THRESHOLD)

            # â”€â”€ 4. CSV logs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            _log_train_scratch(self.train_csv, epoch, lr_now, train_losses)
            _log_val(self.val_csv, epoch, val_metrics)

            # â”€â”€ 5. Checkpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            self._save(last_ckpt, epoch)

            if val_metrics.map > self.best_map:
                self.best_map   = val_metrics.map
                self.no_improve = 0
                self._save(best_ckpt, epoch)
                print(
                    f"  âœ“ New best scratch mAP@{KD_IOU_THRESHOLD:.2f}: "
                    f"{self.best_map:.4f}"
                )
            else:
                self.no_improve += 1
                print(
                    f"  No improvement "
                    f"{self.no_improve}/{KD_EARLY_STOP_PATIENCE}"
                )
                if self.no_improve >= KD_EARLY_STOP_PATIENCE:
                    print(f"\n[Scratch] Early stopping at epoch {epoch}.")
                    break

        if self.pruner is not None:
            self.pruner.report_sparsity()

        print(
            f"\n[Scratch] Training complete.  "
            f"Best student mAP@{KD_IOU_THRESHOLD:.2f} = {self.best_map:.4f}"
        )

        return self._load_student(best_ckpt, device=DEVICE)

    # -----------------------------------------------------------------------
    # Test evaluation  (shared logic â€” identical to StandaloneKDTrainer.test)
    # -----------------------------------------------------------------------

    def test(self, student: StudentEdgeTurkeyNet) -> PerClassMetrics:
        """Evaluate student on the held-out test set, write result files."""
        print(
            f"\n[Scratch] â”€â”€ Test Evaluation â”€â”€"
            f"  (batch={TEST_BATCH_SIZE}, workers={TEST_NUM_WORKERS})"
        )
        test_loader  = get_test_loader(TEST_BATCH_SIZE, TEST_NUM_WORKERS)
        test_metrics = evaluate_map(
            student, test_loader, DEVICE,
            iou_threshold   = KD_IOU_THRESHOLD,
            score_threshold = KD_SCORE_THRESHOLD,
        )
        print("[Scratch] Test results:")
        test_metrics.print_table(iou_threshold=KD_IOU_THRESHOLD)
        self._save_test_results(test_metrics)
        return test_metrics

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _save(self, path: Path, epoch: int) -> None:
        torch.save({
            "epoch":                epoch,
            "model_state":          self.student.state_dict(),
            "optimizer_state":      self.optimizer.state_dict(),
            "best_map":             self.best_map,
            "backbone":             self.student_backbone,
            "backbone_width_mult":  self.backbone_width_mult,
            "backbone_depth_mult":  self.backbone_depth_mult,
            "mode":                 "scratch",
        }, path)

    def _load_student(
        self,
        path: Path,
        device: torch.device = DEVICE,
    ) -> StudentEdgeTurkeyNet:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        s = StudentEdgeTurkeyNet(
            backbone_name       = ckpt.get("backbone",            self.student_backbone),
            num_classes         = NUM_CLASSES,
            width_mult          = 0.5,
            depth_mult          = 0.5,
            backbone_width_mult = ckpt.get("backbone_width_mult", self.backbone_width_mult),
            backbone_depth_mult = ckpt.get("backbone_depth_mult", self.backbone_depth_mult),
            input_size          = KD_INPUT_SIZE,
        )
        s.load_state_dict(ckpt["model_state"])
        s.to(device).eval()
        print(
            f"[Scratch] Loaded student: {path.name}  "
            f"backbone={ckpt.get('backbone', self.student_backbone)}  "
            f"backbone_width={ckpt.get('backbone_width_mult', self.backbone_width_mult):.2f}  "
            f"backbone_depth={ckpt.get('backbone_depth_mult', self.backbone_depth_mult):.2f}  "
            f"mAP={ckpt.get('best_map', 0.0):.4f}"
        )
        return s

    def _save_test_results(self, metrics: PerClassMetrics) -> None:
        data = {
            "mode":                "scratch",
            "student_backbone":    self.student_backbone,
            "backbone_width_mult": self.backbone_width_mult,
            "backbone_depth_mult": self.backbone_depth_mult,
            "iou_threshold":       KD_IOU_THRESHOLD,
            "score_threshold":     KD_SCORE_THRESHOLD,
            "map":                 round(metrics.map, 6),
            "ap":                  {k: round(v, 6) for k, v in metrics.ap.items()},
            "precision":           {k: round(v, 6) for k, v in metrics.precision.items()},
            "recall":              {k: round(v, 6) for k, v in metrics.recall.items()},
        }
        json_path = self.run_dir / "test_results.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Scratch] test_results.json â†’ {json_path}")

        buf = io.StringIO()
        with redirect_stdout(buf):
            metrics.print_table(iou_threshold=KD_IOU_THRESHOLD)

        txt_path = self.run_dir / "test_report.txt"
        with open(txt_path, "w") as f:
            f.write(f"Mode                 : scratch (no distillation)\n")
            f.write(f"Student backbone     : {self.student_backbone}\n")
            f.write(f"Backbone width mult  : {self.backbone_width_mult}\n")
            f.write(f"Backbone depth mult  : {self.backbone_depth_mult}\n")
            f.write(f"IoU threshold        : {KD_IOU_THRESHOLD}\n\n")
            f.write(buf.getvalue())
        print(f"[Scratch] test_report.txt  â†’ {txt_path}")


# ===========================================================================
# DATASET VERIFICATION
# ===========================================================================

def _verify_dataset() -> bool:
    """Check that all dataset split directories exist before training begins."""
    from dataset import (
        TRAIN_IMAGES, TRAIN_LABELS,
        VAL_IMAGES,   VAL_LABELS,
        TEST_IMAGES,  TEST_LABELS,
    )
    splits = {
        "train images": TRAIN_IMAGES, "train labels": TRAIN_LABELS,
        "val images":   VAL_IMAGES,   "val labels":   VAL_LABELS,
        "test images":  TEST_IMAGES,  "test labels":  TEST_LABELS,
    }
    print("\n[KD] Verifying dataset ...")
    all_ok = True
    for name, path in splits.items():
        ok = path.exists()
        print(f"  {'âœ“' if ok else 'âœ— MISSING':<12} {name:<14}  {path}")
        if not ok:
            all_ok = False
    if all_ok:
        from edgeturkeynet.dataset import TRAIN_IMAGES, VAL_IMAGES, TEST_IMAGES
        print(
            f"\n  train={len(list(TRAIN_IMAGES.glob('*')))}  "
            f"val={len(list(VAL_IMAGES.glob('*')))}  "
            f"test={len(list(TEST_IMAGES.glob('*')))} images"
        )
    return all_ok


def _verify_teacher(path: Path) -> bool:
    """Confirm the teacher checkpoint file exists."""
    if not path.exists():
        print(f"\n[KD] âœ— Teacher checkpoint not found: {path}")
        print(
            "     Train a full EdgeTurkeyNet first with main.py, "
            "then set TEACHER_CHECKPOINT to its best.pth path."
        )
        return False
    print(f"[KD] âœ“ Teacher checkpoint: {path}")
    return True


# ===========================================================================
# MAIN RUN FUNCTION
# ===========================================================================

def run_distillation(
    use_distillation:    bool  = USE_DISTILLATION,
    teacher_checkpoint:  Path  = TEACHER_CHECKPOINT,
    teacher_backbone:    str   = TEACHER_BACKBONE,
    student_backbone:    str   = STUDENT_BACKBONE,
    backbone_width_mult: float = BACKBONE_WIDTH_MULT,
    backbone_depth_mult: float = BACKBONE_DEPTH_MULT,
    output_root:         Path  = KD_OUTPUT_ROOT,
) -> Optional[StudentEdgeTurkeyNet]:
    """
    Unified student training pipeline â€” distillation or scratch, same interface.

    Controlled by ``use_distillation`` (default: USE_DISTILLATION global):

      True  â†’ StandaloneKDTrainer
                Teacher loaded and frozen.
                Loss = w_gtÂ·L_gt + w_fdÂ·L_fd + w_rdÂ·(L_cls_kl + L_reg + L_ctr)
                Outputs â†’ <output_root>/<backbone>_bw<bwm>_bd<bdm>_kd/

      False â†’ StandaloneScratchTrainer
                No teacher.  No adapters.  No extra GPU memory.
                Loss = EdgeTurkeyLoss (focal + CIoU + BCE-ctr) only.
                Outputs â†’ <output_root>/<backbone>_bw<bwm>_bd<bdm>_scratch/

    Both modes run the same two phases:
      Phase 1 â€” Training  : KD_EPOCHS epochs, early stopping, periodic pruning,
                            per-epoch CSV logging, best/last checkpoints.
      Phase 2 â€” Testing   : evaluate_map on held-out test set,
                            test_results.json + test_report.txt.

    Backbone scaling (ShuffleNetV2 only):
      backbone_width_mult â€” scales stage channel counts relative to x0.5 reference.
                            1.0 â†’ [48,96,192]  |  0.5 â†’ [24,48,96]  |  0.25 â†’ [12,24,48]
      backbone_depth_mult â€” scales InvertedResidual blocks per stage.
                            1.0 â†’ [4,8,4]  |  0.5 â†’ [2,4,2]  |  0.25 â†’ [1,2,1]

    Args:
        use_distillation:    True = KD mode, False = scratch mode.
        teacher_checkpoint:  Path to frozen teacher .pth (KD mode only).
        teacher_backbone:    Backbone name matching that checkpoint (KD mode only).
        student_backbone:    'shufflenetv2' | 'mobilenetv3'.
        backbone_width_mult: ShuffleNetV2 channel multiplier (ignored for other backbones).
        backbone_depth_mult: ShuffleNetV2 depth multiplier (ignored for other backbones).
        output_root:         Root directory; mode suffix appended automatically.

    Returns:
        Trained StudentEdgeTurkeyNet on DEVICE in eval mode.
    """
    output_root = Path(output_root)
    mode_label  = "Knowledge Distillation" if use_distillation else "Scratch (no distillation)"

    # â”€â”€ Banner â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if use_distillation:
        print("""
â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
â•‘          EdgeTurkeyNet â€” Knowledge Distillation Runner       â•‘
â• â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•£
â•‘  Teacher  : full-size EdgeTurkeyNet (frozen)                 â•‘
â•‘  Student  : ScaledShuffleNetV2 backbone (width Ã— depth)      â•‘
â•‘  Losses   : FD (MSE) + RD (KL + MSE) + GT (CIoU + focal)    â•‘
â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•""")
    else:
        print("""
â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
â•‘       EdgeTurkeyNet â€” Scaled Student Scratch Training        â•‘
â• â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•£
â•‘  Teacher  : none                                             â•‘
â•‘  Student  : ScaledShuffleNetV2 backbone (width Ã— depth)      â•‘
â•‘  Loss     : EdgeTurkeyLoss â€” GT only (CIoU + focal + BCE)    â•‘
â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•""")

    print(f"\n  Mode                 : {mode_label}")
    print(f"  Device               : {DEVICE}")
    print(f"  Student backbone     : {student_backbone}")
    print(f"  Backbone width mult  : {backbone_width_mult}")
    print(f"  Backbone depth mult  : {backbone_depth_mult}")
    if use_distillation:
        print(f"  Teacher backbone     : {teacher_backbone}")
        print(f"  Teacher ckpt         : {teacher_checkpoint}")
    print(f"  Output root          : {output_root}")

    # â”€â”€ 1. Verify dataset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if not _verify_dataset():
        print("\n[run] Aborting: dataset incomplete.")
        sys.exit(1)

    # â”€â”€ 2. Verify teacher (KD mode only) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if use_distillation and not _verify_teacher(Path(teacher_checkpoint)):
        sys.exit(1)

    # â”€â”€ 3. Create run directory â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    run_dir = _make_run_dir(
        output_root, student_backbone,
        backbone_width_mult, backbone_depth_mult,
        distillation=use_distillation,
    )

    # â”€â”€ 4. Training phase â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n" + "=" * 65)
    print(f"  PHASE 1 â€” TRAINING  [{mode_label.upper()}]")
    print("=" * 65)

    if use_distillation:
        trainer = StandaloneKDTrainer(
            teacher_checkpoint  = Path(teacher_checkpoint),
            teacher_backbone    = teacher_backbone,
            student_backbone    = student_backbone,
            backbone_width_mult = backbone_width_mult,
            backbone_depth_mult = backbone_depth_mult,
            run_dir             = run_dir,
        )
    else:
        trainer = StandaloneScratchTrainer(
            student_backbone    = student_backbone,
            backbone_width_mult = backbone_width_mult,
            backbone_depth_mult = backbone_depth_mult,
            run_dir             = run_dir,
        )

    student = trainer.train()   # returns best model on DEVICE, eval mode

    # â”€â”€ 5. Testing phase â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n" + "=" * 65)
    print("  PHASE 2 â€” TEST EVALUATION")
    print("=" * 65)

    test_metrics = trainer.test(student)

    # â”€â”€ Final summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n" + "=" * 65)
    print(f"  {mode_label} â€” Complete")
    print("=" * 65)
    print(
        f"  Student  {student_backbone:<14}  "
        f"bw={backbone_width_mult}  bd={backbone_depth_mult}  "
        f"mode={'kd' if use_distillation else 'scratch'}  "
        f"mAP={test_metrics.map:.4f}  "
        f"body={test_metrics.ap.get('body', 0):.4f}  "
        f"neck={test_metrics.ap.get('neck', 0):.4f}"
    )
    print(f"\n  Outputs â†’ {run_dir}/")
    print("=" * 65)

    return student

