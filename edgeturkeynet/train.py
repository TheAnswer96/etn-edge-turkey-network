"""
Training loop for EdgeTurkeyNet — parametric, run-isolated variant.

All hyperparameters come from a ``RunConfig`` instance (see config.py).
All artefacts (checkpoints, CSV logs) are written to the run-scoped
directories managed by ``RunLogger`` (see logger.py).

Features
--------
- Backbone selection: mobilenetv3 / shufflenetv2 / mobilenetv1
- Mixed Precision Training (AMP) — halves GPU memory on CUDA
- Cosine LR schedule with linear warmup
- Per-epoch CSV logging of losses and per-class val metrics
- Periodic progressive channel pruning
- Early stopping on validation mAP@0.5
- Gradient clipping for stability
- Fully reproducible via seed in RunConfig
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING
import io

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

from .dataset import get_train_loader, get_val_loader
from .evaluate import evaluate_map, PerClassMetrics
from .loss import EdgeTurkeyLoss
from .model import ChannelPruner, EdgeTurkeyNet, NUM_CLASSES

if TYPE_CHECKING:
    from .config import RunConfig
    from .logger import RunLogger


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(
    epoch: int,
    cfg: "RunConfig",
) -> float:
    """
    Compute learning rate for the current epoch.

    Linear warmup for ``cfg.warmup_epochs`` epochs, then cosine decay
    from ``cfg.base_lr`` down to ``cfg.min_lr`` over the remainder.

    Args:
        epoch: Current epoch index (0-based).
        cfg:   RunConfig with lr / schedule parameters.

    Returns:
        Learning rate float.
    """
    if epoch < cfg.warmup_epochs:
        return cfg.base_lr * (epoch + 1) / max(1, cfg.warmup_epochs)
    progress = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
    cosine   = 0.5 * (1.0 + np.cos(np.pi * progress))
    return cfg.min_lr + (cfg.base_lr - cfg.min_lr) * cosine


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    End-to-end training manager for EdgeTurkeyNet.

    All configuration is read from ``RunConfig``; all outputs are written
    to run-scoped directories via ``RunLogger``.

    Backbone selection
    ------------------
    The backbone is chosen by ``cfg.backbone`` and passed to
    ``EdgeTurkeyNet(backbone=cfg.backbone)``.  Available options:
      mobilenetv3  — MobileNetV3-Small, pretrained (default, best accuracy)
      shufflenetv2 — ShuffleNetV2-0.5x, pretrained (fastest on ARM)
      mobilenetv1  — MobileNetV1-1.0, scratch (simplest, ReLU6 quant-friendly)

    Periodic pruning schedule
    -------------------------
    A single ``ChannelPruner`` is created once and called every
    ``cfg.prune_interval`` epochs after epoch ``cfg.prune_start_epoch``.
    Each call zeroes ``cfg.prune_per_call`` of remaining active channels,
    compounding toward the ``cfg.prune_max_sparsity`` ceiling.

    Args:
        cfg:        Full pipeline configuration.
        logger:     Run-scoped logger for CSV metrics and checkpoint paths.
        resume_from: Optional explicit checkpoint path (overrides logger paths).
    """

    def __init__(
        self,
        cfg: "RunConfig",
        logger: "RunLogger",
        resume_from: Optional[Path] = None,
    ) -> None:
        self.cfg    = cfg
        self.logger = logger

        set_seed(cfg.seed)

        # ── Model ──────────────────────────────────────────────────────────
        self.model = EdgeTurkeyNet(
            num_classes=NUM_CLASSES,
            pretrained_backbone=cfg.pretrained,
            input_size=cfg.input_size,
            backbone=cfg.backbone,
        ).to(DEVICE)

        print(
            f"[Trainer] EdgeTurkeyNet  backbone={cfg.backbone}  "
            f"classes={NUM_CLASSES}  params={self.model.get_parameter_count():,}  "
            f"device={DEVICE}"
        )

        # ── Loss ───────────────────────────────────────────────────────────
        self.criterion = EdgeTurkeyLoss(
            num_classes=NUM_CLASSES,
            lambda_cls=cfg.lambda_cls,
            lambda_reg=cfg.lambda_reg,
            lambda_ctr=cfg.lambda_ctr,
            input_size=cfg.input_size,
        )

        # ── Optimiser — differential LR for backbone vs head ───────────────
        backbone_params = list(self.model.backbone.parameters())
        backbone_ids    = {id(p) for p in backbone_params}
        head_params     = [p for p in self.model.parameters()
                           if id(p) not in backbone_ids]

        self.optimizer = optim.AdamW(
            [
                {"params": backbone_params, "lr": cfg.base_lr * 0.1},
                {"params": head_params,     "lr": cfg.base_lr},
            ],
            weight_decay=cfg.weight_decay,
        )

        # ── AMP scaler ─────────────────────────────────────────────────────
        self.scaler = GradScaler('cuda', enabled=torch.cuda.is_available())

        # ── Data loaders ───────────────────────────────────────────────────
        self.train_loader = get_train_loader(
            batch_size=cfg.batch_size, num_workers=cfg.num_workers
        )
        self.val_loader = get_val_loader(
            batch_size=cfg.batch_size, num_workers=cfg.num_workers
        )

        # ── Pruner ─────────────────────────────────────────────────────────
        self.pruner = ChannelPruner(
            self.model,
            per_call_ratio=cfg.prune_per_call,
            max_sparsity=cfg.prune_max_sparsity,
        )

        # ── State ──────────────────────────────────────────────────────────
        self.start_epoch    = 0
        self.best_map       = 0.0
        self.no_improve_cnt = 0

        # ── Resume ─────────────────────────────────────────────────────────
        ckpt_path = resume_from or cfg.resume_from
        if ckpt_path and Path(ckpt_path).exists():
            self._load_checkpoint(Path(ckpt_path))

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.start_epoch    = ckpt.get("epoch", 0) + 1
        self.best_map       = ckpt.get("best_map", 0.0)
        self.no_improve_cnt = ckpt.get("no_improve_cnt", 0)
        print(
            f"[Trainer] Resumed from epoch {self.start_epoch - 1}, "
            f"best mAP={self.best_map:.4f}"
        )

    def _save_checkpoint(self, path: Path, epoch: int) -> None:
        torch.save({
            "epoch":           epoch,
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_map":        self.best_map,
            "no_improve_cnt":  self.no_improve_cnt,
            "backbone":        self.cfg.backbone,
        }, path)

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Run one full training epoch.

        Returns:
            Dict of mean loss components: total, cls, reg, ctr.
        """
        self.model.train()

        lr = get_lr(epoch, self.cfg)
        self.optimizer.param_groups[0]["lr"] = lr * 0.1   # backbone
        self.optimizer.param_groups[1]["lr"] = lr

        totals: Dict[str, float] = {"total": 0.0, "cls": 0.0, "reg": 0.0, "ctr": 0.0}
        n_batches = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            images     = batch["images"].to(DEVICE, non_blocking=True)
            gt_boxes   = batch["boxes"]
            gt_cls_ids = batch["class_ids"]

            self.optimizer.zero_grad(set_to_none=True)

            with autocast('cuda',enabled=torch.cuda.is_available()):
                cls_preds, reg_preds, ctr_preds = self.model(images)
                losses = self.criterion(
                    cls_preds, reg_preds, ctr_preds,
                    gt_boxes, gt_cls_ids,
                )

            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.gradient_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for k in totals:
                totals[k] += losses[k].item()
            n_batches += 1

            if (batch_idx + 1) % 20 == 0:
                elapsed = time.time() - t0
                print(
                    f"  [E{epoch:03d}] Batch {batch_idx+1}/{len(self.train_loader)} | "
                    f"Loss={losses['total'].item():.4f} | LR={lr:.6f} | "
                    f"t={elapsed:.1f}s"
                )

        return {k: v / max(1, n_batches) for k, v in totals.items()}

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def _maybe_prune(self, epoch: int) -> None:
        """Fire one pruning round if the periodic schedule triggers."""
        cfg = self.cfg
        if epoch < cfg.prune_start_epoch:
            return
        if (epoch - cfg.prune_start_epoch) % cfg.prune_interval != 0:
            return
        print(f"\n[Trainer] ── Periodic pruning at epoch {epoch} ──")
        self.pruner.prune()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self) -> EdgeTurkeyNet:
        """
        Run the full training loop.

        Per epoch:
          1. Train one epoch (AMP, grad clipping)
          2. Optional periodic channel pruning
          3. Validate — per-class mAP@IoU
          4. Log epoch row to CSV via RunLogger
          5. Save best / last checkpoints into run directory
          6. Early stopping check

        Returns:
            The trained model (best weights loaded).
        """
        cfg = self.cfg
        print(f"\n{'='*65}")
        print(
            f"  EdgeTurkeyNet Training  |  backbone={cfg.backbone}  "
            f"classes={NUM_CLASSES}  epochs={cfg.epochs}  device={DEVICE}"
        )
        print(
            f"  Pruning: every {cfg.prune_interval} epochs after epoch "
            f"{cfg.prune_start_epoch}, per_call={cfg.prune_per_call*100:.0f}%, "
            f"ceiling={cfg.prune_max_sparsity*100:.0f}%"
        )
        print(f"  Run dir: {self.logger.run_dir}")
        print(f"{'='*65}\n")

        for epoch in range(self.start_epoch, cfg.epochs):

            # ── 1. Train ──────────────────────────────────────────────
            train_losses = self._train_one_epoch(epoch)
            lr_now = get_lr(epoch, cfg)
            print(
                f"\n[E{epoch:03d}] Train | "
                f"total={train_losses['total']:.4f}  "
                f"cls={train_losses['cls']:.4f}  "
                f"reg={train_losses['reg']:.4f}  "
                f"ctr={train_losses['ctr']:.4f}"
            )

            # ── 2. Periodic pruning ───────────────────────────────────
            self._maybe_prune(epoch)

            # ── 3. Validate ───────────────────────────────────────────
            val_metrics: PerClassMetrics = evaluate_map(
                self.model, self.val_loader, DEVICE,
                iou_threshold=cfg.iou_threshold,
                score_threshold=cfg.score_threshold, input_size=cfg.input_size
            )
            val_metrics.print_table(iou_threshold=cfg.iou_threshold)

            # ── 4. CSV logging ────────────────────────────────────────
            self.logger.log_epoch(epoch, lr_now, train_losses, val_metrics)

            # ── 5. Checkpoints ────────────────────────────────────────
            self._save_checkpoint(self.logger.last_model_path, epoch)

            if val_metrics.map > self.best_map:
                self.best_map       = val_metrics.map
                self.no_improve_cnt = 0
                self._save_checkpoint(self.logger.best_model_path, epoch)
                print(f"  ✓ New best mAP@{cfg.iou_threshold:.2f}: {self.best_map:.4f}")
            else:
                self.no_improve_cnt += 1
                print(
                    f"  No improvement "
                    f"{self.no_improve_cnt}/{cfg.early_stop_patience}"
                )
                if self.no_improve_cnt >= cfg.early_stop_patience:
                    print(f"\n[Trainer] Early stopping at epoch {epoch}.")
                    break

        # End-of-training reports
        self.pruner.report_sparsity()
        print(f"\n[Trainer] Done.  Best mAP@{cfg.iou_threshold:.2f} = {self.best_map:.4f}")
        print(f"[Trainer] Best checkpoint: {self.logger.best_model_path}")

        return self.load_best_model()

    # ------------------------------------------------------------------
    # Load best model
    # ------------------------------------------------------------------

    def load_best_model(self) -> EdgeTurkeyNet:
        """
        Load the best checkpoint saved during this run.

        Returns:
            EdgeTurkeyNet in eval mode on CPU.
        """
        best_path = self.logger.best_model_path
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        backbone = ckpt.get("backbone", self.cfg.backbone)
        model = EdgeTurkeyNet(
            num_classes=NUM_CLASSES,
            pretrained_backbone=False,
            backbone=backbone,
            input_size=self.cfg.input_size
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print(
            f"[Checkpoint] Loaded {best_path}  "
            f"(backbone={backbone}  mAP={ckpt.get('best_map', 0.0):.4f})"
        )
        return model


# ---------------------------------------------------------------------------
# Standalone model loader (used by main.py when --skip-train)
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(path: Path, cfg: "RunConfig") -> EdgeTurkeyNet:
    """
    Load a saved EdgeTurkeyNet checkpoint, restoring the correct backbone.

    Args:
        path: Path to the .pth checkpoint.
        cfg:  RunConfig (backbone name used as fallback if not in checkpoint).

    Returns:
        EdgeTurkeyNet in eval mode on CPU.
    """
    # buffer = io.BytesIO(path)
    print(path)
    ckpt     = torch.load(path, map_location="cpu", weights_only=False)
    backbone = ckpt.get("backbone", cfg.backbone)
    model    = EdgeTurkeyNet(
        num_classes=NUM_CLASSES,
        pretrained_backbone=False,
        backbone=backbone,
        input_size=cfg.input_size
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(
        f"[Checkpoint] Loaded {path}  "
        f"(backbone={backbone}  mAP={ckpt.get('best_map', 0.0):.4f})"
    )
    return model
