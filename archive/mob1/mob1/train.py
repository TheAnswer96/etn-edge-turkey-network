# =============================================================================
# train.py
# Fixes:
#   - Teacher output unpacking (ultralytics returns tuple of 2, not 3)
#   - Teacher cls logits extracted from raw head output correctly
# Efficiency upgrades:
#   - AMP (automatic mixed precision) — 1.5-2x faster on CUDA
#   - Gradient accumulation — larger effective batch without more VRAM
#   - EMA (exponential moving average) — better generalisation, free
#   - OneCycleLR — faster convergence than warmup+exponential
#   - Validation frequency control — run eval every N epochs, not every epoch
#   - Backbone/head split LR — head learns faster, backbone stays stable
#   - torch.compile (PyTorch 2.x) — ~20% speedup, opt-in via config
# =============================================================================

import os
import json
import time
import copy
import random
import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR

from mob1.config import (
    EPOCHS, LEARNING_RATE, WEIGHT_DECAY, SEED,
    QAT_START_EPOCH, CHECKPOINTS_DIR, LOGS_DIR, RUNS_DIR,
    TEACHER_WEIGHTS, DISTILL_ALPHA,
)
from mob1.head import NanoDetTurkey
from mob1.dataloader import get_dataloader
from mob1.loss import DetectionLoss
from mob1.evaluate import evaluate, print_metrics

# Optional config keys with safe defaults
import mob1.config as _cfg
GRAD_ACCUM_STEPS = getattr(_cfg, 'GRAD_ACCUM_STEPS', 2)
VAL_EVERY        = getattr(_cfg, 'VAL_EVERY',        5)
USE_COMPILE      = getattr(_cfg, 'USE_COMPILE',      False)
EMA_DECAY        = getattr(_cfg, 'EMA_DECAY',        0.9998)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # fixed input size → safe to enable


# ---------------------------------------------------------------------------
# EMA — shadow model updated every optimiser step.
# Use ema.eval_model() for validation; it generalises better than raw weights.
# ---------------------------------------------------------------------------

class ModelEMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay  = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(m.data, alpha=1.0 - self.decay)
        for s, m in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(m)

    def eval_model(self):
        return self.shadow


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(model, ema, optimizer, scheduler, epoch, metrics, path):
    torch.save({
        "epoch":      epoch,
        "state_dict": model.state_dict(),
        "ema":        ema.shadow.state_dict() if ema else None,
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict() if scheduler else None,
        "metrics":    metrics,
    }, path)


def load_checkpoint(model, ema, optimizer, scheduler, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    if ema and ckpt.get("ema"):
        ema.shadow.load_state_dict(ckpt["ema"])
    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception:
            pass  # non-fatal on resume
    return ckpt["epoch"], ckpt.get("metrics", {})


# ---------------------------------------------------------------------------
# Teacher — ultralytics YOLO.model.forward() in inference mode returns:
#   (pred_tensor, feature_list)   — a 2-tuple, NOT 3.
# pred_tensor shape: (B, 4 + num_classes, num_anchors)
#   first 4 channels = box, remaining = cls logits
# ---------------------------------------------------------------------------

def get_teacher(weights_path, device):
    try:
        from ultralytics import YOLO
        yolo    = YOLO(weights_path)
        teacher = yolo.model.eval().to(device)
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"  Teacher loaded: {weights_path}")
        return teacher
    except Exception as e:
        print(f"  Teacher load failed ({e}). Distillation disabled.")
        return None


def teacher_cls_logits(teacher, imgs):
    """
    Returns (B, N_student_anchors, C) cls logits interpolated to match
    the student's anchor count so KL-div shapes always align.
    """
    out  = teacher(imgs)                          # 2-tuple
    pred = out[0] if isinstance(out, (tuple, list)) else out
    # pred: (B, 4+C, N_teacher)
    cls  = pred[:, 4:, :].permute(0, 2, 1)       # (B, N_teacher, C)
    return cls


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train(resume_from: str = None):
    set_seed()
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,        exist_ok=True)
    os.makedirs(RUNS_DIR,        exist_ok=True)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device={device}  AMP={use_amp}  "
          f"GradAccum={GRAD_ACCUM_STEPS}  ValEvery={VAL_EVERY}")

    # ---- model ----
    model = NanoDetTurkey().to(device)
    if USE_COMPILE and hasattr(torch, "compile"):
        model = torch.compile(model)
        print("  torch.compile() applied")

    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    ema = ModelEMA(model, decay=EMA_DECAY)

    # ---- data ----
    train_loader = get_dataloader("train", augment=True)
    val_loader   = get_dataloader("val",   augment=False)
    steps_per_epoch = len(train_loader)

    # ---- loss ----
    criterion = DetectionLoss(grid=model.grid, stride=model.stride)

    # ---- optimizer: split LR — backbone slower, head faster ----
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    backbone_params = [p for p in model.parameters() if id(p) in backbone_ids]
    head_params     = [p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": LEARNING_RATE * 0.1},
        {"params": head_params,     "lr": LEARNING_RATE},
    ], weight_decay=WEIGHT_DECAY)

    # ---- OneCycleLR — warmup+cosine in one schedule, no manual warmup needed ----
    effective_steps = (EPOCHS * steps_per_epoch + GRAD_ACCUM_STEPS - 1) // GRAD_ACCUM_STEPS
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[LEARNING_RATE * 0.1, LEARNING_RATE],
        total_steps=effective_steps,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=25,
        final_div_factor=1e4,
    )

    # ---- AMP scaler ----
    scaler = GradScaler('cuda', enabled=use_amp)

    # ---- optional teacher ----
    teacher = None
    if TEACHER_WEIGHTS:
        teacher = get_teacher(TEACHER_WEIGHTS, device)

    # ---- resume ----
    start_epoch = 0
    best_map50  = 0.0
    log_history = []

    if resume_from and os.path.exists(resume_from):
        start_epoch, prev = load_checkpoint(
            model, ema, optimizer, scheduler, resume_from, device)
        best_map50 = prev.get("mAP50", 0.0)
        print(f"  Resumed epoch {start_epoch}  best_mAP50={best_map50:.4f}")

    # ---- training loop ----
    opt_step = 0   # counts optimiser (not batch) steps for scheduler

    for epoch in range(start_epoch, EPOCHS):
        t0 = time.time()

        # --- QAT switch ---
        if epoch == QAT_START_EPOCH:
            print(f"\n[Epoch {epoch+1}] Enabling QAT...")
            model.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")
            torch.ao.quantization.prepare_qat(model, inplace=True)

        model.train()
        running   = {"total": 0., "cls": 0., "reg": 0., "dfl": 0., "fg": 0.}
        n_batches = 0
        optimizer.zero_grad()

        for batch_idx, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            is_accum_step = (batch_idx + 1) % GRAD_ACCUM_STEPS == 0 or \
                            (batch_idx + 1) == steps_per_epoch

            with autocast('cuda', enabled=use_amp):
                cls_pred, reg_pred, boxes = model(imgs)
                loss, bd = criterion(cls_pred, reg_pred, boxes, targets)

                if teacher is not None:
                    with torch.no_grad():
                        t_cls = teacher_cls_logits(teacher, imgs)
                    # Align anchor counts if teacher/student differ
                    N_s, N_t = cls_pred.shape[1], t_cls.shape[1]
                    if N_s != N_t:
                        t_cls = torch.nn.functional.interpolate(
                            t_cls.permute(0, 2, 1).unsqueeze(-1),
                            size=(N_s, 1), mode="nearest",
                        ).squeeze(-1).permute(0, 2, 1)
                    kd = torch.nn.functional.kl_div(
                        torch.nn.functional.log_softmax(cls_pred, dim=-1),
                        torch.nn.functional.softmax(t_cls, dim=-1),
                        reduction="batchmean",
                    )
                    loss = (1.0 - DISTILL_ALPHA) * loss + DISTILL_ALPHA * kd

                loss_scaled = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss_scaled).backward()

            if is_accum_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if opt_step < effective_steps:
                    scheduler.step()
                opt_step += 1
                ema.update(model)

            running["total"] += loss.item()
            running["cls"]   += bd["cls"]
            running["reg"]   += bd["reg"]
            running["dfl"]   += bd["dfl"]
            running["fg"]    += bd["num_fg"]
            n_batches        += 1

        avg    = {k: v / n_batches for k, v in running.items()}
        cur_lr = optimizer.param_groups[1]["lr"]
        print(
            f"[{epoch+1:>3}/{EPOCHS}]  "
            f"loss={avg['total']:.4f} "
            f"cls={avg['cls']:.3f} reg={avg['reg']:.3f} "
            f"dfl={avg['dfl']:.3f} fg={avg['fg']:.1f}  "
            f"lr={cur_lr:.2e}  {time.time()-t0:.0f}s"
        )

        # ---- validation every VAL_EVERY epochs + always on last ----
        run_val = ((epoch + 1) % VAL_EVERY == 0) or (epoch + 1 == EPOCHS)
        metrics = {}

        if run_val:
            metrics = evaluate(ema.eval_model(), val_loader, device)
            map50   = metrics["mAP50"]
            print_metrics(metrics, prefix="val")

            last_path = os.path.join(CHECKPOINTS_DIR, "last.pt")
            save_checkpoint(model, ema, optimizer, scheduler,
                            epoch + 1, metrics, last_path)

            if map50 > best_map50:
                best_map50 = map50
                save_checkpoint(model, ema, optimizer, scheduler, epoch + 1,
                                metrics, os.path.join(CHECKPOINTS_DIR, "best.pt"))
                print(f"  ✓ best mAP50={best_map50:.4f}")
        else:
            save_checkpoint(model, ema, optimizer, scheduler,
                            epoch + 1, {}, os.path.join(CHECKPOINTS_DIR, "last.pt"))

        log_history.append({
            "epoch":   epoch + 1,
            "loss":    {k: round(v, 5) for k, v in avg.items()},
            "metrics": metrics,
            "lr":      round(cur_lr, 8),
            "time_s":  round(time.time() - t0, 1),
        })
        with open(os.path.join(LOGS_DIR, "history.json"), "w") as f:
            json.dump(log_history, f, indent=2)

    # ---- finalize QAT ----
    if EPOCHS > QAT_START_EPOCH:
        try:
            torch.ao.quantization.convert(model, inplace=True)
            qat_path = os.path.join(CHECKPOINTS_DIR, "best_qat.pt")
            torch.save(model.state_dict(), qat_path)
            print(f"QAT model → {qat_path}")
        except Exception as e:
            print(f"QAT convert skipped: {e}")

    print(f"\nDone.  Best mAP50={best_map50:.4f}")
    return best_map50