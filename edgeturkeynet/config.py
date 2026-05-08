"""
EdgeTurkeyNet — Centralised Configuration.

All tunable parameters live here as plain module-level variables.
Edit this file to change any aspect of the pipeline, then run:

    python scripts/run_training.py

Groups
------
BACKBONE          — architecture selection
TRAINING          — epochs, batch size, learning rate, regularisation
PRUNING           — progressive channel sparsity schedule
INFERENCE         — confidence and IoU thresholds
LOSS WEIGHTS      — per-component loss coefficients
PIPELINE CONTROL  — toggle training / export / visualisation stages
PATHS             — run output root, dataset root, optional resume checkpoint
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


# ===========================================================================
# BACKBONE
# Choices: "mobilenetv3"  — MobileNetV3-Small, pretrained ImageNet (default)
#          "shufflenetv2" — ShuffleNetV2-0.5x, pretrained ImageNet (fastest ARM)
#          "mobilenetv1"  — MobileNetV1-1.0,   scratch only (ReLU6, quant-friendly)
# ===========================================================================
BACKBONE   = "shufflenetv2"
PRETRAINED = True           # Load pretrained ImageNet weights where available.
                            # Has no effect for mobilenetv1 (no torchvision weights).

# ===========================================================================
# TRAINING
# ===========================================================================
EPOCHS               = 100
BATCH_SIZE           = 16
NUM_WORKERS          = 8
BASE_LR              = 1e-3    # Peak LR reached after linear warmup.
MIN_LR               = 1e-5    # Floor of cosine decay schedule.
WEIGHT_DECAY         = 5e-4    # AdamW L2 regularisation.
WARMUP_EPOCHS        = 5       # Epochs for linear LR ramp-up from 0 → BASE_LR.
GRADIENT_CLIP        = 10.0    # Max gradient norm before clipping.
EARLY_STOP_PATIENCE  = 30      # Epochs without val mAP improvement before stopping.
SEED                 = 42      # Random seed (numpy, torch, python).

# ===========================================================================
# PRUNING  —  progressive cumulative channel sparsity
# Schedule: fire every PRUNE_INTERVAL epochs starting from PRUNE_START_EPOCH.
# Each call zeroes PRUNE_PER_CALL of each layer's remaining active channels.
# Cumulative sparsity after k calls = 1 - (1 - PRUNE_PER_CALL)^k
# Example (defaults): k=1→15%  k=2→27%  k=3→38%  k=4→47%
# ===========================================================================
PRUNE_START_EPOCH  = 100    # Do not prune before this epoch.
PRUNE_INTERVAL     = 20    # Epochs between pruning calls (20, 30, 40, …).
PRUNE_PER_CALL     = 0.40  # Fraction of remaining active channels zeroed per call.
PRUNE_MAX_SPARSITY = 0.99  # Hard ceiling — layers at or above this are skipped.

# ===========================================================================
# INFERENCE / EVALUATION
# ===========================================================================
SCORE_THRESHOLD = 0.30   # Minimum per-class score × centerness to keep a detection.
IOU_THRESHOLD   = 0.50   # IoU threshold for AP computation and DIoU-NMS suppression.
INPUT_SIZE      = (640, 640)   # Model input resolution (H, W).

# ===========================================================================
# LOSS WEIGHTS
# ===========================================================================
LAMBDA_CLS = 1.0   # Classification loss weight (focal loss).
LAMBDA_REG = 2.0   # Regression loss weight (GIoU loss).
LAMBDA_CTR = 0.5   # Centerness loss weight (BCE loss).

# ===========================================================================
# PIPELINE CONTROL
# Set any flag to True to skip that stage (useful for eval-only or fast runs).
# ===========================================================================
SKIP_TRAIN     = False   # Skip training; load checkpoint from RESUME_FROM.
SKIP_EXPORT    = False   # Skip ONNX + INT8 export stages.
SKIP_VISUALISE = False   # Skip test-set inference visualisation.
MAX_VIS_IMAGES = 20      # Maximum images rendered during visualisation.

# ===========================================================================
# PATHS
# ===========================================================================
RUNS_ROOT    = Path("outputs/runs")          # Parent directory for all run folders.
DATASET_ROOT = Path("data/dataset_split")    # Root of the YOLO-format dataset.
RESUME_FROM  = None                          # Path | None — explicit checkpoint to resume.
                                             # e.g. Path("outputs/runs/20260303_120000_mobilenetv3/checkpoints/best.pth")


# ---------------------------------------------------------------------------
# Backbone registry (used for validation only — do not edit)
# ---------------------------------------------------------------------------
SUPPORTED_BACKBONES = ["mobilenetv3", "shufflenetv2", "mobilenetv1"]


# ---------------------------------------------------------------------------
# RunConfig dataclass  —  consumed by Trainer, RunLogger, and main.py
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """
    Typed snapshot of the configuration for one pipeline run.

    Built automatically from the module-level globals above by
    ``get_config()``.  All downstream code (Trainer, RunLogger, main)
    receives a ``RunConfig`` instance — nothing reads the globals directly
    after startup, so the dataclass acts as a validated, immutable record
    of the run's settings.
    """

    # Backbone
    backbone:   str  = "mobilenetv3"
    pretrained: bool = True

    # Training
    epochs:              int   = 2
    batch_size:          int   = 8
    num_workers:         int   = 4
    base_lr:             float = 1e-3
    min_lr:              float = 1e-5
    weight_decay:        float = 5e-4
    warmup_epochs:       int   = 5
    gradient_clip:       float = 10.0
    early_stop_patience: int   = 20
    seed:                int   = 42

    # Pruning
    prune_start_epoch:  int   = 20
    prune_interval:     int   = 10
    prune_per_call:     float = 0.15
    prune_max_sparsity: float = 0.50

    # Inference / evaluation
    score_threshold: float            = 0.30
    iou_threshold:   float            = 0.50
    input_size:      Tuple[int, int]  = (640, 640)

    # Loss weights
    lambda_cls: float = 1.0
    lambda_reg: float = 2.0
    lambda_ctr: float = 0.5

    # Pipeline control
    skip_train:     bool = False
    skip_export:    bool = False
    skip_visualise: bool = False
    max_vis_images: int  = 20

    # Paths
    runs_root:    Path           = field(default_factory=lambda: Path("outputs/runs"))
    dataset_root: Path           = field(default_factory=lambda: Path("data/dataset_split"))
    resume_from:  Optional[Path] = None

    def __post_init__(self) -> None:
        if self.backbone not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unknown backbone '{self.backbone}'. "
                f"Choose from: {SUPPORTED_BACKBONES}"
            )
        self.runs_root    = Path(self.runs_root)
        self.dataset_root = Path(self.dataset_root)
        if self.resume_from is not None:
            self.resume_from = Path(self.resume_from)

    def summary(self) -> str:
        """Return a human-readable multi-line configuration summary."""
        sep = "-" * 46
        lines = [
            sep,
            "  EdgeTurkeyNet — Run Configuration",
            sep,
            f"  backbone:            {self.backbone}",
            f"  pretrained:          {self.pretrained}",
            f"  epochs:              {self.epochs}",
            f"  batch_size:          {self.batch_size}",
            f"  base_lr / min_lr:    {self.base_lr} / {self.min_lr}",
            f"  warmup_epochs:       {self.warmup_epochs}",
            f"  weight_decay:        {self.weight_decay}",
            f"  gradient_clip:       {self.gradient_clip}",
            f"  early_stop_patience: {self.early_stop_patience}",
            f"  seed:                {self.seed}",
            sep,
            f"  prune_start_epoch:   {self.prune_start_epoch}",
            f"  prune_interval:      {self.prune_interval}",
            f"  prune_per_call:      {self.prune_per_call}",
            f"  prune_max_sparsity:  {self.prune_max_sparsity}",
            sep,
            f"  score_threshold:     {self.score_threshold}",
            f"  iou_threshold:       {self.iou_threshold}",
            f"  input_size:          {self.input_size}",
            sep,
            f"  lambda_cls/reg/ctr:  {self.lambda_cls} / {self.lambda_reg} / {self.lambda_ctr}",
            sep,
            f"  skip_train:          {self.skip_train}",
            f"  skip_export:         {self.skip_export}",
            f"  skip_visualise:      {self.skip_visualise}",
            f"  runs_root:           {self.runs_root}",
            f"  resume_from:         {self.resume_from}",
            sep,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factory — call this once at startup to produce the validated RunConfig
# ---------------------------------------------------------------------------

def get_config() -> RunConfig:
    """
    Build and validate a RunConfig from the module-level globals above.

    This is the only function main.py needs to call.  All parameters
    come from the variables defined at the top of this file.

    Returns:
        Validated RunConfig instance.

    Raises:
        ValueError: If BACKBONE is not in SUPPORTED_BACKBONES.
    """
    return RunConfig(
        backbone              = BACKBONE,
        pretrained            = PRETRAINED,
        epochs                = EPOCHS,
        batch_size            = BATCH_SIZE,
        num_workers           = NUM_WORKERS,
        base_lr               = BASE_LR,
        min_lr                = MIN_LR,
        weight_decay          = WEIGHT_DECAY,
        warmup_epochs         = WARMUP_EPOCHS,
        gradient_clip         = GRADIENT_CLIP,
        early_stop_patience   = EARLY_STOP_PATIENCE,
        seed                  = SEED,
        prune_start_epoch     = PRUNE_START_EPOCH,
        prune_interval        = PRUNE_INTERVAL,
        prune_per_call        = PRUNE_PER_CALL,
        prune_max_sparsity    = PRUNE_MAX_SPARSITY,
        score_threshold       = SCORE_THRESHOLD,
        iou_threshold         = IOU_THRESHOLD,
        input_size            = INPUT_SIZE,
        lambda_cls            = LAMBDA_CLS,
        lambda_reg            = LAMBDA_REG,
        lambda_ctr            = LAMBDA_CTR,
        skip_train            = SKIP_TRAIN,
        skip_export           = SKIP_EXPORT,
        skip_visualise        = SKIP_VISUALISE,
        max_vis_images        = MAX_VIS_IMAGES,
        runs_root             = RUNS_ROOT,
        dataset_root          = DATASET_ROOT,
        resume_from           = RESUME_FROM,
    )
