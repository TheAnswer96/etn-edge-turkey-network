# =============================================================================
# config.py — Global configuration for NanoDetPlus-Turkey
# =============================================================================

import os

# --- Paths ---
ROOT_DIR        = os.path.dirname(os.path.pardir)
DATA_DIR        = os.path.join(ROOT_DIR, "src", "dataset_split")
TRAIN_DIR       = os.path.join(DATA_DIR, "train")
VAL_DIR         = os.path.join(DATA_DIR, "val")
TEST_DIR        = os.path.join(DATA_DIR, "test")

RUNS_DIR        = os.path.join(ROOT_DIR, "runs")
CHECKPOINTS_DIR = os.path.join(RUNS_DIR, "checkpoints")
LOGS_DIR        = os.path.join(RUNS_DIR, "logs")
RESULTS_DIR     = os.path.join(RUNS_DIR, "results")

# --- Dataset ---
NUM_CLASSES     = 2
CLASS_NAMES     = ["body", "neck"]

# --- Model ---
WIDTH_MULT      = 0.25      # MobileNetV1 width multiplier
STRIDE          = 16        # Single-scale feature map stride
REG_MAX         = 7         # GFL regression bins (box = reg_max * 4 values)
SHARED_CHANNELS = 64        # Channels in shared 1x1 conv before cls/reg heads

# --- Input ---
INPUT_SIZE      = 256       # Square input resolution

# --- Training ---
EPOCHS          = 150
BATCH_SIZE      = 32
NUM_WORKERS     = 4
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 5e-4
WARMUP_EPOCHS   = 5
LR_DECAY_GAMMA  = 0.97      # Exponential LR decay per epoch after warmup

# Loss weights
LAMBDA_CLS      = 1.0
LAMBDA_REG      = 2.0
LAMBDA_DFL      = 0.5       # Distribution Focal Loss weight

# Label assignment
# fg=870 with batch=32 on a 256-anchor grid = ~27 anchors/GT — far too many.
# Large top-view turkeys fill many grid cells so the in_gt box filter alone
# is not tight enough. Fix: cap TOPK + add center-radius spatial filter.
TOPK_CANDIDATES = 3         # hard cap: max fg anchors per GT (was 13)
CENTER_RADIUS   = 2.5       # only anchors within N strides of GT center pass

# Augmentation
MOSAIC_PROB     = 0.5
FLIP_PROB       = 0.5
HSV_H           = 0.015
HSV_S           = 0.7
HSV_V           = 0.4
DEGREES         = 5.0       # mild rotation (top-view)
TRANSLATE       = 0.1
SCALE           = 0.3

# --- Distillation (set TEACHER_WEIGHTS=None to disable) ---
# TEACHER_WEIGHTS = os.path.join(os.getcwd(), "runs", "yolo11_experiment", "baseline", "weights", "best.pt")
TEACHER_WEIGHTS = None

DISTILL_ALPHA   = 0.5       # weight of distillation loss vs hard loss

# --- Evaluation ---
IOU_THRESHOLD   = 0.5
CONF_THRESHOLD  = 0.25
NMS_IOU         = 0.45

# --- Quantization-aware training ---
QAT_START_EPOCH = 140       # Switch to QAT in final epochs (set > EPOCHS to skip)

# --- Training efficiency ---
GRAD_ACCUM_STEPS = 4       # accumulate N batches before optimiser step (effective batch = BATCH_SIZE * N)
VAL_EVERY        = 5       # run validation every N epochs (saves ~30% wall time)
EMA_DECAY        = 0.9998  # EMA shadow model decay; higher = smoother but slower to adapt
USE_COMPILE      = False   # torch.compile() — PyTorch 2.x only, ~20% speedup, set True if supported

# --- Seed ---
SEED            = 42