from pathlib import Path

# =========================
# MODEL CONFIGURATION
# =========================
# Options: "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"
MODEL_SIZE = "yolo11n"

# =========================
# TRAINING PARAMETERS
# =========================
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16

# =========================
# INFERENCE PARAMETERS
# =========================
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5

# =========================
# DATASET PATHS (YOLO YAML)
# =========================
DATA_YAML = Path("data/dataset_split/data.yaml")

# =========================
# OUTPUT
# =========================
PROJECT_DIR = Path("outputs/runs/yolo11_experiment")
RUN_NAME = "baseline"
