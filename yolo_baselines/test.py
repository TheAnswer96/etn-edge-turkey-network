from ultralytics import YOLO
from pathlib import Path
from .config import (
    MODEL_SIZE,
    DATA_YAML,
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    PROJECT_DIR,
    RUN_NAME,
)


def get_best_weights():
    weights_path = Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"
    if not weights_path.exists():
        raise FileNotFoundError("Best weights not found. Train the model first.")
    return weights_path


def test():

    weights = get_best_weights()
    model = YOLO(str(weights))

    metrics = model.val(
        data=str(DATA_YAML),
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        split="test",
    )
    return metrics
