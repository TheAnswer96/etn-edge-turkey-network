from ultralytics import YOLO
from .config import (
    MODEL_SIZE,
    DATA_YAML,
    EPOCHS,
    IMG_SIZE,
    BATCH_SIZE,
    PROJECT_DIR,
    RUN_NAME,
)


def train():
    model = YOLO(f"{MODEL_SIZE}.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=True,
    )

    return model
