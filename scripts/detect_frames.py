"""
Run ShuffleNetV2 EdgeTurkeyNet on data/raw/frame/90 frames.

Saves annotated images + detections.csv to outputs/predictions/frame_90/.
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from edgeturkeynet.model import EdgeTurkeyNet, CLASS_NAMES
from edgeturkeynet.dataset import letterbox, INPUT_SIZE
from edgeturkeynet.evaluate import predict, SCORE_THRESHOLD, NMS_IOU_THRESHOLD

CHECKPOINT  = _ROOT / "outputs/runs/20260309_154046_shufflenetv2/checkpoints/best.pth"
FRAMES_DIR  = _ROOT / "data/raw/frame/90"
OUTPUT_DIR  = _ROOT / "outputs/predictions/frame_90"

# body=green, neck=orange
CLASS_COLORS = {0: (0, 200, 0), 1: (0, 140, 255)}


def load_model(ckpt_path: Path) -> EdgeTurkeyNet:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backbone = ck.get("backbone", "shufflenetv2")
    model = EdgeTurkeyNet(num_classes=2, pretrained_backbone=False, backbone=backbone)
    model.load_state_dict(ck["model_state"])
    model.eval()
    print(f"Loaded {backbone} checkpoint (epoch {ck.get('epoch', '?')}, "
          f"best_mAP={ck.get('best_map', 0.0):.4f})")
    return model


def preprocess(image_path: Path):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise IOError(f"Cannot read {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]
    img_lb, scale, padding = letterbox(img_rgb, INPUT_SIZE)
    tensor = torch.from_numpy(
        img_lb.astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor, scale, padding, (orig_h, orig_w)


def unletterbox(boxes_lb: np.ndarray, scale: float,
                padding: tuple, orig_shape: tuple) -> np.ndarray:
    if len(boxes_lb) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    pad_top, pad_left = padding
    b = boxes_lb.copy().astype(np.float32)
    b[:, 0] -= pad_left
    b[:, 1] -= pad_top
    b[:, 2] -= pad_left
    b[:, 3] -= pad_top
    b /= scale
    orig_h, orig_w = orig_shape
    b[:, 0::2] = np.clip(b[:, 0::2], 0, orig_w)
    b[:, 1::2] = np.clip(b[:, 1::2], 0, orig_h)
    return b


def draw_boxes(img_bgr: np.ndarray, boxes: np.ndarray,
               class_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    out = img_bgr.copy()
    for box, cls_id, score in zip(boxes, class_ids, scores):
        x1, y1, x2, y2 = box.astype(int)
        color = CLASS_COLORS.get(int(cls_id), (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES[int(cls_id)]} {score:.2f}"
        cv2.putText(out, label, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model(CHECKPOINT)

    image_paths = sorted(
        p for p in FRAMES_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not image_paths:
        print(f"No images in {FRAMES_DIR}")
        return

    csv_path = OUTPUT_DIR / "detections.csv"
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame", "class_id", "class_name", "score",
                         "x1", "y1", "x2", "y2"])

        total_dets = 0
        with torch.inference_mode():
            for img_path in image_paths:
                tensor, scale, padding, orig_shape = preprocess(img_path)

                results = predict(
                    model, tensor,
                    score_threshold=SCORE_THRESHOLD,
                    nms_threshold=NMS_IOU_THRESHOLD,
                    input_size=INPUT_SIZE,
                )
                boxes_lb, class_ids, scores = results[0]

                boxes_orig = unletterbox(
                    boxes_lb.numpy(), scale, padding, orig_shape
                )
                class_ids_np = class_ids.numpy()
                scores_np    = scores.numpy()

                img_bgr = cv2.imread(str(img_path))
                annotated = draw_boxes(img_bgr, boxes_orig, class_ids_np, scores_np)
                cv2.imwrite(str(OUTPUT_DIR / img_path.name), annotated)

                for box, cls_id, score in zip(boxes_orig, class_ids_np, scores_np):
                    x1, y1, x2, y2 = box
                    writer.writerow([
                        img_path.name, int(cls_id), CLASS_NAMES[int(cls_id)],
                        f"{score:.4f}", f"{x1:.1f}", f"{y1:.1f}",
                        f"{x2:.1f}", f"{y2:.1f}",
                    ])

                total_dets += len(boxes_orig)
                print(f"  {img_path.name}: {len(boxes_orig)} det(s)")

    print(f"\nDone. {len(image_paths)} frames, {total_dets} total detections.")
    print(f"Output: {OUTPUT_DIR}")
    print(f"CSV:    {csv_path}")


if __name__ == "__main__":
    main()
