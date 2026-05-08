"""
Evaluate best.pt (YOLO11 segmentation, class 0 = turkey) on the test set.

For each predicted mask the tight axis-aligned bounding box is derived from
result.masks.xyn (normalised polygon points), then matched against GT class-0
bounding boxes.

Metrics: Precision, Recall, AP50 (VOC 2010+ all-points interpolation).

Usage:
    python scripts/evaluate_best.py
    python scripts/evaluate_best.py --conf 0.25 --iou 0.5
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2
import numpy as np

REPO        = Path(__file__).resolve().parent.parent
MODEL_PATH  = REPO / "best.pt"
TEST_IMAGES = REPO / "data" / "dataset_split" / "test" / "images"
TEST_LABELS = REPO / "data" / "dataset_split" / "test" / "labels"
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# GT loader
# ---------------------------------------------------------------------------

def load_gt_xyxy(label_path: Path) -> np.ndarray:
    """YOLO .txt → class-0 boxes as normalised (x1,y1,x2,y2). Shape [N,4]."""
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32)
    rows = []
    with open(label_path) as fh:
        for line in fh:
            p = line.strip().split()
            if len(p) < 5 or int(p[0]) != 0:
                continue
            cx, cy, w, h = float(p[1]), float(p[2]), float(p[3]), float(p[4])
            if w > 0 and h > 0:
                rows.append([cx - w / 2, cy - h / 2,
                             cx + w / 2, cy + h / 2])
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 4), dtype=np.float32)


# ---------------------------------------------------------------------------
# Geometry + metrics
# ---------------------------------------------------------------------------

def iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """[N,4] × [M,4] xyxy normalised → [N,M] IoU."""
    ix1 = np.maximum(pred[:, None, 0], gt[None, :, 0])
    iy1 = np.maximum(pred[:, None, 1], gt[None, :, 1])
    ix2 = np.minimum(pred[:, None, 2], gt[None, :, 2])
    iy2 = np.minimum(pred[:, None, 3], gt[None, :, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    a_p = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    a_g = (gt[:, 2]  - gt[:, 0])  * (gt[:, 3]  - gt[:, 1])
    return (inter / (a_p[:, None] + a_g[None, :] - inter + 1e-9)).astype(np.float32)


def compute_ap50(
    scores:   np.ndarray,
    tp:       np.ndarray,
    total_gt: int,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """VOC 2010+ all-points AP50. Returns (ap50, precisions, recalls)."""
    if total_gt == 0 or len(scores) == 0:
        return 0.0, np.array([]), np.array([])
    order  = np.argsort(-scores)
    tp_s   = tp[order]
    tp_cum = np.cumsum(tp_s)
    fp_cum = np.cumsum(1 - tp_s)
    rec = tp_cum / total_gt
    pre = tp_cum / (tp_cum + fp_cum + 1e-9)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[1.0], pre, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx  = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])), pre, rec


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model_path:  Path  = MODEL_PATH,
    test_images: Path  = TEST_IMAGES,
    test_labels: Path  = TEST_LABELS,
    iou_thresh:  float = 0.50,
    conf_thresh: float = 0.25,
    iou_nms:     float = 0.50,
    imgsz:       int   = 640,
) -> Dict[str, float]:
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print(f"  Model  : {model_path.name}")
    print(f"  Images : {test_images}")
    print(f"  IoU@50 : {iou_thresh}  |  NMS: {iou_nms}  |  conf: {conf_thresh}")
    print(f"{'='*60}\n")

    model = YOLO(str(model_path))

    image_paths: List[Path] = sorted([
        p for p in test_images.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    ])
    if not image_paths:
        raise FileNotFoundError(f"No images in {test_images}")
    print(f"Found {len(image_paths)} test images.\n")

    scores_list: List[float] = []
    tp_list:     List[int]   = []
    total_gt = 0
    t0 = time.perf_counter()

    for idx, img_path in enumerate(image_paths, 1):

        # Load image with cv2 — avoids path-parser issues with spaces in filename
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARNING: cannot read {img_path.name}", flush=True)
            continue

        # Official ultralytics prediction API
        results = model(img, conf=conf_thresh, iou=iou_nms,
                        imgsz=imgsz, verbose=False)

        # GT for this image (class 0 only)
        label_path = test_labels / (img_path.stem + ".txt")
        gt_xyxy    = load_gt_xyxy(label_path)
        n_gt       = len(gt_xyxy)
        total_gt  += n_gt

        # Extract tight bounding boxes from mask polygons
        pred_boxes: List[List[float]] = []
        pred_confs: List[float]       = []

        for result in results:
            if result.masks is None:
                continue
            confs = result.boxes.conf.cpu().numpy()          # [P]
            for poly, c in zip(result.masks.xyn, confs):    # poly: [K,2] normalised (x,y)
                if len(poly) == 0:
                    continue
                x1, y1 = float(poly[:, 0].min()), float(poly[:, 1].min())
                x2, y2 = float(poly[:, 0].max()), float(poly[:, 1].max())
                pred_boxes.append([x1, y1, x2, y2])
                pred_confs.append(float(c))

        if idx % 100 == 0:
            print(f"  [{idx}/{len(image_paths)}]  gt={n_gt}  pred={len(pred_confs)}", flush=True)

        if len(pred_confs) == 0:
            continue

        pred_arr  = np.array(pred_boxes, dtype=np.float32)  # [P,4]
        conf_arr  = np.array(pred_confs, dtype=np.float32)  # [P]

        # Sort descending by confidence
        order    = np.argsort(-conf_arr)
        pred_arr = pred_arr[order]
        conf_arr = conf_arr[order]

        if n_gt == 0:
            scores_list.extend(conf_arr.tolist())
            tp_list.extend([0] * len(conf_arr))
            continue

        # Greedy IoU matching (descending confidence)
        iou_mat: np.ndarray = iou_matrix(pred_arr, gt_xyxy)
        matched: Set[int]   = set()

        for i in range(len(conf_arr)):
            scores_list.append(float(conf_arr[i]))
            best_g   = int(np.argmax(iou_mat[i]))
            best_iou = float(iou_mat[i, best_g])
            if best_iou >= iou_thresh and best_g not in matched:
                tp_list.append(1)
                matched.add(best_g)
            else:
                tp_list.append(0)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s  ({elapsed / len(image_paths) * 1000:.1f} ms/image)\n")

    scores_arr = np.array(scores_list, dtype=np.float64)
    tp_arr     = np.array(tp_list,     dtype=np.float64)

    ap50, _, _ = compute_ap50(scores_arr, tp_arr, total_gt)

    # Precision and recall at the fixed operating point (conf=conf_thresh).
    # All predictions already have conf >= conf_thresh (filtered by the model),
    # so we evaluate the full accumulated TP/FP counts directly.
    n_tp = int(tp_arr.sum())
    n_fp = len(tp_arr) - n_tp
    prec = n_tp / (n_tp + n_fp + 1e-9) if (n_tp + n_fp) > 0 else 0.0
    rec  = n_tp / (total_gt + 1e-9)    if total_gt > 0          else 0.0
    f1   = 2 * prec * rec / (prec + rec + 1e-9)

    print(f"{'='*60}")
    print(f"  GT boxes (class 0) : {total_gt}")
    print(f"  Total predictions  : {len(scores_list)}")
    print(f"  AP50               : {ap50:.4f}  ({ap50 * 100:.2f}%)")
    print(f"  Precision  @conf={conf_thresh} : {prec:.4f}")
    print(f"  Recall     @conf={conf_thresh} : {rec:.4f}")
    print(f"  F1         @conf={conf_thresh} : {f1:.4f}")
    print(f"{'='*60}\n")

    return {
        "n_images":  float(len(image_paths)),
        "n_gt":      float(total_gt),
        "n_pred":    float(len(scores_list)),
        "ap50":      ap50,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate best.pt segmentation model on turkey test set."
    )
    p.add_argument("--model",   type=Path,  default=MODEL_PATH)
    p.add_argument("--images",  type=Path,  default=TEST_IMAGES)
    p.add_argument("--labels",  type=Path,  default=TEST_LABELS)
    p.add_argument("--iou",     type=float, default=0.50)
    p.add_argument("--conf",    type=float, default=0.1)
    p.add_argument("--iou-nms", type=float, default=0.75)
    p.add_argument("--imgsz",   type=int,   default=640)
    return p.parse_args()


if __name__ == "__main__":
    a = _args()
    evaluate(
        model_path  = a.model,
        test_images = a.images,
        test_labels = a.labels,
        iou_thresh  = a.iou,
        conf_thresh = a.conf,
        iou_nms     = a.iou_nms,
        imgsz       = a.imgsz,
    )
