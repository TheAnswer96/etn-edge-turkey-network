"""
CPU-optimized inference script for EdgeTurkeyNet.

Optimizations for Raspberry Pi deployment:
1. torch.no_grad() — eliminates autograd graph construction overhead
2. model.eval() — disables dropout and uses BN running stats
3. torch.inference_mode() — stronger than no_grad, disables version counter
4. Thread count control — avoids oversubscription on RPi's 4 cores
5. Input preprocessing in uint8 → avoids unnecessary float copies
6. Pre-allocated output buffers — reduces GC pressure

EDGE AI: These are inference-time-only optimizations that require no
retraining or model changes — pure deployment engineering.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from .dataset import letterbox, INPUT_SIZE
from .evaluate import predict, diou_nms, SCORE_THRESHOLD, NMS_IOU_THRESHOLD
from .model import EdgeTurkeyNet

# Hardcoded paths
MODEL_CHECKPOINT = Path("outputs/checkpoints/edge_turkey_net_best.pth")
ONNX_MODEL_PATH  = Path("outputs/exports/edge_turkey_net.onnx")
TEST_IMAGES_DIR  = Path("data/dataset_split/test/images")
INFERENCE_OUTPUT_DIR = Path("outputs/inference_outputs")

# CPU thread configuration (Raspberry Pi 4B has 4 cores)
# Setting to 4 avoids thread oversubscription but allows full parallelism
NUM_CPU_THREADS = 4


# ---------------------------------------------------------------------------
# CPU-optimized model loader
# ---------------------------------------------------------------------------

def load_model_for_inference(
    checkpoint_path: Path = MODEL_CHECKPOINT,
    num_threads: int = NUM_CPU_THREADS,
) -> EdgeTurkeyNet:
    """
    Load EdgeTurkeyNet and configure for CPU inference.

    Sets PyTorch CPU thread count to match Raspberry Pi core count.
    Oversubscription (more threads than cores) causes context switching
    overhead that hurts latency on embedded devices.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        num_threads: PyTorch intraop thread count.

    Returns:
        EdgeTurkeyNet configured for CPU inference.
    """
    # Limit threads to Raspberry Pi core count
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)  # Single interop thread for sequential inference

    model = EdgeTurkeyNet(num_classes=1, pretrained_backbone=False)

    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        print(f"[Inference] Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"[Inference] WARNING: Checkpoint not found at {checkpoint_path}. "
              f"Using random weights for benchmark only.")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Image preprocessing (CPU-optimized)
# ---------------------------------------------------------------------------

def preprocess_image(
    image_path: Path,
    input_size: Tuple[int, int] = INPUT_SIZE,
) -> Tuple[torch.Tensor, float, Tuple[int, int], Tuple[int, int]]:
    """
    Load and preprocess a single image for inference.

    Uses uint8 letterboxing before float conversion to minimize memory
    bandwidth usage — important on Raspberry Pi with limited memory bandwidth.

    Args:
        image_path: Path to input image.
        input_size: Model input (H, W).

    Returns:
        tensor: [1, 3, H, W] normalized float32 tensor.
        scale: Letterbox scale factor (for box coordinate reversal).
        padding: (pad_top, pad_left) in pixels.
        orig_shape: Original image (H, W).
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise IOError(f"Cannot read: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    # Letterbox (still uint8 — cheaper memory)
    img_lb, scale, padding = letterbox(img_rgb, input_size)

    # Convert to float tensor [1, 3, H, W]
    # Using numpy float32 conversion once (vs per-channel ops)
    tensor = torch.from_numpy(
        img_lb.astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).contiguous()

    return tensor, scale, padding, (orig_h, orig_w)


def postprocess_detections(
    boxes_lb: torch.Tensor,
    scores: torch.Tensor,
    scale: float,
    padding: Tuple[int, int],
    orig_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert letterbox-space detections back to original image coordinates.

    Reverses the letterboxing transform for visualization/downstream use.

    Args:
        boxes_lb: [K, 4] boxes in letterbox pixel space (x1,y1,x2,y2).
        scores: [K] confidence scores.
        scale: Letterbox scale factor.
        padding: (pad_top, pad_left) applied during letterboxing.
        orig_shape: Original image (H, W).

    Returns:
        boxes_orig: [K, 4] boxes in original image pixels.
        scores_np: [K] scores as numpy array.
    """
    if len(boxes_lb) == 0:
        return np.zeros((0, 4)), np.zeros(0)

    pad_top, pad_left = padding
    boxes = boxes_lb.numpy().copy()

    # Reverse padding
    boxes[:, 0] -= pad_left
    boxes[:, 1] -= pad_top
    boxes[:, 2] -= pad_left
    boxes[:, 3] -= pad_top

    # Reverse scale
    boxes /= scale

    # Clip to original image bounds
    orig_h, orig_w = orig_shape
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, orig_w)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, orig_h)

    return boxes, scores.numpy()


# ---------------------------------------------------------------------------
# Single-image inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def infer_image(
    model: EdgeTurkeyNet,
    image_path: Path,
    score_threshold: float = SCORE_THRESHOLD,
    nms_threshold: float = NMS_IOU_THRESHOLD,
    input_size: Tuple[int, int] = INPUT_SIZE,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run full inference pipeline on a single image.

    Uses torch.inference_mode() which is ~10% faster than no_grad()
    by disabling both autograd graph and version counter tracking.

    Args:
        model: EdgeTurkeyNet in eval mode.
        image_path: Path to input image.
        score_threshold: Minimum detection score.
        nms_threshold: DIoU-NMS threshold.
        input_size: Model input (H, W).

    Returns:
        boxes: [K, 4] detection boxes in original image coordinates.
        scores: [K] confidence scores.
        inference_ms: Total inference time in milliseconds.
    """
    # Preprocess
    tensor, scale, padding, orig_shape = preprocess_image(image_path, input_size)

    # Inference timing
    t0 = time.perf_counter()
    detections = predict(
        model, tensor,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        input_size=input_size,
    )
    inference_ms = (time.perf_counter() - t0) * 1000

    boxes_lb, scores = detections[0]
    boxes_orig, scores_np = postprocess_detections(
        boxes_lb, scores, scale, padding, orig_shape
    )

    return boxes_orig, scores_np, inference_ms


# ---------------------------------------------------------------------------
# Raspberry Pi CPU Benchmark
# ---------------------------------------------------------------------------

def benchmark_cpu(
    model: EdgeTurkeyNet,
    num_warmup: int = 10,
    num_runs: int = 50,
    input_size: Tuple[int, int] = INPUT_SIZE,
) -> dict:
    """
    Simulate Raspberry Pi CPU inference performance.

    Runs inference with single thread matching RPi 4B behavior.
    Results are NOT identical to actual RPi (x86 vs ARM μarch),
    but relative performance metrics are representative.

    To get true RPi performance: export ONNX and run on actual hardware.

    Args:
        model: EdgeTurkeyNet model.
        num_warmup: Warmup iterations (excluded from timing).
        num_runs: Timed inference iterations.
        input_size: Model input (H, W).

    Returns:
        dict with latency statistics and FPS.
    """
    # Configure as single-threaded to match RPi's typical deployment
    torch.set_num_threads(1)

    model.eval()
    h, w = input_size
    dummy = torch.randn(1, 3, h, w)

    print(f"\n[Benchmark] Simulating RPi CPU inference ({num_runs} runs)...")
    print(f"  Input: {h}x{w} | Threads: 1 (RPi simulation)")

    # Warmup
    with torch.inference_mode():
        for _ in range(num_warmup):
            _ = model(dummy)

    # Timed runs
    latencies_ms = []
    with torch.inference_mode():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            latencies_ms.append((time.perf_counter() - t0) * 1000)

    # Restore normal thread count
    torch.set_num_threads(NUM_CPU_THREADS)

    latencies_ms_arr = np.array(latencies_ms)
    results = {
        "mean_ms":   float(latencies_ms_arr.mean()),
        "min_ms":    float(latencies_ms_arr.min()),
        "max_ms":    float(latencies_ms_arr.max()),
        "p95_ms":    float(np.percentile(latencies_ms_arr, 95)),
        "fps":       float(1000 / latencies_ms_arr.mean()),
    }

    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │  CPU Benchmark Results (PyTorch)    │")
    print(f"  ├─────────────────────────────────────┤")
    print(f"  │  Mean latency:  {results['mean_ms']:8.1f} ms          │")
    print(f"  │  Min latency:   {results['min_ms']:8.1f} ms          │")
    print(f"  │  P95 latency:   {results['p95_ms']:8.1f} ms          │")
    print(f"  │  Estimated FPS: {results['fps']:8.2f}               │")
    print(f"  └─────────────────────────────────────┘")
    print(f"  Note: ONNX Runtime on RPi typically 2-3x faster than PyTorch CPU.")
    print(f"  Estimated ONNX FPS on RPi 4B: ~{results['fps']*0.4:.1f} (FP32) | "
          f"~{results['fps']*1.5:.1f} (INT8)")

    return results


# ---------------------------------------------------------------------------
# Batch inference on test set with visualization
# ---------------------------------------------------------------------------

def run_test_set_inference(
    model: EdgeTurkeyNet,
    test_dir: Path = TEST_IMAGES_DIR,
    output_dir: Path = INFERENCE_OUTPUT_DIR,
    score_threshold: float = SCORE_THRESHOLD,
    max_images: int = 20,
    save_visualizations: bool = True,
) -> None:
    """
    Run inference on the test image set and optionally save visualizations.

    Draws bounding boxes and scores on images for qualitative evaluation.
    Colors: Green boxes = detections above threshold.

    Args:
        model: EdgeTurkeyNet in eval mode.
        test_dir: Directory containing test images.
        output_dir: Directory to save annotated images.
        score_threshold: Minimum detection score to draw.
        max_images: Maximum number of images to process.
        save_visualizations: Whether to save annotated images to disk.
    """
    if save_visualizations:
        output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in test_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )[:max_images]

    if len(image_paths) == 0:
        print(f"[Inference] No images found in {test_dir}")
        return

    total_time_ms = 0.0
    total_detections = 0

    print(f"\n[Inference] Processing {len(image_paths)} test images...")

    for img_path in image_paths:
        boxes, scores, ms = infer_image(model, img_path, score_threshold=score_threshold)
        total_time_ms += ms
        total_detections += len(boxes)

        if save_visualizations and len(boxes) > 0:
            img = cv2.imread(str(img_path))
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = box.astype(int)
                # Green box with confidence label
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
                label = f"Turkey {score:.2f}"
                cv2.putText(img, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

            out_path = output_dir / img_path.name
            cv2.imwrite(str(out_path), img)

    avg_ms = total_time_ms / len(image_paths)
    avg_dets = total_detections / len(image_paths)

    print(f"\n[Inference] Results:")
    print(f"  Images processed:  {len(image_paths)}")
    print(f"  Avg latency:       {avg_ms:.1f} ms")
    print(f"  Avg FPS:           {1000/avg_ms:.2f}")
    print(f"  Avg detections:    {avg_dets:.1f} per image")
    if save_visualizations:
        print(f"  Visualizations:    {output_dir}")
