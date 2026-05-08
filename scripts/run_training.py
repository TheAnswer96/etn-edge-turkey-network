"""
EdgeTurkeyNet — Parametric End-to-End Pipeline.

Every aspect of the pipeline is controlled via CLI flags (see edgeturkeynet/config.py).
Each run creates a unique timestamped directory that contains:

  outputs/runs/<YYYYMMDD_HHMMSS>_<backbone>/
    config.json               — full RunConfig as JSON
    train_metrics.csv         — per-epoch losses + per-class val metrics
    test_results.csv          — final per-class test AP/P/R + benchmark
    summary.json              — key results for easy scripted comparison
    checkpoints/
      best.pth
      last.pth
    exports/
      edge_turkey_net.onnx
      edge_turkey_net_int8.pth
    inference_outputs/
      *.jpg

Usage
-----
# Default (ShuffleNetV2, 100 epochs)
python scripts/run_training.py

# ShuffleNetV2 backbone, 50 epochs, custom pruning
python scripts/run_training.py --backbone shufflenetv2 --epochs 50 --prune-per-call 0.20

# MobileNetV1, lower confidence, tighter IoU
python scripts/run_training.py --backbone mobilenetv1 --score-threshold 0.25 --iou-threshold 0.45

# Skip training, evaluate existing checkpoint
python scripts/run_training.py --skip-train --resume-from outputs/runs/.../checkpoints/best.pth

# Fast eval-only run (no export, no visualisation)
python scripts/run_training.py --skip-export --skip-visualise
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from edgeturkeynet.config import get_config, RunConfig
from edgeturkeynet.dataset import (
    TRAIN_IMAGES, TRAIN_LABELS,
    VAL_IMAGES,   VAL_LABELS,
    TEST_IMAGES,  TEST_LABELS,
    get_test_loader,
)
from edgeturkeynet.evaluate import evaluate_map, PerClassMetrics
from edgeturkeynet.export import (
    export_to_onnx,
    estimate_model_stats,
    validate_onnx_runtime,
    quantize_dynamic_int8,
)
from edgeturkeynet.inference import benchmark_cpu, load_model_for_inference, run_test_set_inference
from edgeturkeynet.logger import RunLogger
from edgeturkeynet.model import EdgeTurkeyNet, NUM_CLASSES, CLASS_NAMES
from edgeturkeynet.train import DEVICE, Trainer, load_model_from_checkpoint

# ---------------------------------------------------------------------------
# Dataset verification
# ---------------------------------------------------------------------------

def verify_dataset() -> bool:
    """Verify that all required dataset directories exist."""
    required = [
        TRAIN_IMAGES, TRAIN_LABELS,
        VAL_IMAGES,   VAL_LABELS,
        TEST_IMAGES,  TEST_LABELS,
    ]
    print("\n[Dataset] Verifying dataset structure...")
    all_ok = True
    for d in required:
        ok = d.exists()
        print(f"  {'✓' if ok else '✗ MISSING'}  {d}")
        if not ok:
            all_ok = False

    if all_ok:
        print(
            f"\n  Train: {len(list(TRAIN_IMAGES.glob('*')))} images | "
            f"Val: {len(list(VAL_IMAGES.glob('*')))} images | "
            f"Test: {len(list(TEST_IMAGES.glob('*')))} images"
        )
        print(f"  Classes: {CLASS_NAMES}  (0={CLASS_NAMES[0]}, 1={CLASS_NAMES[1]})")
    else:
        print("\n  [!] Dataset missing. Ensure YOLO-format dataset at data/dataset_split/")
    return all_ok


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def print_banner(cfg: RunConfig) -> None:
    labels = {
        "mobilenetv3":  "MobileNetV3-Small  (pretrained)",
        "shufflenetv2": "ShuffleNetV2-0.5x  (pretrained)",
        "mobilenetv1":  "MobileNetV1-1.0    (from scratch)",
    }
    label = labels.get(cfg.backbone, cfg.backbone)
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║          EdgeTurkeyNet — Aerial Turkey Detection             ║
║          Novel Lightweight Edge AI Detection Pipeline        ║
╠══════════════════════════════════════════════════════════════╣
║  Classes:      body (0)  |  neck (1)                         ║
║  Backbone:     {label:<46}                                   ║
║  Novelty:      Oval-FCOS + DIoU-NMS + Periodic Pruning       ║
║  Deployment:   RPi 4B CPU — FP32 ONNX + INT8 dynamic quant   ║
╚══════════════════════════════════════════════════════════════╝
""")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full parametric pipeline."""

    # # ── 0. Parse arguments → RunConfig ──────────────────────────────────
    cfg = get_config()
    print_banner(cfg)
    print(cfg.summary())
    #
    # # ── 1. Dataset verification ──────────────────────────────────────────
    if not verify_dataset():
        print("\n[Main] Aborting: dataset not found.")
        sys.exit(1)
    #
    # # ── 2. RunLogger — creates timestamped run directory ─────────────────
    logger = RunLogger(cfg)
    #
    # # ── 3. Model architecture stats ───────────────────────────────────────
    print("\n[Main] Model statistics (before training)...")
    _stats_model = EdgeTurkeyNet(
        num_classes=NUM_CLASSES,
        pretrained_backbone=False,   # no download for stats pass
        backbone=cfg.backbone,
        input_size=cfg.input_size,
    )
    estimate_model_stats(_stats_model, input_size=cfg.input_size)
    del _stats_model

    # ── 4. Training ───────────────────────────────────────────────────────
    if cfg.skip_train:
        # Resolve checkpoint: explicit path > run-dir best > any existing
        if cfg.resume_from and Path(cfg.resume_from).exists():
            print("Resuming...")
            best_ckpt = Path(cfg.resume_from)
        elif logger.best_model_path.exists():
            best_ckpt = logger.best_model_path
        else:
            candidates = sorted(cfg.runs_root.glob("*/checkpoints/best.pth"))
            if candidates:
                best_ckpt = candidates[-1]
                print(f"[Main] --skip-train: using {best_ckpt}")
            else:
                print("[Main] --skip-train but no checkpoint found. Aborting.")
                sys.exit(1)
        best_model = load_model_from_checkpoint(best_ckpt, cfg)
    else:
        print("\n[Main] Starting training...")
        trainer   = Trainer(cfg=cfg, logger=logger)
        best_model = trainer.train()            # returns Path to best.pth

    best_model = best_model.to(DEVICE)

    # ── 5. Test set evaluation ────────────────────────────────────────────
    print("\n[Main] Evaluating on test set...")
    test_loader = get_test_loader(
        batch_size=min(cfg.batch_size, 4),
        num_workers=min(cfg.num_workers, 2),
    )
    test_metrics: PerClassMetrics = evaluate_map(
        best_model, test_loader, DEVICE,
        iou_threshold=cfg.iou_threshold,
        score_threshold=cfg.score_threshold,
        input_size=cfg.input_size
    )
    print("\n  Final Test Set Evaluation")
    test_metrics.print_table(iou_threshold=cfg.iou_threshold)
    print(f"  Parameters: {best_model.get_parameter_count():,}")


    # ── 6. FP32 CPU benchmark ─────────────────────────────────────────────
    print("\n[Main] FP32 CPU benchmark (simulated RPi single-thread)...")
    cpu_model  = best_model.cpu()
    bench_fp32 = benchmark_cpu(cpu_model, num_warmup=10, num_runs=30)

    # ── 7. ONNX export + INT8 quantisation ───────────────────────────────
    onnx_size_mb = float("nan")
    int8_size_mb = float("nan")

    if not cfg.skip_export:
        print("\n[Main] Exporting FP32 model to ONNX...")
        try:
            onnx_path = export_to_onnx(
                cpu_model,
                fuse_bn=True,
                output_path=logger.onnx_path,
            )
            onnx_size_mb = onnx_path.stat().st_size / (1024 ** 2)
            print(f"[Main] ONNX saved: {onnx_path}  ({onnx_size_mb:.2f} MB)")
            print("\n[Main] ONNX Runtime validation...")
            validate_onnx_runtime(onnx_path)
        except (ImportError, TypeError) as e:
            print(f"[Main] ONNX export skipped ({e}). "
                  "Install: pip install onnx onnxruntime")

        print("\n[Main] Applying INT8 dynamic quantization...")
        quantize_dynamic_int8(
            cpu_model,
            benchmark=True,
            num_benchmark_runs=30,
            output_path=logger.int8_path,
        )
        if logger.int8_path.exists():
            int8_size_mb = logger.int8_path.stat().st_size / (1024 ** 2)
    else:
        print("\n[Main] --skip-export: skipping ONNX and INT8 stages.")

    # ── 8. Write test results CSV (now benchmark values are known) ────────
    logger.log_test_results(
        test_metrics,
        fp32_fps=bench_fp32.get("fps", float("nan")),
        onnx_mb=onnx_size_mb,
        int8_mb=int8_size_mb,
    )

    # ── 9. Test set visualisations ────────────────────────────────────────
    if not cfg.skip_visualise:
        print("\n[Main] Generating test set visualisations...")
        vis_model = load_model_for_inference(best_ckpt)
        run_test_set_inference(
            vis_model,
            test_dir=TEST_IMAGES,
            output_dir=logger.vis_dir,
            max_images=cfg.max_vis_images,
            save_visualizations=True,
        )
    else:
        print("\n[Main] --skip-visualise: skipping inference visualisations.")

    # ── 10. Write summary.json ────────────────────────────────────────────
    summary = {
        "backbone":          cfg.backbone,
        "epochs":            cfg.epochs,
        "iou_threshold":     cfg.iou_threshold,
        "score_threshold":   cfg.score_threshold,
        "prune_per_call":    cfg.prune_per_call,
        "prune_max_sparsity": cfg.prune_max_sparsity,
        "parameters":        best_model.get_parameter_count(),
        "fp32_fps":          round(bench_fp32.get("fps", 0.0), 2),
        "fp32_latency_ms":   round(bench_fp32.get("mean_ms", 0.0), 2),
        "onnx_size_mb":      round(onnx_size_mb, 3) if onnx_size_mb == onnx_size_mb else None,
        "int8_size_mb":      round(int8_size_mb, 3) if int8_size_mb == int8_size_mb else None,
        "test_map":          round(test_metrics.map, 6),
        "test_ap":           {k: round(v, 6) for k, v in test_metrics.ap.items()},
        "test_precision":    {k: round(v, 6) for k, v in test_metrics.precision.items()},
        "test_recall":       {k: round(v, 6) for k, v in test_metrics.recall.items()},
    }
    summary_path = logger.run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Main] Summary written → {summary_path}")

    # ── 11. Close logger ──────────────────────────────────────────────────
    logger.close()

    # ── 12. Final console summary ─────────────────────────────────────────
    print("\n" + "="*65)
    print("  EdgeTurkeyNet — Pipeline Complete")
    print("="*65)
    print(f"  Run directory:         {logger.run_dir}")
    print(f"  Backbone:              {cfg.backbone}")
    print(f"  Classes:               {CLASS_NAMES}  ({NUM_CLASSES} total)")
    print(f"  mAP@{cfg.iou_threshold:.2f}:               {test_metrics.map:.4f}")
    for cls_name in CLASS_NAMES:
        print(
            f"  AP[{cls_name}]:            "
            f"{test_metrics.ap.get(cls_name, 0.0):.4f}  "
            f"P={test_metrics.precision.get(cls_name, 0.0):.4f}  "
            f"R={test_metrics.recall.get(cls_name, 0.0):.4f}"
        )
    print(f"  Parameters:            {best_model.get_parameter_count():,}")
    print(f"  FP32 CPU FPS:          {bench_fp32.get('fps', 0.0):.2f}")
    if not cfg.skip_export:
        print(f"  FP32 ONNX size:        {onnx_size_mb:.2f} MB")
        print(f"  INT8 model size:       {int8_size_mb:.2f} MB")
    print(f"  train_metrics.csv:     {logger.train_csv}")
    print(f"  test_results.csv:      {logger.test_csv}")
    print(f"  summary.json:          {summary_path}")
    print("="*65)
    if not cfg.skip_export:
        print("\n  Raspberry Pi deployment steps:")
        print(f"  1. Copy {logger.onnx_path.name}  → RPi  (FP32, easy)")
        print("  2. pip install onnxruntime  → run with ORT CPU provider")
        print("  3. For INT8: copy int8.pth, use load_int8_model() from export.py")
        print("  4. Set torch.backends.quantized.engine='qnnpack' on RPi for ARM NEON")


if __name__ == "__main__":
    main()
