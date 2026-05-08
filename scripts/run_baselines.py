"""
run_baselines.py — Independent Baseline & Ablation Runner.

Owns its entire execution path.  No dependency on config.py, logger.py,
train.py, or main.py.

Execution order
---------------
  1. Verify dataset split directories.
  2. Build train, val, and test DataLoaders.
  3. TRAINING phase — each enabled experiment trains its model to completion:
       A. SSDTrainer          → ssd_best.pth
       B. StandardFCOSTrainer → stdloss_best.pth
  4. TESTING phase — each trained model is evaluated on the held-out test set:
       A. _evaluate_ssd(ssd_model,  test_loader)
       B. evaluate_map(std_model,   test_loader)
       C. DenseNMSBenchmark.run(ssd_model, test_loader)  ← real predictions
  5. Print unified comparison table.
  6. Write summary.json + report.txt to BASELINES_ROOT.

Experiments
-----------
  A. MobileNetV3 + SSD
     Same MobileNetV3-Small backbone, anchor-based multi-scale SSD head,
     no PAN-Lite neck.  Isolates the head-design contribution.

  B. Standard BCE + GIoU loss ablation
     Full EdgeTurkeyNet architecture (MobileNetV3 + PAN-Lite + FCOS head)
     but with both custom loss terms disabled:
       CIoU        →  GIoU          (removes aspect-ratio penalty term v)
       Oval-biased →  standard ctr  (removes 1.2:1 prior from centerness target)
     Isolates the custom loss contribution.

  C. DIoU-NMS vs IoU-NMS dense-subset benchmark
     Runs both NMS variants on dense-overlap subsets from real Exp A
     predictions, reporting ΔF1, Δprecision, Δrecall, Δkept-detections.
     Uses the trained SSD model so candidates are realistic.

Configuration
-------------
Edit the CAPITALS at the top of this file.  No CLI arguments are used.

Outputs
-------
  BASELINES_ROOT/
    ssd/
      train_metrics.csv   per-epoch train losses + val mAP
      ssd_best.pth        best checkpoint (val mAP)
      ssd_last.pth        last checkpoint
    standard_loss/
      train_metrics.csv
      stdloss_best.pth
      stdloss_last.pth
    summary.json          all test results, machine-readable
    report.txt            comparison table, human-readable

Usage
-----
  python scripts/run_baselines.py

  from scripts.run_baselines import run
  report = run()
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Project root on path so this file runs from any working directory
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from edgeturkeynet.baseline import (
    BaselineReport,
    BL_IOU_THRESHOLD,
    BL_SCORE_THRESHOLD,
    DEVICE,
    DenseNMSBenchmark,
    ExtendedBaselineReport,
    MobileNetV3SSD,
    NanoDetTrainer,
    YOLOv5NanoTrainer,
    YOLOv8NanoTrainer,
    SSDTrainer,
    StandardFCOSTrainer,
    _evaluate_nanodet,
    _evaluate_ssd,
    _evaluate_yolov5,
    _evaluate_yolov8,
)
from edgeturkeynet.dataset import (
    TRAIN_IMAGES, TRAIN_LABELS,
    VAL_IMAGES,   VAL_LABELS,
    TEST_IMAGES,  TEST_LABELS,
    get_train_loader,
    get_val_loader,
    get_test_loader,
)
from edgeturkeynet.evaluate import evaluate_map


# ===========================================================================
# CONFIGURATION — edit these variables to change any behaviour.
# ===========================================================================

# Output root for all checkpoints, CSVs, and result files
BASELINES_ROOT = Path("outputs/baselines")

# DataLoader settings
TRAIN_BATCH_SIZE  = 16
TRAIN_NUM_WORKERS = 8
VAL_BATCH_SIZE    = 16
VAL_NUM_WORKERS   = 8
TEST_BATCH_SIZE   = 4   # ≤ 4 keeps peak memory predictable on evaluation
TEST_NUM_WORKERS  = 2

# Which experiments to run
RUN_SSD      = False   # A: MobileNetV3 + SSD
RUN_STDLOSS  = False   # B: standard BCE + GIoU loss ablation
RUN_NMS      = False   # C: DIoU-NMS vs IoU-NMS benchmark
RUN_NANODET  = True   # D: NanoDet  (GFL anchor-free, ShuffleNetV2 PAN)
RUN_YOLOV5N  = True   # E: YOLOv5-nano  (anchor-based, CSP-Tiny PAN)
RUN_YOLOV8N  = True   # F: YOLOv8-nano  (anchor-free decoupled, C2f PAN)

# Backbone initialisation
SSD_PRETRAINED     = True   # pretrained MobileNetV3 for Exp A
STDLOSS_PRETRAINED = True   # pretrained MobileNetV3 for Exp B


# ===========================================================================
# DATASET VERIFICATION
# ===========================================================================

def _verify_dataset() -> bool:
    """
    Confirm all six dataset split directories exist before any work begins.

    Prints a ✓ / ✗ line per directory and a file count summary on success.
    Returns True if all directories are present, False otherwise.
    """
    splits = {
        "train images": TRAIN_IMAGES,
        "train labels": TRAIN_LABELS,
        "val images":   VAL_IMAGES,
        "val labels":   VAL_LABELS,
        "test images":  TEST_IMAGES,
        "test labels":  TEST_LABELS,
    }
    print("\n[Dataset] Verifying split directories ...")
    ok = True
    for name, path in splits.items():
        exists = path.exists()
        print(f"  {'✓' if exists else '✗ MISSING':<12} {name:<14}  {path}")
        if not exists:
            ok = False

    if ok:
        print(
            f"\n  train={len(list(TRAIN_IMAGES.glob('*')))} images  "
            f"val={len(list(VAL_IMAGES.glob('*')))} images  "
            f"test={len(list(TEST_IMAGES.glob('*')))} images"
        )
    else:
        print(
            "\n  [!] Dataset incomplete — "
            "expected YOLO-format split at data/dataset_split/."
        )
    return ok


# ===========================================================================
# OUTPUT HELPERS
# ===========================================================================

def _metrics_to_dict(m) -> Optional[dict]:
    """Serialise a PerClassMetrics to a plain dict, or None if absent."""
    if m is None:
        return None
    return {
        "map":       round(m.map, 6),
        "ap":        {k: round(v, 6) for k, v in m.ap.items()},
        "precision": {k: round(v, 6) for k, v in m.precision.items()},
        "recall":    {k: round(v, 6) for k, v in m.recall.items()},
    }


def _nms_to_dict(r) -> Optional[dict]:
    """Serialise a NMSBenchmarkResult to a plain dict, or None if absent."""
    if r is None:
        return None
    return {
        "n_subsets":          r.n_subsets,
        "diou_avg_kept":      round(r.diou_avg_kept,      4),
        "iou_avg_kept":       round(r.iou_avg_kept,       4),
        "diou_avg_precision": round(r.diou_avg_precision, 4),
        "iou_avg_precision":  round(r.iou_avg_precision,  4),
        "diou_avg_recall":    round(r.diou_avg_recall,    4),
        "iou_avg_recall":     round(r.iou_avg_recall,     4),
        "diou_avg_f1":        round(r.diou_avg_f1,        4),
        "iou_avg_f1":         round(r.iou_avg_f1,         4),
        "delta_kept":         round(r.delta_kept,         4),
        "delta_f1":           round(r.delta_f1,           4),
    }


def _save_outputs(report, output_dir: Path) -> None:
    """
    Write summary.json and report.txt to output_dir.

    summary.json  — all numeric results, machine-readable.
    report.txt    — exact capture of the comparison table printed to stdout.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "exp_a_ssd":           _metrics_to_dict(report.ssd_metrics),
        "exp_b_standard_loss": _metrics_to_dict(report.stdloss_metrics),
        "exp_c_nms_benchmark": _nms_to_dict(report.nms_result),
        "reference_edge":      _metrics_to_dict(report.edge_metrics),
        "exp_d_nanodet":       _metrics_to_dict(getattr(report, 'nanodet_metrics', None)),
        "exp_e_yolov5n":       _metrics_to_dict(getattr(report, 'yolov5n_metrics', None)),
        "exp_f_yolov8n":       _metrics_to_dict(getattr(report, 'yolov8n_metrics', None)),
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  summary.json → {summary_path}")

    buf = io.StringIO()
    with redirect_stdout(buf):
        report.print_summary()
    report_path = output_dir / "report.txt"
    with open(report_path, "w") as f:
        f.write(buf.getvalue())
    print(f"  report.txt   → {report_path}")


# ===========================================================================
# MAIN RUN FUNCTION
# ===========================================================================

def run(
    run_ssd:            bool = RUN_SSD,
    run_stdloss:        bool = RUN_STDLOSS,
    run_nms:            bool = RUN_NMS,
    run_nanodet:        bool = RUN_NANODET,
    run_yolov5n:        bool = RUN_YOLOV5N,
    run_yolov8n:        bool = RUN_YOLOV8N,
    baselines_root:     Path = BASELINES_ROOT,
    ssd_pretrained:     bool = SSD_PRETRAINED,
    stdloss_pretrained: bool = STDLOSS_PRETRAINED,
) -> ExtendedBaselineReport:
    """
    Train all enabled baseline models, then evaluate each on the test set.

    Experiments
    -----------
    A. MobileNetV3 + SSD              (anchor-based head, pretrained backbone)
    B. Standard BCE + GIoU            (loss ablation, same arch as EdgeTurkeyNet)
    C. DIoU-NMS vs IoU-NMS            (dense-flock NMS benchmark)
    D. NanoDet                        (GFL anchor-free, ShuffleNetV2 PAN)
    E. YOLOv5-nano                    (anchor-based, CSP-Tiny + PANNet)
    F. YOLOv8-nano                    (anchor-free decoupled, C2f + C2f-PAN)

    The two phases are kept strictly separate:
      TRAINING  — each enabled trainer runs to completion (or early stopping).
      TESTING   — each best checkpoint evaluated on the held-out test set.

    Returns:
        ExtendedBaselineReport with all experiment results.
    """
    baselines_root = Path(baselines_root)

    # ── Banner ────────────────────────────────────────────────────────────
    print("""
╔══════════════════════════════════════════════════════════════╗
║     EdgeTurkeyNet — Baseline & Ablation Experiment Runner    ║
╠══════════════════════════════════════════════════════════════╣
║  A. MobileNetV3 + SSD          (anchor-based head)           ║
║  B. Standard BCE + GIoU loss   (loss ablation)               ║
║  C. DIoU-NMS vs IoU-NMS        (dense-flock NMS benchmark)   ║
║  D. NanoDet                    (GFL anchor-free)             ║
║  E. YOLOv5-nano                (anchor-based CSP-Tiny)       ║
║  F. YOLOv8-nano                (anchor-free C2f decoupled)   ║
╚══════════════════════════════════════════════════════════════╝""")

    active = " ".join(filter(None, [
        "A(SSD)"      if run_ssd      else "",
        "B(StdLoss)"  if run_stdloss  else "",
        "C(NMS)"      if run_nms      else "",
        "D(NanoDet)"  if run_nanodet  else "",
        "E(YOLOv5n)"  if run_yolov5n  else "",
        "F(YOLOv8n)"  if run_yolov8n  else "",
    ])) or "none"
    print(f"\n  Device      : {DEVICE}")
    print(f"  Output root : {baselines_root}")
    print(f"  Experiments : {active}")

    # ── 1. Dataset verification ───────────────────────────────────────────
    if not _verify_dataset():
        print("\n[Baselines] Aborting — dataset incomplete.")
        sys.exit(1)

    # ── 2. Build DataLoaders ──────────────────────────────────────────────
    print("\n[Baselines] Building DataLoaders ...")
    train_loader = get_train_loader(TRAIN_BATCH_SIZE, TRAIN_NUM_WORKERS)
    val_loader   = get_val_loader(VAL_BATCH_SIZE,     VAL_NUM_WORKERS)
    test_loader  = get_test_loader(TEST_BATCH_SIZE,   TEST_NUM_WORKERS)
    print(
        f"  train={len(train_loader)} batches  "
        f"val={len(val_loader)} batches  "
        f"test={len(test_loader)} batches"
    )

    ssd_model      = None
    std_model      = None
    nanodet_model  = None
    yolov5n_model  = None
    yolov8n_model  = None

    # =========================================================================
    # TRAINING PHASE
    # =========================================================================
    print("\n" + "=" * 65)
    print("  PHASE 1 — TRAINING")
    print("=" * 65)

    if run_ssd:
        print("\n[A] Training MobileNetV3 + SSD ...")
        trainer_a = SSDTrainer(
            output_dir = baselines_root / "ssd",
            pretrained = ssd_pretrained,
        )
        trainer_a.train_loader = train_loader
        trainer_a.val_loader   = val_loader
        ssd_model = trainer_a.train()
        print(f"[A] Done — best val mAP: {trainer_a.best_map:.4f}")

    if run_stdloss:
        print("\n[B] Training EdgeTurkeyNet + Standard BCE + GIoU loss ...")
        trainer_b = StandardFCOSTrainer(
            output_dir = baselines_root / "standard_loss",
            pretrained = stdloss_pretrained,
        )
        trainer_b.train_loader = train_loader
        trainer_b.val_loader   = val_loader
        std_model = trainer_b.train()
        print(f"[B] Done — best val mAP: {trainer_b.best_map:.4f}")

    if run_nanodet:
        print("\n[D] Training NanoDet ...")
        trainer_d = NanoDetTrainer(output_dir=baselines_root / "nanodet")
        trainer_d.train_loader = train_loader
        trainer_d.val_loader   = val_loader
        nanodet_model = trainer_d.train()
        print(f"[D] Done — best val mAP: {trainer_d.best_map:.4f}")

    if run_yolov5n:
        print("\n[E] Training YOLOv5-nano ...")
        trainer_e = YOLOv5NanoTrainer(output_dir=baselines_root / "yolov5n")
        trainer_e.train_loader = train_loader
        trainer_e.val_loader   = val_loader
        yolov5n_model = trainer_e.train()
        print(f"[E] Done — best val mAP: {trainer_e.best_map:.4f}")

    if run_yolov8n:
        print("\n[F] Training YOLOv8-nano ...")
        trainer_f = YOLOv8NanoTrainer(output_dir=baselines_root / "yolov8n")
        trainer_f.train_loader = train_loader
        trainer_f.val_loader   = val_loader
        yolov8n_model = trainer_f.train()
        print(f"[F] Done — best val mAP: {trainer_f.best_map:.4f}")

    # =========================================================================
    # TESTING PHASE
    # =========================================================================
    print("\n" + "=" * 65)
    print("  PHASE 2 — TESTING")
    print("=" * 65)

    report = ExtendedBaselineReport()

    if run_ssd and ssd_model is not None:
        print("\n[A] Evaluating MobileNetV3+SSD on test set ...")
        report.ssd_metrics = _evaluate_ssd(
            ssd_model, test_loader,
            score_threshold=BL_SCORE_THRESHOLD, iou_threshold=BL_IOU_THRESHOLD,
        )
        report.ssd_metrics.print_table(BL_IOU_THRESHOLD)

    if run_stdloss and std_model is not None:
        print("\n[B] Evaluating standard-loss model on test set ...")
        report.stdloss_metrics = evaluate_map(
            std_model, test_loader, DEVICE,
            iou_threshold=BL_IOU_THRESHOLD, score_threshold=BL_SCORE_THRESHOLD,
        )
        report.stdloss_metrics.print_table(BL_IOU_THRESHOLD)

    if run_nms:
        nms_model = ssd_model if ssd_model is not None else std_model
        mode = "real model predictions" if nms_model is not None else "synthetic subsets"
        print(f"\n[C] Running DIoU-NMS vs IoU-NMS benchmark ({mode}) ...")
        report.nms_result = DenseNMSBenchmark().run(
            model=nms_model,
            test_loader=test_loader if nms_model is not None else None,
        )

    if run_nanodet and nanodet_model is not None:
        print("\n[D] Evaluating NanoDet on test set ...")
        report.nanodet_metrics = _evaluate_nanodet(
            nanodet_model, test_loader,
            score_threshold=BL_SCORE_THRESHOLD, iou_threshold=BL_IOU_THRESHOLD,
        )
        report.nanodet_metrics.print_table(BL_IOU_THRESHOLD)

    if run_yolov5n and yolov5n_model is not None:
        print("\n[E] Evaluating YOLOv5-nano on test set ...")
        report.yolov5n_metrics = _evaluate_yolov5(
            yolov5n_model, test_loader,
            score_threshold=BL_SCORE_THRESHOLD, iou_threshold=BL_IOU_THRESHOLD,
        )
        report.yolov5n_metrics.print_table(BL_IOU_THRESHOLD)

    if run_yolov8n and yolov8n_model is not None:
        print("\n[F] Evaluating YOLOv8-nano on test set ...")
        report.yolov8n_metrics = _evaluate_yolov8(
            yolov8n_model, test_loader,
            score_threshold=BL_SCORE_THRESHOLD, iou_threshold=BL_IOU_THRESHOLD,
        )
        report.yolov8n_metrics.print_table(BL_IOU_THRESHOLD)

    # ── Print unified comparison table ───────────────────────────────────
    report.print_summary()

    # ── Persist outputs ───────────────────────────────────────────────────
    print("\n[Baselines] Writing outputs ...")
    _save_outputs(report, baselines_root)

    # ── Final one-line-per-experiment summary ─────────────────────────────
    print("\n" + "=" * 65)
    print("  Baseline Runner — Complete")
    print("=" * 65)
    for label, metrics in [
        ("Exp A  MobileNetV3+SSD  ", report.ssd_metrics),
        ("Exp B  Standard BCE+GIoU", report.stdloss_metrics),
        ("Exp D  NanoDet          ", report.nanodet_metrics),
        ("Exp E  YOLOv5-nano      ", report.yolov5n_metrics),
        ("Exp F  YOLOv8-nano      ", report.yolov8n_metrics),
    ]:
        if metrics:
            print(
                f"  {label}  mAP={metrics.map:.4f}  "
                f"body={metrics.ap.get('body', 0):.4f}  "
                f"neck={metrics.ap.get('neck', 0):.4f}"
            )
    if report.nms_result:
        r = report.nms_result
        print(
            f"  Exp C  DIoU vs IoU NMS    "
            f"ΔF1={r.delta_f1:+.4f}  Δkept={r.delta_kept:+.4f}  n={r.n_subsets}"
        )
    print(f"\n  Outputs → {baselines_root}/")
    print("=" * 65)

    return report


if __name__ == "__main__":
    run()
