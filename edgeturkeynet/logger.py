"""
RunLogger — per-run directory management and CSV metric logging.

Each pipeline execution gets its own timestamped directory under
``outputs/runs/<timestamp>_<backbone>/`` that contains:

  config.json               — full RunConfig serialised as JSON
  train_metrics.csv         — per-epoch training losses + val metrics
  test_results.csv          — final per-class test metrics
  checkpoints/              — best.pth and last.pth
  exports/                  — ONNX + INT8 artefacts
  inference_outputs/        — visualisation images

CSV schemas
-----------
train_metrics.csv
  epoch, lr, train_total, train_cls, train_reg, train_ctr,
  val_map, val_ap_body, val_ap_neck,
  val_p_body, val_p_neck, val_r_body, val_r_neck

test_results.csv
  class, ap, precision, recall, map
  (one row per class + one summary row with mAP)
"""

from __future__ import annotations

import csv
import dataclasses
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import RunConfig
    from .evaluate import PerClassMetrics


# ---------------------------------------------------------------------------
# RunLogger
# ---------------------------------------------------------------------------

class RunLogger:
    """
    Creates and manages a unique timestamped run directory and writes
    all training/validation/test metrics to CSV files inside it.

    Directory layout
    ----------------
    outputs/runs/
      20240315_143022_mobilenetv3/
        config.json
        train_metrics.csv
        test_results.csv
        checkpoints/
          best.pth
          last.pth
        exports/
          edge_turkey_net.onnx
          edge_turkey_net_int8.pth
        inference_outputs/
          *.jpg / *.png

    Args:
        cfg: The RunConfig for this run.
    """

    # CSV column headers ────────────────────────────────────────────────
    _TRAIN_HEADER = [
        "epoch",
        "lr",
        "train_total",
        "train_cls",
        "train_reg",
        "train_ctr",
        "val_map",
        "val_ap_body",
        "val_ap_neck",
        "val_p_body",
        "val_p_neck",
        "val_r_body",
        "val_r_neck",
    ]

    _TEST_HEADER = [
        "class",
        "ap",
        "precision",
        "recall",
        "map",
        "fp32_fps",
        "onnx_mb",
        "int8_mb",
    ]

    def __init__(self, cfg: "RunConfig") -> None:
        self.cfg = cfg

        # Build unique run directory name: YYYYMMDD_HHMMSS_<backbone>
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{ts}_{cfg.backbone}"
        self.run_dir: Path = cfg.runs_root / run_name

        # Sub-directories
        self.checkpoint_dir:  Path = self.run_dir / "checkpoints"
        self.export_dir:      Path = self.run_dir / "exports"
        self.vis_dir:         Path = self.run_dir / "inference_outputs"

        # CSV file paths
        self.train_csv: Path = self.run_dir / "train_metrics.csv"
        self.test_csv:  Path = self.run_dir / "test_results.csv"

        # Internal handles (opened lazily)
        self._train_fh  = None
        self._train_writer = None

        self._create_dirs()
        self._write_config()
        self._init_train_csv()

        print(f"[RunLogger] Run directory: {self.run_dir.resolve()}")

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _create_dirs(self) -> None:
        """Create all run subdirectories."""
        for d in (
            self.run_dir,
            self.checkpoint_dir,
            self.export_dir,
            self.vis_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def _write_config(self) -> None:
        """Serialise RunConfig to config.json inside the run directory."""
        cfg_dict = dataclasses.asdict(self.cfg)
        # Path objects are not JSON-serialisable
        for k, v in cfg_dict.items():
            if isinstance(v, Path):
                cfg_dict[k] = str(v)
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(cfg_dict, f, indent=2)

    def _init_train_csv(self) -> None:
        """Create train_metrics.csv and write the header row."""
        self._train_fh     = open(self.train_csv, "w", newline="", buffering=1)
        self._train_writer = csv.DictWriter(
            self._train_fh, fieldnames=self._TRAIN_HEADER
        )
        self._train_writer.writeheader()
        self._train_fh.flush()

    # ------------------------------------------------------------------
    # Public API — training / validation
    # ------------------------------------------------------------------

    def log_epoch(
        self,
        epoch: int,
        lr: float,
        train_losses: Dict[str, float],
        val_metrics: "PerClassMetrics",
    ) -> None:
        """
        Append one row to train_metrics.csv.

        Args:
            epoch:        Current epoch index (0-based).
            lr:           Learning rate used this epoch.
            train_losses: Dict with keys 'total', 'cls', 'reg', 'ctr'.
            val_metrics:  PerClassMetrics from evaluate_map().
        """
        row = {
            "epoch":       epoch,
            "lr":          f"{lr:.8f}",
            "train_total": f"{train_losses.get('total', 0.0):.6f}",
            "train_cls":   f"{train_losses.get('cls',   0.0):.6f}",
            "train_reg":   f"{train_losses.get('reg',   0.0):.6f}",
            "train_ctr":   f"{train_losses.get('ctr',   0.0):.6f}",
            "val_map":     f"{val_metrics.map:.6f}",
            "val_ap_body": f"{val_metrics.ap.get('body',  0.0):.6f}",
            "val_ap_neck": f"{val_metrics.ap.get('neck',  0.0):.6f}",
            "val_p_body":  f"{val_metrics.precision.get('body', 0.0):.6f}",
            "val_p_neck":  f"{val_metrics.precision.get('neck', 0.0):.6f}",
            "val_r_body":  f"{val_metrics.recall.get('body', 0.0):.6f}",
            "val_r_neck":  f"{val_metrics.recall.get('neck', 0.0):.6f}",
        }
        self._train_writer.writerow(row)
        self._train_fh.flush()   # flush so results are visible even if training crashes

    # ------------------------------------------------------------------
    # Public API — test results
    # ------------------------------------------------------------------

    def log_test_results(
        self,
        metrics: "PerClassMetrics",
        fp32_fps: float = float("nan"),
        onnx_mb: float = float("nan"),
        int8_mb: float = float("nan"),
    ) -> None:
        """
        Write per-class test results to test_results.csv.

        Writes one row per class (body, neck) plus a summary row
        that includes mAP and optional benchmark values.

        Args:
            metrics:  PerClassMetrics returned by evaluate_map() on test set.
            fp32_fps: FP32 CPU frames-per-second (from benchmark_cpu).
            onnx_mb:  ONNX file size in MB (nan if export skipped).
            int8_mb:  INT8 state-dict size in MB (nan if export skipped).
        """
        from .model import CLASS_NAMES

        def _fmt(v: float) -> str:
            return f"{v:.6f}" if v == v else ""  # nan → empty string

        with open(self.test_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._TEST_HEADER)
            writer.writeheader()

            for cls_name in CLASS_NAMES:
                writer.writerow({
                    "class":     cls_name,
                    "ap":        _fmt(metrics.ap.get(cls_name, 0.0)),
                    "precision": _fmt(metrics.precision.get(cls_name, 0.0)),
                    "recall":    _fmt(metrics.recall.get(cls_name, 0.0)),
                    "map":       "",
                    "fp32_fps":  "",
                    "onnx_mb":   "",
                    "int8_mb":   "",
                })

            # Summary row — mAP + benchmark figures
            writer.writerow({
                "class":     "ALL",
                "ap":        "",
                "precision": _fmt(metrics.mean_precision),
                "recall":    _fmt(metrics.mean_recall),
                "map":       _fmt(metrics.map),
                "fp32_fps":  _fmt(fp32_fps),
                "onnx_mb":   _fmt(onnx_mb),
                "int8_mb":   _fmt(int8_mb),
            })

        print(f"[RunLogger] Test results → {self.test_csv}")

    # ------------------------------------------------------------------
    # Checkpoint path helpers
    # ------------------------------------------------------------------

    @property
    def best_model_path(self) -> Path:
        """Path to the best checkpoint inside this run's directory."""
        return self.checkpoint_dir / "best.pth"

    @property
    def last_model_path(self) -> Path:
        """Path to the latest checkpoint inside this run's directory."""
        return self.checkpoint_dir / "last.pth"

    @property
    def onnx_path(self) -> Path:
        """Path for the exported ONNX model."""
        return self.export_dir / "edge_turkey_net.onnx"

    @property
    def int8_path(self) -> Path:
        """Path for the INT8 quantised state-dict."""
        return self.export_dir / "edge_turkey_net_int8.pth"

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close any open CSV file handles."""
        if self._train_fh is not None:
            self._train_fh.flush()
            self._train_fh.close()
            self._train_fh     = None
            self._train_writer = None

    def __del__(self) -> None:
        self.close()
