import os
import sys
import json
import torch

from mob1.config import CHECKPOINTS_DIR, RESULTS_DIR, INPUT_SIZE, NUM_CLASSES
from mob1.head import NanoDetTurkey
from mob1.dataloader import get_dataloader
from mob1.train import train
from mob1.evaluate import evaluate, print_metrics


# ---------------------------------------------------------------------------
# Stage: Evaluation on test set
# ---------------------------------------------------------------------------

def run_eval(split="test", weights="best.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = NanoDetTurkey().to(device)

    ckpt_path = os.path.join(CHECKPOINTS_DIR, weights)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded weights from {ckpt_path} (epoch {ckpt.get('epoch','?')})")

    loader  = get_dataloader(split, augment=False)
    metrics = evaluate(model, loader, device)
    print_metrics(metrics, prefix=f"{split.upper()}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{split}_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {out_path}")
    return metrics


# ---------------------------------------------------------------------------
# Stage: Export to TFLite INT8 for Raspberry Pi Zero
# ---------------------------------------------------------------------------

def run_export(weights="best.pt"):
    """
    Export pipeline:
      PyTorch → ONNX → TFLite INT8 (representative dataset calibration)

    Requirements:
      pip install onnx onnx-tf tensorflow
    """
    try:
        import onnx
        import onnxruntime
    except ImportError:
        print("onnx / onnxruntime not installed. Skipping export.")
        print("  pip install onnx onnxruntime onnx-tf tensorflow")
        return

    device   = torch.device("cpu")
    model    = NanoDetTurkey().to(device)
    ckpt     = torch.load(os.path.join(CHECKPOINTS_DIR, weights),
                          map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    onnx_path  = os.path.join(RESULTS_DIR, "nanodet_turkey.onnx")
    tflite_path = os.path.join(RESULTS_DIR, "nanodet_turkey_int8.tflite")

    # --- ONNX export ---
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"],
        output_names=["cls_pred", "reg_pred", "boxes"],
        opset_version=12,
        dynamic_axes={"input": {0: "batch"}},
    )
    print(f"ONNX model saved to {onnx_path}")

    # --- ONNX → TF → TFLite INT8 ---
    try:
        from onnx_tf.backend import prepare
        import tensorflow as tf

        tf_model_dir = os.path.join(RESULTS_DIR, "tf_model")
        tf_rep = prepare(onnx.load(onnx_path))
        tf_rep.export_graph(tf_model_dir)

        # Representative dataset for INT8 calibration
        cal_loader = get_dataloader("val", augment=False)

        def representative_dataset():
            for imgs, _ in cal_loader:
                for i in range(len(imgs)):
                    yield [imgs[i:i+1].numpy()]

        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)
        converter.optimizations           = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset  = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type    = tf.uint8
        converter.inference_output_type   = tf.float32

        tflite_model = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        size_kb = os.path.getsize(tflite_path) / 1024
        print(f"TFLite INT8 model saved to {tflite_path} ({size_kb:.1f} KB)")

    except Exception as e:
        print(f"TFLite conversion failed: {e}")
        print("ONNX model is still available for manual conversion.")


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

def run_pipeline(stage: str = "all", resume_from: str = None):
    """
    stage: "all" | "train" | "eval" | "export"
    resume_from: path to a checkpoint .pt file to resume training
    """
    print("=" * 60)
    print("  NanoDetPlus-Turkey Pipeline")
    print(f"  Stage: {stage}")
    print("=" * 60)

    if stage in ("all", "train"):
        print("\n[1/3] TRAINING")
        train(resume_from=resume_from)

    if stage in ("all", "eval"):
        print("\n[2/3] EVALUATION (test set)")
        metrics = run_eval(split="test", weights="best.pt")
        print(f"\nFinal test results:")
        print_metrics(metrics, prefix="TEST")

    if stage in ("all", "export"):
        print("\n[3/3] EXPORT → TFLite INT8")
        run_export(weights="best.pt")

    print("\nDone.")


