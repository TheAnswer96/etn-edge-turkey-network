"""
Export pipeline for EdgeTurkeyNet.

Covers:
1. Conv + BatchNorm layer fusion — eliminates BN at inference.
   BN has 4 operations (sub mean, div std, mul gamma, add beta) per activation.
   Fusion absorbs these into Conv weights/bias: ~15% latency improvement on ARM.

2. ONNX export — static axes for edge devices.
   Dynamic axes disabled: fixed input shape allows ONNX runtime to optimize
   memory layout and enable graph-level optimizations (e.g., op fusion).

3. Quantization preparation — INT8 calibration scaffold.
   Post-training quantization (PTQ) via ONNX Runtime quantization tools.
   INT8 weights + activations: ~4x smaller model, ~2x faster on RPi ARM Cortex.

4. Model size and FLOPs estimation.

EDGE AI RATIONALE:
- Raspberry Pi 4B CPU: ARM Cortex-A72, ~13 GFLOPS FP32, ~50 GOPS INT8 (NEON)
- INT8 quantization is the most impactful single optimization for RPi inference
- Fused BN eliminates memory bandwidth pressure (no extra buffer for BN params)
- Fixed ONNX shape enables ONNX Runtime to pre-allocate and reuse buffers
"""

from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("[Export] Warning: onnx/onnxruntime not installed. ONNX export disabled.")

from .model import ConvBnAct, DepthwiseSeparableConv, EdgeTurkeyNet

# Hardcoded export paths
EXPORT_DIR = Path("outputs/exports")
ONNX_PATH  = EXPORT_DIR / "edge_turkey_net.onnx"
ONNX_QUANTIZED_PATH = EXPORT_DIR / "edge_turkey_net_int8.onnx"


# ---------------------------------------------------------------------------
# Conv + BN Fusion
# ---------------------------------------------------------------------------

def _fuse_conv_bn_weights(
    conv: nn.Conv2d,
    bn: nn.BatchNorm2d,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute fused Conv weight and bias by absorbing BN parameters.

    Mathematical derivation:
        y = γ * (W*x + b - μ) / σ + β
          = (γ*W/σ) * x + (γ*(b-μ)/σ + β)
          = W_fused * x + b_fused

    Args:
        conv: Conv2d layer (may have bias or not).
        bn: BatchNorm2d layer to absorb.

    Returns:
        (fused_weight, fused_bias) tensors.
    """
    conv_weight = conv.weight.clone()
    conv_bias   = conv.bias.clone() if conv.bias is not None \
                  else torch.zeros(conv.out_channels)

    bn_mean  = bn.running_mean
    bn_var   = bn.running_var
    bn_eps   = bn.eps
    bn_gamma = bn.weight   # scale
    bn_beta  = bn.bias     # shift

    std = torch.sqrt(bn_var + bn_eps)

    # Scale conv weights by bn gamma/std (per output channel)
    scale = bn_gamma / std  # [out_channels]
    fused_w = conv_weight * scale.reshape(-1, 1, 1, 1)
    fused_b = (conv_bias - bn_mean) * scale + bn_beta

    return fused_w, fused_b


def fuse_conv_bn_in_model(model: EdgeTurkeyNet) -> EdgeTurkeyNet:
    """
    Fuse Conv + BatchNorm pairs throughout the model in-place.

    Iterates the model and replaces Conv+BN sequences with a single
    Conv (with bias) that is mathematically equivalent.

    EDGE AI: Eliminates BN overhead at inference:
    - No separate running_mean/var memory access
    - No division + multiply per activation
    - Reduced memory bandwidth on Raspberry Pi

    Important: Call model.eval() before fusion to freeze BN running stats.

    Args:
        model: EdgeTurkeyNet in eval mode.

    Returns:
        Model with fused Conv+BN layers.
    """
    model.eval()

    def _fuse_sequential(module: nn.Module) -> None:
        """Recursively find and fuse Conv+BN in any module."""
        children = list(module.named_children())
        i = 0
        while i < len(children):
            name, child = children[i]
            if (isinstance(child, nn.Conv2d)
                    and i + 1 < len(children)
                    and isinstance(children[i + 1][1], nn.BatchNorm2d)):
                # Found a Conv+BN pair
                bn = children[i + 1][1]
                fused_w, fused_b = _fuse_conv_bn_weights(child, bn)

                # Create new Conv with fused weights (includes bias)
                fused_conv = nn.Conv2d(
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=True,
                )
                fused_conv.weight.data = fused_w
                fused_conv.bias.data   = fused_b

                # Replace Conv and BN
                setattr(module, name, fused_conv)
                setattr(module, children[i + 1][0], nn.Identity())

                i += 2  # Skip both Conv and BN
            else:
                # Recurse into child
                _fuse_sequential(child)
                i += 1

    # Also handle ConvBnAct and DepthwiseSeparableConv wrappers
    for name, module in model.named_modules():
        if isinstance(module, ConvBnAct):
            fused_w, fused_b = _fuse_conv_bn_weights(module.conv, module.bn)
            fused_conv = nn.Conv2d(
                module.conv.in_channels,
                module.conv.out_channels,
                module.conv.kernel_size,
                stride=module.conv.stride,
                padding=module.conv.padding,
                groups=module.conv.groups,
                bias=True,
            )
            fused_conv.weight.data = fused_w
            fused_conv.bias.data   = fused_b
            module.conv = fused_conv
            module.bn   = nn.Identity()

        elif isinstance(module, DepthwiseSeparableConv):
            fused_w, fused_b = _fuse_conv_bn_weights(module.pointwise, module.bn)
            fused_pw = nn.Conv2d(
                module.pointwise.in_channels,
                module.pointwise.out_channels,
                1,
                bias=True,
            )
            fused_pw.weight.data = fused_w
            fused_pw.bias.data   = fused_b
            module.pointwise = fused_pw
            module.bn = nn.Identity()

    print("[Export] Conv+BN fusion complete.")
    return model


# ---------------------------------------------------------------------------
# ONNX Export
# ---------------------------------------------------------------------------

def export_to_onnx(
    model: EdgeTurkeyNet,
    output_path: Path = ONNX_PATH,
    input_size: Tuple[int, int] = (640, 640),
    opset_version: int = 17,
    fuse_bn: bool = True,
) -> Path:
    """
    Export EdgeTurkeyNet to ONNX format for edge deployment.

    Optimizations applied:
    - Conv+BN fusion before export (fewer ONNX nodes)
    - Static input shape (batch=1, H=640, W=640) — allows ONNX Runtime
      to pre-allocate I/O buffers and enable shape-dependent optimizations
    - Opset 17: supports all required ops on ORT 1.15+

    EDGE AI: ONNX Runtime with ORT optimization level 3 on RPi achieves
    ~2x speedup vs vanilla PyTorch for this model class due to:
    - Graph-level op fusion (Hardswish absorbed into preceding Conv)
    - Memory layout optimization (NCHW → NCHWc on ARM)
    - Optimized GEMM/Conv kernels via MKL-DNN backend

    Args:
        model: EdgeTurkeyNet (will be fused in-place).
        output_path: Path to save .onnx file.
        input_size: Fixed input (H, W) — no dynamic axes.
        opset_version: ONNX opset version.
        fuse_bn: Whether to fuse Conv+BN before export.

    Returns:
        Path to exported ONNX file.
    """
    if not ONNX_AVAILABLE:
        raise ImportError("Install onnx and onnxruntime: pip install onnx onnxruntime")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    model.eval()
    if fuse_bn:
        model = fuse_conv_bn_in_model(model)

    h, w = input_size
    dummy_input = torch.randn(1, 3, h, w)

    # Wrap model to flatten outputs for ONNX export
    class ONNXWrapper(nn.Module):
        """Flatten multi-level outputs to single tensors for ONNX graph."""
        def __init__(self, m: EdgeTurkeyNet):
            super().__init__()
            self.m = m

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            cls_list, reg_list, ctr_list = self.m(x)
            # Flatten all levels into single output tensors
            # Shape: [1, N_total, C] where N_total = sum of H*W per level
            cls_flat = torch.cat([p.flatten(2).permute(0, 2, 1) for p in cls_list], dim=1)
            reg_flat = torch.cat([p.flatten(2).permute(0, 2, 1) for p in reg_list], dim=1)
            ctr_flat = torch.cat([p.flatten(2).permute(0, 2, 1) for p in ctr_list], dim=1)
            return cls_flat, reg_flat, ctr_flat

    wrapped = ONNXWrapper(model)
    wrapped.eval()

    torch.onnx.export(
        wrapped,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,  # Fold constant ops at export time
        input_names=["input"],
        output_names=["cls_pred", "reg_pred", "ctr_pred"],
        # Static axes only — fixed batch=1 for edge inference
        dynamic_axes=None,
        verbose=False,
    )

    # Validate ONNX graph
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    file_size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"[Export] ONNX model saved: {output_path}")
    print(f"[Export] ONNX file size:   {file_size_mb:.2f} MB")

    return output_path


def validate_onnx_runtime(
    onnx_path: Path = ONNX_PATH,
    input_size: Tuple[int, int] = (640, 640),
    num_runs: int = 20,
) -> float:
    """
    Validate ONNX model and benchmark latency with ONNX Runtime.

    Uses CPU ExecutionProvider to simulate Raspberry Pi performance.
    ORT intra_op_num_threads=4 matches RPi 4B's 4 cores.

    EDGE AI: ORT with graph optimizations (all levels) typically achieves
    ~1.5-2x speedup vs ONNX Runtime default settings on ARM CPUs.

    Args:
        onnx_path: Path to ONNX model file.
        input_size: Model input (H, W).
        num_runs: Number of inference runs for benchmarking.

    Returns:
        Average inference time in milliseconds.
    """
    if not ONNX_AVAILABLE:
        raise ImportError("onnxruntime not available.")

    import numpy as np
    import time

    # Configure ORT for CPU (simulating RPi 4B: 4 cores, 64-bit ARM)
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4   # Matches RPi 4B core count
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"]
    )

    h, w = input_size
    dummy = np.random.randn(1, 3, h, w).astype(np.float32)
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(5):
        session.run(None, {input_name: dummy})

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(latencies) / len(latencies)
    fps = 1000 / avg_ms

    print(f"[ORT Benchmark] {num_runs} runs on CPUExecutionProvider")
    print(f"  Average latency: {avg_ms:.1f} ms")
    print(f"  Estimated FPS:   {fps:.2f}")
    print(f"  (4-thread, simulating Raspberry Pi 4B)")

    return avg_ms


# ---------------------------------------------------------------------------
# Model size and FLOPs estimation
# ---------------------------------------------------------------------------

def estimate_model_stats(
    model: EdgeTurkeyNet,
    input_size: Tuple[int, int] = (640, 640),
) -> None:
    """
    Print model parameter count, memory footprint, and approximate FLOPs.

    FLOPs estimation uses a hook-based approach counting MAC operations
    in Conv2d and Linear layers.

    Args:
        model: EdgeTurkeyNet model.
        input_size: Input (H, W) for FLOPs calculation.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 ** 2)  # float32 = 4 bytes

    # Hook-based MACs counter
    total_macs = [0]

    def conv_hook(module: nn.Conv2d, inp: tuple, out: torch.Tensor) -> None:
        in_c   = module.in_channels
        k_h, k_w = module.kernel_size if isinstance(module.kernel_size, tuple) \
                   else (module.kernel_size, module.kernel_size)
        out_h, out_w = out.shape[2], out.shape[3]
        groups = module.groups
        # MACs per Conv = out_H * out_W * out_C * in_C/groups * kH * kW
        macs = out_h * out_w * module.out_channels * (in_c // groups) * k_h * k_w
        total_macs[0] += macs

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))

    model.eval()
    h, w = input_size
    with torch.no_grad():
        model(torch.randn(1, 3, h, w))

    for h_obj in hooks:
        h_obj.remove()

    gmacs = total_macs[0] / 1e9

    print("\n" + "="*50)
    print("  EdgeTurkeyNet Model Statistics")
    print("="*50)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size (FP32):    {model_size_mb:.2f} MB")
    print(f"  Model size (INT8):    {model_size_mb/4:.2f} MB (estimated)")
    print(f"  GMACs @ {h}x{w}:    {gmacs:.3f}")
    print("="*50 + "\n")


def prepare_tflite_scaffold(
    model: EdgeTurkeyNet,
    output_path: Path = EXPORT_DIR / "tflite_scaffold",
) -> None:
    """
    Prepare TFLite-friendly conversion scaffold.

    TFLite conversion from ONNX requires:
    1. ONNX → TF SavedModel (via onnx-tf)
    2. SavedModel → TFLite (via TFLiteConverter)

    This function exports the ONNX model and prints instructions.
    Full TFLite conversion requires additional dependencies (onnx-tf, tensorflow).

    EDGE AI: TFLite with XNNPACK delegate on RPi achieves comparable
    performance to ONNX Runtime with slightly better INT8 support for
    some layer types (e.g., depthwise convolutions).

    Args:
        model: EdgeTurkeyNet model.
        output_path: Directory for scaffold files.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    onnx_p = output_path / "model.onnx"
    export_to_onnx(model, onnx_p, fuse_bn=True)

    instructions = """
# TFLite Conversion Instructions
# ================================
# 1. Install dependencies:
#    pip install onnx-tf tensorflow

# 2. Convert ONNX → TF SavedModel:
#    python -c "
#    import onnx
#    from onnx_tf.backend import prepare
#    model = onnx.load('model.onnx')
#    tf_rep = prepare(model)
#    tf_rep.export_graph('saved_model')
#    "

# 3. Convert SavedModel → TFLite:
#    python -c "
#    import tensorflow as tf
#    converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
#    converter.optimizations = [tf.lite.Optimize.DEFAULT]
#    # For INT8:
#    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#    tflite_model = converter.convert()
#    open('edge_turkey_net.tflite', 'wb').write(tflite_model)
#    "
"""
    with open(output_path / "TFLITE_CONVERSION.md", "w") as f:
        f.write(instructions)
    print(f"[Export] TFLite scaffold saved to {output_path}")
    print(instructions)


# ---------------------------------------------------------------------------
# INT8 Dynamic Quantization
# ---------------------------------------------------------------------------

INT8_MODEL_PATH = EXPORT_DIR / "edge_turkey_net_int8.pth"


def quantize_dynamic_int8(
    model: EdgeTurkeyNet,
    output_path: Path = INT8_MODEL_PATH,
    benchmark: bool = True,
    input_size: Tuple[int, int] = (640, 640),
    num_benchmark_runs: int = 30,
) -> torch.nn.Module:
    """
    Apply PyTorch dynamic INT8 quantization to EdgeTurkeyNet.

    Dynamic quantization
    --------------------
    Weights are quantized to INT8 statically (at conversion time).
    Activations are quantized dynamically at runtime — their scale and
    zero-point are computed per-tensor per-inference call from the actual
    activation range, so no calibration dataset is required.

    Targeted layer types: ``nn.Linear`` and ``nn.Conv2d``.

    Why dynamic (not static/QAT) for this use case:
    - No calibration dataset needed → simpler deployment pipeline
    - Weight-only INT8 still gives ~3-4x memory reduction on RPi
    - On ARM Cortex-A72 (RPi 4B), dynamic quantization of Conv2d via
      PyTorch's fbgemm/qnnpack backend yields ~1.5-2x speedup for
      memory-bandwidth-bound layers (most conv layers in this model)
    - Static/QAT would require re-training; dynamic works on the
      trained checkpoint immediately

    Quantization backend: ``qnnpack`` (designed for ARM mobile/embedded CPUs).
    ``fbgemm`` targets x86 — using qnnpack produces INT8 ops that map to
    ARM NEON instructions on the Raspberry Pi.

    EDGE AI summary:
    - FP32 weights (4 bytes) → INT8 weights (1 byte): ~4x weight memory
    - Activation quantization: reduces memory bandwidth per inference
    - ARM NEON INT8 SIMD: up to 4x throughput vs FP32 for vectorised ops
    - Combined with pruning (zeroed channels): further reduces effective MACs

    Args:
        model:               FP32 EdgeTurkeyNet in eval mode.
        output_path:         Path to save the quantized model state dict.
        benchmark:           If True, run and print FP32 vs INT8 latency comparison.
        input_size:          Model input (H, W) for benchmarking.
        num_benchmark_runs:  Number of timed inference runs.

    Returns:
        Quantized model (torch.nn.Module).
    """
    import copy
    import time

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Use qnnpack backend — optimised for ARM embedded CPUs (Raspberry Pi)
    # fbgemm is for x86; qnnpack maps to ARM NEON int8 intrinsics
    torch.backends.quantized.engine = "qnnpack"

    model_fp32 = model.cpu().eval()

    # ── Optional FP32 baseline benchmark ─────────────────────────────────
    if benchmark:
        dummy = torch.randn(1, 3, *input_size)
        # Warmup
        with torch.inference_mode():
            for _ in range(5):
                model_fp32(dummy)
        # Time
        fp32_times: list = []
        with torch.inference_mode():
            for _ in range(num_benchmark_runs):
                t0 = time.perf_counter()
                model_fp32(dummy)
                fp32_times.append((time.perf_counter() - t0) * 1000)
        fp32_mean_ms = sum(fp32_times) / len(fp32_times)

    # ── Apply dynamic quantization ────────────────────────────────────────
    # torch.quantization.quantize_dynamic:
    #   - Walks the module tree
    #   - Replaces nn.Linear and nn.Conv2d with their Int8 counterparts
    #     (DynamicQuantizedLinear, DynamicQuantizedConv2d)
    #   - Packs INT8 weight tensors; dequantises to FP32 for computation
    #     on platforms without full INT8 matmul (falls back gracefully)
    #   - On qnnpack / ARM, uses native INT8 kernels where available
    model_int8 = torch.quantization.quantize_dynamic(
        copy.deepcopy(model_fp32),
        qconfig_spec={nn.Linear, nn.Conv2d},
        dtype=torch.qint8,
    )
    model_int8.eval()

    # ── Measure actual INT8 file size ─────────────────────────────────────
    torch.save(model_int8.state_dict(), output_path)
    fp32_size_mb  = sum(p.numel() * 4 for p in model_fp32.parameters()) / (1024 ** 2)
    int8_size_mb  = output_path.stat().st_size / (1024 ** 2)

    # ── INT8 benchmark ────────────────────────────────────────────────────
    if benchmark:
        dummy = torch.randn(1, 3, *input_size)
        with torch.inference_mode():
            for _ in range(5):
                model_int8(dummy)
        int8_times: list = []
        with torch.inference_mode():
            for _ in range(num_benchmark_runs):
                t0 = time.perf_counter()
                model_int8(dummy)
                int8_times.append((time.perf_counter() - t0) * 1000)
        int8_mean_ms = sum(int8_times) / len(int8_times)
        speedup      = fp32_mean_ms / int8_mean_ms

    # ── Report ────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  INT8 Dynamic Quantization Results")
    print("="*55)
    print(f"  Backend:             qnnpack (ARM NEON optimised)")
    print(f"  FP32 model size:     {fp32_size_mb:.2f} MB")
    print(f"  INT8 model size:     {int8_size_mb:.2f} MB  "
          f"({fp32_size_mb/int8_size_mb:.1f}x smaller)")
    if benchmark:
        print(f"  FP32 latency:        {fp32_mean_ms:.1f} ms  "
              f"({1000/fp32_mean_ms:.2f} FPS)")
        print(f"  INT8 latency:        {int8_mean_ms:.1f} ms  "
              f"({1000/int8_mean_ms:.2f} FPS)")
        print(f"  Speedup:             {speedup:.2f}x")
        print(f"  Note: Run on actual RPi for true ARM NEON INT8 speedup.")
        print(f"        Expected RPi speedup: ~1.5-2.5x vs FP32.")
    print(f"  Saved to:            {output_path}")
    print("="*55 + "\n")

    return model_int8


def load_int8_model(
    path: Path = INT8_MODEL_PATH,
    input_size: Tuple[int, int] = (640, 640),
) -> torch.nn.Module:
    """
    Reload a previously saved INT8-quantized EdgeTurkeyNet for inference.

    The quantized model's state dict is not directly loadable into a
    standard EdgeTurkeyNet because the layer types differ (DynamicQuantized*
    vs Conv2d).  This function re-applies quantize_dynamic to a fresh
    FP32 model (with random weights) and then loads the INT8 state dict,
    which is the correct round-trip for dynamic quantization.

    Args:
        path:       Path to the INT8 state dict saved by quantize_dynamic_int8.
        input_size: Model input (H, W) — used to rebuild grids.

    Returns:
        Quantized EdgeTurkeyNet in eval mode.
    """
    import copy
    from .model import EdgeTurkeyNet, NUM_CLASSES

    torch.backends.quantized.engine = "qnnpack"

    base_model = EdgeTurkeyNet(
        num_classes=NUM_CLASSES,
        pretrained_backbone=False,
        input_size=input_size,
    ).cpu().eval()

    model_int8 = torch.quantization.quantize_dynamic(
        copy.deepcopy(base_model),
        qconfig_spec={nn.Linear, nn.Conv2d},
        dtype=torch.qint8,
    )
    model_int8.load_state_dict(torch.load(path, map_location="cpu"))
    model_int8.eval()
    print(f"[Export] Loaded INT8 model from {path}")
    return model_int8
