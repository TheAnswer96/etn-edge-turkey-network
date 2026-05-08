# EdgeTurkeyNet

Lightweight aerial turkey detector for edge deployment on Raspberry Pi. Detects turkeys from drone top-down footage in two classes: **body** (class 0) and **neck** (class 1).

Built around a novel **Oval-FCOS** detection head with **DIoU-NMS** and **periodic channel pruning**, targeting 3–8 FPS on RPi CPU.

---

## Architecture

```
Backbone (interchangeable)
    ↓
PAN-Lite Neck  (depthwise separable convs + SE attention)
    P3 (128ch) · P4 (96ch) · P5 (64ch)
    ↓
Oval-FCOS Head  (anchor-free, per-pixel cls + reg + oval centerness)
    ↓
DIoU-NMS  (per-class, cross-class suppression disabled)
```

### Backbones

| Flag | Backbone | Init |
|------|----------|------|
| `mobilenetv3` | MobileNetV3-Small | pretrained ImageNet | 
| `shufflenetv2` | ShuffleNetV2-0.5x | pretrained ImageNet | 
| `mobilenetv1` | MobileNetV1-1.0 | from scratch | 

### Novel contributions

- **Oval-biased centerness** — aspect ratio tuned for top-down turkey body shape
- **CIoU regression loss** — aspect-ratio aware bounding box regression
- **DIoU-NMS** — centroid-distance-aware suppression reduces missed detections in dense flocks
- **Periodic channel pruning** — structured sparsity during training for RPi deployment

### Loss

| Term | Function | Weight |
|------|----------|--------|
| Classification | Focal loss | 1.0 |
| Regression | CIoU | 2.0 |
| Centerness | BCE | 0.5 |

---

## Setup

```bash
pip install -r requirements.txt
```

**Dependencies:** `torch`, `torchvision`, `opencv-python`, `numpy`, `onnx`, `onnxruntime`, `ultralytics`

Dataset expected at `data/dataset_split/{train,val,test}/{images,labels}` in YOLO format (`class_id cx cy w h` normalized).

---

## Usage

### Train

```bash
# Default (ShuffleNetV2-0.5x, 100 epochs)
python scripts/run_training.py

# ShuffleNetV2, custom hyperparameters
python scripts/run_training.py --backbone shufflenetv2 --epochs 50 --batch-size 16

# MobileNetV1, tighter thresholds
python scripts/run_training.py --backbone mobilenetv1 --score-threshold 0.25 --iou-threshold 0.45
```

### Evaluate existing checkpoint

```bash
python scripts/run_training.py --skip-train --resume-from outputs/runs/<timestamp>/checkpoints/best.pth

# Skip export and visualisation steps
python scripts/run_training.py --skip-train --skip-export --skip-visualise \
    --resume-from outputs/runs/<timestamp>/checkpoints/best.pth
```

### Ablation baselines

```bash
python scripts/run_baselines.py
```

Runs up to 6 experiments (toggle `RUN_*` flags at top of file):

| ID | Experiment | Purpose |
|----|-----------|---------|
| A | MobileNetV3 + SSD | isolates head design vs. FCOS |
| B | Standard BCE + GIoU loss | isolates loss novelty |
| C | DIoU-NMS vs IoU-NMS | dense-flock NMS benchmark |
| D | NanoDet (GFL anchor-free) | external baseline |
| E | YOLOv5-nano | anchor-based external baseline |
| F | YOLOv8-nano | anchor-free external baseline |

### Knowledge distillation

```bash
python scripts/run_distillation.py
```

Trains a scaled-down student (ShuffleNetV2, configurable width/depth multipliers) against a frozen teacher checkpoint. Runs KD student and scratch student in parallel for direct comparison.

### Frame inference

```bash
python scripts/detect_frames.py
```

Runs EdgeTurkeyNet on `data/raw/frame/90`, writes annotated images + `detections.csv` to `outputs/predictions/frame_90/`.

---

## Outputs

Every training run creates a timestamped directory:

```
outputs/runs/<YYYYMMDD_HHMMSS>_<backbone>/
    config.json             full RunConfig snapshot
    train_metrics.csv       per-epoch losses + per-class val metrics
    test_results.csv        final per-class AP/P/R + benchmark
    summary.json            key results for scripted comparison
    checkpoints/
        best.pth
        last.pth
    exports/
        edge_turkey_net.onnx        FP32 ONNX (for ORT on RPi)
        edge_turkey_net_int8.pth    INT8 dynamic quantized
    inference_outputs/
        *.jpg
```

## Project Structure

```
edgeturkeynet/      Model package — config, model, loss, train, evaluate, export, inference, logger
yolo_baselines/     Ultralytics YOLO11 training/testing wrapper
data_pipeline/      Frame extraction, VOC→YOLO conversion, dataset statistics
scripts/            Entry-point scripts
data/               Dataset (not tracked in git)
outputs/            Run artifacts (not tracked in git)
archive/            Legacy implementations (mob1, nn) — reference only
```

---

## Configuration

All hyperparameters live in `edgeturkeynet/config.py` as module-level vars and are exposed as CLI flags. Run `python scripts/run_training.py --help` for the full list.

Key parameters:

| Flag | Default | Description |
|------|---------|-------------|
| `--backbone` | `mobilenetv3` | `mobilenetv3` / `shufflenetv2` / `mobilenetv1` |
| `--epochs` | `100` | Training epochs |
| `--batch-size` | `16` | Batch size |
| `--base-lr` | `1e-3` | Base learning rate |
| `--score-threshold` | `0.30` | Detection confidence threshold |
| `--iou-threshold` | `0.50` | NMS and eval IoU threshold |
| `--prune-per-call` | `0.15` | Fraction of channels pruned per pruning step |
| `--prune-max-sparsity` | `0.50` | Maximum cumulative channel sparsity |
