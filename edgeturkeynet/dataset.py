"""
Dataset loader for YOLO-format aerial turkey detection dataset.

Two detection classes (YOLO label index → class name):
  0 — body  (large oval torso region, primary detection target)
  1 — neck  (smaller elongated protrusion, secondary detail class)

Handles:
- YOLO label format (.txt with class_id cx cy w h, all normalised)
- Letterbox resizing (preserves aspect ratio with padding)
- Bounding box + class-id coordinate scaling
- Per-box class_id tensor returned alongside boxes
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Paths — hardcoded as required
# ---------------------------------------------------------------------------

DATASET_ROOT = Path("data/dataset_split")

TRAIN_IMAGES = DATASET_ROOT / "train" / "images"
TRAIN_LABELS = DATASET_ROOT / "train" / "labels"
VAL_IMAGES   = DATASET_ROOT / "val"   / "images"
VAL_LABELS   = DATASET_ROOT / "val"   / "labels"
TEST_IMAGES  = DATASET_ROOT / "test"  / "images"
TEST_LABELS  = DATASET_ROOT / "test"  / "labels"

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Training input resolution
INPUT_SIZE = (640, 640)  # (H, W)

# Letterbox padding color (ImageNet mean for pretrained backbone consistency)
PAD_COLOR = (114, 114, 114)


# ---------------------------------------------------------------------------
# Utility: Letterbox resize
# ---------------------------------------------------------------------------

def letterbox(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    pad_color: Tuple[int, int, int] = PAD_COLOR,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image with letterboxing (maintaining aspect ratio).

    Letterboxing prevents shape distortion of top-down turkey bodies.
    Distortion would cause oval bodies to appear as circles/rectangles,
    degrading centerness predictions and IoU during training.

    Args:
        image: Input BGR/RGB image as numpy array [H, W, 3].
        target_size: Target (height, width) for output.
        pad_color: RGB padding color tuple.

    Returns:
        image_lb: Letterboxed image [target_H, target_W, 3].
        scale: Scale factor applied to original image.
        padding: (pad_top, pad_left) added in pixels.
    """
    src_h, src_w = image.shape[:2]
    tgt_h, tgt_w = target_size

    # Compute scale preserving aspect ratio
    scale = min(tgt_w / src_w, tgt_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))

    # Resize
    image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to target size
    pad_top  = (tgt_h - new_h) // 2
    pad_left = (tgt_w - new_w) // 2
    pad_bottom = tgt_h - new_h - pad_top
    pad_right  = tgt_w - new_w - pad_left

    image_lb = cv2.copyMakeBorder(
        image_resized,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=pad_color,
    )

    return image_lb, scale, (pad_top, pad_left)


def scale_boxes_to_letterbox(
    boxes_yolo: np.ndarray,
    src_shape: Tuple[int, int],
    scale: float,
    padding: Tuple[int, int],
    target_size: Tuple[int, int] = (640, 640),
) -> np.ndarray:
    """
    Convert YOLO-format boxes to absolute pixel boxes after letterboxing.

    YOLO format: [cx, cy, w, h] normalized to [0,1] in original image.
    Output format: [cx, cy, w, h] normalized to [0,1] in letterboxed image.

    Args:
        boxes_yolo: [N, 4] normalized YOLO boxes in original image space.
        src_shape: Original image (H, W).
        scale: Letterbox scale factor.
        padding: (pad_top, pad_left) in pixels.
        target_size: Letterboxed image (H, W).

    Returns:
        boxes_scaled: [N, 4] normalized boxes in letterboxed image space.
    """
    if len(boxes_yolo) == 0:
        return boxes_yolo

    src_h, src_w = src_shape
    tgt_h, tgt_w = target_size
    pad_top, pad_left = padding

    # Denormalize to original pixel space
    cx = boxes_yolo[:, 0] * src_w
    cy = boxes_yolo[:, 1] * src_h
    w  = boxes_yolo[:, 2] * src_w
    h  = boxes_yolo[:, 3] * src_h

    # Apply letterbox scale and shift
    cx = cx * scale + pad_left
    cy = cy * scale + pad_top
    w  = w  * scale
    h  = h  * scale

    # Renormalize to letterboxed image space
    boxes_scaled = np.stack([
        cx / tgt_w,
        cy / tgt_h,
        w  / tgt_w,
        h  / tgt_h,
    ], axis=-1)

    # Clip to valid range
    boxes_scaled = np.clip(boxes_scaled, 0.0, 1.0)

    return boxes_scaled


# ---------------------------------------------------------------------------
# YOLO Dataset Class
# ---------------------------------------------------------------------------

class TurkeyDataset(Dataset):
    """
    PyTorch Dataset for YOLO-format single-class turkey detection.

    Loads images and corresponding .txt label files.
    Applies letterbox resizing with proper bbox coordinate transformation.
    Handles missing labels (images with no turkeys) gracefully.

    Label format per line: <class_id> <cx> <cy> <w> <h> (normalized 0-1).
    Single-class optimization: class_id is always 0, so we skip class
    dimension parsing and hardcode it — reduces label parsing overhead.

    Args:
        images_dir: Path to images directory.
        labels_dir: Path to labels directory.
        input_size: Model input (H, W).
        augment: Whether to apply basic augmentations (for training).
    """

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        input_size: Tuple[int, int] = INPUT_SIZE,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.input_size = input_size
        self.augment = augment

        # Collect all valid image paths
        self.image_paths: List[Path] = sorted([
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ])

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {self.images_dir}. "
                f"Supported extensions: {IMAGE_EXTENSIONS}"
            )

        print(f"[TurkeyDataset] Loaded {len(self.image_paths)} images from {images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_label(self, image_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load YOLO-format label for a given image.

        Parses the class id from column 0 and keeps it alongside the box.
        Valid class ids are 0 (body) and 1 (neck); any other value is
        silently discarded so mislabelled or legacy single-class files
        that only contain class 0 still load correctly.

        Args:
            image_path: Path to the image file.

        Returns:
            boxes:    [N, 4] float32 array of normalised (cx, cy, w, h).
            class_ids:[N]    int64 array of class indices {0, 1}.
            Both arrays are empty ([0,4] / [0]) when no label file exists.
        """
        label_path = self.labels_dir / (image_path.stem + ".txt")

        if not label_path.exists():
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)

        boxes: List[List[float]] = []
        class_ids: List[int] = []

        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                if cls_id not in (0, 1):
                    # Ignore unexpected class ids gracefully
                    continue
                cx, cy, w, h = (float(parts[1]), float(parts[2]),
                                float(parts[3]), float(parts[4]))
                if w > 0 and h > 0:
                    boxes.append([cx, cy, w, h])
                    class_ids.append(cls_id)

        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)

        return (np.array(boxes, dtype=np.float32),
                np.array(class_ids, dtype=np.int64))

    def _apply_augmentation(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        class_ids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply training augmentations (geometric only — no augmentation research).

        - Horizontal flip  (50 % probability)
        - Vertical flip    (20 % probability — less common in aerial top-down views)

        Class ids are unaffected by flips; they are passed through unchanged
        so the return signature stays consistent.

        Args:
            image:     [H, W, 3] letterboxed image.
            boxes:     [N, 4] normalised (cx, cy, w, h).
            class_ids: [N] int64 class indices.

        Returns:
            Augmented (image, boxes, class_ids).
        """
        if np.random.random() < 0.5:
            image = np.fliplr(image).copy()
            if len(boxes) > 0:
                boxes[:, 0] = 1.0 - boxes[:, 0]  # cx → 1 - cx

        if np.random.random() < 0.2:
            image = np.flipud(image).copy()
            if len(boxes) > 0:
                boxes[:, 1] = 1.0 - boxes[:, 1]  # cy → 1 - cy

        return image, boxes, class_ids

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and preprocess a single sample.

        Returns:
            dict with keys:
                'image':     [3, H, W] float32 tensor normalised to [0, 1].
                'boxes':     [N, 4] float32 tensor (normalised cx, cy, w, h).
                'class_ids': [N]    int64 tensor of class indices {0=body, 1=neck}.
                'num_boxes': scalar int64 tensor.
        """
        image_path = self.image_paths[idx]

        # Load image
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise IOError(f"Cannot read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Load labels (boxes + class ids)
        boxes, class_ids = self._load_label(image_path)

        # Letterbox resize
        src_h, src_w = image_rgb.shape[:2]
        image_lb, scale, padding = letterbox(image_rgb, self.input_size)

        # Scale boxes to letterboxed coordinate space
        if len(boxes) > 0:
            boxes = scale_boxes_to_letterbox(
                boxes, (src_h, src_w), scale, padding, self.input_size
            )

        # Training augmentations (class_ids passed through unchanged)
        if self.augment:
            image_lb, boxes, class_ids = self._apply_augmentation(
                image_lb, boxes, class_ids
            )

        # Convert to tensors
        image_tensor = torch.from_numpy(
            image_lb.astype(np.float32) / 255.0
        ).permute(2, 0, 1).contiguous()

        if len(boxes) > 0:
            boxes_tensor     = torch.from_numpy(boxes)
            class_ids_tensor = torch.from_numpy(class_ids)
        else:
            boxes_tensor     = torch.zeros((0, 4), dtype=torch.float32)
            class_ids_tensor = torch.zeros(0, dtype=torch.int64)

        return {
            "image":     image_tensor,
            "boxes":     boxes_tensor,
            "class_ids": class_ids_tensor,
            "num_boxes": torch.tensor(len(boxes), dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Collate function (handles variable-length box lists)
# ---------------------------------------------------------------------------

def collate_fn(
    batch: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
    """
    Custom collate function for variable number of boxes per image.

    Standard torch.stack() requires equal-size tensors.
    Boxes and class_ids are kept as lists of per-image tensors.

    Args:
        batch: List of dataset samples.

    Returns:
        dict with:
            'images':    [B, 3, H, W] stacked image tensor.
            'boxes':     List[B] of [N_i, 4] box tensors.
            'class_ids': List[B] of [N_i] int64 class-id tensors.
            'num_boxes': [B] tensor of box counts.
    """
    images     = torch.stack([item["image"]     for item in batch], dim=0)
    boxes      = [item["boxes"]     for item in batch]
    class_ids  = [item["class_ids"] for item in batch]
    num_boxes  = torch.stack([item["num_boxes"] for item in batch], dim=0)

    return {
        "images":    images,
        "boxes":     boxes,
        "class_ids": class_ids,
        "num_boxes": num_boxes,
    }


# ---------------------------------------------------------------------------
# DataLoader factory functions
# ---------------------------------------------------------------------------

def get_train_loader(
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create training DataLoader with augmentation.

    Args:
        batch_size: Training batch size.
        num_workers: DataLoader worker processes.
        shuffle: Shuffle dataset each epoch.

    Returns:
        Configured DataLoader.
    """
    dataset = TurkeyDataset(
        images_dir=TRAIN_IMAGES,
        labels_dir=TRAIN_LABELS,
        input_size=INPUT_SIZE,
        augment=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,  # Avoid batch size=1 with BN layers
    )


def get_val_loader(
    batch_size: int = 8,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create validation DataLoader (no augmentation).

    Args:
        batch_size: Validation batch size.
        num_workers: DataLoader worker processes.

    Returns:
        Configured DataLoader.
    """
    dataset = TurkeyDataset(
        images_dir=VAL_IMAGES,
        labels_dir=VAL_LABELS,
        input_size=INPUT_SIZE,
        augment=False,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def get_test_loader(
    batch_size: int = 4,
    num_workers: int = 2,
) -> DataLoader:
    """
    Create test DataLoader for final evaluation.

    Args:
        batch_size: Test batch size.
        num_workers: DataLoader worker processes.

    Returns:
        Configured DataLoader.
    """
    dataset = TurkeyDataset(
        images_dir=TEST_IMAGES,
        labels_dir=TEST_LABELS,
        input_size=INPUT_SIZE,
        augment=False,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
