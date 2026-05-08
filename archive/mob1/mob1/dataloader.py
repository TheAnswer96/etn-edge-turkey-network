# =============================================================================
# dataloader.py — YOLO-format dataset with mosaic + standard augmentations
# =============================================================================

import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from mob1.config import (
    INPUT_SIZE, BATCH_SIZE, NUM_WORKERS, SEED,
    MOSAIC_PROB, FLIP_PROB, HSV_H, HSV_S, HSV_V,
    DEGREES, TRANSLATE, SCALE,
    TRAIN_DIR, VAL_DIR, TEST_DIR,
)


def load_label(label_path):
    """Returns (N,5) array: [cls, cx, cy, w, h] normalized, or empty."""
    if not os.path.exists(label_path):
        return np.zeros((0, 5), dtype=np.float32)
    with open(label_path) as f:
        lines = f.read().strip().splitlines()
    if not lines:
        return np.zeros((0, 5), dtype=np.float32)
    return np.array([list(map(float, l.split())) for l in lines],
                    dtype=np.float32)


def xywhn_to_xyxy(boxes, w, h):
    """Convert normalized cxcywh → pixel xyxy."""
    out = np.zeros_like(boxes)
    out[:, 0] = (boxes[:, 1] - boxes[:, 3] / 2) * w
    out[:, 1] = (boxes[:, 2] - boxes[:, 4] / 2) * h
    out[:, 2] = (boxes[:, 1] + boxes[:, 3] / 2) * w
    out[:, 3] = (boxes[:, 2] + boxes[:, 4] / 2) * h
    return out


def xyxy_to_xywhn(boxes, w, h):
    """Convert pixel xyxy → normalized cxcywh."""
    out = np.zeros_like(boxes)
    out[:, 0] = ((boxes[:, 0] + boxes[:, 2]) / 2) / w
    out[:, 1] = ((boxes[:, 1] + boxes[:, 3]) / 2) / h
    out[:, 2] = (boxes[:, 2] - boxes[:, 0]) / w
    out[:, 3] = (boxes[:, 3] - boxes[:, 1]) / h
    return out


def augment_hsv(img):
    r = np.random.uniform(-1, 1, 3) * [HSV_H, HSV_S, HSV_V] + 1
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lut = np.arange(256, dtype=np.float32)
    lut_h = ((lut * r[0]) % 180).astype(np.uint8)
    lut_s = np.clip(lut * r[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(lut * r[2], 0, 255).astype(np.uint8)
    img_hsv[..., 0] = cv2.LUT(img_hsv[..., 0], lut_h)
    img_hsv[..., 1] = cv2.LUT(img_hsv[..., 1], lut_s)
    img_hsv[..., 2] = cv2.LUT(img_hsv[..., 2], lut_v)
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


def random_affine(img, labels, degrees, translate, scale):
    """Mild affine: rotation + translation + scale. No shear for top-view."""
    h, w = img.shape[:2]
    angle  = random.uniform(-degrees, degrees)
    t      = random.uniform(-translate, translate)
    s      = random.uniform(1 - scale, 1 + scale)

    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, s)
    M[0, 2] += t * w
    M[1, 2] += t * h

    img_out = cv2.warpAffine(img, M, (w, h), borderValue=(114, 114, 114))

    if len(labels) == 0:
        return img_out, labels

    # Transform box corners
    cls  = labels[:, 0:1]
    xyxy = xywhn_to_xyxy(
        np.hstack([labels[:, 0:1] * 0, labels[:, 1:]]), w, h
    )
    n = len(xyxy)
    corners = np.ones((n * 4, 3))
    corners[:, :2] = np.array([
        [xyxy[:, 0], xyxy[:, 1]],
        [xyxy[:, 2], xyxy[:, 1]],
        [xyxy[:, 2], xyxy[:, 3]],
        [xyxy[:, 0], xyxy[:, 3]],
    ]).T.reshape(-1, 2)

    corners[:, :2] = corners[:, :2] @ M[:, :2].T + M[:, 2]
    corners = corners[:, :2].reshape(n, 4, 2)

    x1 = corners[:, :, 0].min(axis=1)
    y1 = corners[:, :, 1].min(axis=1)
    x2 = corners[:, :, 0].max(axis=1)
    y2 = corners[:, :, 1].max(axis=1)

    x1 = np.clip(x1, 0, w)
    y1 = np.clip(y1, 0, h)
    x2 = np.clip(x2, 0, w)
    y2 = np.clip(y2, 0, h)

    keep = (x2 - x1 > 2) & (y2 - y1 > 2)
    if not keep.any():
        return img_out, np.zeros((0, 5), dtype=np.float32)

    new_xyxy = np.stack([x1[keep], y1[keep], x2[keep], y2[keep]], axis=1)
    new_norm  = xyxy_to_xywhn(new_xyxy, w, h)
    new_labels = np.hstack([cls[keep], new_norm])
    return img_out, new_labels


class TurkeyDataset(Dataset):
    def __init__(self, split_dir: str, augment: bool = False):
        self.img_dir   = os.path.join(split_dir, "images")
        self.lbl_dir   = os.path.join(split_dir, "labels")
        self.augment   = augment
        self.size      = INPUT_SIZE
        self.img_files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

    def __len__(self):
        return len(self.img_files)

    def _load_sample(self, idx):
        name   = os.path.splitext(self.img_files[idx])[0]
        img    = cv2.imread(os.path.join(self.img_dir, self.img_files[idx]))
        labels = load_label(os.path.join(self.lbl_dir, name + ".txt"))
        return img, labels

    def _resize_pad(self, img, labels):
        """Letterbox resize to INPUT_SIZE × INPUT_SIZE."""
        h, w = img.shape[:2]
        r    = self.size / max(h, w)
        nw, nh = int(w * r), int(h * r)
        img  = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        pad_x, pad_y = (self.size - nw) // 2, (self.size - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = img

        if len(labels):
            labels = labels.copy()
            labels[:, 1] = (labels[:, 1] * w * r + pad_x) / self.size
            labels[:, 2] = (labels[:, 2] * h * r + pad_y) / self.size
            labels[:, 3] = labels[:, 3] * w * r / self.size
            labels[:, 4] = labels[:, 4] * h * r / self.size

        return canvas, labels

    def _mosaic(self, idx):
        """4-image mosaic centered at (s/2, s/2)."""
        s    = self.size
        yc   = xc = s // 2
        idxs = [idx] + random.choices(range(len(self)), k=3)
        mosaic_img    = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        mosaic_labels = []

        for i, ii in enumerate(idxs):
            img, labels = self._load_sample(ii)
            h0, w0 = img.shape[:2]
            r = s / max(h0, w0)
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)))
            h, w = img.shape[:2]

            if i == 0:   x1a,y1a,x2a,y2a = xc-w, yc-h, xc, yc
            elif i == 1: x1a,y1a,x2a,y2a = xc, yc-h, xc+w, yc
            elif i == 2: x1a,y1a,x2a,y2a = xc-w, yc, xc, yc+h
            else:        x1a,y1a,x2a,y2a = xc, yc, xc+w, yc+h

            x1a=max(x1a,0); y1a=max(y1a,0)
            x2a=min(x2a,2*s); y2a=min(y2a,2*s)

            x1b = w - (x2a - x1a); y1b = h - (y2a - y1a)
            x1b=max(x1b,0); y1b=max(y1b,0)

            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y1b+(y2a-y1a),
                                                 x1b:x1b+(x2a-x1a)]
            if len(labels):
                lbl = labels.copy()
                lbl[:, 1] = (lbl[:, 1] * w0 * r - x1b + x1a)
                lbl[:, 2] = (lbl[:, 2] * h0 * r - y1b + y1a)
                lbl[:, 3] = lbl[:, 3] * w0 * r
                lbl[:, 4] = lbl[:, 4] * h0 * r
                mosaic_labels.append(lbl)

        # Crop to center INPUT_SIZE
        x0 = xc - s // 2; y0 = yc - s // 2
        mosaic_img = mosaic_img[y0:y0+s, x0:x0+s]

        if mosaic_labels:
            all_lbl = np.concatenate(mosaic_labels, axis=0)
            # Adjust coords to crop
            all_lbl[:, 1] -= x0; all_lbl[:, 2] -= y0
            # Convert back to normalized
            all_lbl[:, 1] /= s; all_lbl[:, 2] /= s
            all_lbl[:, 3] /= s; all_lbl[:, 4] /= s
            all_lbl = np.clip(all_lbl, 0, 1)
            # Filter out degenerate boxes
            valid = (all_lbl[:, 3] > 0.01) & (all_lbl[:, 4] > 0.01)
            all_lbl = all_lbl[valid]
        else:
            all_lbl = np.zeros((0, 5), dtype=np.float32)

        return mosaic_img, all_lbl

    def __getitem__(self, idx):
        if self.augment and random.random() < MOSAIC_PROB:
            img, labels = self._mosaic(idx)
        else:
            img, labels = self._load_sample(idx)
            img, labels = self._resize_pad(img, labels)

        if self.augment:
            img = augment_hsv(img)
            img, labels = random_affine(img, labels, DEGREES, TRANSLATE, SCALE)
            if random.random() < FLIP_PROB:
                img = img[:, ::-1].copy()
                if len(labels):
                    labels[:, 1] = 1.0 - labels[:, 1]

        # Ensure correct size after augmentation
        if img.shape[:2] != (self.size, self.size):
            img, labels = self._resize_pad(img, labels)

        # BGR→RGB, HWC→CHW, normalize to [0,1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        # labels: (N, 5) [cls, cx, cy, w, h] normalized
        if len(labels) == 0:
            labels = torch.zeros((0, 5), dtype=torch.float32)
        else:
            labels = torch.from_numpy(labels.astype(np.float32))

        return img, labels


def collate_fn(batch):
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs, 0)
    # Return list of per-image label tensors (variable length)
    return imgs, list(targets)


def get_dataloader(split: str, augment: bool = None):
    dirs = {"train": TRAIN_DIR, "val": VAL_DIR, "test": TEST_DIR}
    if augment is None:
        augment = (split == "train")
    ds = TurkeyDataset(dirs[split], augment=augment)
    shuffle = (split == "train")
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == "train"),
    )


# if __name__ == "__main__":
#     loader = get_dataloader("train")
#     imgs, targets = next(iter(loader))
#     print(f"imgs:    {imgs.shape}")
#     print(f"targets: {[t.shape for t in targets[:3]]}")