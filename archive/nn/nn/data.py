import os
import numpy as np
from pathlib import Path
from xml.etree import ElementTree as ET
from typing import Tuple, List, Dict
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import nn.config as config


def collate_batch(batch: List[Dict]) -> Dict:
    """Custom collate function for variable-sized batches (different number of boxes per image)."""
    images = torch.stack([item['image'] for item in batch])
    heatmaps = torch.stack([item['heatmap'] for item in batch])
    offsets = torch.stack([item['offset'] for item in batch])
    n_objects = torch.tensor([item['n_objects'] for item in batch])

    # Don't stack boxes - keep as list (variable size)
    boxes = [item['boxes'] for item in batch]

    return {
        'image': images,
        'heatmap': heatmaps,
        'offset': offsets,
        'boxes': boxes,
        'n_objects': n_objects,
    }


class VOCDataset(Dataset):
    """Load VOC XML annotations and generate density heatmaps."""

    def __init__(
            self,
            image_dir: str = config.IMAGE_DIR,
            xml_dir: str = config.XML_DIR,
            img_size: Tuple[int, int] = config.IMG_SIZE,
            heatmap_stride: int = config.HEATMAP_STRIDE,
            gaussian_sigma: float = config.GAUSSIAN_SIGMA,
            augment: bool = config.AUGMENT,
    ):
        self.image_dir = Path(image_dir)
        self.xml_dir = Path(xml_dir)
        self.img_size = img_size
        # The model's total stride is ~32 (416→13 width, 312→10 height)
        self.heatmap_stride = 32
        self.gaussian_sigma = gaussian_sigma
        self.augment = augment

        # Model feature map output is 13×10 (not calculated from stride)
        self.hmap_h = 10
        self.hmap_w = 13

        self.image_files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        if self.augment:
            self.transform = transforms.Compose([
                transforms.ColorJitter(
                    brightness=config.AUG_BRIGHTNESS,
                    contrast=config.AUG_CONTRAST,
                    saturation=config.AUG_SATURATION
                ),
                transforms.RandomRotation(degrees=config.AUG_ROTATION_DEGREES),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict:
        img_name = self.image_files[idx]
        img_path = self.image_dir / img_name
        xml_path = self.xml_dir / img_name.replace('.jpg', '.xml').replace('.png', '.xml')

        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        img = cv2.resize(img, self.img_size)

        boxes, classes = self._parse_xml(xml_path, orig_h, orig_w)

        scale_x = self.img_size[0] / orig_w
        scale_y = self.img_size[1] / orig_h
        if len(boxes) > 0:
            boxes = boxes * np.array([scale_x, scale_y, scale_x, scale_y])

        heatmap, offset = self._generate_targets(boxes)

        if self.augment and self.transform is not None:
            img = np.asarray(self.transform(transforms.ToPILImage()(img)))

        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        heatmap = torch.from_numpy(heatmap).float()
        offset = torch.from_numpy(offset).float()

        return {
            'image': img,
            'heatmap': heatmap,
            'offset': offset,
            'boxes': torch.from_numpy(boxes).float() if len(boxes) > 0 else torch.tensor([]),
            'n_objects': len(boxes),
        }

    def _parse_xml(self, xml_path: Path, img_h: int, img_w: int) -> Tuple[np.ndarray, List[str]]:
        """Parse VOC XML and return bounding boxes (normalized coordinates)."""
        boxes = []
        classes = []

        if not xml_path.exists():
            return np.array([]), []

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for obj in root.findall('object'):
                class_name = obj.find('name').text.lower()
                if class_name not in ['body', 'neck']:
                    continue

                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text) / img_w
                ymin = float(bndbox.find('ymin').text) / img_h
                xmax = float(bndbox.find('xmax').text) / img_w
                ymax = float(bndbox.find('ymax').text) / img_h

                boxes.append([xmin, ymin, xmax, ymax])
                classes.append(class_name)

        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")
            return np.array([]), []

        if not boxes:
            return np.array([]), []

        return np.array(boxes), classes

    def _generate_targets(self, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate density heatmap and offset field from bounding boxes."""
        heatmap = np.zeros((1, self.hmap_h, self.hmap_w), dtype=np.float32)
        offset = np.zeros((2, self.hmap_h, self.hmap_w), dtype=np.float32)

        for box in boxes:
            cx = (box[0] + box[2]) / 2 * self.img_size[0]
            cy = (box[1] + box[3]) / 2 * self.img_size[1]

            hx = cx / self.heatmap_stride
            hy = cy / self.heatmap_stride

            if not (0 <= hx < self.hmap_w and 0 <= hy < self.hmap_h):
                continue

            gaussian = self._generate_gaussian(hx, hy, self.gaussian_sigma)
            heatmap[0] = np.maximum(heatmap[0], gaussian)

            hx_int = int(hx)
            hy_int = int(hy)
            dx = hx - hx_int
            dy = hy - hy_int

            if 0 <= hx_int < self.hmap_w and 0 <= hy_int < self.hmap_h:
                offset[0, hy_int, hx_int] = dx
                offset[1, hy_int, hx_int] = dy

        return heatmap, offset

    def _generate_gaussian(self, cx: float, cy: float, sigma: float, size_multiplier: int = 3) -> np.ndarray:
        """Generate 2D Gaussian kernel."""
        size = int(sigma * size_multiplier) * 2 + 1
        x = np.arange(0, size) - (size - 1) / 2
        y = np.arange(0, size) - (size - 1) / 2
        xx, yy = np.meshgrid(x, y)
        gaussian = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))

        hmap = np.zeros((self.hmap_h, self.hmap_w), dtype=np.float32)
        sx = int(cx - size / 2)
        sy = int(cy - size / 2)

        gx_start = max(0, -sx)
        gy_start = max(0, -sy)
        gx_end = min(size, self.hmap_w - sx)
        gy_end = min(size, self.hmap_h - sy)

        sx = max(0, sx)
        sy = max(0, sy)
        ex = min(self.hmap_w, sx + gx_end - gx_start)
        ey = min(self.hmap_h, sy + gy_end - gy_start)

        if ex > sx and ey > sy:
            hmap[sy:ey, sx:ex] = gaussian[gy_start:gy_end, gx_start:gx_end]

        return hmap


def create_loaders(
        image_dir: str = config.IMAGE_DIR,
        xml_dir: str = config.XML_DIR,
        batch_size: int = config.BATCH_SIZE,
        num_workers: int = config.NUM_WORKERS,
        test_split: float = config.TEST_SPLIT,
        img_size: Tuple[int, int] = config.IMG_SIZE,
        seed: int = config.SEED,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/test dataloaders with stratified split."""

    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = VOCDataset(
        image_dir=image_dir,
        xml_dir=xml_dir,
        img_size=img_size,
        augment=True,
    )

    n_total = len(dataset)
    indices = np.arange(n_total)
    np.random.shuffle(indices)

    split_idx = int(n_total * (1 - test_split))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    test_dataset.dataset.augment = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        collate_fn=collate_batch,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        collate_fn=collate_batch,
    )

    print(f"Dataset: {n_total} images")
    print(f"Train: {len(train_indices)}, Test: {len(test_indices)}")

    return train_loader, test_loader