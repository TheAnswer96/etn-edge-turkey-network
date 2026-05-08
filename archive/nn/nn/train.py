import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

import nn.config as config
from nn.data import create_loaders
from nn.model import DenseTurkeyDetector, PrecisionAwareLoss


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            device: str = config.DEVICE,
            lr: float = config.LEARNING_RATE,
            weight_decay: float = config.WEIGHT_DECAY,
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = PrecisionAwareLoss(
            alpha_heatmap=config.LOSS_ALPHA_HEATMAP,
            alpha_offset=config.LOSS_ALPHA_OFFSET,
            alpha_spatial=config.LOSS_ALPHA_SPATIAL,
        )
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode=config.SCHEDULER_MODE,
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE,
            verbose=True,
        )

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_precision': [],
            'val_recall': [],
            'best_epoch': 0,
            'best_val_loss': float('inf'),
        }

    def train_epoch(self, train_loader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc='Training'):
            images = batch['image'].to(self.device)
            gt_heatmap = batch['heatmap'].to(self.device)
            gt_offset = batch['offset'].to(self.device)

            pred_heatmap, pred_offset = self.model(images)
            loss = self.criterion(pred_heatmap, gt_heatmap, pred_offset, gt_offset)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=config.GRADIENT_CLIP_MAX_NORM)
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def validate(self, val_loader) -> dict:
        """Validate on test set."""
        self.model.eval()
        total_loss = 0.0

        all_pred_peaks = []
        all_gt_peaks = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating'):
                images = batch['image'].to(self.device)
                gt_heatmap = batch['heatmap'].to(self.device)
                gt_offset = batch['offset'].to(self.device)

                pred_heatmap, pred_offset = self.model(images)
                loss = self.criterion(pred_heatmap, gt_heatmap, pred_offset, gt_offset)
                total_loss += loss.item()

                for i in range(pred_heatmap.shape[0]):
                    pred_peaks = self._extract_peaks(
                        pred_heatmap[i:i + 1],
                        pred_offset[i:i + 1],
                        threshold=config.HEATMAP_THRESHOLD,
                    )
                    gt_peaks = self._extract_peaks(
                        gt_heatmap[i:i + 1],
                        None,
                        threshold=0.1,
                    )

                    all_pred_peaks.append(pred_peaks)
                    all_gt_peaks.append(gt_peaks)

        avg_loss = total_loss / len(val_loader)
        precision, recall = self._compute_metrics(all_pred_peaks, all_gt_peaks)

        return {
            'loss': avg_loss,
            'precision': precision,
            'recall': recall,
        }

    def _extract_peaks(
            self,
            heatmap: torch.Tensor,
            offset: torch.Tensor = None,
            threshold: float = 0.5,
    ) -> np.ndarray:
        """Extract peaks from heatmap with optional offset refinement."""
        hmap = heatmap.cpu().numpy()[0, 0]

        peaks = []
        for y in range(1, hmap.shape[0] - 1):
            for x in range(1, hmap.shape[1] - 1):
                if hmap[y, x] > threshold:
                    window = hmap[y - 1:y + 2, x - 1:x + 2]
                    if hmap[y, x] == window.max():
                        peaks.append([x, y, hmap[y, x]])

        if offset is not None and len(peaks) > 0:
            off = offset.cpu().numpy()[0]
            peaks = np.array(peaks)
            for i, (x, y, conf) in enumerate(peaks):
                y_int, x_int = int(y), int(x)
                if 0 <= y_int < off.shape[1] and 0 <= x_int < off.shape[2]:
                    dx = off[0, y_int, x_int]
                    dy = off[1, y_int, x_int]
                    peaks[i, 0] += dx * 0.5
                    peaks[i, 1] += dy * 0.5
            return peaks

        return np.array(peaks) if peaks else np.array([]).reshape(0, 3)

    def _compute_metrics(self, pred_peaks_list, gt_peaks_list, iou_dist=3.0) -> tuple:
        """Compute precision and recall for counting task."""
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for pred_peaks, gt_peaks in zip(pred_peaks_list, gt_peaks_list):
            if len(pred_peaks) == 0 and len(gt_peaks) == 0:
                continue

            if len(pred_peaks) == 0:
                total_fn += len(gt_peaks)
                continue

            if len(gt_peaks) == 0:
                total_fp += len(pred_peaks)
                continue

            matched_gt = set()
            for pred in sorted(pred_peaks, key=lambda p: -p[2]):
                best_gt = None
                best_dist = iou_dist

                for gt_idx, gt in enumerate(gt_peaks):
                    if gt_idx in matched_gt:
                        continue

                    dist = np.sqrt((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2)
                    if dist < best_dist:
                        best_dist = dist
                        best_gt = gt_idx

                if best_gt is not None:
                    total_tp += 1
                    matched_gt.add(best_gt)
                else:
                    total_fp += 1

            total_fn += len(gt_peaks) - len(matched_gt)

        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)

        return precision, recall

    def fit(
            self,
            train_loader,
            val_loader,
            epochs: int = config.NUM_EPOCHS,
            checkpoint_dir: str = config.CHECKPOINT_DIR,
    ):
        """Train for multiple epochs."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)

        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")

            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            print(f"Train Loss: {train_loss:.4f}")

            val_metrics = self.validate(val_loader)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])

            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")

            self.scheduler.step(val_metrics['loss'])

            if val_metrics['loss'] < self.history['best_val_loss']:
                self.history['best_val_loss'] = val_metrics['loss']
                self.history['best_epoch'] = epoch

                ckpt_path = checkpoint_dir / config.CHECKPOINT_FILENAME
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state': self.model.state_dict(),
                        'optimizer_state': self.optimizer.state_dict(),
                        'metrics': val_metrics,
                        'history': self.history,
                    },
                    ckpt_path,
                )
                print(f"Saved checkpoint: {ckpt_path}")

        history_path = checkpoint_dir / config.HISTORY_FILENAME
        with open(history_path, 'w') as f:
            history = {
                k: [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in vals]
                if isinstance(vals, list) else vals
                for k, vals in self.history.items()
            }
            json.dump(history, f, indent=2)

        print(f"\nTraining complete. Best epoch: {self.history['best_epoch']}")
        print(f"Best validation loss: {self.history['best_val_loss']:.4f}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, default=config.IMAGE_DIR)
    parser.add_argument('--xml-dir', type=str, default=config.XML_DIR)
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--checkpoint-dir', type=str, default=config.CHECKPOINT_DIR)
    parser.add_argument('--device', type=str, default=config.DEVICE)
    args = parser.parse_args()

    train_loader, val_loader = create_loaders(
        image_dir=args.image_dir,
        xml_dir=args.xml_dir,
        batch_size=args.batch_size,
    )

    model = DenseTurkeyDetector()
    trainer = Trainer(model, device=args.device, lr=args.lr)

    trainer.fit(
        train_loader,
        val_loader,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
    )