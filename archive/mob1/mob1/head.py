# =============================================================================
# head.py — Decoupled anchor-free detection head + full NanoDetTurkey model
# GFL (Generalized Focal Loss) regression eliminates exp() for INT8 safety
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from mob1.backbone import MobileNetV1Backbone
from mob1.config import (NUM_CLASSES, SHARED_CHANNELS, REG_MAX,
                    INPUT_SIZE, STRIDE, WIDTH_MULT)


class DetHead(nn.Module):
    """
    Decoupled head: shared 1×1 conv → parallel cls + reg branches.
    reg_max bins per side × 4 sides = 4*(reg_max+1) regression outputs.
    """
    def __init__(self, in_channels: int, num_classes: int,
                 shared_ch: int, reg_max: int):
        super().__init__()
        self.reg_max = reg_max
        self.num_classes = num_classes

        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, shared_ch, 1, bias=False),
            nn.BatchNorm2d(shared_ch),
            nn.ReLU6(inplace=True),
        )
        self.cls_head = nn.Conv2d(shared_ch, num_classes, 1)
        self.reg_head = nn.Conv2d(shared_ch, 4 * (reg_max + 1), 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_head.weight, std=0.01)
        nn.init.constant_(self.cls_head.bias, -4.0)  # low initial confidence
        nn.init.normal_(self.reg_head.weight, std=0.01)
        nn.init.zeros_(self.reg_head.bias)

    def forward(self, x):
        feat = self.shared(x)
        cls  = self.cls_head(feat)   # (B, num_classes, H, W)
        reg  = self.reg_head(feat)   # (B, 4*(reg_max+1), H, W)
        return cls, reg


class NanoDetTurkey(nn.Module):
    """
    Full detector:
      Input 256×256 → MobileNetV1-0.25 → single 16×16 feature map → DetHead
    """
    def __init__(self):
        super().__init__()
        self.stride     = STRIDE
        self.num_classes = NUM_CLASSES
        self.reg_max    = REG_MAX

        self.backbone = MobileNetV1Backbone(width_mult=WIDTH_MULT)
        self.head     = DetHead(
            in_channels=self.backbone.out_channels,
            num_classes=NUM_CLASSES,
            shared_ch=SHARED_CHANNELS,
            reg_max=REG_MAX,
        )

        # Precompute distribution projection vector in PIXEL space.
        # Each bin i represents a distance of i * stride pixels from the anchor.
        proj = torch.arange(REG_MAX + 1, dtype=torch.float32) * STRIDE
        self.register_buffer("proj", proj)

        # Precompute grid anchors for decode_boxes
        grid_size = INPUT_SIZE // STRIDE
        self._build_grid(grid_size)

    def _build_grid(self, grid_size):
        ys, xs = torch.meshgrid(
            torch.arange(grid_size, dtype=torch.float32),
            torch.arange(grid_size, dtype=torch.float32),
            indexing="ij",
        )
        # (grid_size*grid_size, 2) — [x_center, y_center] in pixel space
        grid = torch.stack([xs, ys], dim=-1).reshape(-1, 2)
        grid = (grid + 0.5) * self.stride
        self.register_buffer("grid", grid)  # (N, 2)

    def decode_boxes(self, reg_pred):
        """
        Decode GFL distribution to (x1,y1,x2,y2) in pixel space.
        reg_pred: (B, N, 4*(reg_max+1))
        returns:  (B, N, 4)  xyxy pixels
        """
        B, N, _ = reg_pred.shape
        reg = reg_pred.reshape(B, N, 4, self.reg_max + 1)
        reg = F.softmax(reg, dim=-1)
        # dot with [0, stride, 2*stride, ...] → predicted distance in pixels
        dist = (reg * self.proj).sum(dim=-1)  # (B, N, 4)  already in pixels

        grid = self.grid.unsqueeze(0)  # (1, N, 2)
        x1 = grid[..., 0] - dist[..., 0]
        y1 = grid[..., 1] - dist[..., 1]
        x2 = grid[..., 0] + dist[..., 2]
        y2 = grid[..., 1] + dist[..., 3]

        return torch.stack([x1, y1, x2, y2], dim=-1)

    def forward(self, x):
        """
        Returns:
          cls_pred: (B, N, num_classes)  — raw logits
          reg_pred: (B, N, 4*(reg_max+1)) — GFL distribution logits
          boxes:    (B, N, 4)            — decoded xyxy pixels (detached from grad during train is handled in loss)
        where N = (INPUT_SIZE/STRIDE)^2
        """
        feat           = self.backbone(x)
        cls_map, reg_map = self.head(feat)

        B, C, H, W = cls_map.shape
        # Flatten spatial dims: (B, H*W, ...)
        cls_pred = cls_map.permute(0, 2, 3, 1).reshape(B, H * W, C)
        reg_pred = reg_map.permute(0, 2, 3, 1).reshape(B, H * W, 4 * (self.reg_max + 1))

        boxes = self.decode_boxes(reg_pred)

        return cls_pred, reg_pred, boxes


# if __name__ == "__main__":
#     model = NanoDetTurkey()
#     x = torch.randn(1, 3, 256, 256)
#     cls, reg, boxes = model(x)
#     print(f"cls_pred : {cls.shape}")    # (1, 256, 2)
#     print(f"reg_pred : {reg.shape}")    # (1, 256, 32)
#     print(f"boxes    : {boxes.shape}")  # (1, 256, 4)
#     total = sum(p.numel() for p in model.parameters())
#     print(f"Total params: {total:,}")