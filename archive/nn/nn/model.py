
import torch
import torch.nn as nn
import torch.nn.functional as F
import nn.config as config


class ShuffleNetV2Block(nn.Module):
    """ShuffleNetV2 basic block with channel shuffle."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.stride = stride

        hidden_ch = out_ch // 2

        if stride == 1:
            assert in_ch == out_ch
            self.branch1 = nn.Sequential(
                nn.Conv2d(hidden_ch, hidden_ch, 1, bias=False),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_ch, hidden_ch, 3, stride, 1, groups=hidden_ch, bias=False),
                nn.BatchNorm2d(hidden_ch),
                nn.Conv2d(hidden_ch, hidden_ch, 1, bias=False),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_ch, hidden_ch, 1, bias=False),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_ch, hidden_ch, 3, stride, 1, groups=hidden_ch, bias=False),
                nn.BatchNorm2d(hidden_ch),
                nn.Conv2d(hidden_ch, hidden_ch, 1, bias=False),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU(inplace=True),
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_ch, hidden_ch, 3, stride, 1, groups=in_ch, bias=False),
                nn.BatchNorm2d(hidden_ch),
                nn.Conv2d(hidden_ch, hidden_ch, 1, bias=False),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            x = torch.cat([x2, self.branch1(x1)], dim=1)
        else:
            x = torch.cat([self.branch2(x), self.branch1(x)], dim=1)

        B, C, H, W = x.shape
        x = x.reshape(B, 2, C // 2, H, W).permute(0, 2, 1, 3, 4).reshape(B, C, H, W)
        return x


class ShuffleNetV2_05(nn.Module):
    """ShuffleNetV2 0.5x backbone for mobile/edge deployment."""

    def __init__(self, num_classes=1000):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, 3, 2, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2 = self._make_stage(24, 48, 4, stride=2)
        self.stage3 = self._make_stage(48, 96, 8, stride=2)
        self.stage4 = self._make_stage(96, 192, 4, stride=2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(192, 1024, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

        self._init_weights()

    def _make_stage(self, in_ch: int, out_ch: int, n_blocks: int, stride: int):
        layers = [ShuffleNetV2Block(in_ch, out_ch, stride)]
        for _ in range(1, n_blocks):
            layers.append(ShuffleNetV2Block(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        """Return only feature map without classification."""
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        return x


class DenseTurkeyDetector(nn.Module):
    """Lightweight dense detection model for turkey counting."""

    def __init__(
        self,
        backbone_ch: int = config.BACKBONE_CHANNELS,
        heatmap_ch: int = config.HEATMAP_CHANNELS
    ):
        super().__init__()

        self.backbone = ShuffleNetV2_05()

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(backbone_ch, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, heatmap_ch, kernel_size=1),
            nn.Sigmoid(),
        )

        self.offset_head = nn.Sequential(
            nn.Conv2d(backbone_ch, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, kernel_size=1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in [self.heatmap_head, self.offset_head]:
            for layer in m:
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        heatmap = self.heatmap_head(features)
        offset = self.offset_head(features)
        return heatmap, offset


class PrecisionAwareLoss(nn.Module):
    """Loss function optimized for precision (FP >> FN)."""

    def __init__(
        self,
        alpha_heatmap: float = config.LOSS_ALPHA_HEATMAP,
        alpha_offset: float = config.LOSS_ALPHA_OFFSET,
        alpha_spatial: float = config.LOSS_ALPHA_SPATIAL,
        min_distance: int = config.LOSS_MIN_DISTANCE,
    ):
        super().__init__()
        self.alpha_heatmap = alpha_heatmap
        self.alpha_offset = alpha_offset
        self.alpha_spatial = alpha_spatial
        self.min_distance = min_distance

    def forward(
        self,
        pred_heatmap: torch.Tensor,
        gt_heatmap: torch.Tensor,
        pred_offset: torch.Tensor,
        gt_offset: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_heatmap: (B, 1, H, W) predicted density
            gt_heatmap: (B, 1, H, W) target density
            pred_offset: (B, 2, H, W) predicted offsets
            gt_offset: (B, 2, H, W) target offsets
        """

        heatmap_loss = self._focal_loss(
            pred_heatmap,
            gt_heatmap,
            alpha=config.LOSS_FOCAL_ALPHA,
            gamma=config.LOSS_FOCAL_GAMMA
        )

        confidence_mask = (gt_heatmap > config.LOSS_OFFSET_CONFIDENCE_THRESHOLD).float()
        offset_loss = self._smooth_l1_loss(pred_offset, gt_offset, confidence_mask)

        spatial_loss = self._spatial_regularization(pred_heatmap)

        total_loss = (
            self.alpha_heatmap * heatmap_loss +
            self.alpha_offset * offset_loss +
            self.alpha_spatial * spatial_loss
        )

        return total_loss

    def _focal_loss(self, pred, target, alpha=2.0, gamma=4.0):
        """Focal loss for dense heatmap prediction."""
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target > 0.5, pred, 1 - pred)
        focal_weight = (1 - pt) ** gamma
        weight = torch.where(target > 0.5, 1.0, alpha)
        loss = weight * focal_weight * bce
        return loss.mean()

    def _smooth_l1_loss(self, pred, target, mask):
        """Smooth L1 loss for offset field."""
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < 1.0,
            0.5 * diff ** 2,
            diff - 0.5
        )
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss

    def _spatial_regularization(self, pred_heatmap):
        """Penalize spurious nearby peaks."""
        B, C, H, W = pred_heatmap.shape
        max_pool = F.max_pool2d(pred_heatmap, kernel_size=3, stride=1, padding=1)
        is_peak = (pred_heatmap == max_pool).float()
        kernel = torch.ones(1, 1, 5, 5, device=pred_heatmap.device) / 25.0
        peak_density = F.conv2d(is_peak, kernel, padding=2)
        loss = torch.clamp(peak_density - 1.0, min=0).mean()
        return loss