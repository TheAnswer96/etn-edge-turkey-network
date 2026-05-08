"""
Teacher-Student Knowledge Distillation for EdgeTurkeyNet.

Overview
--------
A pretrained, frozen EdgeTurkeyNet acts as the *teacher*.  A smaller
*student* â€” same architecture but with scaled-down neck and head channels
(controlled by ``width_mult``) â€” is trained to mimic the teacher while
simultaneously learning from ground-truth labels.

Student architecture
--------------------
The student reuses EdgeTurkeyNet's backbone, PANLiteNeck, and AnchorFreeHead
but all internal channel widths are multiplied by ``width_mult``:

  Teacher neck channels: P3=128, P4=96, P5=64
  Student neck channels (width_mult=0.5): P3=64, P4=48, P5=32

The backbone is shared in type (mobilenetv3 / shufflenetv2 / mobilenetv1)
but is always initialised from scratch for the student â€” backbone weights
transfer implicitly through distillation, not direct weight copying.

Distillation losses
-------------------
Three complementary signals are combined:

1. Feature distillation (FD)
   MSE between teacher and student neck outputs at each FPN level.
   Student features are projected to teacher channel count via a learned
   1Ã—1 adapter before comparison, so no architecture constraint is imposed.

   L_fd = mean over levels of MSE(adapter(student_feat), teacher_feat.detach())

2. Response distillation (RD)
   Soft-target KL divergence on classification logits (temperature-scaled)
   + MSE on regression and centerness predictions.

   L_cls_soft = KLDiv(softmax(student_cls / T), softmax(teacher_cls / T)) * TÂ²
   L_reg_soft = MSE(student_reg, teacher_reg.detach())
   L_ctr_soft = MSE(student_ctr, teacher_ctr.detach())

   Temperature T > 1 softens the teacher's class distribution, exposing
   inter-class similarity information that hard labels cannot convey.

3. Ground-truth detection loss (GT)
   Standard EdgeTurkeyLoss computed against GT boxes and class ids.
   Ensures the student converges to correct predictions, not just to
   mimic the teacher's errors.

   L_gt = Î»_clsÂ·FocalLoss + Î»_regÂ·CIoU + Î»_ctrÂ·BCE

Total loss
----------
L_total = w_gt * L_gt + w_fd * L_fd + w_rd * (L_cls_soft + L_reg_soft + L_ctr_soft)

Default weights: w_gt=1.0, w_fd=0.5, w_rd=1.0

EDGE AI RATIONALE
-----------------
Knowledge distillation is the primary compression strategy complementary
to pruning and quantisation:
- Pruning (in train.py): removes unimportant channels â€” structured sparsity
- Quantisation (in export.py): reduces weight precision â€” INT8
- Distillation (here): transfers semantic knowledge to a smaller network

A student with width_mult=0.5 has roughly 4Ã— fewer parameters in the neck
and head than the teacher, directly reducing inference MACs on RPi 4B.
The backbone remains the same type, but since it is always the smallest
available option (ShuffleNetV2-0.5x â†’ 1.4M params), backbone choice
dominates; the neck/head savings on top still matter for latency.

Usage
-----
from distill import build_student, KnowledgeDistillationTrainer
from config import get_config

cfg = get_config()
# Distillation-specific parameters are taken from KD_CONFIG at the top of
# this file â€” edit those globals to tune the distillation run.

trainer = KnowledgeDistillationTrainer(
    teacher_checkpoint="runs/20260303_120000_mobilenetv3/checkpoints/best.pth",
    cfg=cfg,
    logger=logger,   # a RunLogger pointing to a fresh timestamped run dir
)
student = trainer.train()
"""

from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast

from .dataset import get_train_loader, get_val_loader
from .evaluate import evaluate_map, PerClassMetrics
from .loss import EdgeTurkeyLoss
from .model import (
    AnchorFreeHead,
    ChannelPruner,
    ConvBnAct,
    DepthwiseSeparableConv,
    EdgeTurkeyNet,
    NUM_CLASSES,
    PANLiteNeck,
    SqueezeExcitation,
    build_backbone,
)
from .train import get_lr, set_seed

if TYPE_CHECKING:
    from .config import RunConfig
    from .logger import RunLogger


# ===========================================================================
# DISTILLATION CONFIGURATION â€” edit these globals to tune a distillation run
# ===========================================================================

# Student architecture
KD_WIDTH_MULT   = 0.5    # Channel width multiplier for student neck + head towers.
                          # 0.5 â†’ ~4Ã— fewer neck/head params, ~2Ã— faster neck.
                          # Must be in (0.0, 1.0].

# Loss weights
KD_W_GT  = 1.0   # Weight for ground-truth detection loss (EdgeTurkeyLoss).
KD_W_FD  = 0.5   # Weight for feature distillation loss (FPN-level MSE).
KD_W_RD  = 1.0   # Weight for response distillation loss (cls KL + reg/ctr MSE).

# Response distillation temperature
KD_TEMPERATURE = 4.0   # Softmax temperature for cls KL divergence.
                        # Higher â†’ softer teacher distribution â†’ more inter-class info.
                        # Typical range: 2â€“8.

# Training duration for the student
KD_EPOCHS              = 80    # Maximum student training epochs.
KD_EARLY_STOP_PATIENCE = 15    # Epochs without val mAP improvement before stopping.

# Pruning during student training (same progressive schedule as teacher)
KD_PRUNE_STUDENT = True   # Whether to also apply periodic pruning to the student.


# ===========================================================================
# DEVICE
# ===========================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Scaled ShuffleNetV2 backbone â€” parametric width Ã— depth
# ---------------------------------------------------------------------------

class _InvertedResidual(nn.Module):
    """
    ShuffleNetV2 InvertedResidual block.

    Two branches:
      - Stride-1: identity (left) + depthwise-separable transform (right),
        concatenated then channel-shuffled.
      - Stride-2: both branches are strided, outputs concatenated.

    This is a faithful re-implementation of the torchvision block so the
    student backbone can use arbitrary channel counts rather than being
    locked to the four fixed torchvision variants (x0.5/x1.0/x1.5/x2.0).

    Args:
        in_channels:  Input channel count.
        out_channels: Output channel count (must be even; split equally).
        stride:       1 or 2.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        assert stride in (1, 2), f"stride must be 1 or 2, got {stride}"
        assert out_channels % 2 == 0, "out_channels must be even for channel split"

        self.stride    = stride
        branch_out     = out_channels // 2

        if stride == 1:
            # Left branch: identity (in_channels == branch_out required)
            # Right branch: pw â†’ dw â†’ pw
            assert in_channels == out_channels, (
                "stride-1 block requires in_channels == out_channels"
            )
            branch_in = in_channels // 2
            self.branch2 = nn.Sequential(
                nn.Conv2d(branch_in, branch_in, 1, bias=False),
                nn.BatchNorm2d(branch_in),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_in, branch_in, 3, stride=1, padding=1,
                          groups=branch_in, bias=False),
                nn.BatchNorm2d(branch_in),
                nn.Conv2d(branch_in, branch_out, 1, bias=False),
                nn.BatchNorm2d(branch_out),
                nn.ReLU(inplace=True),
            )
        else:
            # Left branch: dw â†’ pw (strided, processes full input)
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1,
                          groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, branch_out, 1, bias=False),
                nn.BatchNorm2d(branch_out),
                nn.ReLU(inplace=True),
            )
            # Right branch: pw â†’ dw â†’ pw (strided)
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1,
                          groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, branch_out, 1, bias=False),
                nn.BatchNorm2d(branch_out),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def _channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.view(B, groups, C // groups, H, W)
        x = x.transpose(1, 2).contiguous()
        return x.view(B, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat([x1, self.branch2(x2)], dim=1)
        else:
            out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        return self._channel_shuffle(out, 2)


class ScaledShuffleNetV2Backbone(nn.Module):
    """
    Parametric ShuffleNetV2 backbone with independent width and depth scaling.

    Unlike the fixed torchvision variants (x0.5 / x1.0 / x1.5 / x2.0),
    this backbone accepts continuous multipliers applied to the x0.5 reference:

      backbone_width_mult â€” scales all stage output channel counts.
        1.0 â†’ reference channels  [48, 96, 192]
        0.5 â†’ half channels       [24, 48,  96]
        0.25â†’ quarter channels    [12, 24,  48]

      backbone_depth_mult â€” scales the number of InvertedResidual blocks per stage.
        The x0.5 reference has [4, 8, 4] blocks in stages 2/3/4.
        1.0 â†’ [4, 8, 4]  (reference)
        0.5 â†’ [2, 4, 2]  (half depth)
        0.25â†’ [1, 2, 1]  (quarter depth, minimum 1 per stage)

    Always trained from scratch â€” no pretrained weights exist for non-standard
    channel configurations. Knowledge transfers via distillation.

    Feature scales:
      P3: stride 8  â€” stage2 output
      P4: stride 16 â€” stage3 output
      P5: stride 32 â€” stage4 output

    Args:
        backbone_width_mult: Channel width multiplier in (0, 1].
        backbone_depth_mult: Stage block count multiplier in (0, 1].
    """

    # Reference channel counts and stage depths (ShuffleNetV2 x0.5)
    _REF_CHANNELS: List[int] = [48, 96, 192]   # stage2, stage3, stage4
    _REF_DEPTHS:   List[int] = [4,  8,  4]     # blocks per stage

    def __init__(
        self,
        backbone_width_mult: float = 1.0,
        backbone_depth_mult: float = 1.0,
    ) -> None:
        super().__init__()

        if not 0.0 < backbone_width_mult <= 1.0:
            raise ValueError(
                f"backbone_width_mult must be in (0, 1], got {backbone_width_mult}"
            )
        if not 0.0 < backbone_depth_mult <= 1.0:
            raise ValueError(
                f"backbone_depth_mult must be in (0, 1], got {backbone_depth_mult}"
            )

        self.backbone_width_mult = backbone_width_mult
        self.backbone_depth_mult = backbone_depth_mult

        # Scale channels â€” round to nearest even (channel split requires even)
        def _ch(base: int) -> int:
            return max(2, round(base * backbone_width_mult / 2) * 2)

        # Scale depths â€” minimum 1 block per stage
        def _depth(base: int) -> int:
            return max(1, round(base * backbone_depth_mult))

        ch2 = _ch(self._REF_CHANNELS[0])   # stage2 out
        ch3 = _ch(self._REF_CHANNELS[1])   # stage3 out
        ch4 = _ch(self._REF_CHANNELS[2])   # stage4 out

        d2 = _depth(self._REF_DEPTHS[0])
        d3 = _depth(self._REF_DEPTHS[1])
        d4 = _depth(self._REF_DEPTHS[2])

        self.out_channels = [ch2, ch3, ch4]

        # Stem: conv3Ã—3/2 â†’ BN â†’ ReLU â†’ maxpool/2  (â†’ stride 4 total)
        STEM_OUT = max(2, round(24 * backbone_width_mult / 2) * 2)
        self.stem = nn.Sequential(
            nn.Conv2d(3, STEM_OUT, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(STEM_OUT),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # Stage 2 â€” first block: stride-2 (_InvertedResidual stride=2)
        #           rest:        stride-1 (in == out == ch2)
        self.stage2 = self._make_stage(STEM_OUT, ch2, d2)
        self.stage3 = self._make_stage(ch2,      ch3, d3)
        self.stage4 = self._make_stage(ch3,      ch4, d4)

        self._init_weights()

        print(
            f"[ScaledShuffleNetV2] width={backbone_width_mult}  "
            f"depth={backbone_depth_mult}  "
            f"channels={self.out_channels}  "
            f"depths=[{d2},{d3},{d4}]  "
            f"params={sum(p.numel() for p in self.parameters()):,}"
        )

    @staticmethod
    def _make_stage(
        in_channels: int, out_channels: int, n_blocks: int
    ) -> nn.Sequential:
        """Build one ShuffleNetV2 stage: one stride-2 block + (n-1) stride-1 blocks."""
        layers = [_InvertedResidual(in_channels, out_channels, stride=2)]
        for _ in range(n_blocks - 1):
            layers.append(_InvertedResidual(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x  = self.stem(x)       # /4
        p3 = self.stage2(x)     # /8
        p4 = self.stage3(p3)    # /16
        p5 = self.stage4(p4)    # /32
        return p3, p4, p5


# ---------------------------------------------------------------------------
# Student neck â€” same topology as PANLiteNeck but channels scaled by width_mult
# ---------------------------------------------------------------------------

class StudentPANLiteNeck(nn.Module):
    """
    Scaled-down PAN-Lite neck for the student model.

    Identical topology to PANLiteNeck (top-down + bottom-up pathways,
    SE attention, DepthwiseSeparableConv merges) but all internal channel
    counts are multiplied by ``width_mult``.

    Args:
        backbone_channels: [p3_ch, p4_ch, p5_ch] from the student backbone.
        width_mult:        Channel scaling factor in (0, 1].
    """

    def __init__(
        self,
        backbone_channels: List[int],
        width_mult: float = 0.5,
    ) -> None:
        super().__init__()

        # Scale neck output channels, round up to nearest even number
        def _ch(base: int) -> int:
            return max(1, round(base * width_mult / 2) * 2)

        self.nc = {
            'p3': _ch(PANLiteNeck.NECK_CHANNELS['p3']),   # default 128 â†’ 64
            'p4': _ch(PANLiteNeck.NECK_CHANNELS['p4']),   # default  96 â†’ 48
            'p5': _ch(PANLiteNeck.NECK_CHANNELS['p5']),   # default  64 â†’ 32
        }
        self.out_channels = list(self.nc.values())   # expose for adapter building

        in_p3, in_p4, in_p5 = backbone_channels
        nc = self.nc

        # Lateral projections
        self.lat_p3 = ConvBnAct(in_p3, nc['p3'], 1)
        self.lat_p4 = ConvBnAct(in_p4, nc['p4'], 1)
        self.lat_p5 = ConvBnAct(in_p5, nc['p5'], 1)

        # Top-down pathway
        self.td_p4_conv = DepthwiseSeparableConv(nc['p5'] + nc['p4'], nc['p4'])
        self.td_p4_se   = SqueezeExcitation(nc['p4'])
        self.td_p3_conv = DepthwiseSeparableConv(nc['p4'] + nc['p3'], nc['p3'])
        self.td_p3_se   = SqueezeExcitation(nc['p3'])

        # Bottom-up pathway
        self.bu_p4_conv = DepthwiseSeparableConv(nc['p3'] + nc['p4'], nc['p4'])
        self.bu_p4_se   = SqueezeExcitation(nc['p4'])
        self.bu_p5_conv = DepthwiseSeparableConv(nc['p4'] + nc['p5'], nc['p5'])
        self.bu_p5_se   = SqueezeExcitation(nc['p5'])

        self.upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.MaxPool2d(2, 2)

    def forward(
        self,
        p3: torch.Tensor,
        p4: torch.Tensor,
        p5: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p3 = self.lat_p3(p3)
        p4 = self.lat_p4(p4)
        p5 = self.lat_p5(p5)

        td_p4 = self.td_p4_se(
            self.td_p4_conv(torch.cat([self.upsample(p5), p4], dim=1))
        )
        td_p3 = self.td_p3_se(
            self.td_p3_conv(torch.cat([self.upsample(td_p4), p3], dim=1))
        )
        bu_p4 = self.bu_p4_se(
            self.bu_p4_conv(torch.cat([self.downsample(td_p3), td_p4], dim=1))
        )
        bu_p5 = self.bu_p5_se(
            self.bu_p5_conv(torch.cat([self.downsample(bu_p4), p5], dim=1))
        )

        return td_p3, bu_p4, bu_p5


# ---------------------------------------------------------------------------
# Student head â€” same topology as AnchorFreeHead, towers scaled by width_mult
# ---------------------------------------------------------------------------

class StudentAnchorFreeHead(nn.Module):
    """
    Scaled-down FCOS detection head for the student model.

    Tower depth is controlled by ``depth_mult`` applied to the teacher's
    fixed tower depth of 2 DSConv layers, rounded to the nearest integer
    (minimum 1).  Output projection Conv2d layers always keep the same
    output channels as the teacher (num_classes, 4, 1) so predictions are
    directly comparable for response distillation regardless of depth.

    Teacher reference depth: 2 DSConv layers per tower per scale.

      depth_mult=1.0 â†’ 2 layers  (same as teacher)
      depth_mult=0.5 â†’ 1 layer   (default, half depth)
      depth_mult=0.25â†’ 1 layer   (rounds up to minimum 1)

    Args:
        in_channels_list: Per-scale input channel counts [p3, p4, p5]
                          from the StudentPANLiteNeck.
        num_classes:      Detection class count (must match teacher).
        depth_mult:       Tower depth multiplier applied to teacher depth=2.
                          Must be in (0, 1].  Default 0.5 â†’ 1 layer.
    """

    ASPECT_RATIO_PRIOR: float = 1.2
    TEACHER_TOWER_DEPTH: int  = 2   # teacher AnchorFreeHead has 2 DSConv per tower

    def __init__(
        self,
        in_channels_list: List[int],
        num_classes: int  = NUM_CLASSES,
        depth_mult: float = 0.5,
    ) -> None:
        super().__init__()

        if not 0.0 < depth_mult <= 1.0:
            raise ValueError(f"depth_mult must be in (0, 1], got {depth_mult}")

        self.num_classes = num_classes
        self.depth_mult  = depth_mult

        # Number of DSConv layers per tower â€” minimum 1
        self.tower_depth: int = max(
            1, round(self.TEACHER_TOWER_DEPTH * depth_mult)
        )

        self.cls_towers = nn.ModuleList()
        self.reg_towers = nn.ModuleList()
        self.cls_preds  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()
        self.ctr_preds  = nn.ModuleList()

        for in_ch in in_channels_list:
            self.cls_towers.append(nn.Sequential(
                *[DepthwiseSeparableConv(in_ch, in_ch)
                  for _ in range(self.tower_depth)]
            ))
            self.reg_towers.append(nn.Sequential(
                *[DepthwiseSeparableConv(in_ch, in_ch)
                  for _ in range(self.tower_depth)]
            ))
            self.cls_preds.append(nn.Conv2d(in_ch, num_classes, 1))
            self.reg_preds.append(nn.Conv2d(in_ch, 4, 1))
            self.ctr_preds.append(nn.Conv2d(in_ch, 1, 1))

        self.scales = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in in_channels_list
        ])
        self._initialize_biases()

    def _initialize_biases(self) -> None:
        prior_prob = 0.01
        bias_val   = -math.log((1 - prior_prob) / prior_prob)
        for pred in self.cls_preds:
            nn.init.constant_(pred.bias, bias_val)

    def forward(
        self,
        features: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        cls_out, reg_out, ctr_out = [], [], []
        for i, feat in enumerate(features):
            c = self.cls_towers[i](feat)
            r = self.reg_towers[i](feat)
            cls_out.append(self.cls_preds[i](c))
            reg_out.append(F.relu(self.reg_preds[i](r)) * self.scales[i])
            ctr_out.append(self.ctr_preds[i](c))
        return cls_out, reg_out, ctr_out


# ---------------------------------------------------------------------------
# Student model
# ---------------------------------------------------------------------------

class StudentEdgeTurkeyNet(nn.Module):
    """
    Reduced-capacity student for knowledge distillation.

    Mirrors EdgeTurkeyNet's three-part structure
    (backbone â†’ neck â†’ head) but with four independent scaling axes:

      backbone_width_mult â€” scales ShuffleNetV2 stage channel counts.
                            Applied to x0.5 reference channels [48, 96, 192].
                            Only active when backbone_name='shufflenetv2'.
                            MobileNetV3 has no torchvision width API; ignored.

      backbone_depth_mult â€” scales ShuffleNetV2 blocks per stage.
                            Applied to x0.5 reference depths [4, 8, 4].
                            Only active when backbone_name='shufflenetv2'.

      width_mult          â€” scales neck + head tower channel counts.
                            Applied to PANLiteNeck reference channels [128,96,64].

      depth_mult          â€” scales head tower DSConv layer count.
                            Applied to AnchorFreeHead reference depth of 2.

    For shufflenetv2 the backbone is always built from scratch via
    ScaledShuffleNetV2Backbone.  For other backbones, build_backbone() is
    used as before (backbone scaling parameters are silently ignored).

    Args:
        backbone_name:        One of 'mobilenetv3', 'shufflenetv2', 'mobilenetv1'.
        num_classes:          Must match teacher (default 2).
        width_mult:           Neck + head channel multiplier (default 0.5).
        depth_mult:           Head tower depth multiplier (default 0.5).
        backbone_width_mult:  ShuffleNetV2 channel multiplier (default 1.0 = x0.5 ref).
        backbone_depth_mult:  ShuffleNetV2 block-depth multiplier (default 1.0).
        input_size:           Model input (H, W) (must match teacher).
    """

    STRIDES = [8, 16, 32]

    def __init__(
        self,
        backbone_name: str = "mobilenetv3",
        num_classes: int   = NUM_CLASSES,
        width_mult: float  = 0.5,
        depth_mult: float  = 0.5,
        backbone_width_mult: float = 1.0,
        backbone_depth_mult: float = 1.0,
        input_size: Tuple[int, int] = (640, 640),
    ) -> None:
        super().__init__()

        if not 0.0 < width_mult <= 1.0:
            raise ValueError(f"width_mult must be in (0, 1], got {width_mult}")
        if not 0.0 < depth_mult <= 1.0:
            raise ValueError(f"depth_mult must be in (0, 1], got {depth_mult}")

        self.num_classes          = num_classes
        self.width_mult           = width_mult
        self.depth_mult           = depth_mult
        self.backbone_width_mult  = backbone_width_mult
        self.backbone_depth_mult  = backbone_depth_mult
        self.input_size           = input_size
        self.backbone_name        = backbone_name

        # Backbone â€” scaled variant for shufflenetv2, standard otherwise
        if backbone_name.lower() == "shufflenetv2":
            self.backbone = ScaledShuffleNetV2Backbone(
                backbone_width_mult = backbone_width_mult,
                backbone_depth_mult = backbone_depth_mult,
            )
        else:
            # MobileNetV3 / MobileNetV1 â€” always from scratch, no backbone scaling
            self.backbone = build_backbone(backbone_name, pretrained=False)

        # Scaled neck
        self.neck = StudentPANLiteNeck(
            self.backbone.out_channels, width_mult=width_mult
        )

        # Scaled head â€” both width (via neck out_channels) and depth
        self.head = StudentAnchorFreeHead(
            self.neck.out_channels, num_classes, depth_mult=depth_mult
        )

        # Precompute FCOS grids (identical to EdgeTurkeyNet._register_grids)
        self._grids: List[torch.Tensor] = []
        self._register_grids()

    # ------------------------------------------------------------------
    # Grid registration (mirrors EdgeTurkeyNet)
    # ------------------------------------------------------------------

    def _register_grids(self) -> None:
        h, w = self.input_size
        for stride in self.STRIDES:
            fh, fw = h // stride, w // stride
            yv, xv = torch.meshgrid(
                torch.arange(fh, dtype=torch.float32),
                torch.arange(fw, dtype=torch.float32),
                indexing='ij',
            )
            grid = (
                torch.stack([xv, yv], dim=-1).reshape(-1, 2) * stride + stride / 2
            )
            self._grids.append(grid)

    # ------------------------------------------------------------------
    # Shared with EdgeTurkeyNet (copied verbatim for standalone use)
    # ------------------------------------------------------------------

    def decode_predictions(
        self,
        cls_preds: List[torch.Tensor],
        reg_preds: List[torch.Tensor],
        ctr_preds: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode raw head outputs to boxes + scores (identical to teacher)."""
        all_boxes, all_scores, all_ctr = [], [], []
        device = cls_preds[0].device

        for i, (cls_p, reg_p, ctr_p) in enumerate(
            zip(cls_preds, reg_preds, ctr_preds)
        ):
            B, C, H, W = cls_p.shape
            grid = self._grids[i].to(device)

            cls_flat = cls_p.reshape(B, C, -1).permute(0, 2, 1)
            reg_flat = reg_p.reshape(B, 4, -1).permute(0, 2, 1)
            ctr_flat = ctr_p.reshape(B, 1, -1).permute(0, 2, 1)

            cls_scores = torch.sigmoid(cls_flat)
            ctr_scores = torch.sigmoid(ctr_flat)

            cx = grid[:, 0].unsqueeze(0)
            cy = grid[:, 1].unsqueeze(0)

            x1 = (cx - reg_flat[..., 0]).unsqueeze(-1)
            y1 = (cy - reg_flat[..., 1]).unsqueeze(-1)
            x2 = (cx + reg_flat[..., 2]).unsqueeze(-1)
            y2 = (cy + reg_flat[..., 3]).unsqueeze(-1)
            boxes = torch.cat([x1, y1, x2, y2], dim=-1)

            all_boxes.append(boxes)
            all_scores.append(cls_scores)
            all_ctr.append(ctr_scores)

        return (
            torch.cat(all_boxes,  dim=1),
            torch.cat(all_scores, dim=1),
            torch.cat(all_ctr,    dim=1),
        )

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Standard forward â€” same signature as EdgeTurkeyNet.forward()."""
        p3, p4, p5 = self.backbone(x)
        p3, p4, p5 = self.neck(p3, p4, p5)
        return self.head([p3, p4, p5])

    def forward_with_features(
        self, x: torch.Tensor
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],   # neck features
        List[torch.Tensor],                                  # cls preds
        List[torch.Tensor],                                  # reg preds
        List[torch.Tensor],                                  # ctr preds
    ]:
        """
        Forward pass that additionally returns neck feature maps.

        Used by KnowledgeDistillationTrainer to compute feature distillation
        loss between student and teacher neck outputs.

        Returns:
            neck_feats: (p3, p4, p5) feature tensors after neck fusion.
            cls_preds, reg_preds, ctr_preds: raw head outputs.
        """
        p3_b, p4_b, p5_b = self.backbone(x)
        p3, p4, p5        = self.neck(p3_b, p4_b, p5_b)
        cls_p, reg_p, ctr_p = self.head([p3, p4, p5])
        return (p3, p4, p5), cls_p, reg_p, ctr_p


# ---------------------------------------------------------------------------
# Feature adapters â€” project student neck channels â†’ teacher neck channels
# ---------------------------------------------------------------------------

class FeatureAdapters(nn.Module):
    """
    Per-level 1Ã—1 convolution adapters that project student neck feature maps
    to the same channel count as the teacher before computing MSE.

    Without adapters, comparing features of different depths is invalid.
    Adapters are learnable, allowing the student to find the best linear
    projection of its feature space into the teacher's.

    Args:
        student_channels: [p3_ch, p4_ch, p5_ch] from StudentPANLiteNeck.
        teacher_channels: [p3_ch, p4_ch, p5_ch] from PANLiteNeck.
    """

    def __init__(
        self,
        student_channels: List[int],
        teacher_channels: List[int],
    ) -> None:
        super().__init__()
        self.adapters = nn.ModuleList([
            nn.Conv2d(s_ch, t_ch, kernel_size=1, bias=False)
            for s_ch, t_ch in zip(student_channels, teacher_channels)
        ])
        self._init_weights()

    def _init_weights(self) -> None:
        for adapter in self.adapters:
            nn.init.kaiming_normal_(adapter.weight, mode='fan_out')

    def forward(
        self,
        student_feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Project each student feature level to teacher channel dimension.

        Args:
            student_feats: (p3, p4, p5) tensors from StudentPANLiteNeck.

        Returns:
            List of projected tensors, one per FPN level.
        """
        return [
            adapter(feat)
            for adapter, feat in zip(self.adapters, student_feats)
        ]


# ---------------------------------------------------------------------------
# Distillation loss
# ---------------------------------------------------------------------------

class DistillationLoss(nn.Module):
    """
    Combined knowledge distillation loss.

    Combines three signals:
      1. Feature distillation (FD): MSE on projected neck outputs.
      2. Response distillation (RD): soft-target KL on cls logits +
         MSE on reg and ctr outputs.
      3. Ground-truth loss (GT): standard EdgeTurkeyLoss.

    All teacher tensors are detached before loss computation â€” the teacher
    is frozen and its gradient graph is never constructed.

    Args:
        num_classes:   Detection class count.
        temperature:   Softmax temperature for cls KL divergence.
        w_gt:          Weight for ground-truth loss.
        w_fd:          Weight for feature distillation loss.
        w_rd:          Weight for response distillation loss.
        lambda_cls/reg/ctr: EdgeTurkeyLoss component weights (GT branch).
        input_size:    Model input (H, W).
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        temperature: float = 4.0,
        w_gt:  float = 1.0,
        w_fd:  float = 0.5,
        w_rd:  float = 1.0,
        lambda_cls: float = 1.0,
        lambda_reg: float = 2.0,
        lambda_ctr: float = 0.5,
        input_size: Tuple[int, int] = (640, 640),
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.T    = temperature
        self.w_gt = w_gt
        self.w_fd = w_fd
        self.w_rd = w_rd

        # FPN strides â€” used for stride-normalisation of regression MSE.
        # These match EdgeTurkeyNet.STRIDES and are fixed for a 640Ã—640 input.
        self.strides: List[float] = [8.0, 16.0, 32.0]

        # GT branch reuses the existing FCOS loss
        self.gt_loss = EdgeTurkeyLoss(
            num_classes=num_classes,
            lambda_cls=lambda_cls,
            lambda_reg=lambda_reg,
            lambda_ctr=lambda_ctr,
            input_size=input_size,
        )

    # ------------------------------------------------------------------
    # Sub-losses
    # ------------------------------------------------------------------

    def _feature_distillation_loss(
        self,
        student_feats_adapted: List[torch.Tensor],
        teacher_feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        MSE between adapted student features and detached teacher features.

        Normalises by spatial size so loss magnitude is independent of
        FPN level resolution.

        Args:
            student_feats_adapted: Adapter-projected student features [B, T_ch, H, W].
            teacher_feats:         Raw teacher neck outputs (detached inside).

        Returns:
            Scalar mean FD loss across all FPN levels.
        """
        total = torch.tensor(0.0, device=student_feats_adapted[0].device)
        for s_feat, t_feat in zip(student_feats_adapted, teacher_feats):
            total = total + F.mse_loss(s_feat, t_feat.detach())
        return total / len(student_feats_adapted)

    def _response_distillation_loss(
        self,
        student_cls:  List[torch.Tensor],
        student_reg:  List[torch.Tensor],
        student_ctr:  List[torch.Tensor],
        teacher_cls:  List[torch.Tensor],
        teacher_reg:  List[torch.Tensor],
        teacher_ctr:  List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Soft-target response distillation across all FPN levels.

        Classification â€” soft KL divergence
        ------------------------------------
        KL( softmax(S/T) || softmax(T_teach/T) ) * TÂ²

        TÂ² rescaling restores the gradient magnitude to the same order as
        the hard-label loss (Hinton et al., 2015).  With only two classes
        (body / neck) the inter-class signal is limited, so temperature in
        the range 3â€“5 is appropriate.

        Regression â€” stride-normalised Smooth L1
        -----------------------------------------
        Raw (l, t, r, b) values live in pixel space [0, ~300] on a 640Ã—640
        input.  Plain MSE on those values produces losses of 200â€“300, which
        dwarf the classification KL term (typically 0.1â€“1.0) and make the
        KD_W_RD weight uninterpretable.

        Fix: divide both student and teacher predictions by the FPN stride
        before computing the loss.  This maps values into cell-relative
        space, roughly [0, 20] for typical turkey boxes, making rd_reg
        commensurable with rd_cls and rd_ctr.

        Smooth L1 (Huber) is used instead of MSE because:
          - It is linear for errors > beta, capping the gradient for large
            deviations early in training when the student is far from the
            teacher.
          - It matches the regression loss used in most detection frameworks
            including the GT branch (CIoU also avoids squared pixel error).
          - beta=1.0 in cell-relative space â‰ˆ 8 pixels absolute at P3,
            16 px at P4, 32 px at P5 â€” a sensible transition point.

        Centerness â€” MSE
        -----------------
        Centerness is already in [0, 1] (pre-sigmoid logits are bounded by
        training dynamics).  MSE is appropriate and commensurable with the
        other terms without further normalisation.

        Args:
            student_cls/reg/ctr: Raw student head outputs per FPN level
                                 [B, C/4/1, H, W].
            teacher_cls/reg/ctr: Raw teacher head outputs per FPN level
                                 (detached inside this function).

        Returns:
            Dict with scalar tensors:
              'cls' â€” mean soft KL across FPN levels (TÂ²-rescaled)
              'reg' â€” mean stride-normalised Smooth L1 across FPN levels
              'ctr' â€” mean MSE across FPN levels
        """
        T = self.T
        device    = student_cls[0].device
        total_cls = torch.zeros(1, device=device)
        total_reg = torch.zeros(1, device=device)
        total_ctr = torch.zeros(1, device=device)

        for idx, (s_cls, s_reg, s_ctr, t_cls, t_reg, t_ctr) in enumerate(zip(
            student_cls, student_reg, student_ctr,
            teacher_cls, teacher_reg, teacher_ctr,
        )):
            # â”€â”€ Classification: soft KL divergence â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # Flatten spatial dims: [B, C, H, W] â†’ [B*H*W, C]
            B, C, H, W = s_cls.shape
            s_flat = s_cls.reshape(B, C, -1).permute(0, 2, 1).reshape(-1, C)
            t_flat = t_cls.reshape(B, C, -1).permute(0, 2, 1).reshape(-1, C)

            s_log_soft = F.log_softmax(s_flat / T, dim=-1)
            t_soft     = F.softmax(t_flat.detach() / T, dim=-1)
            # kl_div(log_input, target); batchmean divides by batch size
            cls_kl     = F.kl_div(s_log_soft, t_soft, reduction='batchmean') * (T ** 2)
            total_cls  = total_cls + cls_kl

            # â”€â”€ Regression: stride-normalised Smooth L1 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # Divide by stride to convert pixel distances â†’ cell-relative.
            # s_reg / stride puts values in roughly [0, 20] range, making
            # rd_reg the same order of magnitude as rd_cls and rd_ctr.
            stride     = self.strides[idx]
            s_reg_norm = s_reg         / stride
            t_reg_norm = t_reg.detach() / stride
            reg_loss   = F.smooth_l1_loss(s_reg_norm, t_reg_norm, beta=1.0)
            total_reg  = total_reg + reg_loss

            # â”€â”€ Centerness: plain MSE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # Logits are bounded by training dynamics; no normalisation needed.
            total_ctr = total_ctr + F.mse_loss(s_ctr, t_ctr.detach())

        n = len(student_cls)
        return {
            'cls': total_cls / n,
            'reg': total_reg / n,
            'ctr': total_ctr / n,
        }

    # ------------------------------------------------------------------
    # Combined forward
    # ------------------------------------------------------------------

    def forward(
        self,
        # Student outputs
        student_feats_adapted: List[torch.Tensor],
        student_cls:  List[torch.Tensor],
        student_reg:  List[torch.Tensor],
        student_ctr:  List[torch.Tensor],
        # Teacher outputs (will be detached inside)
        teacher_feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        teacher_cls:   List[torch.Tensor],
        teacher_reg:   List[torch.Tensor],
        teacher_ctr:   List[torch.Tensor],
        # Ground-truth labels
        gt_boxes_batch:     List[torch.Tensor],
        gt_class_ids_batch: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute full distillation + GT loss.

        Returns:
            Dict with keys:
              'total'    â€” weighted sum of all components
              'gt'       â€” ground-truth EdgeTurkeyLoss total
              'fd'       â€” feature distillation loss
              'rd_cls'   â€” response distillation classification KL
              'rd_reg'   â€” response distillation regression MSE
              'rd_ctr'   â€” response distillation centerness MSE
        """
        # â”€â”€ 1. Feature distillation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        fd_loss = self._feature_distillation_loss(
            student_feats_adapted, teacher_feats
        )

        # â”€â”€ 2. Response distillation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        rd_losses = self._response_distillation_loss(
            student_cls, student_reg, student_ctr,
            teacher_cls, teacher_reg, teacher_ctr,
        )
        rd_total = rd_losses['cls'] + rd_losses['reg'] + rd_losses['ctr']

        # â”€â”€ 3. Ground-truth detection loss â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        gt_losses = self.gt_loss(
            student_cls, student_reg, student_ctr,
            gt_boxes_batch, gt_class_ids_batch,
        )

        # â”€â”€ 4. Weighted total â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        total = (
            self.w_gt * gt_losses['total'] +
            self.w_fd * fd_loss +
            self.w_rd * rd_total
        )

        return {
            'total':  total,
            'gt':     gt_losses['total'],
            'fd':     fd_loss,
            'rd_cls': rd_losses['cls'],
            'rd_reg': rd_losses['reg'],
            'rd_ctr': rd_losses['ctr'],
        }


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_student(
    backbone_name:       str   = "mobilenetv3",
    width_mult:          float = KD_WIDTH_MULT,
    depth_mult:          float = 0.5,
    backbone_width_mult: float = 1.0,
    backbone_depth_mult: float = 1.0,
    num_classes:         int   = NUM_CLASSES,
    input_size:          Tuple[int, int] = (640, 640),
) -> StudentEdgeTurkeyNet:
    """
    Construct a StudentEdgeTurkeyNet.

    Args:
        backbone_name:       Backbone architecture ('shufflenetv2' | 'mobilenetv3' | 'mobilenetv1').
        width_mult:          Neck + head channel multiplier.
        depth_mult:          Head tower depth multiplier (teacher depth=2).
        backbone_width_mult: ShuffleNetV2 stage channel multiplier (ignored for other backbones).
        backbone_depth_mult: ShuffleNetV2 stage block-depth multiplier (ignored for other backbones).
        num_classes:         Detection class count (must match teacher).
        input_size:          Model input (H, W).

    Returns:
        StudentEdgeTurkeyNet ready for distillation or scratch training.
    """
    return StudentEdgeTurkeyNet(
        backbone_name       = backbone_name,
        num_classes         = num_classes,
        width_mult          = width_mult,
        depth_mult          = depth_mult,
        backbone_width_mult = backbone_width_mult,
        backbone_depth_mult = backbone_depth_mult,
        input_size          = input_size,
    )


def load_frozen_teacher(checkpoint_path: Path, cfg: "RunConfig") -> EdgeTurkeyNet:
    """
    Load a trained EdgeTurkeyNet checkpoint and freeze all parameters.

    The teacher is set to eval mode and all requires_grad flags are set
    to False.  No gradients are ever computed through the teacher, so
    memory and compute overhead is minimal.

    Args:
        checkpoint_path: Path to the .pth checkpoint saved by Trainer.
        cfg:             RunConfig (backbone used as fallback).

    Returns:
        Frozen EdgeTurkeyNet on DEVICE in eval mode.
    """
    ckpt     = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    backbone = ckpt.get("backbone", cfg.backbone)

    teacher = EdgeTurkeyNet(
        num_classes=NUM_CLASSES,
        pretrained_backbone=False,
        backbone=backbone,
    )
    teacher.load_state_dict(ckpt["model_state"])
    teacher.eval()

    for param in teacher.parameters():
        param.requires_grad = False

    teacher = teacher.to(DEVICE)
    print(
        f"[KD] Teacher loaded: {checkpoint_path.name}  "
        f"backbone={backbone}  "
        f"params={sum(p.numel() for p in teacher.parameters()):,}  "
        f"[FROZEN]"
    )
    return teacher


def _teacher_neck_features(
    teacher: EdgeTurkeyNet,
    images: torch.Tensor,
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
]:
    """
    Run teacher forward pass and extract neck features + head outputs.

    Hooks into the teacher's backbone and neck to capture intermediate
    feature maps without modifying the teacher's architecture.

    Returns:
        neck_feats: (p3, p4, p5) neck output tensors.
        cls_preds, reg_preds, ctr_preds: raw teacher head outputs.
    """
    neck_out: Dict[str, torch.Tensor] = {}

    def _hook(name):
        def fn(module, inp, out):
            neck_out[name] = out
        return fn

    h = teacher.neck.register_forward_hook(
        lambda m, i, o: neck_out.update({'feats': o})
    )
    cls_p, reg_p, ctr_p = teacher(images)
    h.remove()

    feats = neck_out['feats']   # tuple (p3, p4, p5)
    return feats, cls_p, reg_p, ctr_p


# ---------------------------------------------------------------------------
# Knowledge Distillation Trainer
# ---------------------------------------------------------------------------

class KnowledgeDistillationTrainer:
    """
    Trains a StudentEdgeTurkeyNet using knowledge distillation from a
    frozen teacher EdgeTurkeyNet.

    Training loop per epoch
    -----------------------
    1. Run images through the frozen teacher (no grad).
       Capture: neck features (p3, p4, p5) + head outputs (cls, reg, ctr).
    2. Run images through the student (with grad).
       Capture: neck features + head outputs.
    3. Project student neck features through FeatureAdapters.
    4. Compute DistillationLoss (FD + RD + GT).
    5. Back-propagate through student + adapters only.
    6. Validate student mAP@IoU after each epoch.
    7. Log epoch row to CSV via RunLogger.
    8. Save best/last student checkpoint.
    9. Optional periodic pruning of the student.

    Args:
        teacher_checkpoint: Path to the frozen teacher's best.pth.
        cfg:                RunConfig from config.py.
        logger:             RunLogger for a fresh timestamped run directory.
        width_mult:         Student width multiplier (default KD_WIDTH_MULT).
        temperature:        KL divergence temperature (default KD_TEMPERATURE).
        w_gt / w_fd / w_rd: Loss component weights.
        epochs:             Max student training epochs (default KD_EPOCHS).
        early_stop_patience: Patience for early stopping (default KD_EARLY_STOP_PATIENCE).
        prune_student:      Apply periodic pruning to student (default KD_PRUNE_STUDENT).
    """

    def __init__(
        self,
        teacher_checkpoint: Path,
        cfg: "RunConfig",
        logger: "RunLogger",
        width_mult: float = KD_WIDTH_MULT,
        temperature: float = KD_TEMPERATURE,
        w_gt:  float = KD_W_GT,
        w_fd:  float = KD_W_FD,
        w_rd:  float = KD_W_RD,
        epochs: int = KD_EPOCHS,
        early_stop_patience: int = KD_EARLY_STOP_PATIENCE,
        prune_student: bool = KD_PRUNE_STUDENT,
    ) -> None:
        self.cfg    = cfg
        self.logger = logger

        set_seed(cfg.seed)

        # â”€â”€ Teacher (frozen) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.teacher = load_frozen_teacher(Path(teacher_checkpoint), cfg)

        # â”€â”€ Student â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.student = build_student(
            backbone_name=cfg.backbone,
            width_mult=width_mult,
            num_classes=NUM_CLASSES,
            input_size=cfg.input_size,
        ).to(DEVICE)

        print(
            f"[KD] Student: backbone={cfg.backbone}  "
            f"width_mult={width_mult}  "
            f"params={self.student.get_parameter_count():,}"
        )

        # â”€â”€ Feature adapters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        teacher_neck_chs = list(PANLiteNeck.NECK_CHANNELS.values())  # [128, 96, 64]
        student_neck_chs = self.student.neck.out_channels             # scaled
        self.adapters = FeatureAdapters(
            student_neck_chs, teacher_neck_chs
        ).to(DEVICE)

        # â”€â”€ Loss â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.criterion = DistillationLoss(
            num_classes=NUM_CLASSES,
            temperature=temperature,
            w_gt=w_gt,
            w_fd=w_fd,
            w_rd=w_rd,
            lambda_cls=cfg.lambda_cls,
            lambda_reg=cfg.lambda_reg,
            lambda_ctr=cfg.lambda_ctr,
            input_size=cfg.input_size,
        )

        # â”€â”€ Optimiser â€” student + adapters, backbone gets lower LR â”€â”€â”€â”€
        backbone_params = list(self.student.backbone.parameters())
        backbone_ids    = {id(p) for p in backbone_params}
        other_params    = (
            [p for p in self.student.parameters() if id(p) not in backbone_ids]
            + list(self.adapters.parameters())
        )

        self.optimizer = optim.AdamW(
            [
                {"params": backbone_params, "lr": cfg.base_lr * 0.1},
                {"params": other_params,    "lr": cfg.base_lr},
            ],
            weight_decay=cfg.weight_decay,
        )

        self.scaler = GradScaler('cuda',enabled=torch.cuda.is_available())

        # â”€â”€ Data loaders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.train_loader = get_train_loader(
            batch_size=cfg.batch_size, num_workers=cfg.num_workers
        )
        self.val_loader = get_val_loader(
            batch_size=cfg.batch_size, num_workers=cfg.num_workers
        )

        # â”€â”€ Pruner (optional) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.prune_student = prune_student
        if prune_student:
            # Wrap student in a thin shim so ChannelPruner can locate head preds
            self._student_for_pruner = _StudentPrunerShim(self.student)
            self.pruner = ChannelPruner(
                self._student_for_pruner,
                per_call_ratio=cfg.prune_per_call,
                max_sparsity=cfg.prune_max_sparsity,
            )

        # â”€â”€ Training state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.epochs              = epochs
        self.early_stop_patience = early_stop_patience
        self.best_map            = 0.0
        self.no_improve_cnt      = 0

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_student_checkpoint(self, path: Path, epoch: int) -> None:
        torch.save({
            "epoch":                epoch,
            "model_state":          self.student.state_dict(),
            "adapter_state":        self.adapters.state_dict(),
            "optimizer_state":      self.optimizer.state_dict(),
            "best_map":             self.best_map,
            "no_improve_cnt":       self.no_improve_cnt,
            "backbone":             self.cfg.backbone,
            "width_mult":           self.student.width_mult,
            "depth_mult":           self.student.depth_mult,
            "backbone_width_mult":  self.student.backbone_width_mult,
            "backbone_depth_mult":  self.student.backbone_depth_mult,
        }, path)

    # ------------------------------------------------------------------
    # One training epoch
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Run one full distillation training epoch.

        The teacher is run in a torch.no_grad() context to avoid
        storing any intermediate activations for the teacher graph.

        Returns:
            Dict of mean loss components: total, gt, fd, rd_cls, rd_reg, rd_ctr.
        """
        self.student.train()
        self.adapters.train()

        # Fake RunConfig for get_lr
        lr = get_lr(epoch, self.cfg)
        # Override epoch count with KD epochs for correct cosine schedule
        _cfg_kd = _KDCfgProxy(self.cfg, self.epochs)
        lr = get_lr(epoch, _cfg_kd)

        self.optimizer.param_groups[0]["lr"] = lr * 0.1
        self.optimizer.param_groups[1]["lr"] = lr

        totals: Dict[str, float] = {
            "total": 0.0, "gt": 0.0, "fd": 0.0,
            "rd_cls": 0.0, "rd_reg": 0.0, "rd_ctr": 0.0,
        }
        n_batches = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            images     = batch["images"].to(DEVICE, non_blocking=True)
            gt_boxes   = batch["boxes"]
            gt_cls_ids = batch["class_ids"]

            # â”€â”€ Teacher forward (no grad) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            with torch.no_grad():
                t_feats, t_cls, t_reg, t_ctr = _teacher_neck_features(
                    self.teacher, images
                )

            # â”€â”€ Student forward (with grad) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            self.optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=torch.cuda.is_available()):
                s_feats, s_cls, s_reg, s_ctr = (
                    self.student.forward_with_features(images)
                )
                s_feats_adapted = self.adapters(s_feats)

                losses = self.criterion(
                    student_feats_adapted=s_feats_adapted,
                    student_cls=s_cls,
                    student_reg=s_reg,
                    student_ctr=s_ctr,
                    teacher_feats=t_feats,
                    teacher_cls=t_cls,
                    teacher_reg=t_reg,
                    teacher_ctr=t_ctr,
                    gt_boxes_batch=gt_boxes,
                    gt_class_ids_batch=gt_cls_ids,
                )

            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                list(self.student.parameters()) + list(self.adapters.parameters()),
                self.cfg.gradient_clip,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for k in totals:
                totals[k] += losses[k].item()
            n_batches += 1

            if (batch_idx + 1) % 20 == 0:
                print(
                    f"  [KD E{epoch:03d}] Batch {batch_idx+1}/"
                    f"{len(self.train_loader)} | "
                    f"Loss={losses['total'].item():.4f}  "
                    f"gt={losses['gt'].item():.4f}  "
                    f"fd={losses['fd'].item():.4f}  "
                    f"rd_cls={losses['rd_cls'].item():.4f} | "
                    f"LR={lr:.6f} | t={time.time()-t0:.1f}s"
                )

        return {k: v / max(1, n_batches) for k, v in totals.items()}

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def _maybe_prune_student(self, epoch: int) -> None:
        if not self.prune_student:
            return
        cfg = self.cfg
        if epoch < cfg.prune_start_epoch:
            return
        if (epoch - cfg.prune_start_epoch) % cfg.prune_interval != 0:
            return
        print(f"\n[KD] â”€â”€ Periodic student pruning at epoch {epoch} â”€â”€")
        self.pruner.prune()

    # ------------------------------------------------------------------
    # Main distillation loop
    # ------------------------------------------------------------------

    def train(self) -> StudentEdgeTurkeyNet:
        """
        Run the full knowledge distillation training loop.

        Returns:
            Best-checkpoint StudentEdgeTurkeyNet in eval mode on CPU.
        """
        cfg = self.cfg
        print(f"\n{'='*65}")
        print(
            f"  Knowledge Distillation  |  backbone={cfg.backbone}  "
            f"width_mult={self.student.width_mult}  "
            f"epochs={self.epochs}  device={DEVICE}"
        )
        print(
            f"  Teacher params: {sum(p.numel() for p in self.teacher.parameters()):,}  "
            f"Student params: {self.student.get_parameter_count():,}"
        )
        print(f"  Run dir: {self.logger.run_dir}")
        print(f"{'='*65}\n")

        best_ckpt  = self.logger.checkpoint_dir / "student_best.pth"
        last_ckpt  = self.logger.checkpoint_dir / "student_last.pth"

        for epoch in range(self.epochs):

            # â”€â”€ 1. Train â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            train_losses = self._train_one_epoch(epoch)
            _cfg_kd = _KDCfgProxy(cfg, self.epochs)
            lr_now = get_lr(epoch, _cfg_kd)

            print(
                f"\n[KD E{epoch:03d}] Train | "
                f"total={train_losses['total']:.4f}  "
                f"gt={train_losses['gt']:.4f}  "
                f"fd={train_losses['fd']:.4f}  "
                f"rd_cls={train_losses['rd_cls']:.4f}  "
                f"rd_reg={train_losses['rd_reg']:.4f}  "
                f"rd_ctr={train_losses['rd_ctr']:.4f}"
            )

            # â”€â”€ 2. Optional student pruning â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            self._maybe_prune_student(epoch)

            # â”€â”€ 3. Validate student â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            val_metrics: PerClassMetrics = evaluate_map(
                self.student, self.val_loader, DEVICE,
                iou_threshold=cfg.iou_threshold,
                score_threshold=cfg.score_threshold,
            )
            val_metrics.print_table(iou_threshold=cfg.iou_threshold)

            # â”€â”€ 4. CSV logging â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # Re-use RunLogger.log_epoch with KD losses mapped to train_losses keys
            kd_losses_for_csv = {
                "total": train_losses["total"],
                "cls":   train_losses["rd_cls"],
                "reg":   train_losses["rd_reg"],
                "ctr":   train_losses["rd_ctr"],
            }
            self.logger.log_epoch(epoch, lr_now, kd_losses_for_csv, val_metrics)

            # â”€â”€ 5. Checkpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            self._save_student_checkpoint(last_ckpt, epoch)

            if val_metrics.map > self.best_map:
                self.best_map       = val_metrics.map
                self.no_improve_cnt = 0
                self._save_student_checkpoint(best_ckpt, epoch)
                print(
                    f"  âœ“ New best student mAP@{cfg.iou_threshold:.2f}: "
                    f"{self.best_map:.4f}"
                )
            else:
                self.no_improve_cnt += 1
                print(
                    f"  No improvement "
                    f"{self.no_improve_cnt}/{self.early_stop_patience}"
                )
                if self.no_improve_cnt >= self.early_stop_patience:
                    print(f"\n[KD] Early stopping at epoch {epoch}.")
                    break

        print(
            f"\n[KD] Done.  Best student mAP@{cfg.iou_threshold:.2f} = "
            f"{self.best_map:.4f}"
        )

        return self.load_best_student(best_ckpt)

    # ------------------------------------------------------------------
    # Load best student
    # ------------------------------------------------------------------

    def load_best_student(self, path: Path) -> StudentEdgeTurkeyNet:
        """
        Load best student checkpoint.

        Returns:
            StudentEdgeTurkeyNet in eval mode on CPU.
        """
        ckpt                = torch.load(path, map_location="cpu", weights_only=False)
        backbone            = ckpt.get("backbone",            self.cfg.backbone)
        width_mult          = ckpt.get("width_mult",          self.student.width_mult)
        depth_mult          = ckpt.get("depth_mult",          self.student.depth_mult)
        backbone_width_mult = ckpt.get("backbone_width_mult", self.student.backbone_width_mult)
        backbone_depth_mult = ckpt.get("backbone_depth_mult", self.student.backbone_depth_mult)

        student = StudentEdgeTurkeyNet(
            backbone_name       = backbone,
            num_classes         = NUM_CLASSES,
            width_mult          = width_mult,
            depth_mult          = depth_mult,
            backbone_width_mult = backbone_width_mult,
            backbone_depth_mult = backbone_depth_mult,
            input_size          = self.cfg.input_size,
        )
        student.load_state_dict(ckpt["model_state"])
        student.eval()
        print(
            f"[KD] Best student loaded: {path.name}  "
            f"backbone={backbone}  width_mult={width_mult}  "
            f"depth_mult={depth_mult}  "
            f"backbone_width_mult={backbone_width_mult}  "
            f"backbone_depth_mult={backbone_depth_mult}  "
            f"mAP={ckpt.get('best_map', 0.0):.4f}"
        )
        return student


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _StudentPrunerShim:
    """
    Thin shim so ChannelPruner.model.head.{cls,reg,ctr}_preds resolves
    correctly for the StudentEdgeTurkeyNet (which uses StudentAnchorFreeHead).
    """

    def __init__(self, student: StudentEdgeTurkeyNet) -> None:
        self.head = student.head

    def modules(self):
        return self.head.parameters.__self__.modules() if False else iter([])

    # ChannelPruner iterates self.model.modules() â€” delegate to student
    def __getattr__(self, name):
        return getattr(self._student, name)

    def __init__(self, student: StudentEdgeTurkeyNet) -> None:  # type: ignore[no-redef]
        self._student = student
        self.head     = student.head

    def modules(self):  # type: ignore[override]
        return self._student.modules()


class _KDCfgProxy:
    """
    Lightweight proxy that overrides only ``epochs`` on a RunConfig so that
    ``get_lr()`` uses the KD epoch count rather than the full training count.
    """

    def __init__(self, cfg: "RunConfig", epochs: int) -> None:
        self._cfg   = cfg
        self.epochs = epochs

    def __getattr__(self, name: str):
        return getattr(self._cfg, name)
