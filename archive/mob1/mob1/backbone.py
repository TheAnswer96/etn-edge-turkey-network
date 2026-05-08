# =============================================================================
# backbone.py — MobileNetV1-0.25 backbone (ReLU6, no skip connections)
# Quantization-friendly: Conv→BN→ReLU6 only, no Hardswish/SiLU/residuals
# =============================================================================

import torch
import torch.nn as nn
from mob1.config import WIDTH_MULT


def _make_divisible(v, divisor=8):
    return max(divisor, int(v + divisor / 2) // divisor * divisor)


def conv_bn_relu6(in_c, out_c, stride=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1,
                  groups=groups, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU6(inplace=True),
    )


def conv1x1_bn_relu6(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU6(inplace=True),
    )


class DepthwiseSeparable(nn.Module):
    """Depthwise separable conv block — the core MobileNetV1 unit."""
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.dw = conv_bn_relu6(in_c, in_c, stride=stride, groups=in_c)
        self.pw = conv1x1_bn_relu6(in_c, out_c)

    def forward(self, x):
        return self.pw(self.dw(x))


class MobileNetV1Backbone(nn.Module):
    def __init__(self, width_mult: float = WIDTH_MULT):
        super().__init__()

        def c(n): return _make_divisible(int(n * width_mult))

        # Standard conv: stride 2 → 128×128
        self.stem = conv_bn_relu6(3, c(32), stride=2)

        # Depthwise separable blocks
        self.layers = nn.Sequential(
            DepthwiseSeparable(c(32),  c(64),  stride=1),   # 128×128
            DepthwiseSeparable(c(64),  c(128), stride=2),   # 64×64
            DepthwiseSeparable(c(128), c(128), stride=1),   # 64×64
            DepthwiseSeparable(c(128), c(256), stride=2),   # 32×32
            DepthwiseSeparable(c(256), c(256), stride=1),   # 32×32
            DepthwiseSeparable(c(256), c(512), stride=2),   # 16×16  ← output
        )

        self.out_channels = c(512)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        return x   # (B, out_channels, H/16, W/16)


if __name__ == "__main__":
    m = MobileNetV1Backbone()
    x = torch.randn(1, 3, 256, 256)
    y = m(x)
    print(f"Backbone output: {y.shape}")   # expect (1, 128, 16, 16)
    total = sum(p.numel() for p in m.parameters())
    print(f"Backbone params: {total:,}")