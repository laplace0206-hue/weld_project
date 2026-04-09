from __future__ import annotations

import torch
from torch import nn

from .modules.adaptive_preprocess import AdaptivePreprocessModule
from .modules.attention import build_attention


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WeldAdaptiveBlock(nn.Module):
    def __init__(self, c1: int, c2: int, hidden_channels: int = 16, iterations: int = 4) -> None:
        super().__init__()
        self.project = ConvBNAct(c1, c2, kernel_size=1) if c1 != c2 else nn.Identity()
        self.preprocess = AdaptivePreprocessModule(channels=c2, hidden_channels=hidden_channels, iterations=iterations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project(x)
        return self.preprocess(x)


class WeldEMA(nn.Module):
    def __init__(self, c1: int, c2: int, reduction: int = 8) -> None:
        super().__init__()
        self.project = ConvBNAct(c1, c2, kernel_size=1) if c1 != c2 else nn.Identity()
        self.attention = build_attention("ema", c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project(x)
        return self.attention(x)


class LiteFPNRefine(nn.Module):
    def __init__(self, c1: int, c2: int, expansion: float = 0.5, use_shortcut: bool = True) -> None:
        super().__init__()
        hidden = max(int(c2 * expansion), 16)
        self.reduce = ConvBNAct(c1, hidden, kernel_size=1)
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.expand = ConvBNAct(hidden, c2, kernel_size=1)
        self.shortcut = use_shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.reduce(x)
        x = self.depthwise(x)
        x = self.expand(x)
        if self.shortcut:
            x = x + identity
        return x
