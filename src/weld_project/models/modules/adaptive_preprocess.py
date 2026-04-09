from __future__ import annotations

import torch
from torch import nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)


class AdaptivePreprocessModule(nn.Module):
    def __init__(self, channels: int = 3, hidden_channels: int = 16, iterations: int = 4) -> None:
        super().__init__()
        self.iterations = iterations
        self.encoder = nn.Sequential(
            DepthwiseSeparableConv(channels, hidden_channels),
            DepthwiseSeparableConv(hidden_channels, hidden_channels),
            DepthwiseSeparableConv(hidden_channels, hidden_channels),
        )
        self.curve_head = nn.Sequential(
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        curve = self.curve_head(self.encoder(x))
        enhanced = x
        for _ in range(self.iterations):
            enhanced = enhanced + curve * (enhanced.square() - enhanced)
        return enhanced.clamp(0.0, 1.0)
