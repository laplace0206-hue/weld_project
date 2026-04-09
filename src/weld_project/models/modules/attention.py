from __future__ import annotations

import torch
from torch import nn


class SimAMAttention(nn.Module):
    def __init__(self, epsilon: float = 1e-4) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        variance = (x - mean).pow(2)
        score = variance / (4 * (variance.mean(dim=(2, 3), keepdim=True) + self.epsilon)) + 0.5
        return x * torch.sigmoid(score)


class EMAAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.shared = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.conv_h = nn.Conv2d(hidden, channels, kernel_size=1)
        self.conv_w = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        context_h = x.mean(dim=3, keepdim=True)
        context_w = x.mean(dim=2, keepdim=True).transpose(2, 3)
        context = torch.cat([context_h, context_w], dim=2)
        context = self.shared(context)
        height, width = x.shape[2:]
        attention_h, attention_w = torch.split(context, [height, width], dim=2)
        attention_w = attention_w.transpose(2, 3)
        attention_h = self.conv_h(attention_h).sigmoid()
        attention_w = self.conv_w(attention_w).sigmoid()
        return identity * attention_h * attention_w


def build_attention(name: str, channels: int) -> nn.Module:
    normalized = name.lower()
    if normalized == "simam":
        return SimAMAttention()
    if normalized == "ema":
        return EMAAttention(channels)
    raise ValueError(f"不支持的注意力类型: {name}")
