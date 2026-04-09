from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WeightedFusion(nn.Module):
    def __init__(self, num_inputs: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        self.epsilon = 1e-4

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        weight = F.relu(self.weights)
        normalized = weight / (weight.sum() + self.epsilon)
        return sum(w * x for w, x in zip(normalized, inputs))


class BiFPNLite(nn.Module):
    def __init__(self, in_channels: list[int], out_channels: int = 128) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(nn.Conv2d(ch, out_channels, kernel_size=1) for ch in in_channels)
        self.top_down_fuse = nn.ModuleList(WeightedFusion(2) for _ in range(len(in_channels) - 1))
        self.bottom_up_fuse = nn.ModuleList(WeightedFusion(2) for _ in range(len(in_channels) - 1))
        self.output_convs = nn.ModuleList(ConvBNAct(out_channels, out_channels, 3) for _ in in_channels)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        pyramids = [layer(feature) for layer, feature in zip(self.lateral, features)]

        for index in range(len(pyramids) - 2, -1, -1):
            upsampled = F.interpolate(pyramids[index + 1], size=pyramids[index].shape[-2:], mode="nearest")
            pyramids[index] = self.output_convs[index](self.top_down_fuse[index]([pyramids[index], upsampled]))

        for index in range(1, len(pyramids)):
            pooled = F.max_pool2d(pyramids[index - 1], kernel_size=2, stride=2)
            pyramids[index] = self.output_convs[index](self.bottom_up_fuse[index - 1]([pyramids[index], pooled]))

        return pyramids
