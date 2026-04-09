from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import mobilenet_v3_small

from .modules import AdaptivePreprocessModule, BiFPNLite, build_attention


@dataclass
class DetectorOutputs:
    cls_logits: list[torch.Tensor]
    box_reg: list[torch.Tensor]
    obj_logits: list[torch.Tensor]


class MobileNetFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = mobilenet_v3_small(weights=None).features
        self.stem = backbone[:4]
        self.stage3 = backbone[4:7]
        self.stage4 = backbone[7:10]
        self.stage5 = backbone[10:]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return [c3, c4, c5]


class DetectionHead(nn.Module):
    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        self.cls_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, num_classes, 1),
        )
        self.box_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 4, 1),
        )
        self.obj_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.cls_head(x), self.box_head(x), self.obj_head(x)


class ImprovedWeldDetector(nn.Module):
    def __init__(
        self,
        num_classes: int,
        use_preprocess: bool = True,
        use_attention: bool = True,
        attention_type: str = "ema",
        neck_out_channels: int = 128,
    ) -> None:
        super().__init__()
        self.preprocess = AdaptivePreprocessModule() if use_preprocess else nn.Identity()
        self.backbone = MobileNetFeatureExtractor()
        self.neck = BiFPNLite(in_channels=[40, 96, 576], out_channels=neck_out_channels)
        self.use_attention = use_attention
        if use_attention:
            self.attention_blocks = nn.ModuleList(build_attention(attention_type, neck_out_channels) for _ in range(3))
        else:
            self.attention_blocks = nn.ModuleList(nn.Identity() for _ in range(3))
        self.heads = nn.ModuleList(DetectionHead(neck_out_channels, num_classes) for _ in range(3))

    def forward(self, x: torch.Tensor) -> DetectorOutputs:
        x = self.preprocess(x)
        features = self.backbone(x)
        features = self.neck(features)
        features = [attention(feature) for attention, feature in zip(self.attention_blocks, features)]

        cls_logits: list[torch.Tensor] = []
        box_reg: list[torch.Tensor] = []
        obj_logits: list[torch.Tensor] = []
        for head, feature in zip(self.heads, features):
            cls_pred, box_pred, obj_pred = head(feature)
            cls_logits.append(cls_pred)
            box_reg.append(box_pred)
            obj_logits.append(obj_pred)

        return DetectorOutputs(cls_logits=cls_logits, box_reg=box_reg, obj_logits=obj_logits)
