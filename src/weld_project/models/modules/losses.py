from __future__ import annotations

import torch
import torch.nn.functional as F


def sigmoid_focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    probability = logits.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = probability * targets + (1 - probability) * (1 - targets)
    focal_weight = (alpha * targets + (1 - alpha) * (1 - targets)) * (1 - pt).pow(gamma)
    return (ce_loss * focal_weight).mean()


def _box_edges(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_center, y_center, width, height = boxes.unbind(dim=-1)
    half_w = width / 2
    half_h = height / 2
    return x_center - half_w, y_center - half_h, x_center + half_w, y_center + half_h


def eiou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    pred_x1, pred_y1, pred_x2, pred_y2 = _box_edges(pred_boxes)
    tgt_x1, tgt_y1, tgt_x2, tgt_y2 = _box_edges(target_boxes)

    inter_x1 = torch.maximum(pred_x1, tgt_x1)
    inter_y1 = torch.maximum(pred_y1, tgt_y1)
    inter_x2 = torch.minimum(pred_x2, tgt_x2)
    inter_y2 = torch.minimum(pred_y2, tgt_y2)

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
    tgt_area = (tgt_x2 - tgt_x1).clamp(min=0) * (tgt_y2 - tgt_y1).clamp(min=0)
    union = pred_area + tgt_area - inter + eps
    iou = inter / union

    enclose_x1 = torch.minimum(pred_x1, tgt_x1)
    enclose_y1 = torch.minimum(pred_y1, tgt_y1)
    enclose_x2 = torch.maximum(pred_x2, tgt_x2)
    enclose_y2 = torch.maximum(pred_y2, tgt_y2)

    cw = (enclose_x2 - enclose_x1).clamp(min=eps)
    ch = (enclose_y2 - enclose_y1).clamp(min=eps)

    center_distance = (pred_boxes[..., 0] - target_boxes[..., 0]).pow(2) + (pred_boxes[..., 1] - target_boxes[..., 1]).pow(2)
    width_distance = (pred_boxes[..., 2] - target_boxes[..., 2]).pow(2)
    height_distance = (pred_boxes[..., 3] - target_boxes[..., 3]).pow(2)

    return (1 - iou + center_distance / (cw.pow(2) + ch.pow(2)) + width_distance / cw.pow(2) + height_distance / ch.pow(2)).mean()


def focal_eiou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    cls_logits: torch.Tensor,
    cls_targets: torch.Tensor,
    box_weight: float = 2.0,
    cls_weight: float = 1.0,
) -> torch.Tensor:
    box_loss = eiou_loss(pred_boxes, target_boxes)
    cls_loss = sigmoid_focal_loss(cls_logits, cls_targets)
    return box_weight * box_loss + cls_weight * cls_loss
