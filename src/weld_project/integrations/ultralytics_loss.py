from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..models.modules.losses import sigmoid_focal_loss


def _xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    width = (x2 - x1).clamp(min=0)
    height = (y2 - y1).clamp(min=0)
    x_center = x1 + width / 2
    y_center = y1 + height / 2
    return torch.stack((x_center, y_center, width, height), dim=-1)


def weighted_eiou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, weight: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    pred_xywh = _xyxy_to_xywh(pred_boxes)
    target_xywh = _xyxy_to_xywh(target_boxes)

    pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes.unbind(dim=-1)
    tgt_x1, tgt_y1, tgt_x2, tgt_y2 = target_boxes.unbind(dim=-1)

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

    center_distance = (pred_xywh[..., 0] - target_xywh[..., 0]).pow(2) + (pred_xywh[..., 1] - target_xywh[..., 1]).pow(2)
    width_distance = (pred_xywh[..., 2] - target_xywh[..., 2]).pow(2)
    height_distance = (pred_xywh[..., 3] - target_xywh[..., 3]).pow(2)

    eiou = 1 - iou + center_distance / (cw.pow(2) + ch.pow(2)) + width_distance / cw.pow(2) + height_distance / ch.pow(2)
    return (eiou.unsqueeze(-1) * weight).sum()


class CustomBboxLoss(nn.Module):
    def __init__(self, reg_max: int = 16):
        super().__init__()
        from ultralytics.utils.loss import DFLoss

        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from ultralytics.utils.loss import bbox2dist

        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        loss_iou = weighted_eiou_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask], weight) / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.zeros(1, device=pred_dist.device, dtype=pred_dist.dtype).squeeze(0)
        return loss_iou, loss_dfl


class CustomV8DetectionLoss:
    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):
        from ultralytics.utils.loss import v8DetectionLoss

        self.base = v8DetectionLoss(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.hyp = self.base.hyp
        self.device = self.base.device
        self.bbox_loss = CustomBboxLoss(self.base.reg_max).to(self.device)
        self.base.bbox_loss = self.bbox_loss

    def __getattr__(self, item: str) -> Any:
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(item)
        return getattr(base, item)

    def __deepcopy__(self, memo: dict[int, Any]) -> "CustomV8DetectionLoss":
        copied = self.__class__.__new__(self.__class__)
        memo[id(self)] = copied
        for key, value in self.__dict__.items():
            setattr(copied, key, copy.deepcopy(value, memo))
        return copied

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        loss = torch.zeros(3, device=self.device)
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = self.base.make_anchors(preds["feats"], self.base.stride, 0.5) if hasattr(self.base, 'make_anchors') else (None, None)
        if anchor_points is None:
            from ultralytics.utils.tal import make_anchors
            anchor_points, stride_tensor = make_anchors(preds["feats"], self.base.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.base.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.base.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.base.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.base.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = sigmoid_focal_loss(pred_scores, target_scores.to(dtype))

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), loss, loss.detach()

    def parse_output(self, preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]) -> torch.Tensor:
        return self.base.parse_output(preds)

    def __call__(self, preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        preds = self.parse_output(preds)
        batch_size = preds["boxes"].shape[0]
        _, loss, loss_detach = self.get_assigned_targets_and_loss(preds, batch)
        return loss * batch_size, loss_detach
