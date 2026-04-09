from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..config import ExperimentConfig
from .. import register_legacy_import_aliases
from ..integrations import register_custom_ultralytics_modules
from ..models import ImprovedWeldDetector
from ..utils.io import ensure_dir, save_yaml


def _stage_flags(config: ExperimentConfig, stage: str) -> dict[str, Any]:
    if stage not in config.ablation:
        raise ValueError(f"未知实验阶段: {stage}")
    return config.ablation[stage]


def write_stage_recipe(config: ExperimentConfig, stage: str) -> Path:
    output_dir = ensure_dir(config.output_dir / "recipes")
    recipe_path = output_dir / f"{stage}.yaml"
    flags = _stage_flags(config, stage)
    save_yaml(
        recipe_path,
        {
            "stage": stage,
            "baseline": config.model.baseline_name,
            "image_size": config.model.image_size,
            "modules": {
                "adaptive_preprocess": bool(flags["preprocess"]),
                "enhanced_fpn": bool(flags["neck"]),
                "attention": bool(flags["attention"]),
                "loss": flags["loss"],
                "attention_type": config.model.attention_type,
                "neck_type": config.model.neck_type,
            },
        },
    )
    return recipe_path


def build_research_model(config: ExperimentConfig, stage: str) -> ImprovedWeldDetector:
    flags = _stage_flags(config, stage)
    model = ImprovedWeldDetector(
        num_classes=config.dataset.num_classes,
        use_preprocess=bool(flags["preprocess"]),
        use_attention=bool(flags["attention"]),
        attention_type=config.model.attention_type,
    )
    return model


def train_with_ultralytics(config: ExperimentConfig, runtime_dataset_yaml: str | Path, stage: str) -> Any:
    from ultralytics import YOLO

    output_dir = ensure_dir(config.output_dir)
    recipe_path = write_stage_recipe(config, stage)
    flags = _stage_flags(config, stage)
    loss_type = str(flags.get("loss", config.model.loss_type)).lower()

    register_legacy_import_aliases()
    register_custom_ultralytics_modules(loss_type)

    model = YOLO(config.model.baseline_name)

    return model.train(
        data=str(runtime_dataset_yaml),
        epochs=config.train.epochs,
        imgsz=config.model.image_size,
        batch=config.train.batch,
        workers=config.train.workers,
        device=config.train.device,
        optimizer=config.train.optimizer,
        lr0=config.train.lr0,
        patience=config.train.patience,
        pretrained=config.train.pretrained,
        cache=config.train.cache,
        amp=config.train.amp,
        close_mosaic=config.train.close_mosaic,
        project=str(output_dir),
        name=stage,
        exist_ok=True,
        plots=True,
        verbose=True,
    )


def validate_with_ultralytics(config: ExperimentConfig, runtime_dataset_yaml: str | Path, weights: str | Path) -> Any:
    from ultralytics import YOLO

    register_legacy_import_aliases()
    register_custom_ultralytics_modules(config.model.loss_type)

    model = YOLO(str(weights))
    return model.val(
        data=str(runtime_dataset_yaml),
        split=config.eval.split,
        conf=config.eval.conf,
        iou=config.eval.iou,
        imgsz=config.model.image_size,
        project=str(config.output_dir),
        name="val",
        exist_ok=True,
        plots=True,
    )


def export_with_ultralytics(config: ExperimentConfig, weights: str | Path) -> Any:
    from ultralytics import YOLO

    register_legacy_import_aliases()
    register_custom_ultralytics_modules(config.model.loss_type)

    model = YOLO(str(weights))
    return model.export(
        format=config.export.format,
        imgsz=config.export.imgsz,
        dynamic=config.export.dynamic,
    )


def summarize_model(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total_params": total, "trainable_params": trainable}
