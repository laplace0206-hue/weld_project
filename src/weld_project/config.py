from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils.io import load_yaml


@dataclass
class DatasetConfig:
    config_path: Path
    dataset_root: Path
    source_yaml: Path
    train_images: Path
    val_images: Path
    test_images: Path
    class_names: list[str]
    num_classes: int


@dataclass
class ModelConfig:
    baseline_name: str
    image_size: int
    use_adaptive_preprocess: bool
    use_enhanced_fpn: bool
    use_attention: bool
    loss_type: str
    attention_type: str
    neck_type: str


@dataclass
class TrainConfig:
    epochs: int
    batch: int
    workers: int
    device: str
    optimizer: str
    lr0: float
    patience: int
    pretrained: bool
    cache: bool
    amp: bool
    close_mosaic: int


@dataclass
class EvalConfig:
    split: str
    conf: float
    iou: float


@dataclass
class ExportConfig:
    format: str
    imgsz: int
    dynamic: bool


@dataclass
class ExperimentConfig:
    project_name: str
    output_dir: Path
    dataset: DatasetConfig
    model: ModelConfig
    train: TrainConfig
    eval: EvalConfig
    export: ExportConfig
    ablation: dict[str, dict[str, Any]]
    project_root: Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / value).resolve()


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    project_root = _project_root()
    config_file = _resolve_path(project_root, str(config_path))
    raw = load_yaml(config_file)

    dataset_cfg_path = _resolve_path(project_root, raw["dataset"]["config_path"])
    dataset_raw = load_yaml(dataset_cfg_path)

    dataset = DatasetConfig(
        config_path=dataset_cfg_path,
        dataset_root=_resolve_path(project_root, dataset_raw["dataset_root"]),
        source_yaml=_resolve_path(project_root, dataset_raw["source_yaml"]),
        train_images=_resolve_path(project_root, dataset_raw["train_images"]),
        val_images=_resolve_path(project_root, dataset_raw["val_images"]),
        test_images=_resolve_path(project_root, dataset_raw["test_images"]),
        class_names=list(dataset_raw["class_names"]),
        num_classes=int(dataset_raw["num_classes"]),
    )

    model = ModelConfig(**raw["model"])
    train = TrainConfig(**raw["train"])
    eval_config = EvalConfig(**raw["eval"])
    export = ExportConfig(**raw["export"])

    project = raw["project"]
    return ExperimentConfig(
        project_name=project["name"],
        output_dir=_resolve_path(project_root, project["output_dir"]),
        dataset=dataset,
        model=model,
        train=train,
        eval=eval_config,
        export=export,
        ablation=dict(raw["ablation"]),
        project_root=project_root,
    )
