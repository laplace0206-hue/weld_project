from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.weld_project import register_legacy_import_aliases
from src.weld_project.config import load_experiment_config
from src.weld_project.integrations import register_custom_ultralytics_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证当前训练链路实际使用的损失函数")
    parser.add_argument("--config", default="configs/experiment.yaml", help="实验配置文件")
    parser.add_argument("--stage", default="full", help="消融阶段名称，默认 full")
    parser.add_argument("--model-config", default="configs/models/yolov8_weld_custom.yaml", help="自定义模型 YAML；为空则使用配置中的 baseline_name")
    parser.add_argument("--weights", default=None, help="已有权重路径；传入后优先加载权重")
    parser.add_argument("--loss-type", default=None, help="覆盖损失类型，如 ciou 或 focal_eiou")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    stage_flags = config.ablation.get(args.stage, {})
    loss_type = (args.loss_type or stage_flags.get("loss") or config.model.loss_type).lower()

    register_legacy_import_aliases()
    register_custom_ultralytics_modules(loss_type)

    from ultralytics import YOLO

    if args.weights:
        model_source = str((PROJECT_ROOT / args.weights).resolve()) if not Path(args.weights).is_absolute() else args.weights
    elif args.model_config:
        model_cfg = Path(args.model_config)
        model_source = str((PROJECT_ROOT / model_cfg).resolve()) if not model_cfg.is_absolute() else str(model_cfg)
    else:
        model_source = config.model.baseline_name

    model = YOLO(model_source)
    criterion = model.model.init_criterion()
    bbox_loss = getattr(criterion, "bbox_loss", None)

    print(f"[Model Source] {model_source}")
    print(f"[Loss Type] {loss_type}")
    print(f"[Criterion Class] {criterion.__class__.__name__}")
    print(f"[BBox Loss Class] {bbox_loss.__class__.__name__ if bbox_loss is not None else 'None'}")


if __name__ == "__main__":
    main()