from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.weld_project.config import load_experiment_config
from src.weld_project.data import build_runtime_dataset_yaml
from src.weld_project.integrations import register_custom_ultralytics_modules
from src.weld_project.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练自定义焊缝缺陷 YOLOv8 模型")
    parser.add_argument("--config", default="configs/experiment.yaml", help="实验配置文件")
    parser.add_argument(
        "--model-config",
        default="configs/models/yolov8_weld_custom.yaml",
        help="自定义 YOLO 模型 YAML",
    )
    parser.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数")
    parser.add_argument("--batch", type=int, default=None, help="覆盖 batch 大小")
    parser.add_argument("--imgsz", type=int, default=None, help="覆盖输入图像尺寸")
    parser.add_argument("--workers", type=int, default=None, help="覆盖数据加载线程数")
    parser.add_argument("--device", default=None, help="覆盖训练设备，如 0、cpu")
    parser.add_argument("--name", default="custom_yolo", help="训练运行名称")
    parser.add_argument("--loss-type", default=None, help="覆盖损失类型，如 ciou 或 focal_eiou")
    parser.add_argument("--fraction", type=float, default=1.0, help="训练数据使用比例，便于快速冒烟验证")
    parser.add_argument("--skip-val", action="store_true", help="跳过每个 epoch 后的验证，便于快速冒烟")
    parser.add_argument("--dry-run", action="store_true", help="只构建模型与数据配置，不实际训练")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    model_config_path = Path(args.model_config)
    if not model_config_path.is_absolute():
        model_config_path = (PROJECT_ROOT / model_config_path).resolve()
    artifacts_dir = ensure_dir(PROJECT_ROOT / "artifacts")
    runtime_yaml = build_runtime_dataset_yaml(config.dataset.dataset_root, config.dataset.class_names, artifacts_dir)

    loss_type = (args.loss_type or config.model.loss_type).lower()

    register_custom_ultralytics_modules(loss_type)

    from ultralytics import YOLO

    model = YOLO(str(model_config_path))

    epochs = args.epochs if args.epochs is not None else config.train.epochs
    batch = args.batch if args.batch is not None else config.train.batch
    imgsz = args.imgsz if args.imgsz is not None else config.model.image_size
    workers = args.workers if args.workers is not None else config.train.workers
    device = args.device if args.device is not None else config.train.device

    print(f"[Custom Model YAML] {model_config_path}")
    print(f"[Dataset YAML] {runtime_yaml}")
    print(f"[Loss Type] {loss_type}")
    print(
        f"[Train Args] epochs={epochs}, batch={batch}, imgsz={imgsz}, workers={workers}, "
        f"device={device}, fraction={args.fraction}, val={not args.skip_val}"
    )

    if args.dry_run:
        print("Custom YOLO dry-run 完成，未启动训练。")
        return

    results = model.train(
        data=str(runtime_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device,
        optimizer=config.train.optimizer,
        lr0=config.train.lr0,
        patience=config.train.patience,
        pretrained=False,
        cache=config.train.cache,
        amp=config.train.amp,
        fraction=args.fraction,
        val=not args.skip_val,
        close_mosaic=config.train.close_mosaic,
        project=str(config.output_dir),
        name=args.name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    print(results)


if __name__ == "__main__":
    main()
