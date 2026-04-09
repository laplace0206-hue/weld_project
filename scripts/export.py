from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from weld_project.config import load_experiment_config
from weld_project.training import export_with_ultralytics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出焊缝缺陷检测模型")
    parser.add_argument("--config", default="configs/experiment.yaml", help="实验配置文件")
    parser.add_argument("--weights", required=True, help="待导出的权重文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    result = export_with_ultralytics(config, args.weights)
    print(f"导出完成: {result}")


if __name__ == "__main__":
    main()
