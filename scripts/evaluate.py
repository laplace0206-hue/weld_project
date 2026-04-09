from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from weld_project.config import load_experiment_config
from weld_project.data import build_runtime_dataset_yaml
from weld_project.evaluation import summarize_ultralytics_metrics
from weld_project.training import validate_with_ultralytics
from weld_project.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估焊缝缺陷检测模型")
    parser.add_argument("--config", default="configs/experiment.yaml", help="实验配置文件")
    parser.add_argument("--weights", required=True, help="待评估权重文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    artifacts_dir = ensure_dir(PROJECT_ROOT / "artifacts")
    runtime_yaml = build_runtime_dataset_yaml(config.dataset.dataset_root, config.dataset.class_names, artifacts_dir)

    results = validate_with_ultralytics(config, runtime_yaml, args.weights)
    summary = summarize_ultralytics_metrics(results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
