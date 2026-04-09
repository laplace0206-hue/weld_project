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
from weld_project.data import analyze_dataset
from weld_project.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析焊缝缺陷数据集")
    parser.add_argument("--config", default="configs/experiment.yaml", help="实验配置文件")
    parser.add_argument("--dataset", default=None, help="数据集根目录，可覆盖配置")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    dataset_root = Path(args.dataset).resolve() if args.dataset else config.dataset.dataset_root
    report = analyze_dataset(dataset_root, config.dataset.class_names)

    artifacts_dir = ensure_dir(PROJECT_ROOT / "artifacts")
    report_path = artifacts_dir / "dataset_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n数据集分析结果已保存: {report_path}")


if __name__ == "__main__":
    main()
