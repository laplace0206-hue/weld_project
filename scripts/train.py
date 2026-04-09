from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.weld_project.config import load_experiment_config
from src.weld_project.data import build_runtime_dataset_yaml
from src.weld_project.training.pipeline import build_research_model, summarize_model, train_with_ultralytics
from src.weld_project.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练焊缝缺陷检测模型")
    parser.add_argument("--config", default="configs/experiment.yaml", help="实验配置文件")
    parser.add_argument(
        "--stage",
        default="baseline",
        choices=["baseline", "preprocess", "neck", "attention", "full"],
        help="消融实验阶段",
    )
    parser.add_argument("--dry-run", action="store_true", help="只构建配置与研究模型，不实际启动训练")
    parser.add_argument("--loss-type", default=None, help="覆盖当前阶段使用的损失类型，如 ciou 或 focal_eiou")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    artifacts_dir = ensure_dir(PROJECT_ROOT / "artifacts")
    runtime_yaml = build_runtime_dataset_yaml(config.dataset.dataset_root, config.dataset.class_names, artifacts_dir)

    research_model = build_research_model(config, args.stage)
    summary = summarize_model(research_model)
    effective_loss = args.loss_type or config.ablation[args.stage].get("loss", config.model.loss_type)
    print(f"[Research Model] stage={args.stage}, params={summary['total_params']:,}")
    print(f"[Dataset YAML] {runtime_yaml}")
    print(f"[Loss Type] {effective_loss}")

    if args.dry_run:
        print("Dry run 完成，未启动 Ultralytics 训练。")
        return

    if args.loss_type:
        config.ablation[args.stage]["loss"] = args.loss_type

    results = train_with_ultralytics(config, runtime_yaml, args.stage)
    print("训练完成。")
    print(results)


if __name__ == "__main__":
    main()
