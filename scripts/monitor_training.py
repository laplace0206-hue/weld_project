from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.weld_project.utils.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实时查看训练进度")
    parser.add_argument(
        "--run-dir",
        default="runs/weld_defect_research/formal_custom_yolo_50e",
        help="训练输出目录，里面应包含 args.yaml 和 results.csv",
    )
    parser.add_argument("--interval", type=int, default=5, help="刷新间隔，单位秒")
    parser.add_argument("--once", action="store_true", help="只输出一次当前进度")
    return parser.parse_args()


def _format_seconds(seconds: float | int) -> str:
    total = int(max(seconds, 0))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _load_progress(run_dir: Path) -> dict[str, object]:
    args_path = run_dir / "args.yaml"
    results_path = run_dir / "results.csv"
    if not args_path.exists():
        raise FileNotFoundError(f"未找到 args.yaml: {args_path}")
    if not results_path.exists():
        raise FileNotFoundError(f"未找到 results.csv: {results_path}")

    args = load_yaml(args_path)
    total_epochs = int(args.get("epochs", 0))

    with results_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        return {
            "total_epochs": total_epochs,
            "finished_epochs": 0,
            "progress_percent": 0.0,
            "elapsed": 0.0,
            "eta": None,
            "best_map50": None,
            "best_map5095": None,
            "latest": None,
        }

    latest = rows[-1]
    finished_epochs = int(float(latest["epoch"]))
    elapsed = float(latest["time"])
    avg_epoch_time = elapsed / max(finished_epochs, 1)
    remaining_epochs = max(total_epochs - finished_epochs, 0)
    eta = avg_epoch_time * remaining_epochs
    best_map50 = max(float(row["metrics/mAP50(B)"]) for row in rows)
    best_map5095 = max(float(row["metrics/mAP50-95(B)"]) for row in rows)

    return {
        "total_epochs": total_epochs,
        "finished_epochs": finished_epochs,
        "progress_percent": (finished_epochs / total_epochs * 100.0) if total_epochs else 0.0,
        "elapsed": elapsed,
        "eta": eta,
        "best_map50": best_map50,
        "best_map5095": best_map5095,
        "latest": latest,
    }


def _render(progress: dict[str, object], run_dir: Path) -> str:
    latest = progress["latest"]
    lines = [
        f"运行目录: {run_dir}",
        f"进度: {progress['finished_epochs']}/{progress['total_epochs']} ({progress['progress_percent']:.1f}%)",
        f"已用时间: {_format_seconds(progress['elapsed'])}",
        f"预计剩余: {_format_seconds(progress['eta']) if progress['eta'] is not None else '--:--:--'}",
        f"最佳 mAP50: {progress['best_map50']:.4f}" if progress["best_map50"] is not None else "最佳 mAP50: --",
        f"最佳 mAP50-95: {progress['best_map5095']:.4f}" if progress["best_map5095"] is not None else "最佳 mAP50-95: --",
    ]

    if latest is not None:
        lines.extend(
            [
                f"当前 epoch: {int(float(latest['epoch']))}",
                f"最新 precision: {float(latest['metrics/precision(B)']):.4f}",
                f"最新 recall: {float(latest['metrics/recall(B)']):.4f}",
                f"最新 mAP50: {float(latest['metrics/mAP50(B)']):.4f}",
                f"最新 mAP50-95: {float(latest['metrics/mAP50-95(B)']):.4f}",
                f"train box/cls/dfl: {float(latest['train/box_loss']):.4f} / {float(latest['train/cls_loss']):.4f} / {float(latest['train/dfl_loss']):.4f}",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (PROJECT_ROOT / run_dir).resolve()

    while True:
        progress = _load_progress(run_dir)
        if not args.once:
            os.system("cls" if os.name == "nt" else "clear")
        print(_render(progress, run_dir))
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
