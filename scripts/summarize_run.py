from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总单个或多个训练 run 的最佳指标")
    parser.add_argument("runs", nargs="+", help="一个或多个 run 目录路径")
    parser.add_argument("--sort-by", default="metrics/mAP50-95(B)", help="按哪一列排序输出")
    return parser.parse_args()


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _to_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")


def summarize_run(run_dir: Path) -> dict[str, object]:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到 results.csv: {csv_path}")

    rows = _read_rows(csv_path)
    if not rows:
        raise ValueError(f"results.csv 为空: {csv_path}")

    best_row = max(rows, key=lambda row: _to_float(row, "metrics/mAP50-95(B)"))
    last_row = rows[-1]

    return {
        "run_dir": str(run_dir.resolve()),
        "epochs_recorded": len(rows),
        "best_epoch": int(float(best_row["epoch"])),
        "best_map50": _to_float(best_row, "metrics/mAP50(B)"),
        "best_map50_95": _to_float(best_row, "metrics/mAP50-95(B)"),
        "best_precision": _to_float(best_row, "metrics/precision(B)"),
        "best_recall": _to_float(best_row, "metrics/recall(B)"),
        "last_epoch": int(float(last_row["epoch"])),
        "last_map50": _to_float(last_row, "metrics/mAP50(B)"),
        "last_map50_95": _to_float(last_row, "metrics/mAP50-95(B)"),
    }


def _sort_value(summary: dict[str, object], sort_by: str) -> float:
    mapping = {
        "metrics/mAP50-95(B)": "best_map50_95",
        "metrics/mAP50(B)": "best_map50",
        "metrics/precision(B)": "best_precision",
        "metrics/recall(B)": "best_recall",
        "best_map50_95": "best_map50_95",
        "best_map50": "best_map50",
        "best_precision": "best_precision",
        "best_recall": "best_recall",
    }
    key = mapping.get(sort_by, "best_map50_95")
    return float(summary.get(key, 0.0))


def main() -> None:
    args = parse_args()
    summaries = [summarize_run(Path(path)) for path in args.runs]
    summaries.sort(key=lambda item: _sort_value(item, args.sort_by), reverse=True)

    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()