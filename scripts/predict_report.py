from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from src.weld_project.inference import format_text_report, resolve_default_weights, run_image_inference
from src.weld_project.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对焊缝照片进行缺陷识别并输出图片与文字分析")
    parser.add_argument("--weights", default=None, help="模型权重路径，如 best.pt 或 yolo26n.pt")
    parser.add_argument("--source", required=True, help="输入图片路径")
    parser.add_argument("--output-dir", default="outputs/inference", help="输出目录")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--device", default="0", help="推理设备，如 0 或 cpu")
    parser.add_argument("--preprocess", action="store_true", help="推理前使用传统机器视觉预处理")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights).expanduser() if args.weights else resolve_default_weights(PROJECT_ROOT)
    source_path = Path(args.source)
    output_dir = ensure_dir(PROJECT_ROOT / args.output_dir)

    if weights_path is None:
        raise FileNotFoundError("未找到可用权重。请显式传入 --weights，或先完成训练生成 best.pt。")
    if not weights_path.is_absolute():
        weights_path = weights_path.resolve()
    if not source_path.is_absolute():
        source_path = source_path.resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"输入图片不存在: {source_path}")

    result = run_image_inference(
        weights_path=weights_path,
        source_path=source_path,
        output_dir=output_dir,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        preprocess=args.preprocess,
    )
    summary = result["summary"]
    output_paths = result["output_paths"]

    print(format_text_report(summary))
    print("")
    print(f"标注图片: {output_paths['annotated_image']}")
    print(f"文字报告: {output_paths['text_report']}")
    print(f"JSON报告: {output_paths['json_report']}")


if __name__ == "__main__":
    main()
