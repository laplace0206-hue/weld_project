from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2

from .. import register_legacy_import_aliases
from ..data.transforms import load_bgr_image, traditional_preprocess
from ..integrations import register_custom_ultralytics_modules
from ..utils.io import ensure_dir
from .reporting import analyze_detection_result, save_detection_outputs

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def resolve_default_weights(project_root: str | Path) -> Path | None:
    root = Path(project_root)
    best_candidates = sorted(root.glob("runs/**/weights/best.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if best_candidates:
        return best_candidates[0]

    fallback = root.parent / "yolo26n.pt"
    if fallback.exists():
        return fallback
    return None


def run_image_inference(
    weights_path: str | Path,
    source_path: str | Path,
    output_dir: str | Path,
    imgsz: int = 640,
    conf: float = 0.25,
    device: str = "0",
    preprocess: bool = False,
) -> dict[str, Any]:
    weights = Path(weights_path).resolve()
    source = Path(source_path).resolve()
    destination = ensure_dir(output_dir)

    if not weights.exists():
        raise FileNotFoundError(f"权重文件不存在: {weights}")
    if not source.exists():
        raise FileNotFoundError(f"输入图片不存在: {source}")

    inference_source = source
    if preprocess:
        pre_dir = ensure_dir(destination / "preprocessed")
        image = load_bgr_image(source)
        enhanced = traditional_preprocess(image)
        inference_source = pre_dir / source.name
        cv2.imwrite(str(inference_source), enhanced)

    register_legacy_import_aliases()
    register_custom_ultralytics_modules()

    from ultralytics import YOLO

    model = YOLO(str(weights))
    results = model.predict(
        source=str(inference_source),
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=False,
        save=False,
    )
    if not results:
        raise RuntimeError("模型未返回任何推理结果。")

    result = results[0]
    summary = analyze_detection_result(result, getattr(result, "names", {}))
    paths = save_detection_outputs(result, summary, destination)

    return {
        "summary": summary,
        "output_paths": paths,
        "weights_path": weights,
        "source_path": source,
        "used_source_path": inference_source,
    }


def run_batch_inference(
    weights_path: str | Path,
    source_dir: str | Path,
    output_dir: str | Path,
    imgsz: int = 640,
    conf: float = 0.25,
    device: str = "0",
    preprocess: bool = False,
) -> dict[str, Any]:
    folder = Path(source_dir).resolve()
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"输入文件夹不存在: {folder}")

    images = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"文件夹中未找到可识别图片: {folder}")

    destination = ensure_dir(output_dir)
    image_results: list[dict[str, Any]] = []
    class_counter: dict[str, int] = {}
    images_with_defects = 0

    for image_path in images:
        result = run_image_inference(
            weights_path=weights_path,
            source_path=image_path,
            output_dir=destination,
            imgsz=imgsz,
            conf=conf,
            device=device,
            preprocess=preprocess,
        )
        summary = result["summary"]
        if summary["total_detections"] > 0:
            images_with_defects += 1
        for item in summary["class_summary"]:
            class_counter[item["class_name"]] = class_counter.get(item["class_name"], 0) + int(item["count"])

        image_results.append(
            {
                "image_name": summary["image_name"],
                "total_detections": summary["total_detections"],
                "risk_level": summary["risk_level"],
                "dominant_class": summary.get("dominant_class") or "None",
                "annotated_image": str(result["output_paths"]["annotated_image"]),
                "text_report": str(result["output_paths"]["text_report"]),
                "json_report": str(result["output_paths"]["json_report"]),
            }
        )

    summary = {
        "source_dir": str(folder),
        "total_images": len(images),
        "images_with_defects": images_with_defects,
        "images_without_defects": len(images) - images_with_defects,
        "class_counter": class_counter,
        "image_results": image_results,
    }

    csv_path = destination / "batch_summary.csv"
    json_path = destination / "batch_summary.json"
    txt_path = destination / "batch_summary.txt"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image_name", "total_detections", "risk_level", "dominant_class", "annotated_image", "text_report", "json_report"],
        )
        writer.writeheader()
        writer.writerows(image_results)

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"输入文件夹: {folder}",
        f"图片总数: {summary['total_images']}",
        f"检测到缺陷的图片数: {summary['images_with_defects']}",
        f"未检测到缺陷的图片数: {summary['images_without_defects']}",
        "",
        "类别总计:",
    ]
    if class_counter:
        for class_name, count in sorted(class_counter.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"- {class_name}: {count}")
    else:
        lines.append("- 未检测到任何缺陷")
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "summary": summary,
        "output_paths": {
            "csv_report": csv_path,
            "json_report": json_path,
            "text_report": txt_path,
        },
    }
