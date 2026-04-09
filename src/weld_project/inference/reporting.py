from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SEVERITY_ORDER = {
    "Crack": 4,
    "Porosity": 3,
    "Spatters": 2,
    "Welding line": 1,
}

SEVERITY_TEXT = {
    4: "高风险",
    3: "中高风险",
    2: "中风险",
    1: "低风险",
    0: "未见明显缺陷",
}

RECOMMENDATIONS = {
    "Crack": "检测到裂纹，建议优先复检并进行返修，避免继续服役。",
    "Porosity": "检测到气孔，建议复核焊接工艺参数与保护气体稳定性。",
    "Spatters": "检测到飞溅，建议清理焊缝表面并检查送丝与电流设置。",
    "Welding line": "已识别焊缝主体，可结合其他缺陷结果综合判断。",
}


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def analyze_detection_result(result: Any, class_names: list[str] | dict[int, str]) -> dict[str, Any]:
    names = class_names if isinstance(class_names, dict) else {index: name for index, name in enumerate(class_names)}
    original_path = Path(getattr(result, "path", "result.jpg"))
    orig_img = getattr(result, "orig_img", None)
    if orig_img is None:
        raise ValueError("推理结果中缺少原始图像。")
    image_h, image_w = orig_img.shape[:2]
    image_area = float(max(image_h * image_w, 1))

    detections: list[dict[str, Any]] = []
    class_counter: Counter[str] = Counter()
    class_confidences: dict[str, list[float]] = defaultdict(list)
    class_areas: dict[str, list[float]] = defaultdict(list)

    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        for box, conf, cls_idx in zip(xyxy, confs, classes):
            x1, y1, x2, y2 = [float(x) for x in box.tolist()]
            class_name = names.get(int(cls_idx), f"class_{cls_idx}")
            width = max(x2 - x1, 0.0)
            height = max(y2 - y1, 0.0)
            area_ratio = (width * height) / image_area
            detections.append(
                {
                    "class_id": int(cls_idx),
                    "class_name": class_name,
                    "confidence": round(_safe_float(conf), 4),
                    "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "area_ratio": round(area_ratio, 6),
                }
            )
            class_counter[class_name] += 1
            class_confidences[class_name].append(_safe_float(conf))
            class_areas[class_name].append(area_ratio)

    dominant_class = max(class_counter, key=class_counter.get) if class_counter else None
    max_severity = max((SEVERITY_ORDER.get(item["class_name"], 0) for item in detections), default=0)
    top_detection = max(detections, key=lambda item: item["confidence"], default=None)

    class_summary: list[dict[str, Any]] = []
    for class_name, count in class_counter.most_common():
        class_summary.append(
            {
                "class_name": class_name,
                "count": count,
                "max_confidence": round(max(class_confidences[class_name]), 4),
                "mean_confidence": round(sum(class_confidences[class_name]) / len(class_confidences[class_name]), 4),
                "mean_area_ratio": round(sum(class_areas[class_name]) / len(class_areas[class_name]), 6),
            }
        )

    recommendations = []
    for class_name, _ in class_counter.most_common():
        text = RECOMMENDATIONS.get(class_name)
        if text and text not in recommendations:
            recommendations.append(text)

    if not recommendations:
        recommendations.append("未检测到明确缺陷，可作为正常样本记录，但建议继续结合人工复核。")

    return {
        "image_name": original_path.name,
        "image_path": str(original_path),
        "image_size": {"width": image_w, "height": image_h},
        "total_detections": len(detections),
        "risk_level": SEVERITY_TEXT.get(max_severity, "未知"),
        "dominant_class": dominant_class,
        "top_detection": top_detection,
        "class_summary": class_summary,
        "detections": detections,
        "recommendations": recommendations,
    }


def format_text_report(summary: dict[str, Any]) -> str:
    lines = [
        f"图像: {summary['image_name']}",
        f"总检测框数量: {summary['total_detections']}",
        f"综合风险等级: {summary['risk_level']}",
    ]

    dominant_class = summary.get("dominant_class")
    if dominant_class:
        lines.append(f"主要缺陷类型: {dominant_class}")

    top_detection = summary.get("top_detection")
    if top_detection:
        lines.append(
            "最高置信度目标: "
            f"{top_detection['class_name']} (conf={top_detection['confidence']:.4f}, bbox={top_detection['bbox_xyxy']})"
        )

    lines.append("")
    lines.append("各类别统计:")
    if summary["class_summary"]:
        for item in summary["class_summary"]:
            lines.append(
                f"- {item['class_name']}: 数量={item['count']}, "
                f"最大置信度={item['max_confidence']:.4f}, 平均置信度={item['mean_confidence']:.4f}, "
                f"平均面积占比={item['mean_area_ratio']:.6f}"
            )
    else:
        lines.append("- 未检测到缺陷目标")

    lines.append("")
    lines.append("建议:")
    for item in summary["recommendations"]:
        lines.append(f"- {item}")

    return "\n".join(lines)


def save_detection_outputs(result: Any, summary: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    stem = Path(summary["image_name"]).stem
    annotated_path = destination / f"{stem}_pred.jpg"
    json_path = destination / f"{stem}_report.json"
    text_path = destination / f"{stem}_report.txt"

    plotted = result.plot()
    if isinstance(plotted, np.ndarray):
        cv2.imwrite(str(annotated_path), plotted)

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    text_path.write_text(format_text_report(summary), encoding="utf-8")

    return {
        "annotated_image": annotated_path,
        "json_report": json_path,
        "text_report": text_path,
    }
