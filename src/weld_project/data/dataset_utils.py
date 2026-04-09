from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import shutil
from typing import Any

from ..utils.io import ensure_dir, save_yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _image_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def _label_file_for_image(image_path: Path) -> Path:
    labels_dir = image_path.parent.parent / "labels"
    return labels_dir / f"{image_path.stem}.txt"


def _polygon_to_box(points: list[float]) -> tuple[float, float, float, float]:
    xs = points[0::2]
    ys = points[1::2]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y
    x_center = min_x + width / 2
    y_center = min_y + height / 2
    return x_center, y_center, width, height


def _parse_label_line(line: str) -> dict[str, Any]:
    values = line.split()
    class_id = int(values[0])
    coordinates = [float(value) for value in values[1:]]
    if len(coordinates) == 4:
        x_center, y_center, width, height = coordinates
        return {
            "class_id": class_id,
            "label_type": "bbox",
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
        }
    if len(coordinates) >= 6 and len(coordinates) % 2 == 0:
        x_center, y_center, width, height = _polygon_to_box(coordinates)
        return {
            "class_id": class_id,
            "label_type": "polygon",
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
            "points": coordinates,
        }
    raise ValueError(f"无法解析标签行: {line}")


def _parse_label_file(label_path: Path) -> list[dict[str, Any]]:
    if not label_path.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(_parse_label_line(line))
    return rows


def _infer_label_mode(images_dir: Path) -> str:
    for image_path in _image_files(images_dir):
        label_path = _label_file_for_image(image_path)
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parsed = _parse_label_line(line)
            return parsed["label_type"]
    return "unknown"


def _safe_link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _split_stats(images_dir: Path, class_names: list[str]) -> dict[str, Any]:
    images = _image_files(images_dir)
    class_counter: Counter[str] = Counter()
    box_areas: list[float] = []
    missing_labels = 0
    polygon_instances = 0
    bbox_instances = 0

    for image_path in images:
        label_path = _label_file_for_image(image_path)
        labels = _parse_label_file(label_path)
        if not label_path.exists():
            missing_labels += 1
        for label in labels:
            class_id = int(label["class_id"])
            width = float(label["width"])
            height = float(label["height"])
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            class_counter[class_name] += 1
            box_areas.append(width * height)
            if label["label_type"] == "polygon":
                polygon_instances += 1
            else:
                bbox_instances += 1

    total_boxes = sum(class_counter.values())
    mean_area = sum(box_areas) / len(box_areas) if box_areas else 0.0
    tiny_boxes = sum(area < 0.01 for area in box_areas)

    return {
        "images": len(images),
        "boxes": total_boxes,
        "missing_labels": missing_labels,
        "label_mode": _infer_label_mode(images_dir),
        "polygon_instances": polygon_instances,
        "bbox_instances": bbox_instances,
        "class_distribution": dict(class_counter),
        "mean_normalized_box_area": round(mean_area, 6),
        "tiny_box_ratio": round(tiny_boxes / len(box_areas), 6) if box_areas else 0.0,
    }


def analyze_dataset(dataset_root: str | Path, class_names: list[str]) -> dict[str, Any]:
    root = Path(dataset_root)
    report = {
        "dataset_root": str(root.resolve()),
        "classes": class_names,
        "train": _split_stats(root / "train" / "images", class_names),
        "valid": _split_stats(root / "valid" / "images", class_names),
        "test": _split_stats(root / "test" / "images", class_names),
    }
    return report


def prepare_detection_dataset(dataset_root: str | Path, output_dir: str | Path) -> Path:
    root = Path(dataset_root).resolve()
    prepared_root = ensure_dir(Path(output_dir) / "prepared_detection_dataset")

    for split in ("train", "valid", "test"):
        source_images = root / split / "images"
        source_labels = root / split / "labels"
        target_images = ensure_dir(prepared_root / split / "images")
        target_labels = ensure_dir(prepared_root / split / "labels")

        for image_path in _image_files(source_images):
            _safe_link_or_copy(image_path, target_images / image_path.name)
            label_path = source_labels / f"{image_path.stem}.txt"
            target_label_path = target_labels / f"{image_path.stem}.txt"
            if not label_path.exists():
                target_label_path.write_text("", encoding="utf-8")
                continue

            converted_lines: list[str] = []
            for label in _parse_label_file(label_path):
                converted_lines.append(
                    f"{label['class_id']} {label['x_center']:.6f} {label['y_center']:.6f} {label['width']:.6f} {label['height']:.6f}"
                )
            target_label_path.write_text("\n".join(converted_lines), encoding="utf-8")

    return prepared_root


def build_runtime_dataset_yaml(dataset_root: str | Path, class_names: list[str], output_dir: str | Path) -> Path:
    root = Path(dataset_root).resolve()
    artifacts = ensure_dir(output_dir)
    sample_mode = _infer_label_mode(root / "train" / "images")
    runtime_root = prepare_detection_dataset(root, artifacts) if sample_mode == "polygon" else root
    runtime_yaml = artifacts / "dataset_runtime.yaml"
    save_yaml(
        runtime_yaml,
        {
            "path": str(runtime_root),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": len(class_names),
            "names": class_names,
        },
    )
    return runtime_yaml
