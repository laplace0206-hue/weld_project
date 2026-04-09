from __future__ import annotations

from typing import Any


def summarize_ultralytics_metrics(results: Any) -> dict[str, float | None]:
    box = getattr(results, "box", None)
    if box is None:
        return {"map50": None, "map50_95": None, "precision": None, "recall": None}

    return {
        "map50": getattr(box, "map50", None),
        "map50_95": getattr(box, "map", None),
        "precision": getattr(box, "mp", None),
        "recall": getattr(box, "mr", None),
    }
