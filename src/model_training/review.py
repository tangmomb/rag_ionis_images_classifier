from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import ensure_dir, read_json, write_json


def _coerce_label(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and int(value) in (0, 1):
        return int(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "footage", "keep", "kept"}:
            return 0
        if normalized in {"1", "graphic", "filter", "filtered"}:
            return 1
    return None


def load_time_labels(path: str | Path, video_name: str | None = None) -> list[tuple[float, int]]:
    raw = read_json(path)
    if isinstance(raw, dict) and video_name and video_name in raw:
        raw = raw[video_name]

    labels: list[tuple[float, int]] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            label = _coerce_label(item.get("label"))
            time_value = item.get("time", item.get("timestamp"))
            if label is not None and time_value is not None:
                labels.append((float(time_value), label))
    elif isinstance(raw, dict):
        for key, value in raw.items():
            label = _coerce_label(value)
            if label is None:
                continue
            try:
                labels.append((float(key), label))
            except ValueError:
                continue
    return sorted(labels)


def nearest_label(time_sec: float, labels: list[tuple[float, int]], tolerance: float = 0.30) -> int | None:
    if not labels:
        return None
    nearest_time, nearest = min(labels, key=lambda item: abs(item[0] - time_sec))
    if abs(nearest_time - time_sec) <= tolerance:
        return nearest
    return None


def export_review_frames(
    predictions: list[dict[str, Any]],
    out_dir: str | Path,
    threshold: float = 0.5,
    top_k: int = 100,
    uncertain_margin: float = 0.10,
    labels: list[tuple[float, int]] | None = None,
    label_tolerance: float = 0.30,
) -> dict[str, Any]:
    out_root = ensure_dir(out_dir)
    labels = labels or []
    selected: list[tuple[str, dict[str, Any]]] = []

    ranked = sorted(
        predictions,
        key=lambda item: abs(float(item.get("prob_graphic", item.get("prob_filter", 0.0))) - threshold),
    )
    for item in ranked:
        prob = float(item.get("prob_graphic", item.get("prob_filter", 0.0)))
        if abs(prob - threshold) <= uncertain_margin or len(selected) < top_k:
            selected.append(("uncertain", item))
        if len([entry for entry in selected if entry[0] == "uncertain"]) >= top_k:
            break

    false_positive = 0
    false_negative = 0
    if labels:
        for item in predictions:
            time_sec = float(item.get("time", 0.0))
            label = nearest_label(time_sec, labels, tolerance=label_tolerance)
            if label is None:
                continue
            pred = int(item.get("smoothed_pred", item.get("pred", 0)))
            if pred == 1 and label == 0:
                selected.append(("false_positive", item))
                false_positive += 1
            elif pred == 0 and label == 1:
                selected.append(("false_negative", item))
                false_negative += 1

    manifest: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for category, item in selected:
        frame_path = item.get("frame_path")
        if not frame_path:
            continue
        source = Path(frame_path)
        if not source.exists():
            continue
        key = (category, str(source.resolve()))
        if key in seen:
            continue
        seen.add(key)
        category_dir = ensure_dir(out_root / category)
        prob = float(item.get("prob_graphic", item.get("prob_filter", 0.0)))
        time_sec = float(item.get("time", 0.0))
        target = category_dir / f"t{time_sec:010.3f}_p{prob:.3f}_{source.name}"
        shutil.copy2(source, target)
        row = dict(item)
        row["review_category"] = category
        row["review_path"] = str(target.resolve())
        manifest.append(row)

    summary = {
        "out_dir": str(out_root.resolve()),
        "count": len(manifest),
        "false_positive": false_positive,
        "false_negative": false_negative,
        "items": manifest,
    }
    write_json(summary, out_root / "manifest.json")
    return summary
