from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


@dataclass(frozen=True)
class FrameRecord:
    path: Path
    label: int | None = None
    time: float | None = None
    frame_index: int | None = None


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def list_images(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def list_videos(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root] if root.suffix.lower() in VIDEO_EXTENSIONS else []
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


def load_image_rgb(path: str | Path, crop_bottom: float = 0.0) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if crop_bottom > 0:
        width, height = image.size
        keep_height = max(1, int(round(height * (1.0 - crop_bottom))))
        image = image.crop((0, 0, width, keep_height))
    return image


def read_frame_rgb(cap: cv2.VideoCapture) -> np.ndarray | None:
    ok, frame_bgr = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def save_rgb_frame(frame_rgb: np.ndarray, path: str | Path, jpeg_quality: int = 95) -> None:
    ensure_dir(Path(path).parent)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])


def file_signature(path: str | Path) -> str:
    p = Path(path)
    stat = p.stat()
    payload = f"{p.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_hash(parts: Iterable[Any]) -> str:
    text = json.dumps(list(parts), sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(data: Any, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def choose_device(requested: str | None = None) -> str:
    if requested:
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def relative_or_name(path: Path, root: Path | None = None) -> str:
    if root is not None:
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return path.name


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default
