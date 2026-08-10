from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def list_images(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def load_image_rgb(path: str | Path, crop_bottom: float = 0.0) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if crop_bottom > 0:
        width, height = image.size
        keep_height = max(1, int(round(height * (1.0 - crop_bottom))))
        image = image.crop((0, 0, width, keep_height))
    return image


def file_signature(path: str | Path) -> str:
    p = Path(path)
    stat = p.stat()
    payload = f"{p.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_hash(parts: Iterable[Any]) -> str:
    text = json.dumps(list(parts), sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def choose_device(requested: str | None = None) -> str:
    if requested:
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
