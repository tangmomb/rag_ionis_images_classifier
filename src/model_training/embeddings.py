from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import ensure_dir, file_signature, list_images, load_image_rgb, stable_hash


DEFAULT_DINO_MODEL = "facebook/dinov2-base"
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


@dataclass(frozen=True)
class EmbeddingConfig:
    dino_model: str = DEFAULT_DINO_MODEL
    clip_model: str = DEFAULT_CLIP_MODEL
    crop_bottom: float = 0.20

    def to_dict(self) -> dict[str, str | float]:
        return {
            "dino_model": self.dino_model,
            "clip_model": self.clip_model,
            "crop_bottom": self.crop_bottom,
        }


class FrozenBackboneEmbedder:
    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        device: str = "cpu",
        use_dino: bool = True,
        use_clip: bool = True,
    ) -> None:
        self.config = config or EmbeddingConfig()
        self.device = torch.device(device)
        self.use_dino = use_dino
        self.use_clip = use_clip

        if not self.use_dino and not self.use_clip:
            raise ValueError("At least one backbone must be enabled")

        self.dino_processor = None
        self.dino_model = None
        if self.use_dino:
            self.dino_processor = AutoImageProcessor.from_pretrained(self.config.dino_model)
            self.dino_model = AutoModel.from_pretrained(self.config.dino_model).to(self.device)
            self.dino_model.eval()
            for parameter in self.dino_model.parameters():
                parameter.requires_grad_(False)

        self.clip_processor = None
        self.clip_model = None
        if self.use_clip:
            self.clip_processor = CLIPProcessor.from_pretrained(self.config.clip_model)
            self.clip_model = CLIPModel.from_pretrained(self.config.clip_model).to(self.device)
            self.clip_model.eval()
            for parameter in self.clip_model.parameters():
                parameter.requires_grad_(False)

    @torch.inference_mode()
    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, 0), dtype=np.float32)

        vectors: list[np.ndarray] = []
        if self.use_dino:
            assert self.dino_processor is not None
            assert self.dino_model is not None
            dino_inputs = self.dino_processor(images=list(images), return_tensors="pt").to(self.device)
            dino_outputs = self.dino_model(**dino_inputs)
            dino_vec = dino_outputs.last_hidden_state[:, 0, :]
            dino_vec = torch.nn.functional.normalize(dino_vec, dim=1)
            vectors.append(dino_vec.detach().cpu().numpy().astype(np.float32))

        if self.use_clip:
            assert self.clip_processor is not None
            assert self.clip_model is not None
            clip_inputs = self.clip_processor(images=list(images), return_tensors="pt").to(self.device)
            clip_outputs = self.clip_model.vision_model(pixel_values=clip_inputs["pixel_values"])
            if hasattr(clip_outputs, "pooler_output") and clip_outputs.pooler_output is not None:
                clip_vec = clip_outputs.pooler_output
            else:
                clip_vec = clip_outputs.last_hidden_state[:, 0, :]
            clip_vec = self.clip_model.visual_projection(clip_vec)
            clip_vec = torch.nn.functional.normalize(clip_vec, dim=1)
            vectors.append(clip_vec.detach().cpu().numpy().astype(np.float32))

        return np.concatenate(vectors, axis=1)


def cache_path_for_image(image_path: Path, cache_dir: Path, config: EmbeddingConfig) -> Path:
    signature = stable_hash(
        [
            "frame-filter-embedding-v1",
            file_signature(image_path),
            config.to_dict(),
        ]
    )
    return cache_dir / f"{signature}.joblib"


def embed_image_paths(
    image_paths: Sequence[str | Path],
    embedder: FrozenBackboneEmbedder,
    batch_size: int = 16,
    cache_dir: str | Path | None = None,
    show_progress: bool = True,
) -> np.ndarray:
    paths = [Path(p) for p in image_paths]
    if not paths:
        return np.empty((0, 0), dtype=np.float32)

    cache_root = ensure_dir(cache_dir) if cache_dir is not None else None
    output: list[np.ndarray | None] = [None] * len(paths)
    missing_indices: list[int] = []

    for idx, path in enumerate(paths):
        cache_file = cache_path_for_image(path, cache_root, embedder.config) if cache_root else None
        if cache_file and cache_file.exists():
            output[idx] = joblib.load(cache_file)
        else:
            missing_indices.append(idx)

    iterator = range(0, len(missing_indices), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Embedding frames", unit="batch")  # type: ignore[assignment]

    for start in iterator:
        batch_indices = missing_indices[start : start + batch_size]
        images = [load_image_rgb(paths[i], crop_bottom=embedder.config.crop_bottom) for i in batch_indices]
        embeddings = embedder.embed_images(images)
        for local_idx, original_idx in enumerate(batch_indices):
            vector = embeddings[local_idx].astype(np.float32)
            output[original_idx] = vector
            if cache_root:
                joblib.dump(vector, cache_path_for_image(paths[original_idx], cache_root, embedder.config))

    ready = [vector for vector in output if vector is not None]
    if len(ready) != len(paths):
        raise RuntimeError("Failed to compute all embeddings")
    return np.vstack(ready).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute frozen DINOv2 + CLIP embeddings for images.")
    parser.add_argument("--images", required=True, help="Image directory.")
    parser.add_argument("--out", required=True, help="Output .npy file.")
    parser.add_argument("--cache-dir", default=None, help="Optional embedding cache directory.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--crop-bottom", type=float, default=0.20)
    parser.add_argument("--device", default=None, help="cuda, cpu, or auto if omitted.")
    parser.add_argument("--dino-model", default=DEFAULT_DINO_MODEL)
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    return parser.parse_args()


def main() -> None:
    from utils import choose_device

    args = parse_args()
    paths = list_images(args.images)
    if not paths:
        raise FileNotFoundError(f"No image found in {args.images}")
    config = EmbeddingConfig(
        dino_model=args.dino_model,
        clip_model=args.clip_model,
        crop_bottom=args.crop_bottom,
    )
    embedder = FrozenBackboneEmbedder(config=config, device=choose_device(args.device))
    embeddings = embed_image_paths(paths, embedder, batch_size=args.batch_size, cache_dir=args.cache_dir)
    ensure_dir(Path(args.out).parent)
    np.save(args.out, embeddings)
    print(f"Saved {embeddings.shape} embeddings to {args.out}")


if __name__ == "__main__":
    main()
