from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import IMAGE_EXTENSIONS, ensure_dir, write_json


DEFAULT_INPUT = Path("outputs/extracted_frames")
DEFAULT_OUTPUT = Path("outputs/clusters")
DEFAULT_LABELS_DIR = Path("data/frames/01_initial_sort_via_script")
FEATURE_KEYS = [
    "gray_std",
    "color_std",
    "edge_ratio",
    "raw_edge_ratio",
    "dominant_hue_ratio",
    "brightness_mean",
    "brightness_std",
    "gray_entropy",
    "flat_region_ratio",
    "largest_flat_region_ratio",
]


@dataclass(frozen=True)
class ClusterSummary:
    cluster: int
    role_hint: str
    count: int
    gray_std: float
    edge_ratio: float
    dominant_hue_ratio: float
    flat_region_ratio: float
    largest_flat_region_ratio: float

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "cluster": self.cluster,
            "role_hint": self.role_hint,
            "count": self.count,
            "gray_std": round(self.gray_std, 6),
            "edge_ratio": round(self.edge_ratio, 6),
            "dominant_hue_ratio": round(self.dominant_hue_ratio, 6),
            "flat_region_ratio": round(self.flat_region_ratio, 6),
            "largest_flat_region_ratio": round(self.largest_flat_region_ratio, 6),
        }


def image_second(path: Path) -> float:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        return float(int(parts[0]) * 60 + int(parts[1]))
    if len(parts) == 3 and all(part.isdigit() for part in parts):
        return float(int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2]))
    return float("inf")


def list_images(input_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: (image_second(path), path.as_posix()),
    )


def group_image_dirs(input_dir: Path) -> list[tuple[str, list[Path]]]:
    direct = [path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS] if input_dir.exists() else []
    if direct:
        return [(input_dir.name, list_images(input_dir))]

    groups: list[tuple[str, list[Path]]] = []
    for child in sorted(input_dir.iterdir()) if input_dir.exists() else []:
        if not child.is_dir():
            continue
        images = list_images(child)
        if images:
            groups.append((child.name, images))
    return groups


def odd_kernel(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def gray_entropy(gray: np.ndarray) -> float:
    histogram = cv2.calcHist([gray], [0], None, [32], [0, 256]).ravel()
    total = float(histogram.sum())
    if total <= 0:
        return 0.0
    probabilities = histogram / total
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum() / 5.0)


def flat_region_stats(image_bgr: np.ndarray, tile_size: int = 32) -> tuple[float, float]:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    scale = min(1.0, 640.0 / max(height, width))
    if scale < 1.0:
        image_rgb = cv2.resize(
            image_rgb,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    work = image_rgb.astype(np.float32) / 255.0
    lab = cv2.cvtColor(work, cv2.COLOR_RGB2Lab)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x * grad_x + grad_y * grad_y) / 4.0

    h, w = image_rgb.shape[:2]
    tile_size = max(8, int(tile_size))
    flat_mask = np.zeros((h, w), dtype=np.uint8)
    for y in range(0, h, tile_size):
        y2 = min(h, y + tile_size)
        for x in range(0, w, tile_size):
            x2 = min(w, x + tile_size)
            tile_lab = lab[y:y2, x:x2]
            pixels = tile_lab.reshape(-1, 3)
            mean_lab = pixels.mean(axis=0)
            delta_e = np.linalg.norm(pixels - mean_lab, axis=1)
            color_spread = float(np.percentile(delta_e, 95))
            mean_grad = float(grad[y:y2, x:x2].mean())
            if color_spread < 4.0 and mean_grad < 0.018:
                flat_mask[y:y2, x:x2] = 255

    if not np.any(flat_mask):
        return 0.0, 0.0
    _, _, stats, _ = cv2.connectedComponentsWithStats((flat_mask > 0).astype(np.uint8), connectivity=8)
    largest_area = int(stats[1:, cv2.CC_STAT_AREA].max()) if len(stats) > 1 else 0
    total = int(flat_mask.size)
    return float(np.mean(flat_mask > 0)), float(largest_area / total)


def image_features(path: Path, feature_size: int, blur_kernel: int, flat_tile_size: int) -> dict[str, float]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image illisible: {path}")

    raw_small = cv2.resize(image, (128, 72), interpolation=cv2.INTER_AREA)
    raw_gray = cv2.cvtColor(raw_small, cv2.COLOR_BGR2GRAY)
    raw_edges = cv2.Canny(raw_gray, 80, 160)

    kernel = odd_kernel(blur_kernel)
    blurred = cv2.GaussianBlur(image, (kernel, kernel), 0) if kernel > 1 else image
    small = cv2.resize(blurred, (feature_size, feature_size), interpolation=cv2.INTER_AREA)
    small_float = small.astype(np.float32) / 255.0
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    brightness = gray.astype(np.float32) / 255.0
    edges = cv2.Canny(gray, 80, 160)

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    saturation = hsv[..., 1].astype(np.float32) / 255.0
    value = hsv[..., 2].astype(np.float32) / 255.0
    chromatic_mask = (saturation >= 0.20) & (value >= 0.18)
    if chromatic_mask.any():
        hue_bins = 18
        bins = np.floor(hsv[..., 0][chromatic_mask].astype(np.float32) / 180.0 * hue_bins).astype(np.int32) % hue_bins
        histogram = np.bincount(bins, minlength=hue_bins)
        smoothed = histogram + np.roll(histogram, 1) + np.roll(histogram, -1)
        dominant_hue_ratio = float(smoothed.max() / (feature_size * feature_size))
    else:
        dominant_hue_ratio = 0.0

    flat_ratio, largest_flat_ratio = flat_region_stats(image, tile_size=flat_tile_size)
    return {
        "gray_std": float(gray.std() / 255.0),
        "color_std": float(small_float.reshape(-1, 3).std(axis=0).mean()),
        "edge_ratio": float(np.mean(edges > 0)),
        "raw_edge_ratio": float(np.mean(raw_edges > 0)),
        "dominant_hue_ratio": dominant_hue_ratio,
        "brightness_mean": float(brightness.mean()),
        "brightness_std": float(brightness.std()),
        "gray_entropy": gray_entropy(raw_gray),
        "flat_region_ratio": flat_ratio,
        "largest_flat_region_ratio": largest_flat_ratio,
    }


def feature_matrix(features: list[dict[str, float]]) -> np.ndarray:
    return np.asarray([[item[key] for key in FEATURE_KEYS] for item in features], dtype=np.float32)


def role_hint(summary: dict[str, float], args: argparse.Namespace) -> str:
    flat = summary["flat_region_ratio"]
    largest_flat = summary["largest_flat_region_ratio"]
    edges = summary["edge_ratio"]
    hue = summary["dominant_hue_ratio"]
    gray_std = summary["gray_std"]

    graphic_score = 0
    if flat >= args.graphic_flat_ratio:
        graphic_score += 1
    if largest_flat >= args.graphic_largest_flat_ratio:
        graphic_score += 1
    if edges <= args.graphic_max_edge_ratio:
        graphic_score += 1
    if hue >= args.graphic_dominant_hue_ratio:
        graphic_score += 1
    if gray_std <= args.graphic_max_gray_std:
        graphic_score += 1

    if graphic_score >= 3:
        return "graphic_candidate"
    if graphic_score <= 1:
        return "footage_candidate"
    return "mixed"


def cluster_summaries(labels: np.ndarray, features: list[dict[str, float]], args: argparse.Namespace) -> list[ClusterSummary]:
    summaries: list[ClusterSummary] = []
    for cluster_id in sorted(set(int(label) for label in labels)):
        indices = np.flatnonzero(labels == cluster_id)
        means = {key: float(np.mean([features[index][key] for index in indices])) for key in FEATURE_KEYS}
        summaries.append(
            ClusterSummary(
                cluster=cluster_id,
                role_hint=role_hint(means, args),
                count=int(len(indices)),
                gray_std=means["gray_std"],
                edge_ratio=means["edge_ratio"],
                dominant_hue_ratio=means["dominant_hue_ratio"],
                flat_region_ratio=means["flat_region_ratio"],
                largest_flat_region_ratio=means["largest_flat_region_ratio"],
            )
        )
    return summaries


def safe_copy(source: Path, target: Path, mode: str) -> None:
    ensure_dir(target.parent)
    if target.exists():
        return
    if mode == "hardlink":
        try:
            target.hardlink_to(source)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def unique_target(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(2, 10_000):
        candidate = path.with_name(f"{stem}_{index:03d}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot find a free target name for {path}")


def label_for_role(role_hint_value: str, mixed_target: str) -> str:
    if role_hint_value == "graphic_candidate":
        return "graphic"
    if role_hint_value == "footage_candidate":
        return "footage"
    return mixed_target


def apply_label_move(
    group_name: str,
    image_paths: list[Path],
    labels: np.ndarray,
    summary_by_cluster: dict[int, ClusterSummary],
    labels_dir: Path,
    mixed_target: str,
    mode: str,
) -> dict[Path, Path]:
    moved: dict[Path, Path] = {}
    for index, source in enumerate(image_paths):
        cluster_id = int(labels[index])
        role = summary_by_cluster[cluster_id].role_hint
        label = label_for_role(role, mixed_target=mixed_target)
        target_dir = ensure_dir(labels_dir / label)
        target = unique_target(target_dir / f"{group_name}__{source.name}")
        if mode == "copy":
            shutil.copy2(source, target)
        else:
            shutil.move(str(source), str(target))
        moved[source] = target
    return moved


def contact_sheet(image_paths: list[Path], out_path: Path, thumb_size: int = 160, max_images: int = 80) -> None:
    selected = image_paths[:max_images]
    if not selected:
        return
    cols = min(8, max(1, math.ceil(math.sqrt(len(selected)))))
    rows = math.ceil(len(selected) / cols)
    label_h = 24
    sheet = Image.new("RGB", (cols * thumb_size, rows * (thumb_size + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(selected):
        row, col = divmod(index, cols)
        x = col * thumb_size
        y = row * (thumb_size + label_h)
        try:
            image = Image.open(path).convert("RGB")
            image.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            offset_x = x + (thumb_size - image.width) // 2
            offset_y = y + (thumb_size - image.height) // 2
            sheet.paste(image, (offset_x, offset_y))
        except Exception:
            draw.rectangle((x, y, x + thumb_size - 1, y + thumb_size - 1), outline="red")
        draw.text((x + 4, y + thumb_size + 4), path.name[:24], fill=(0, 0, 0))
    ensure_dir(out_path.parent)
    sheet.save(out_path, quality=90)


def write_cluster_outputs(
    group_name: str,
    image_paths: list[Path],
    labels: np.ndarray,
    features: list[dict[str, float]],
    summaries: list[ClusterSummary],
    out_root: Path,
    copy_mode: str,
    limit_per_cluster: int | None,
    labels_dir: Path,
    apply_labels: bool,
    label_mode: str,
    mixed_target: str,
) -> Path:
    group_out = out_root / group_name
    if group_out.exists():
        shutil.rmtree(group_out)
    ensure_dir(group_out)

    summary_by_cluster = {item.cluster: item for item in summaries}
    for cluster_id in sorted(summary_by_cluster):
        summary = summary_by_cluster[cluster_id]
        indices = np.flatnonzero(labels == cluster_id)
        cluster_paths = [image_paths[index] for index in indices]
        cluster_dir = group_out / summary.role_hint / f"cluster_{cluster_id:02d}"
        ensure_dir(cluster_dir)
        for rank, source in enumerate(cluster_paths):
            if limit_per_cluster is not None and rank >= limit_per_cluster:
                continue
            target = cluster_dir / source.name
            safe_copy(source, target, mode=copy_mode)
        contact_sheet(cluster_paths, cluster_dir / "contact_sheet.jpg")

    labeled_paths: dict[Path, Path] = {}
    if apply_labels:
        labeled_paths = apply_label_move(
            group_name=group_name,
            image_paths=image_paths,
            labels=labels,
            summary_by_cluster=summary_by_cluster,
            labels_dir=labels_dir,
            mixed_target=mixed_target,
            mode=label_mode,
        )

    items: list[dict[str, object]] = []
    for cluster_id in sorted(summary_by_cluster):
        summary = summary_by_cluster[cluster_id]
        indices = np.flatnonzero(labels == cluster_id)
        for index in indices:
            items.append(
                {
                    "source": str(image_paths[index].resolve()),
                    "labeled_path": str(labeled_paths[image_paths[index]].resolve()) if image_paths[index] in labeled_paths else None,
                    "label": label_for_role(summary.role_hint, mixed_target=mixed_target),
                    "cluster": int(cluster_id),
                    "role_hint": summary.role_hint,
                    "features": {key: round(float(features[index][key]), 6) for key in FEATURE_KEYS},
                }
            )

    manifest = {
        "group": group_name,
        "source_count": len(image_paths),
        "output_dir": str(group_out.resolve()),
        "copy_mode": copy_mode,
        "apply_labels": apply_labels,
        "label_mode": label_mode,
        "labels_dir": str(labels_dir.resolve()),
        "mixed_target": mixed_target,
        "feature_keys": FEATURE_KEYS,
        "clusters": [summary.to_dict() for summary in summaries],
        "items": sorted(items, key=lambda item: str(item["source"])),
    }
    write_json(manifest, group_out / "manifest.json")
    return group_out / "manifest.json"


def cluster_group(group_name: str, images: list[Path], args: argparse.Namespace) -> Path | None:
    if args.sample_every_n > 1:
        images = images[:: args.sample_every_n]
    if args.max_images is not None:
        images = images[: args.max_images]
    if len(images) < 2:
        print(f"[skip] {group_name}: pas assez d'images ({len(images)})")
        return None

    print(f"[features] {group_name}: {len(images)} images")
    features = [image_features(path, args.feature_size, args.blur_kernel, args.flat_tile_size) for path in images]
    matrix = feature_matrix(features)
    scaled = StandardScaler().fit_transform(matrix)
    clusters = min(args.clusters, len(images))
    kmeans = KMeans(n_clusters=clusters, random_state=args.seed, n_init="auto")
    labels = kmeans.fit_predict(scaled)

    silhouette = 0.0
    if len(set(labels.tolist())) > 1 and len(images) > clusters:
        silhouette = float(silhouette_score(scaled, labels))
    summaries = cluster_summaries(labels, features, args)
    manifest_path = write_cluster_outputs(
        group_name=group_name,
        image_paths=images,
        labels=labels,
        features=features,
        summaries=summaries,
        out_root=Path(args.out),
        copy_mode=args.copy_mode,
        limit_per_cluster=args.limit_per_cluster,
        labels_dir=Path(args.labels_dir),
        apply_labels=not args.no_apply_labels,
        label_mode=args.label_mode,
        mixed_target=args.mixed_target,
    )
    print(f"[ok] {group_name}: {clusters} clusters, silhouette={silhouette:.3f} -> {manifest_path}")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster extracted frames to speed up manual footage/graphic sorting.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input directory, usually outputs/extracted_frames.")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT), help="Output directory for clustered copies.")
    parser.add_argument("--labels-dir", default=str(DEFAULT_LABELS_DIR), help="Directory containing footage/ and graphic/.")
    parser.add_argument("--clusters", type=int, default=8, help="Number of clusters per video/group.")
    parser.add_argument("--feature-size", type=int, default=64, help="Reduced image size used for CV features.")
    parser.add_argument("--blur-kernel", type=int, default=31, help="Gaussian blur kernel used before edge/color stats.")
    parser.add_argument("--flat-tile-size", type=int, default=32, help="Tile size used to detect flat graphic regions.")
    parser.add_argument("--sample-every-n", type=int, default=1, help="Cluster one frame every N frames.")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum images per group.")
    parser.add_argument("--limit-groups", type=int, default=None, help="Maximum video/group directories to process.")
    parser.add_argument("--limit-per-cluster", type=int, default=None, help="Copy at most N images per cluster; manifest still lists all.")
    parser.add_argument("--copy-mode", choices=["copy", "hardlink"], default="copy", help="Use hardlinks when possible to save disk.")
    parser.add_argument("--label-mode", choices=["move", "copy"], default="move", help="Move or copy clustered frames into labels-dir/footage and labels-dir/graphic.")
    parser.add_argument("--no-apply-labels", action="store_true", help="Only write cluster review outputs, do not fill footage/graphic.")
    parser.add_argument("--mixed-target", choices=["footage", "graphic"], default="footage", help="Where to place ambiguous mixed clusters.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--graphic-flat-ratio", type=float, default=0.14)
    parser.add_argument("--graphic-largest-flat-ratio", type=float, default=0.06)
    parser.add_argument("--graphic-max-edge-ratio", type=float, default=0.010)
    parser.add_argument("--graphic-dominant-hue-ratio", type=float, default=0.60)
    parser.add_argument("--graphic-max-gray-std", type=float, default=0.18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.clusters < 2:
        raise ValueError("--clusters must be >= 2")
    if args.sample_every_n < 1:
        raise ValueError("--sample-every-n must be >= 1")

    input_dir = Path(args.input)
    groups = group_image_dirs(input_dir)
    if args.limit_groups is not None:
        groups = groups[: args.limit_groups]
    if not groups:
        print(f"Aucune image trouvee dans {input_dir}")
        return

    manifests = []
    for group_name, images in groups:
        manifest = cluster_group(group_name, images, args)
        if manifest:
            manifests.append(str(manifest.resolve()))
    write_json({"input": str(input_dir.resolve()), "groups": len(groups), "manifests": manifests}, Path(args.out) / "index.json")
    print(f"{len(manifests)} groupe(s) clusterise(s). Index: {Path(args.out) / 'index.json'}")


if __name__ == "__main__":
    main()
