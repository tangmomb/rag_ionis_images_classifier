from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

DEFAULT_INPUT = Path("data/frames/01_initial_sort_via_script")
DEFAULT_OUTPUT = Path("data/frames/02_initial_sort_sample")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def sequence_key(path: Path) -> str:
    name = path.name
    if "__" in name:
        return name.split("__", 1)[0]
    return path.stem


def sample_step(image_count: int) -> int:
    if image_count < 100:
        return 5
    if image_count <= 500:
        return 20
    return 200


def copy_sampled_group(source_dir: Path, target_dir: Path) -> tuple[int, int]:
    images = list_images(source_dir)
    grouped: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        grouped[sequence_key(image)].append(image)

    copied = 0
    total = 0
    target_dir.mkdir(parents=True, exist_ok=True)

    for image_paths in grouped.values():
        image_paths.sort()
        total += len(image_paths)
        step = sample_step(len(image_paths))
        for image in image_paths[::step]:
            shutil.copy2(image, target_dir / image.name)
            copied += 1

    return total, copied


def reset_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a lighter copy of initial_sort_via_script by sampling frames per image sequence."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Source directory containing footage/ and graphic/.")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT), help="Output directory for sampled images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.out)

    if not input_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable: {input_dir}")

    reset_output_dir(output_dir)

    total_images = 0
    total_copied = 0
    for child in sorted(input_dir.iterdir()):
        if not child.is_dir():
            continue
        group_total, group_copied = copy_sampled_group(child, output_dir / child.name)
        total_images += group_total
        total_copied += group_copied
        print(f"[ok] {child.name}: {group_copied}/{group_total} images copied")

    print(f"Sampling complete: {total_copied}/{total_images} images in {output_dir}")


if __name__ == "__main__":
    main()
