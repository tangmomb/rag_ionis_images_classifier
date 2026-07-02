from __future__ import annotations

import argparse
import importlib.util
import math
from datetime import datetime
from pathlib import Path
from types import ModuleType

import sys
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_prep.youtube_pipeline import load_video_list
from utils import ensure_dir, write_json


SUMMARY_CLASSES = ["footage", "graphic", "mixture"]
SUMMARY_THUMB_SIZE = (160, 90)
SUMMARY_COLUMNS = 6
SUMMARY_MARGIN = 24
SUMMARY_GAP = 10
SUMMARY_HEADER_HEIGHT = 40


def load_predict_module() -> ModuleType:
    script_path = Path(__file__).with_name("02_predict_video.py")
    spec = importlib.util.spec_from_file_location("predict_video_module", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load prediction script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ask_inputs_with_tkinter(args: argparse.Namespace) -> None:
    if args.video_list and args.model:
        return
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if not args.video_list:
            initial_txt_dir = str(Path("outputs").resolve()) if Path("outputs").exists() else str(Path.cwd())
            selected_list = filedialog.askopenfilename(
                title="Select text file with YouTube URLs",
                initialdir=initial_txt_dir,
                filetypes=[("Text files", "*.txt *.tsv"), ("All files", "*.*")],
            )
            if selected_list:
                args.video_list = selected_list

        if not args.model:
            initial_model_dir = str(Path("models").resolve()) if Path("models").exists() else str(Path.cwd())
            selected_model = filedialog.askopenfilename(
                title="Select trained model",
                initialdir=initial_model_dir,
                filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")],
            )
            if selected_model:
                args.model = selected_model
    finally:
        root.destroy()


def ask_video_list_path(args: argparse.Namespace) -> None:
    if args.video_list:
        return
    while True:
        value = input("TXT file with YouTube URLs: ").strip().strip('"')
        if not value:
            print("Enter a TXT file path.")
            continue
        candidate = Path(value)
        if not candidate.exists():
            print(f"File not found: {candidate}")
            continue
        args.video_list = str(candidate)
        return


def ask_model_path(args: argparse.Namespace) -> None:
    if args.model:
        return
    default_model = Path("models/frame_filter.joblib")
    while True:
        value = input(f"Model path [{default_model}]: ").strip().strip('"')
        candidate = Path(value) if value else default_model
        if not candidate.exists():
            print(f"File not found: {candidate}")
            continue
        args.model = str(candidate)
        return


def build_run_output_dir(model_path: str | Path, root_dir: str | Path = "models_tests") -> Path:
    model_stem = Path(model_path).stem
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return ensure_dir(Path(root_dir) / f"{model_stem}_{timestamp}")


def build_class_patchwork_section(class_dir: Path, class_name: str, font: ImageFont.ImageFont) -> Image.Image:
    image_paths = sorted(path for path in class_dir.glob("*.jpg") if path.is_file())
    thumb_w, thumb_h = SUMMARY_THUMB_SIZE
    columns = SUMMARY_COLUMNS
    rows = max(1, math.ceil(len(image_paths) / columns))
    width = SUMMARY_MARGIN * 2 + columns * thumb_w + max(0, columns - 1) * SUMMARY_GAP
    grid_height = rows * thumb_h + max(0, rows - 1) * SUMMARY_GAP
    height = SUMMARY_MARGIN * 2 + SUMMARY_HEADER_HEIGHT + grid_height
    canvas = Image.new("RGB", (width, height), color=(250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((SUMMARY_MARGIN, SUMMARY_MARGIN), f"{class_name} ({len(image_paths)})", fill=(20, 20, 20), font=font)

    for index, path in enumerate(image_paths):
        row = index // columns
        col = index % columns
        x = SUMMARY_MARGIN + col * (thumb_w + SUMMARY_GAP)
        y = SUMMARY_MARGIN + SUMMARY_HEADER_HEIGHT + row * (thumb_h + SUMMARY_GAP)
        with Image.open(path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail(SUMMARY_THUMB_SIZE)
            tile = Image.new("RGB", SUMMARY_THUMB_SIZE, color=(235, 235, 235))
            paste_x = (thumb_w - thumb.width) // 2
            paste_y = (thumb_h - thumb.height) // 2
            tile.paste(thumb, (paste_x, paste_y))
            canvas.paste(tile, (x, y))
    return canvas


def build_video_summary_patchwork(video_output_dir: Path, summary_dir: Path, video_id: str) -> Path:
    ensure_dir(summary_dir)
    font = ImageFont.load_default()
    sections = [
        build_class_patchwork_section(video_output_dir / class_name, class_name, font)
        for class_name in SUMMARY_CLASSES
    ]
    width = sum(section.width for section in sections) + SUMMARY_GAP * max(0, len(sections) - 1)
    height = max(section.height for section in sections) + SUMMARY_MARGIN * 2
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))

    x = 0
    for section in sections:
        canvas.paste(section, (x, SUMMARY_MARGIN))
        x += section.width + SUMMARY_GAP

    output_path = summary_dir / f"{video_id}_patchwork.jpg"
    canvas.save(output_path, quality=95)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the video prediction model on each YouTube URL listed in a TXT file.")
    parser.add_argument("--video-list", default=None, help="TXT/TSV file containing one YouTube URL per line or the pipeline TSV format.")
    parser.add_argument("--model", default=None, help="Model .joblib file to test.")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--sample-every-n", type=int, default=1)
    parser.add_argument("--crop-bottom", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-dir", default="outputs/cache")
    parser.add_argument("--frames-dir", default=None)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--smoothing-method", choices=["mean", "majority", "none"], default="mean")
    parser.add_argument("--merge-gap", type=float, default=0.5)
    parser.add_argument("--min-segment-duration", type=float, default=0.5)
    parser.add_argument("--export-review", action="store_true")
    parser.add_argument("--review-dir", default="outputs/review")
    parser.add_argument("--review-top-k", type=int, default=100)
    parser.add_argument("--uncertain-margin", type=float, default=0.10)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--label-tolerance", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ask_inputs_with_tkinter(args)
    ask_video_list_path(args)
    ask_model_path(args)

    predict_module = load_predict_module()
    videos = load_video_list(args.video_list)
    if not videos:
        raise FileNotFoundError(f"No YouTube URL found in {args.video_list}")

    run_output_dir = build_run_output_dir(args.model)
    summary_dir = ensure_dir(run_output_dir / "summary")
    model_path = Path(args.model).resolve()
    print(f"Model: {model_path}")
    print(f"Video list: {Path(args.video_list).resolve()}")
    print(f"Output directory: {run_output_dir.resolve()}")

    results: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    for index, (video_id, _, url) in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] Processing {video_id}")
        predict_module.clear_cache_dir(args.cache_dir)

        video_output_dir = run_output_dir / f"{video_id}_classified_frames"
        video_args = argparse.Namespace(
            video=None,
            youtube_url=url,
            video_id=None,
            model=str(model_path),
            out=str(video_output_dir / "predictions.json"),
            classified_frames_dir=str(video_output_dir),
            fps=args.fps,
            sample_every_n=args.sample_every_n,
            crop_bottom=args.crop_bottom,
            threshold=args.threshold,
            batch_size=args.batch_size,
            device=args.device,
            cache_dir=args.cache_dir,
            frames_dir=args.frames_dir,
            jpeg_quality=args.jpeg_quality,
            smoothing_window=args.smoothing_window,
            smoothing_method=args.smoothing_method,
            merge_gap=args.merge_gap,
            min_segment_duration=args.min_segment_duration,
            export_review=args.export_review,
            review_dir=str(run_output_dir / "review" / video_id) if args.export_review else args.review_dir,
            review_top_k=args.review_top_k,
            uncertain_margin=args.uncertain_margin,
            labels=args.labels,
            label_tolerance=args.label_tolerance,
            auto_output_locations=False,
            output_parent_dir=None,
        )
        try:
            predict_module.predict(video_args)
            patchwork_path = build_video_summary_patchwork(video_output_dir, summary_dir, video_id)
            results.append(
                {
                    "video_id": video_id,
                    "youtube_url": url,
                    "predictions_json": str(Path(video_args.out).resolve()),
                    "classified_frames_dir": str(video_output_dir.resolve()),
                    "summary_patchwork": str(patchwork_path.resolve()),
                }
            )
        except Exception as exc:
            print(f"[error] {video_id}: {exc}")
            failures.append({"video_id": video_id, "youtube_url": url, "error": str(exc)})

    summary = {
        "model": str(model_path),
        "video_list": str(Path(args.video_list).resolve()),
        "output_dir": str(run_output_dir.resolve()),
        "success_count": len(results),
        "failure_count": len(failures),
        "results": results,
        "failures": failures,
    }
    write_json(summary, run_output_dir / "summary.json")
    print(f"Saved summary to {run_output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
