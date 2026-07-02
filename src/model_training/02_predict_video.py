from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import yt_dlp
from tqdm import tqdm
from yt_dlp.utils import DownloadError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_prep.youtube_pipeline import existing_download, ffmpeg_exe, load_video_info_from_url
from embeddings import DEFAULT_CLIP_MODEL, DEFAULT_DINO_MODEL, EmbeddingConfig, FrozenBackboneEmbedder, embed_image_paths
from review import export_review_frames, load_time_labels
from temporal_smoothing import build_filter_segments
from utils import choose_device, ensure_dir, stable_hash, write_json


DEFAULT_LABEL_MAPPING = {"footage": 0, "graphic": 1, "mixture": 2}
DEFAULT_CLASS_NAMES = ["footage", "graphic", "mixture"]
DEFAULT_FILTER_CLASS_NAMES = ["graphic", "mixture"]
DEFAULT_YTDLP_FORMAT = "bestvideo[height=720]+bestaudio/best[height=720]/bestvideo[height<=720]+bestaudio/best[height<=720]/best"
DEFAULT_MERGE_FORMAT = "mp4"


def clear_cache_dir(cache_dir: str | Path) -> Path:
    cache_path = Path(cache_dir)
    if cache_path.exists():
        shutil.rmtree(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def probe_video_metadata(video: str | Path) -> tuple[float, float, int]:
    ffmpeg_path = ffmpeg_exe()
    command = [
        str(ffmpeg_path),
        "-i",
        str(video),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    text = (result.stderr or "") + "\n" + (result.stdout or "")

    duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    fps_match = re.search(r"(\d+(?:\.\d+)?)\s*fps\b", text)
    if not duration_match:
        raise RuntimeError(f"Could not determine duration for video: {video}")

    hours = float(duration_match.group(1))
    minutes = float(duration_match.group(2))
    seconds = float(duration_match.group(3))
    duration_sec = hours * 3600.0 + minutes * 60.0 + seconds
    source_fps = float(fps_match.group(1)) if fps_match else 0.0
    nb_frames = 0
    if nb_frames <= 0 and source_fps > 0 and duration_sec > 0:
        nb_frames = max(1, int(round(source_fps * duration_sec)))
    return duration_sec, source_fps, nb_frames


def ffmpeg_extract_frame(
    video: str | Path,
    target_time_sec: float,
    out_path: str | Path,
    jpeg_quality: int = 95,
) -> None:
    ffmpeg_path = ffmpeg_exe()
    qscale = min(31, max(2, int(round((100 - jpeg_quality) * 0.29)) + 2))
    command = [
        str(ffmpeg_path),
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{target_time_sec:.3f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-q:v",
        str(qscale),
        str(out_path),
    ]
    subprocess.run(command, check=True)

def download_youtube_video(url: str, cache_dir: str | Path) -> tuple[str, Path]:
    download_dir = ensure_dir(Path(cache_dir) / "downloads")
    video_id, _, canonical_url = load_video_info_from_url(url)
    existing = existing_download(download_dir, video_id)
    if existing:
        return (video_id, existing.resolve())

    options = {
        "format": DEFAULT_YTDLP_FORMAT,
        "merge_output_format": DEFAULT_MERGE_FORMAT,
        "outtmpl": str(download_dir / "%(id)s.%(ext)s"),
        "ffmpeg_location": str(ffmpeg_exe()),
        "js_runtimes": {"node": {}},
        "remote_components": ["ejs:github"],
        "quiet": False,
        "noplaylist": True,
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            downloader.extract_info(canonical_url, download=True)
    except DownloadError as exc:
        raise RuntimeError(f"yt-dlp failed to download the video: {exc}") from exc

    downloaded = existing_download(download_dir, video_id)
    if not downloaded:
        raise RuntimeError(f"Downloaded video file not found for {video_id}.")
    return (video_id, downloaded.resolve())


def extract_prediction_frames(
    video: str | Path,
    out_dir: str | Path,
    fps: float,
    sample_every_n: int = 1,
    jpeg_quality: int = 95,
    frame_prefix: str | None = None,
) -> list[dict[str, object]]:
    video_path = Path(video)
    prefix = frame_prefix or video_path.stem
    duration_sec, source_fps, frame_count = probe_video_metadata(video_path)
    output_root = ensure_dir(out_dir)
    records: list[dict[str, object]] = []
    if duration_sec <= 0:
        raise RuntimeError(f"Could not determine duration for video: {video_path}")

    interval_sec = max(1e-6, max(1, sample_every_n) / fps) if fps > 0 else float(max(1, sample_every_n))
    intervals = np.arange(0.0, duration_sec, interval_sec, dtype=np.float64)
    seed = int(stable_hash([str(video_path.resolve()), frame_count, source_fps, interval_sec])[:16], 16)
    rng = np.random.default_rng(seed)
    progress = tqdm(total=len(intervals), desc=f"Sample {video_path.name}", unit="sample")

    for start_sec in intervals:
        end_sec = min(start_sec + interval_sec, duration_sec)
        if end_sec <= start_sec:
            progress.update(1)
            continue
        target_sec = float(rng.uniform(start_sec, end_sec))
        frame_index = max(0, int(round(target_sec * source_fps))) if source_fps > 0 else len(records)
        actual_time_sec = frame_index / source_fps if source_fps > 0 else float(target_sec)
        frame_name = f"{prefix}_t{actual_time_sec:010.3f}_f{frame_index:08d}.jpg"
        frame_path = output_root / frame_name
        if not frame_path.exists():
            ffmpeg_extract_frame(video_path, target_sec, frame_path, jpeg_quality=jpeg_quality)
        if frame_path.exists():
            records.append(
                {
                    "frame_path": str(frame_path.resolve()),
                    "time": float(actual_time_sec),
                    "frame_index": int(frame_index),
                }
            )
        progress.update(1)
    progress.close()
    return records


def load_model_payload(path: str | Path) -> dict[str, object]:
    payload = joblib.load(path)
    if not isinstance(payload, dict) or "classifier" not in payload:
        raise ValueError(f"Unsupported model format: {path}")
    return payload


def class_names_from_payload(payload: dict[str, object]) -> list[str]:
    raw_class_names = payload.get("class_names")
    if isinstance(raw_class_names, list) and all(isinstance(item, str) for item in raw_class_names):
        return [str(item) for item in raw_class_names]

    raw_mapping = payload.get("label_mapping", DEFAULT_LABEL_MAPPING)
    if isinstance(raw_mapping, dict):
        pairs = sorted((int(value), str(key)) for key, value in raw_mapping.items())
        return [name for _, name in pairs]
    return DEFAULT_CLASS_NAMES


def align_probabilities(classifier: object, proba: np.ndarray, class_names: list[str]) -> np.ndarray:
    classes = getattr(classifier, "classes_", None)
    if classes is None:
        return proba
    class_ids = [int(value) for value in np.asarray(classes).tolist()]
    if class_ids == list(range(len(class_names))):
        return proba

    aligned = np.zeros((proba.shape[0], len(class_names)), dtype=np.float32)
    for source_index, class_id in enumerate(class_ids):
        if 0 <= class_id < len(class_names):
            aligned[:, class_id] = proba[:, source_index]
    return aligned


def export_frames_by_class(predictions: list[dict[str, object]], out_dir: str | Path, class_names: list[str]) -> dict[str, int]:
    root = ensure_dir(out_dir)
    counts = {class_name: 0 for class_name in class_names}
    for item in predictions:
        class_name = str(item.get("pred_label", ""))
        frame_path = item.get("frame_path")
        if not frame_path or class_name not in counts:
            continue
        source = Path(str(frame_path))
        if not source.exists():
            continue
        target_dir = ensure_dir(root / class_name)
        shutil.copy2(source, target_dir / source.name)
        counts[class_name] += 1
    return counts


def predict(args: argparse.Namespace) -> None:
    youtube_url = args.youtube_url
    if youtube_url:
        print("Downloading YouTube video with yt-dlp...")
        video_id, video_path = download_youtube_video(youtube_url, args.cache_dir)
        args.video_id = video_id
        args.video = str(video_path)
        if getattr(args, "auto_output_locations", False):
            parent_dir = Path(getattr(args, "output_parent_dir", "outputs"))
            classified_dir = parent_dir / f"{args.video_id}_classified_frames"
            args.classified_frames_dir = str(classified_dir)
            args.out = str(classified_dir / "predictions.json")
        print(f"Downloaded video to {args.video}")

    model_payload = load_model_payload(args.model)
    model_backbones = model_payload.get("backbones", {})
    if not isinstance(model_backbones, dict):
        model_backbones = {}

    dino_model = str(model_backbones.get("dino", DEFAULT_DINO_MODEL))
    clip_model = str(model_backbones.get("clip", DEFAULT_CLIP_MODEL))
    crop_bottom = float(args.crop_bottom if args.crop_bottom is not None else model_payload.get("crop_bottom", 0.20))
    threshold = float(args.threshold if args.threshold is not None else model_payload.get("threshold_recommended", 0.5))
    class_names = class_names_from_payload(model_payload)
    filter_class_names = model_payload.get("filter_class_names", DEFAULT_FILTER_CLASS_NAMES)
    if not isinstance(filter_class_names, list):
        filter_class_names = DEFAULT_FILTER_CLASS_NAMES
    filter_indices = [class_names.index(str(name)) for name in filter_class_names if str(name) in class_names]
    if not filter_indices and "graphic" in class_names:
        filter_indices = [class_names.index("graphic")]

    video_path = Path(args.video)
    video_key = str(getattr(args, "video_id", "") or video_path.stem)
    frame_dir = Path(args.frames_dir) if args.frames_dir else Path(args.cache_dir) / "frames" / video_key
    embedding_cache = Path(args.cache_dir) / "embeddings"

    records = extract_prediction_frames(
        video=video_path,
        out_dir=frame_dir,
        fps=args.fps,
        sample_every_n=args.sample_every_n,
        jpeg_quality=args.jpeg_quality,
        frame_prefix=video_key,
    )
    if not records:
        raise RuntimeError(f"No frame extracted from {video_path}")

    device = choose_device(args.device)
    print(f"Using device: {device}")
    config = EmbeddingConfig(dino_model=dino_model, clip_model=clip_model, crop_bottom=crop_bottom)
    embedder = FrozenBackboneEmbedder(config=config, device=device)
    frame_paths = [str(item["frame_path"]) for item in records]
    x = embed_image_paths(frame_paths, embedder, batch_size=args.batch_size, cache_dir=embedding_cache)

    classifier = model_payload["classifier"]
    proba = align_probabilities(classifier, classifier.predict_proba(x), class_names)  # type: ignore[attr-defined]
    filter_score = proba[:, filter_indices].sum(axis=1) if filter_indices else np.zeros(proba.shape[0], dtype=np.float32)
    times = [float(item["time"]) for item in records]
    frame_step = 1.0 / args.fps if args.fps > 0 else None
    smoothed_filter, segments = build_filter_segments(
        times=times,
        probabilities=filter_score,
        threshold=threshold,
        window=args.smoothing_window,
        method=args.smoothing_method,
        merge_gap=args.merge_gap,
        min_duration=args.min_segment_duration,
        frame_step=frame_step,
    )
    smoothed_by_class: dict[str, np.ndarray] = {}
    segments_by_class: dict[str, list[dict[str, float]]] = {}
    for class_index, class_name in enumerate(class_names):
        smoothed_class, class_segments = build_filter_segments(
            times=times,
            probabilities=proba[:, class_index],
            threshold=threshold,
            window=args.smoothing_window,
            method=args.smoothing_method,
            merge_gap=args.merge_gap,
            min_duration=args.min_segment_duration,
            frame_step=frame_step,
        )
        smoothed_by_class[class_name] = smoothed_class
        segments_by_class[class_name] = [segment.to_dict() for segment in class_segments]

    predictions: list[dict[str, object]] = []
    smoothed_matrix = np.column_stack([smoothed_by_class[name] for name in class_names])
    for item, probs, smooth_probs, score, smooth_score in zip(records, proba, smoothed_matrix, filter_score, smoothed_filter):
        raw_pred = int(np.argmax(probs))
        smooth_pred = int(np.argmax(smooth_probs))
        row: dict[str, object] = {
            **item,
            "pred": raw_pred,
            "pred_label": class_names[raw_pred],
            "smoothed_pred": smooth_pred,
            "smoothed_pred_label": class_names[smooth_pred],
            "filter_score": round(float(score), 6),
            "smoothed_filter_score": round(float(smooth_score), 6),
            "filter_pred": int(float(score) >= threshold),
            "smoothed_filter_pred": int(float(smooth_score) >= threshold),
        }
        for class_index, class_name in enumerate(class_names):
            row[f"prob_{class_name}"] = round(float(probs[class_index]), 6)
            row[f"smoothed_prob_{class_name}"] = round(float(smooth_probs[class_index]), 6)
        predictions.append(row)

    output = {
        "video": str(video_path.resolve()),
        "video_id": video_key,
        "youtube_url": youtube_url,
        "model": str(Path(args.model).resolve()),
        "fps": args.fps,
        "sample_every_n": args.sample_every_n,
        "crop_bottom": crop_bottom,
        "threshold": threshold,
        "class_names": class_names,
        "filter_class_names": [class_names[index] for index in filter_indices],
        "backbones": {"dino": dino_model, "clip": clip_model},
        "smoothing": {
            "method": args.smoothing_method,
            "window": args.smoothing_window,
            "merge_gap": args.merge_gap,
            "min_segment_duration": args.min_segment_duration,
        },
        "segments_filter": [segment.to_dict() for segment in segments],
        "segments_by_class": segments_by_class,
        "predictions": predictions,
    }
    ensure_dir(frame_dir)
    write_json(output, args.out)
    print(f"Saved predictions to {args.out}")
    print(f"Filter segments: {len(segments)}")
    if args.classified_frames_dir:
        counts = export_frames_by_class(predictions, args.classified_frames_dir, class_names)
        print(f"Copied classified frames to {args.classified_frames_dir}: {counts}")

    if args.export_review:
        labels = load_time_labels(args.labels, video_path.name) if args.labels else []
        summary = export_review_frames(
            predictions=predictions,
            out_dir=args.review_dir,
            threshold=threshold,
            top_k=args.review_top_k,
            uncertain_margin=args.uncertain_margin,
            labels=labels,
            label_tolerance=args.label_tolerance,
        )
        print(f"Exported {summary['count']} review frames to {summary['out_dir']}")


def ask_inputs_with_tkinter(args: argparse.Namespace) -> None:
    if args.youtube_url and args.model and args.out and args.classified_frames_dir:
        return
    try:
        import tkinter as tk
        from tkinter import filedialog, simpledialog
    except Exception:
        return

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if not args.youtube_url:
            youtube_url = simpledialog.askstring("YouTube URL", "Enter the YouTube video URL to analyze:")
            if youtube_url:
                args.youtube_url = youtube_url.strip()

        if not args.model:
            initial_model_dir = str(Path("models").resolve()) if Path("models").exists() else str(Path.cwd())
            selected_model = filedialog.askopenfilename(
                title="Select trained model",
                initialdir=initial_model_dir,
                filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")],
            )
            if selected_model:
                args.model = selected_model

        if not args.out or not args.classified_frames_dir:
            selected_parent = filedialog.askdirectory(
                title="Select folder where the classified frames folder should be created"
            )
            if selected_parent:
                parent_dir = Path(selected_parent)
                video_stem = "prediction_run"
                classified_dir = parent_dir / f"{video_stem}_classified_frames"
                if not args.classified_frames_dir:
                    args.classified_frames_dir = str(classified_dir)
                if not args.out:
                    args.out = str(classified_dir / "predictions.json")
                args.auto_output_locations = True
                args.output_parent_dir = str(parent_dir)
    finally:
        root.destroy()


def ask_youtube_url(args: argparse.Namespace) -> None:
    if args.youtube_url:
        return
    while True:
        value = input("YouTube URL to analyze: ").strip().strip('"')
        if not value:
            print("Enter a YouTube URL.")
            continue
        if not value.startswith(("http://", "https://")):
            print("Enter a valid URL starting with http:// or https://")
            continue
        args.youtube_url = value
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


def ask_output_locations(args: argparse.Namespace) -> None:
    if args.out and args.classified_frames_dir:
        return
    while True:
        value = input("Folder where the classified frames folder should be created [outputs]: ").strip().strip('"')
        parent_dir = Path(value) if value else Path("outputs")
        parent_dir.mkdir(parents=True, exist_ok=True)
        video_stem = "prediction_run"
        classified_dir = parent_dir / f"{video_stem}_classified_frames"
        if not args.classified_frames_dir:
            args.classified_frames_dir = str(classified_dir)
        if not args.out:
            args.out = str(classified_dir / "predictions.json")
        args.auto_output_locations = True
        args.output_parent_dir = str(parent_dir)
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict footage, graphic and mixture probabilities for a video.")
    parser.add_argument("--video", default=None, help="Local video path. Normally filled automatically after YouTube download.")
    parser.add_argument("--youtube-url", default=None, help="YouTube URL to download with yt-dlp before prediction.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default=None, help="Predictions JSON path. If omitted, it is created in the classified frames folder.")
    parser.add_argument("--classified-frames-dir", default=None, help="Directory where footage/, graphic/ and mixture/ copies are written.")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--sample-every-n", type=int, default=1)
    parser.add_argument("--crop-bottom", type=float, default=None, help="Override model crop; default uses saved value.")
    parser.add_argument("--threshold", type=float, default=None, help="Override model threshold; default uses saved value or 0.5.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None, help="cuda, cpu, or auto if omitted.")
    parser.add_argument("--cache-dir", default="outputs/cache")
    parser.add_argument("--frames-dir", default=None, help="Optional directory for sampled frames.")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--smoothing-method", choices=["mean", "majority", "none"], default="mean")
    parser.add_argument("--merge-gap", type=float, default=0.5)
    parser.add_argument("--min-segment-duration", type=float, default=0.5)
    parser.add_argument("--export-review", action="store_true")
    parser.add_argument("--review-dir", default="outputs/review")
    parser.add_argument("--review-top-k", type=int, default=100)
    parser.add_argument("--uncertain-margin", type=float, default=0.10)
    parser.add_argument("--labels", default=None, help="Optional JSON labels for false positive/negative export.")
    parser.add_argument("--label-tolerance", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clear_cache_dir(args.cache_dir)
    ask_inputs_with_tkinter(args)
    ask_youtube_url(args)
    ask_model_path(args)
    ask_output_locations(args)
    predict(args)


if __name__ == "__main__":
    main()
