from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embeddings import DEFAULT_CLIP_MODEL, DEFAULT_DINO_MODEL, EmbeddingConfig, FrozenBackboneEmbedder, embed_image_paths
from review import export_review_frames, load_time_labels
from temporal_smoothing import build_filter_segments
from utils import choose_device, ensure_dir, read_frame_rgb, save_rgb_frame, stable_hash, write_json


def extract_prediction_frames(
    video: str | Path,
    out_dir: str | Path,
    fps: float,
    sample_every_n: int = 1,
    jpeg_quality: int = 95,
) -> list[dict[str, object]]:
    video_path = Path(video)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0:
        step = max(1, sample_every_n)
    else:
        step = max(1, int(round(source_fps / fps))) if source_fps > 0 else 1
        step *= max(1, sample_every_n)

    output_root = ensure_dir(out_dir)
    records: list[dict[str, object]] = []
    frame_index = 0
    progress = tqdm(total=frame_count or None, desc=f"Sample {video_path.name}", unit="frame")
    while True:
        frame_rgb = read_frame_rgb(cap)
        if frame_rgb is None:
            break
        if frame_index % step == 0:
            time_sec = frame_index / source_fps if source_fps > 0 else float(len(records))
            frame_name = f"{video_path.stem}_t{time_sec:010.3f}_f{frame_index:08d}.jpg"
            frame_path = output_root / frame_name
            if not frame_path.exists():
                save_rgb_frame(frame_rgb, frame_path, jpeg_quality=jpeg_quality)
            records.append(
                {
                    "frame_path": str(frame_path.resolve()),
                    "time": float(time_sec),
                    "frame_index": int(frame_index),
                }
            )
        frame_index += 1
        progress.update(1)
    progress.close()
    cap.release()
    return records


def load_model_payload(path: str | Path) -> dict[str, object]:
    payload = joblib.load(path)
    if not isinstance(payload, dict) or "classifier" not in payload:
        raise ValueError(f"Unsupported model format: {path}")
    return payload


def predict(args: argparse.Namespace) -> None:
    model_payload = load_model_payload(args.model)
    model_backbones = model_payload.get("backbones", {})
    if not isinstance(model_backbones, dict):
        model_backbones = {}

    dino_model = str(model_backbones.get("dino", DEFAULT_DINO_MODEL))
    clip_model = str(model_backbones.get("clip", DEFAULT_CLIP_MODEL))
    crop_bottom = float(args.crop_bottom if args.crop_bottom is not None else model_payload.get("crop_bottom", 0.20))
    threshold = float(args.threshold if args.threshold is not None else model_payload.get("threshold_recommended", 0.5))

    video_path = Path(args.video)
    run_id = stable_hash([str(video_path.resolve()), video_path.stat().st_size, video_path.stat().st_mtime_ns, args.fps, args.sample_every_n])[:16]
    frame_dir = Path(args.frames_dir) if args.frames_dir else Path(args.cache_dir) / "frames" / f"{video_path.stem}_{run_id}"
    embedding_cache = Path(args.cache_dir) / "embeddings"

    records = extract_prediction_frames(
        video=video_path,
        out_dir=frame_dir,
        fps=args.fps,
        sample_every_n=args.sample_every_n,
        jpeg_quality=args.jpeg_quality,
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
    proba = classifier.predict_proba(x)[:, 1]  # type: ignore[attr-defined]
    times = [float(item["time"]) for item in records]
    frame_step = 1.0 / args.fps if args.fps > 0 else None
    smoothed, segments = build_filter_segments(
        times=times,
        probabilities=proba,
        threshold=threshold,
        window=args.smoothing_window,
        method=args.smoothing_method,
        merge_gap=args.merge_gap,
        min_duration=args.min_segment_duration,
        frame_step=frame_step,
    )

    predictions: list[dict[str, object]] = []
    for item, prob, smooth_prob in zip(records, proba, smoothed):
        raw_pred = int(float(prob) >= threshold)
        smooth_pred = int(float(smooth_prob) >= threshold)
        predictions.append(
            {
                **item,
                "prob_footage": round(float(1.0 - prob), 6),
                "prob_graphic": round(float(prob), 6),
                "pred": raw_pred,
                "smoothed_prob_graphic": round(float(smooth_prob), 6),
                "smoothed_pred": smooth_pred,
            }
        )

    output = {
        "video": str(video_path.resolve()),
        "model": str(Path(args.model).resolve()),
        "fps": args.fps,
        "sample_every_n": args.sample_every_n,
        "crop_bottom": crop_bottom,
        "threshold": threshold,
        "backbones": {"dino": dino_model, "clip": clip_model},
        "smoothing": {
            "method": args.smoothing_method,
            "window": args.smoothing_window,
            "merge_gap": args.merge_gap,
            "min_segment_duration": args.min_segment_duration,
        },
        "segments_graphic": [segment.to_dict() for segment in segments],
        "predictions": predictions,
    }
    write_json(output, args.out)
    print(f"Saved predictions to {args.out}")
    print(f"Filter segments: {len(segments)}")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict footage vs graphic probabilities for a video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=float, default=2.0)
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
    predict(parse_args())


if __name__ == "__main__":
    main()
