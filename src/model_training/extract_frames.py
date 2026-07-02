from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import ensure_dir, list_videos, read_frame_rgb, save_rgb_frame


def extract_frames(
    videos: str | Path,
    out: str | Path,
    fps: float = 1.0,
    sample_every_n: int = 1,
    class_dir: str | None = None,
    jpeg_quality: int = 95,
) -> int:
    video_paths = list_videos(videos)
    if not video_paths:
        raise FileNotFoundError(f"No video found in {videos}")

    output_root = Path(out)
    total_saved = 0
    for video_path in video_paths:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Skipping unreadable video: {video_path}")
            continue

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0:
            step = max(1, sample_every_n)
        else:
            step = max(1, int(round(source_fps / fps))) if source_fps > 0 else 1
            step *= max(1, sample_every_n)

        if class_dir:
            target_dir = output_root / class_dir
        else:
            target_dir = output_root / video_path.stem
        ensure_dir(target_dir)

        frame_index = 0
        saved_for_video = 0
        progress = tqdm(total=frame_count or None, desc=f"Extract {video_path.name}", unit="frame")
        while True:
            frame_rgb = read_frame_rgb(cap)
            if frame_rgb is None:
                break
            if frame_index % step == 0:
                time_sec = frame_index / source_fps if source_fps > 0 else float(saved_for_video)
                frame_name = f"{video_path.stem}_t{time_sec:010.3f}_f{frame_index:08d}.jpg"
                save_rgb_frame(frame_rgb, target_dir / frame_name, jpeg_quality=jpeg_quality)
                saved_for_video += 1
            frame_index += 1
            progress.update(1)
        progress.close()
        cap.release()
        total_saved += saved_for_video
        print(f"{video_path.name}: saved {saved_for_video} frames to {target_dir}")
    return total_saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from videos.")
    parser.add_argument("--videos", required=True, help="Video file or directory.")
    parser.add_argument("--out", required=True, help="Output frames directory.")
    parser.add_argument("--fps", type=float, default=1.0, help="Target extraction FPS.")
    parser.add_argument("--sample-every-n", type=int, default=1, help="Keep one extracted frame every N samples.")
    parser.add_argument("--class-dir", choices=["footage", "graphic", "mixture"], default=None, help="Optional label directory.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved = extract_frames(
        videos=args.videos,
        out=args.out,
        fps=args.fps,
        sample_every_n=args.sample_every_n,
        class_dir=args.class_dir,
        jpeg_quality=args.jpeg_quality,
    )
    print(f"Saved {saved} frames")


if __name__ == "__main__":
    main()
