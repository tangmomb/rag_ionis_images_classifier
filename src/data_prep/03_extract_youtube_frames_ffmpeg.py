from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from youtube_pipeline import DEFAULT_EXTRACTED_FRAMES_DIR, DEFAULT_RAW_VIDEO_DIR, ffmpeg_exe, latest_video_dir, rename_frames_with_timecodes, video_files


def extract_frames(video_path: Path, out_root: Path, fps: float = 1.0, force: bool = False) -> int:
    if fps <= 0:
        raise ValueError("--fps must be greater than 0")

    video_images_dir = out_root / video_path.stem
    existing = sorted(video_images_dir.glob("*.jpg")) if video_images_dir.exists() else []
    if existing and not force:
        print(f"[skip] {video_path.name}: {len(existing)} images existent deja dans {video_images_dir}")
        return len(existing)

    if force and video_images_dir.exists():
        shutil.rmtree(video_images_dir)
    video_images_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = video_images_dir / "frame_%06d.jpg"
    command = [
        str(ffmpeg_exe()),
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        str(output_pattern),
    ]
    print(f"[frames] {video_path.name} -> {video_images_dir}")
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    rename_frames_with_timecodes(video_images_dir, interval_seconds=1.0 / fps)
    count = len(list(video_images_dir.glob("*.jpg")))
    print(f"[ok] {count} images")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from downloaded YouTube videos with ffmpeg.")
    parser.add_argument("--video-dir", help="Directory containing videos. Default: latest subdirectory of data/raw_videos.")
    parser.add_argument("--raw-video-dir", default=str(DEFAULT_RAW_VIDEO_DIR), help="Parent directory used when --video-dir is absent.")
    parser.add_argument("--out", default=str(DEFAULT_EXTRACTED_FRAMES_DIR), help="Temporary output directory for extracted frames before sorting.")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to extract. Default: 1.")
    parser.add_argument("--limit", type=int, help="Maximum number of videos to process.")
    parser.add_argument("--force", action="store_true", help="Delete and regenerate already extracted frames.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(args.raw_video_dir)
    videos = video_files(video_dir)
    if args.limit is not None:
        videos = videos[: args.limit]
    if not videos:
        print(f"Aucune video trouvee dans {video_dir}")
        return

    print(f"Dossier videos: {video_dir}")
    out_root = Path(args.out)
    total = 0
    for video_path in videos:
        total += extract_frames(video_path, out_root=out_root, fps=args.fps, force=args.force)
    print(f"{total} images extraites dans {out_root}.")


if __name__ == "__main__":
    main()
