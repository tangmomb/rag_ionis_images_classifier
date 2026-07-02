from __future__ import annotations

import argparse
from pathlib import Path

import yt_dlp
from dotenv import load_dotenv
from yt_dlp.utils import DownloadError

from youtube_pipeline import DEFAULT_RAW_VIDEO_DIR, DEFAULT_VIDEO_LIST, existing_download, ffmpeg_exe, load_video_info_from_url, load_video_list


DEFAULT_FORMAT = "bestvideo[height=720]+bestaudio/best[height=720]/bestvideo[height<=720]+bestaudio/best[height<=720]/best"
DEFAULT_MERGE_FORMAT = "mp4"


def download_video(
    video: tuple[str, str, str],
    download_dir: Path,
    force: bool = False,
    ytdlp_format: str = DEFAULT_FORMAT,
) -> Path:
    youtube_video_id, title, url = video
    video_dir = download_dir
    video_dir.mkdir(parents=True, exist_ok=True)

    if not force:
        existing = existing_download(video_dir, youtube_video_id)
        if existing:
            print(f"[skip] {youtube_video_id} deja telecharge: {existing}")
            return existing

    options = {
        "format": ytdlp_format,
        "merge_output_format": DEFAULT_MERGE_FORMAT,
        "outtmpl": str(video_dir / "%(id)s.%(ext)s"),
        "ffmpeg_location": str(ffmpeg_exe()),
        "js_runtimes": {"node": {}},
        "quiet": False,
        "noplaylist": True,
    }

    print(f"[download] {youtube_video_id} - {title}")
    with yt_dlp.YoutubeDL(options) as downloader:
        downloader.extract_info(url, download=True)

    downloaded = existing_download(video_dir, youtube_video_id)
    if not downloaded:
        raise FileNotFoundError(f"Video telechargee introuvable pour {youtube_video_id}")
    return downloaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download YouTube videos listed in a TSV file using yt-dlp.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, help="Maximum number of videos to download.")
    selection.add_argument("--video-url", help="Download one specific YouTube video.")
    parser.add_argument("--video-list", default=str(DEFAULT_VIDEO_LIST), help="TSV file produced by 01_fetch_youtube_list.py.")
    parser.add_argument("--raw-video-dir", default=str(DEFAULT_RAW_VIDEO_DIR), help="Parent output directory for downloaded videos.")
    parser.add_argument("--format", default=DEFAULT_FORMAT, help="yt-dlp format selector.")
    parser.add_argument("--force", action="store_true", help="Download again even if a video file already exists.")
    parser.add_argument("--dry-run", action="store_true", help="List videos without downloading.")
    return parser.parse_args()


def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    videos = [load_video_info_from_url(args.video_url)] if args.video_url else load_video_list(args.video_list, limit=args.limit)
    if not videos:
        print(f"Aucune video trouvee dans {args.video_list}.")
        return

    if args.dry_run:
        for youtube_video_id, title, url in videos:
            print(f"[dry-run] {youtube_video_id} - {title} - {url}")
        print(f"{len(videos)} videos trouvees.")
        return

    download_dir = Path(args.raw_video_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dossier de telechargement: {download_dir}")

    downloaded_count = 0
    failed: list[str] = []
    for video in videos:
        try:
            download_video(video, download_dir, force=args.force, ytdlp_format=args.format)
            downloaded_count += 1
        except DownloadError as error:
            youtube_video_id = video[0]
            print(f"[error] {youtube_video_id}: {error}")
            failed.append(youtube_video_id)

    print(f"{downloaded_count} videos traitees dans {download_dir}.")
    if failed:
        print(f"{len(failed)} videos en erreur: {', '.join(failed)}")


if __name__ == "__main__":
    main()
