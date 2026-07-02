from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from youtube_pipeline import DEFAULT_CHANNEL, DEFAULT_VIDEO_LIST, fetch_video, fetch_videos, write_video_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch a YouTube video list using the official YouTube Data API.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, help="Maximum number of channel videos to fetch.")
    selection.add_argument("--video-url", help="Fetch one specific YouTube video.")
    parser.add_argument("--channel", default=DEFAULT_CHANNEL, help=f"YouTube channel URL, handle, or channel ID. Default: {DEFAULT_CHANNEL}")
    parser.add_argument("--video-list", default=str(DEFAULT_VIDEO_LIST), help="Output TSV file: video_id, url, title.")
    return parser.parse_args()


def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    videos = fetch_video(args.video_url) if args.video_url else fetch_videos(args.channel, limit=args.limit)
    target = write_video_list(videos, Path(args.video_list))
    print(f"{len(videos)} video(s) written to {target}")


if __name__ == "__main__":
    main()
