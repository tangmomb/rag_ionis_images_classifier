from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import requests
from imageio_ffmpeg import get_ffmpeg_exe


YOUTUBE_API = "https://www.googleapis.com/youtube/v3"
DEFAULT_CHANNEL = "https://www.youtube.com/@IONIS-STM/videos"
DEFAULT_VIDEO_LIST = Path("outputs/video_urls.txt")
DEFAULT_RAW_VIDEO_DIR = Path("data/raw_videos")
DEFAULT_EXTRACTED_FRAMES_DIR = Path("outputs/extracted_frames")
BIN_DIR = Path("downloads/bin")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def youtube(endpoint: str, sleep_seconds: float = 0.5, **params: object) -> dict:
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY is missing. Create .env or export the variable.")

    url = f"{YOUTUBE_API}/{endpoint}"
    query = {**params, "key": api_key}
    print(f"{url}?{urlencode({**query, 'key': '***'})}")
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    response = requests.get(url, params=query, timeout=30)
    response.raise_for_status()
    return response.json()


def get_channel_id(channel: str) -> str:
    channel = channel.strip().rstrip("/")
    if channel.endswith("/videos"):
        channel = channel.removesuffix("/videos")
    if channel.startswith("UC"):
        return channel
    if "/channel/" in channel:
        return channel.split("/channel/", 1)[1].split("/", 1)[0]

    handle = channel.split("/")[-1].removeprefix("@")
    data = youtube("channels", part="id", forHandle=handle)
    items = data.get("items", [])
    if not items:
        raise RuntimeError(f"Channel not found: {channel}")
    return items[0]["id"]


def uploads_playlist_id(channel: str) -> str:
    channel_id = get_channel_id(channel)
    data = youtube("channels", part="contentDetails", id=channel_id)
    items = data.get("items", [])
    if not items:
        raise RuntimeError(f"Uploads playlist not found for channel: {channel}")
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def parse_datetime(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def parse_duration(duration: str) -> int:
    total = 0
    number = ""
    for char in duration.removeprefix("P"):
        if char == "T":
            continue
        if char.isdigit():
            number += char
        elif number:
            total += int(number) * {"D": 86400, "H": 3600, "M": 60, "S": 1}[char]
            number = ""
    return total


def extract_youtube_video_id(value: str) -> str:
    raw_value = value.strip()
    parsed = urlparse(raw_value if "://" in raw_value else f"https://{raw_value}")
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host.startswith("m."):
        host = host[2:]

    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/", 1)[0]
    elif host.endswith("youtube.com"):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        else:
            parts = [part for part in parsed.path.split("/") if part]
            video_id = parts[1] if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live"} else ""
    else:
        video_id = raw_value if len(raw_value) == 11 else ""

    if len(video_id) != 11:
        raise argparse.ArgumentTypeError("Lien YouTube invalide ou ID video introuvable.")
    return video_id


def playlist_video_ids(channel: str, limit: int | None = None) -> list[str]:
    playlist_id = uploads_playlist_id(channel)
    video_ids: list[str] = []
    page_token: str | None = None

    while True:
        page = youtube(
            "playlistItems",
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=page_token or "",
        )
        video_ids += [item["contentDetails"]["videoId"] for item in page.get("items", [])]
        if limit is not None and len(video_ids) >= limit:
            return video_ids[:limit]
        page_token = page.get("nextPageToken")
        if not page_token:
            return video_ids


def chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def fetch_videos(channel: str, limit: int | None = None, random_sample: bool = False, seed: int | None = None) -> list[dict]:
    video_id_list = playlist_video_ids(channel, limit=None if random_sample else limit)
    if random_sample and limit is not None and len(video_id_list) > limit:
        rng = random.Random(seed)
        video_id_list = rng.sample(video_id_list, limit)
    elif limit is not None:
        video_id_list = video_id_list[:limit]

    videos: list[dict] = []
    for video_ids in chunks(video_id_list, 50):
        data = youtube(
            "videos",
            part="snippet,contentDetails,statistics",
            id=",".join(video_ids),
            maxResults=50,
        )
        videos += data.get("items", [])
    return videos


def fetch_video(video_url: str) -> list[dict]:
    video_id = extract_youtube_video_id(video_url)
    data = youtube(
        "videos",
        part="snippet,contentDetails,statistics",
        id=video_id,
        maxResults=1,
    )
    if not data.get("items"):
        raise RuntimeError(f"Video YouTube introuvable: {video_url}")
    return data["items"]


def video_info_payload(video: dict) -> dict:
    snippet = video["snippet"]
    content = video["contentDetails"]
    video_id = video["id"]
    published_at = parse_datetime(snippet.get("publishedAt"))
    return {
        "youtube_video_id": video_id,
        "title": snippet["title"],
        "description": snippet.get("description"),
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "published_at": published_at.isoformat().replace("+00:00", "Z") if published_at else None,
        "duration_seconds": parse_duration(content["duration"]),
        "raw_json": video,
    }


def video_list_row(video: dict) -> str:
    payload = video_info_payload(video)
    title = str(payload["title"]).replace("\t", " ").replace("\r", " ").replace("\n", " ")
    return f"{payload['youtube_video_id']}\t{payload['url']}\t{title}"


def write_video_list(videos: list[dict], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [video_list_row(video) for video in videos]
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return target


def load_video_list(path: str | Path, limit: int | None = None) -> list[tuple[str, str, str]]:
    source = Path(path)
    if not source.exists():
        return []
    videos: list[tuple[str, str, str]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t", 2)
        if len(parts) >= 2:
            youtube_video_id = parts[0].strip()
            url = parts[1].strip()
            title = parts[2].strip() if len(parts) >= 3 else youtube_video_id
        else:
            youtube_video_id = extract_youtube_video_id(parts[0])
            url = f"https://www.youtube.com/watch?v={youtube_video_id}"
            title = youtube_video_id
        videos.append((youtube_video_id, title, url))
    return videos[:limit] if limit is not None else videos


def load_video_info_from_url(video_url: str) -> tuple[str, str, str]:
    youtube_video_id = extract_youtube_video_id(video_url)
    return (
        youtube_video_id,
        youtube_video_id,
        f"https://www.youtube.com/watch?v={youtube_video_id}",
    )


def ffmpeg_exe() -> Path:
    source = Path(get_ffmpeg_exe())
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    target = BIN_DIR / "ffmpeg.exe"
    if not target.exists():
        shutil.copy2(source, target)
    os.environ["PATH"] = f"{target.parent}{os.pathsep}{os.environ.get('PATH', '')}"
    return target


def existing_download(video_dir: Path, youtube_video_id: str) -> Path | None:
    matches = sorted(
        path
        for path in video_dir.glob(f"{youtube_video_id}.*")
        if path.is_file() and path.suffix not in {".part", ".ytdl", ".temp", ".json"}
    )
    return matches[0] if matches else None


def video_files(video_dir: str | Path) -> list[Path]:
    root = Path(video_dir)
    if not root.exists():
        return []
    direct = sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)
    if direct:
        return direct
    videos: list[Path] = []
    for child in sorted(root.iterdir()):
        if child.is_dir():
            videos.extend(sorted(path for path in child.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS))
    return videos


def latest_video_dir(parent_dir: str | Path) -> Path:
    root = Path(parent_dir)
    if video_files(root):
        return root
    candidates = sorted(path for path in root.iterdir() if path.is_dir() and video_files(path))
    if not candidates:
        raise FileNotFoundError(f"Aucun dossier de videos trouve dans {root}")
    return candidates[-1]


def format_timecode(value: float) -> str:
    total_seconds = int(round(value))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}_{minutes:02d}_{seconds:02d}"
    return f"{minutes:02d}_{seconds:02d}"


def rename_frames_with_timecodes(video_images_dir: Path, interval_seconds: float) -> None:
    frames = sorted(video_images_dir.glob("frame_*.jpg"))
    for index, frame in enumerate(frames):
        second = math.floor((index * interval_seconds) * 1000) / 1000
        target = video_images_dir / f"{format_timecode(second)}.jpg"
        frame.rename(target)
