from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from youtube_pipeline import DEFAULT_CHANNEL, DEFAULT_EXTRACTED_FRAMES_DIR, DEFAULT_RAW_VIDEO_DIR, DEFAULT_VIDEO_LIST


DEFAULT_CLUSTERS_OUT = Path("outputs/clusters")
DEFAULT_LABELS_DIR = Path("data/frames/initial_sort_via_script")


def run_step(name: str, command: list[str], dry_run: bool = False) -> None:
    printable = " ".join(command)
    print(f"\n=== {name} ===")
    print(printable)
    if dry_run:
        return
    subprocess.run(command, check=True)


def script_path(name: str) -> str:
    return str(Path(__file__).resolve().parent / name)


def ensure_inside_workspace(path: Path) -> Path:
    workspace = Path.cwd().resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"Refusing to clean outside workspace: {resolved}") from exc
    return resolved


def clear_directory(path: str | Path, dry_run: bool = False) -> None:
    target = ensure_inside_workspace(Path(path))
    print(f"[clean] {target}")
    if dry_run:
        return
    target.mkdir(parents=True, exist_ok=True)
    for child in target.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def clear_file(path: str | Path, dry_run: bool = False) -> None:
    target = ensure_inside_workspace(Path(path))
    print(f"[clean] {target}")
    if dry_run:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()


def remove_path(path: str | Path, dry_run: bool = False) -> None:
    target = ensure_inside_workspace(Path(path))
    print(f"[clean] {target}")
    if dry_run or not target.exists():
        return
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()


def clean_pipeline_outputs(args: argparse.Namespace) -> None:
    print("\n=== clean previous run ===")
    clear_file(args.video_list, dry_run=args.dry_run)
    clear_directory(args.raw_video_dir, dry_run=args.dry_run)
    clear_directory(args.frames_dir, dry_run=args.dry_run)
    clear_directory(args.clusters_out, dry_run=args.dry_run)
    clear_directory(Path(args.labels_dir) / "footage", dry_run=args.dry_run)
    clear_directory(Path(args.labels_dir) / "graphic", dry_run=args.dry_run)
    remove_path(Path(args.labels_dir) / "mixture", dry_run=args.dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full data preparation pipeline: YouTube list, download, frame extraction, clustering.")

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, help="Maximum number of channel videos to process.")
    selection.add_argument("--all", action="store_true", help="Process all videos from the channel.")
    selection.add_argument("--video-url", help="Process one specific YouTube video.")

    parser.add_argument("--channel", default=DEFAULT_CHANNEL, help=f"YouTube channel URL, handle, or channel ID. Default: {DEFAULT_CHANNEL}")
    parser.add_argument("--video-list", default=str(DEFAULT_VIDEO_LIST), help="TSV file used between step 01 and step 02.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible video sampling.")
    parser.add_argument("--sequential", action="store_true", help="With --limit, take videos in channel order instead of random sampling.")
    parser.add_argument("--raw-video-dir", default=str(DEFAULT_RAW_VIDEO_DIR))
    parser.add_argument("--frames-dir", default=str(DEFAULT_EXTRACTED_FRAMES_DIR), help="Temporary extracted frames directory before sorting.")
    parser.add_argument("--clusters-out", default=str(DEFAULT_CLUSTERS_OUT))
    parser.add_argument("--labels-dir", default=str(DEFAULT_LABELS_DIR), help="Directory containing footage/ and graphic/.")

    parser.add_argument("--fps", type=float, default=1.0, help="Frame extraction FPS. Default: 1.")
    parser.add_argument("--clusters", type=int, default=None, help="Number of visual clusters per video/group.")
    parser.add_argument("--cluster-sample-every-n", type=int, default=1, help="Cluster one frame every N frames.")
    parser.add_argument("--cluster-copy-mode", choices=["copy", "hardlink"], default="copy")
    parser.add_argument("--cluster-limit-per-cluster", type=int, default=None)
    parser.add_argument("--cluster-label-mode", choices=["move", "copy"], default="move", help="Move or copy clustered frames into labels-dir/footage and labels-dir/graphic.")
    parser.add_argument("--cluster-mixed-target", choices=["footage", "graphic"], default="footage", help="Where to place ambiguous mixed clusters.")
    parser.add_argument("--no-apply-labels", action="store_true", help="Do not fill footage/graphic during clustering.")

    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-extract", action="store_true")

    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-cluster", action="store_true")
    parser.add_argument("--no-clean", action="store_true", help="Do not delete previous video list, downloads, extracted frames, clusters, footage and graphic.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def ask_selection(args: argparse.Namespace) -> None:
    if args.limit is not None or args.all or args.video_url:
        return
    if args.skip_fetch and args.skip_download:
        return

    while True:
        value = input("Videos a preparer (`all`, nombre, ou lien YouTube): ").strip()
        if not value:
            print("Entre `all`, un nombre, ou un lien YouTube.")
            continue
        if value.lower() == "all":
            args.all = True
            return
        try:
            limit = int(value)
        except ValueError:
            args.video_url = value
            return
        if limit <= 0:
            print("Le nombre doit etre superieur a 0.")
            continue
        args.limit = limit
        return


def ask_clusters(args: argparse.Namespace) -> None:
    if args.clusters is not None or args.skip_cluster:
        return
    while True:
        value = input("Nombre de clusters prealables par video [8]: ").strip()
        if not value:
            args.clusters = 8
            return
        try:
            clusters = int(value)
        except ValueError:
            print("Entre un nombre entier.")
            continue
        if clusters < 2:
            print("Le nombre de clusters doit etre superieur ou egal a 2.")
            continue
        args.clusters = clusters
        return


def main() -> None:
    args = parse_args()
    ask_selection(args)
    ask_clusters(args)
    python = sys.executable

    if not args.no_clean:
        clean_pipeline_outputs(args)

    if not args.skip_fetch:
        command = [
            python,
            script_path("01_fetch_youtube_list.py"),
            "--video-list",
            args.video_list,
        ]
        if args.video_url:
            command += ["--video-url", args.video_url]
        else:
            command += ["--channel", args.channel]
            if args.limit is not None:
                command += ["--limit", str(args.limit)]
                if not args.sequential:
                    command.append("--random-sample")
                if args.seed is not None:
                    command += ["--seed", str(args.seed)]
        run_step("01 fetch YouTube list", command, dry_run=args.dry_run)

    if not args.skip_download:
        command = [
            python,
            script_path("02_download_youtube_videos.py"),
            "--video-list",
            args.video_list,
            "--raw-video-dir",
            args.raw_video_dir,
        ]
        if args.video_url:
            command += ["--video-url", args.video_url]
        elif args.limit is not None:
            command += ["--limit", str(args.limit)]
        if args.force_download:
            command.append("--force")
        run_step("02 download YouTube videos", command, dry_run=args.dry_run)

    if not args.skip_extract:
        command = [
            python,
            script_path("03_extract_youtube_frames_ffmpeg.py"),
            "--raw-video-dir",
            args.raw_video_dir,
            "--out",
            args.frames_dir,
            "--fps",
            str(args.fps),
        ]
        if args.limit is not None and not args.video_url:
            command += ["--limit", str(args.limit)]
        if args.force_extract:
            command.append("--force")
        run_step("03 extract frames with ffmpeg", command, dry_run=args.dry_run)

    if not args.skip_cluster:
        command = [
            python,
            script_path("04_cluster_extracted_frames.py"),
            "--input",
            args.frames_dir,
            "--out",
            args.clusters_out,
            "--labels-dir",
            args.labels_dir,
            "--clusters",
            str(args.clusters),
            "--sample-every-n",
            str(args.cluster_sample_every_n),
            "--copy-mode",
            args.cluster_copy_mode,
            "--label-mode",
            args.cluster_label_mode,
            "--mixed-target",
            args.cluster_mixed_target,
        ]
        if args.cluster_limit_per_cluster is not None:
            command += ["--limit-per-cluster", str(args.cluster_limit_per_cluster)]
        if args.no_apply_labels:
            command.append("--no-apply-labels")
        run_step("04 cluster extracted frames", command, dry_run=args.dry_run)

    print("\nData preparation pipeline complete.")


if __name__ == "__main__":
    main()
