from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from review import export_review_frames, load_time_labels
from utils import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export uncertain and mislabeled frames for review.")
    parser.add_argument("--predictions", required=True, help="Predictions JSON from predict_video.py.")
    parser.add_argument("--out", default="outputs/review")
    parser.add_argument("--threshold", type=float, default=None, help="Override threshold; default reads JSON threshold or 0.5.")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--uncertain-margin", type=float, default=0.10)
    parser.add_argument("--labels", default=None, help="Optional JSON labels.")
    parser.add_argument("--label-tolerance", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = read_json(args.predictions)
    predictions = payload.get("predictions", [])
    if not isinstance(predictions, list):
        raise ValueError("Predictions JSON must contain a list field named 'predictions'")

    threshold = float(args.threshold if args.threshold is not None else payload.get("threshold", 0.5))
    video_name = Path(str(payload.get("video", ""))).name or None
    labels = load_time_labels(args.labels, video_name) if args.labels else []
    summary = export_review_frames(
        predictions=predictions,
        out_dir=args.out,
        threshold=threshold,
        top_k=args.top_k,
        uncertain_margin=args.uncertain_margin,
        labels=labels,
        label_tolerance=args.label_tolerance,
    )
    print(f"Exported {summary['count']} review frames to {summary['out_dir']}")
    print(
        f"False positives: {summary['false_positive']}, "
        f"false negatives: {summary['false_negative']}, "
        f"wrong class: {summary.get('wrong_class', 0)}"
    )


if __name__ == "__main__":
    main()
