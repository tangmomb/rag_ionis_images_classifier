from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


SmoothingMethod = Literal["mean", "majority", "none"]


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    duration: float
    score: float

    def to_dict(self) -> dict[str, float]:
        return {
            "start": round(self.start, 4),
            "end": round(self.end, 4),
            "duration": round(self.duration, 4),
            "score": round(self.score, 6),
        }


def smooth_probabilities(
    probabilities: list[float] | np.ndarray,
    window: int = 5,
    method: SmoothingMethod = "mean",
    threshold: float = 0.5,
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.size == 0 or method == "none":
        return probs
    if window <= 1:
        return probs
    if window % 2 == 0:
        window += 1

    radius = window // 2
    smoothed = np.zeros_like(probs)
    for i in range(probs.size):
        left = max(0, i - radius)
        right = min(probs.size, i + radius + 1)
        chunk = probs[left:right]
        if method == "majority":
            smoothed[i] = float(np.mean(chunk >= threshold))
        else:
            smoothed[i] = float(np.mean(chunk))
    return smoothed


def probabilities_to_segments(
    times: list[float] | np.ndarray,
    probabilities: list[float] | np.ndarray,
    threshold: float = 0.5,
    frame_step: float | None = None,
) -> list[Segment]:
    t = np.asarray(times, dtype=np.float32)
    probs = np.asarray(probabilities, dtype=np.float32)
    if t.size == 0 or probs.size == 0:
        return []
    if t.size != probs.size:
        raise ValueError("times and probabilities must have the same length")

    if frame_step is None:
        if t.size >= 2:
            diffs = np.diff(t)
            positive = diffs[diffs > 0]
            frame_step = float(np.median(positive)) if positive.size else 0.0
        else:
            frame_step = 0.0

    active = probs >= threshold
    segments: list[Segment] = []
    start_idx: int | None = None
    for idx, is_active in enumerate(active):
        if is_active and start_idx is None:
            start_idx = idx
        if start_idx is not None and (not is_active or idx == len(active) - 1):
            end_idx = idx - 1 if not is_active else idx
            start = float(t[start_idx])
            end = float(t[end_idx] + (frame_step or 0.0))
            score = float(np.mean(probs[start_idx : end_idx + 1]))
            segments.append(Segment(start=start, end=end, duration=max(0.0, end - start), score=score))
            start_idx = None
    return segments


def merge_close_segments(segments: list[Segment], merge_gap: float = 0.5) -> list[Segment]:
    if not segments:
        return []
    merged: list[Segment] = [segments[0]]
    for current in segments[1:]:
        previous = merged[-1]
        if current.start - previous.end <= merge_gap:
            total_duration = max(current.end - previous.start, 1e-9)
            weighted_score = (
                previous.score * max(previous.duration, 1e-9)
                + current.score * max(current.duration, 1e-9)
            ) / (max(previous.duration, 1e-9) + max(current.duration, 1e-9))
            merged[-1] = Segment(
                start=previous.start,
                end=current.end,
                duration=total_duration,
                score=float(weighted_score),
            )
        else:
            merged.append(current)
    return merged


def filter_short_segments(segments: list[Segment], min_duration: float = 0.0) -> list[Segment]:
    if min_duration <= 0:
        return segments
    return [segment for segment in segments if segment.duration >= min_duration]


def build_filter_segments(
    times: list[float] | np.ndarray,
    probabilities: list[float] | np.ndarray,
    threshold: float = 0.5,
    window: int = 5,
    method: SmoothingMethod = "mean",
    merge_gap: float = 0.5,
    min_duration: float = 0.5,
    frame_step: float | None = None,
) -> tuple[np.ndarray, list[Segment]]:
    smoothed = smooth_probabilities(probabilities, window=window, method=method, threshold=threshold)
    segments = probabilities_to_segments(times, smoothed, threshold=threshold, frame_step=frame_step)
    segments = merge_close_segments(segments, merge_gap=merge_gap)
    segments = filter_short_segments(segments, min_duration=min_duration)
    return smoothed, segments
