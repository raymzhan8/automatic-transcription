"""Audio-based silent / non-silent section detection for full recordings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import librosa
import numpy as np

DetectionMethod = Literal["rms", "librosa_split"]

DEFAULT_SR = 22050
DEFAULT_FRAME_LENGTH = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_MIN_DURATION = 0.1


@dataclass(frozen=True)
class SilentSegment:
    start: float
    end: float
    is_silent: bool

    @property
    def duration(self) -> float:
        return self.end - self.start


def _frame_times(n_frames: int, hop_length: int, sr: int) -> np.ndarray:
    return librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)


def _merge_runs(
    starts: list[float],
    ends: list[float],
    labels: list[bool],
    min_duration: float,
) -> list[SilentSegment]:
    if not starts:
        return []

    merged: list[SilentSegment] = []
    run_start = starts[0]
    run_end = ends[0]
    run_label = labels[0]

    for start, end, label in zip(starts[1:], ends[1:], labels[1:]):
        if label == run_label:
            run_end = end
            continue
        if run_end - run_start >= min_duration:
            merged.append(SilentSegment(run_start, run_end, run_label))
        run_start, run_end, run_label = start, end, label

    if run_end - run_start >= min_duration:
        merged.append(SilentSegment(run_start, run_end, run_label))
    return merged


def _fill_gaps(segments: list[SilentSegment], total_duration: float) -> list[SilentSegment]:
    if total_duration <= 0:
        return []

    if not segments:
        return [SilentSegment(0.0, total_duration, is_silent=True)]

    filled: list[SilentSegment] = []
    cursor = 0.0
    for segment in sorted(segments, key=lambda s: s.start):
        if segment.start > cursor:
            filled.append(SilentSegment(cursor, segment.start, is_silent=True))
        filled.append(segment)
        cursor = max(cursor, segment.end)

    if cursor < total_duration:
        filled.append(SilentSegment(cursor, total_duration, is_silent=True))
    return filled


def _invert_intervals(
    intervals: list[tuple[float, float]],
    total_duration: float,
    *,
    is_silent: bool,
    min_duration: float,
) -> list[SilentSegment]:
    if total_duration <= 0:
        return []

    segments: list[SilentSegment] = []
    cursor = 0.0
    for start, end in sorted(intervals):
        start = max(0.0, start)
        end = min(total_duration, end)
        if end <= start:
            continue
        if start > cursor and end - cursor >= min_duration:
            segments.append(SilentSegment(cursor, start, is_silent=not is_silent))
        if end - start >= min_duration:
            segments.append(SilentSegment(start, end, is_silent=is_silent))
        cursor = max(cursor, end)

    if cursor < total_duration and total_duration - cursor >= min_duration:
        segments.append(SilentSegment(cursor, total_duration, is_silent=not is_silent))
    return segments


def detect_rms(
    y: np.ndarray,
    sr: int,
    *,
    threshold_db: float = -40.0,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
    min_duration: float = DEFAULT_MIN_DURATION,
) -> list[SilentSegment]:
    """Mark frames silent when RMS is below threshold_db relative to the recording peak."""
    if y.size == 0:
        return []

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    peak = float(np.max(rms))
    if peak <= 0:
        total_duration = len(y) / sr
        return [SilentSegment(0.0, total_duration, is_silent=True)]

    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-12) / peak)
    frame_silent = rms_db <= threshold_db
    times = _frame_times(len(rms), hop_length, sr)
    total_duration = len(y) / sr

    starts: list[float] = []
    ends: list[float] = []
    labels: list[bool] = []
    for idx, is_silent in enumerate(frame_silent):
        start = float(times[idx])
        end = float(times[idx + 1]) if idx + 1 < len(times) else total_duration
        starts.append(start)
        ends.append(end)
        labels.append(bool(is_silent))

    return _fill_gaps(_merge_runs(starts, ends, labels, min_duration), total_duration)


def detect_librosa_split(
    y: np.ndarray,
    sr: int,
    *,
    top_db: float = 40.0,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
    min_duration: float = DEFAULT_MIN_DURATION,
) -> list[SilentSegment]:
    """Use librosa.effects.split and derive complementary silent intervals."""
    if y.size == 0:
        return []

    total_duration = len(y) / sr
    intervals = librosa.effects.split(
        y,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    nonsilent = [
        (float(start / sr), float(end / sr))
        for start, end in intervals
        if end > start
    ]
    segments = _invert_intervals(
        nonsilent,
        total_duration,
        is_silent=False,
        min_duration=min_duration,
    )
    return _fill_gaps(segments, total_duration)


def segment_audio(
    y: np.ndarray,
    sr: int,
    method: DetectionMethod,
    **params: object,
) -> list[SilentSegment]:
    if method == "rms":
        return detect_rms(y, sr, **params)
    if method == "librosa_split":
        return detect_librosa_split(y, sr, **params)
    raise ValueError(f"Unknown detection method: {method!r}")


def segment_audio_path(
    path: Path | str,
    method: DetectionMethod,
    *,
    sr: int = DEFAULT_SR,
    **params: object,
) -> list[SilentSegment]:
    y, loaded_sr = librosa.load(path, sr=sr, mono=True)
    if loaded_sr != sr:
        raise ValueError(f"Expected sample rate {sr}, got {loaded_sr} from {path}")
    return segment_audio(y, sr, method, **params)


def _overlap(start: float, end: float, seg_start: float, seg_end: float) -> float:
    return max(0.0, min(end, seg_end) - max(start, seg_start))


def label_interval(
    segments: list[SilentSegment],
    start: float,
    end: float,
) -> bool:
    """Return True when the majority of [start, end] overlaps silent segments."""
    if end <= start:
        return True

    duration = end - start
    silent_overlap = sum(
        _overlap(start, end, segment.start, segment.end)
        for segment in segments
        if segment.is_silent
    )
    nonsilent_overlap = sum(
        _overlap(start, end, segment.start, segment.end)
        for segment in segments
        if not segment.is_silent
    )
    if silent_overlap == nonsilent_overlap:
        return silent_overlap >= duration / 2.0
    return silent_overlap > nonsilent_overlap
