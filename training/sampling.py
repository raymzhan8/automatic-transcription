"""Valid-anchor excerpt sampling for framewise training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dataset.canonical.schema import HOP_S, MASKED_TRAJECTORY_TYPE

EXCERPT_S = 4.0
EXCERPT_FRAMES = int(EXCERPT_S / HOP_S)


@dataclass(frozen=True)
class ExcerptSlice:
    start_idx: int
    end_idx: int
    padding_mask: np.ndarray
    anchor_idx: int


def choose_excerpt_start(
    anchor_idx: int,
    n_frames: int,
    duration_s: float,
    *,
    rng: np.random.Generator,
    excerpt_s: float = EXCERPT_S,
) -> int:
    """Pick excerpt start so anchor lies inside [start, start + excerpt_s)."""
    anchor_time = (anchor_idx + 0.5) * HOP_S
    lo = max(0.0, anchor_time - excerpt_s)
    hi = min(max(0.0, duration_s - excerpt_s), anchor_time)
    if hi < lo:
        lo = hi = max(0.0, min(anchor_time, duration_s - excerpt_s))
    start_time = float(rng.uniform(lo, hi)) if hi > lo else lo
    return int(round(start_time / HOP_S - 0.5))


def slice_excerpt(
    n_frames: int,
    duration_s: float,
    valid_target: np.ndarray,
    *,
    start_idx: int,
    excerpt_frames: int = EXCERPT_FRAMES,
) -> ExcerptSlice:
    """Slice fixed-length excerpt with padding beyond audio duration."""
    end_idx = start_idx + excerpt_frames
    padding_mask = np.zeros(excerpt_frames, dtype=bool)
    frame_times = (np.arange(start_idx, end_idx) + 0.5) * HOP_S
    padding_mask = frame_times >= duration_s

    return ExcerptSlice(
        start_idx=start_idx,
        end_idx=end_idx,
        padding_mask=padding_mask,
        anchor_idx=-1,
    )


def sample_anchor_index(valid_target: np.ndarray, rng: np.random.Generator) -> int:
    valid_indices = np.flatnonzero(valid_target)
    if len(valid_indices) == 0:
        raise ValueError("no valid anchor frames")
    return int(rng.choice(valid_indices))


def loss_mask(padding_mask: np.ndarray, valid_target: np.ndarray) -> np.ndarray:
    """Frames contributing to loss: in-audio and supervised."""
    return (~padding_mask) & valid_target


def type_labels_for_loss(
    trajectory_type: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Map masked positions to ignore_index=-1."""
    labels = trajectory_type.astype(np.int64).copy()
    labels[~mask] = -1
    return labels
