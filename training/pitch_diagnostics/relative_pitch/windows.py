"""Step 13: shared fixed-size relative-contour windowing, used by both
signals.py's contour-MAE metric and probe.py's T0-T3 classifier features.

Window = WINDOW_FRAMES (21, ~210ms at the 10ms native hop) consecutive
*contiguous* valid frames (no gap larger than 1.5x the native hop inside the
window -- i.e. no invalid-target gap spanned), centered on the labeled
frame. relative_pitch[t] = pitch_cents[t] - pitch_cents[window_start], per
spec formula C; the first (always-zero) entry is dropped, giving a
WINDOW_FRAMES-1 = 20-dim feature vector. No learned representation.
"""

from __future__ import annotations

import numpy as np

WINDOW_FRAMES = 21
HALF = WINDOW_FRAMES // 2
NATIVE_HOP_S = 0.01
GAP_TOL_S = 1.5 * NATIVE_HOP_S


def eligible_centers(record: dict) -> np.ndarray:
    """Indices i such that [i-HALF, i+HALF] is a contiguous run (no gap) and
    the center frame has a valid T0-T3 trajectory_type label."""
    dt = record["dt_seconds"]
    n = len(dt)
    if n < WINDOW_FRAMES:
        return np.array([], dtype=np.int64)
    contiguous = np.ones(n, dtype=bool)
    contiguous[1:] &= dt[1:] <= GAP_TOL_S  # dt[i] = gap before frame i
    # a center i is eligible iff frames i-HALF+1 .. i+HALF are all contiguous
    # with their predecessor (i.e. no gap anywhere inside the window)
    window_ok = np.ones(n, dtype=bool)
    for offset in range(-HALF + 1, HALF + 1):
        shifted = np.roll(contiguous, -offset)
        window_ok &= shifted
    ttype = record["trajectory_type"]
    valid_type = (ttype >= 0) & (ttype <= 3)
    idx = np.arange(n)
    ok = window_ok & valid_type & (idx >= HALF) & (idx < n - HALF)
    return idx[ok]


def relative_contour(path_cents: np.ndarray, center: int) -> np.ndarray:
    window = path_cents[center - HALF: center + HALF + 1]
    rel = window - window[0]
    return rel[1:]  # drop the always-zero first entry -> 20-dim


PRIMITIVE_RESAMPLE_POINTS = 20
MIN_PRIMITIVE_FRAMES = 3


def primitive_segments(record: dict) -> list[tuple[int, int, int]]:
    """Maximal runs of time-contiguous valid frames sharing one T0-T3 label,
    derived directly from the existing per-frame trajectory_type array (no
    canonicalization change -- this only re-derives segment boundaries from
    labels already present in the framewise cache). Returns
    (start_idx, end_idx_exclusive, ttype), end-exclusive, length >= 3 frames.
    """
    ttype = record["trajectory_type"]
    dt = record["dt_seconds"]
    n = len(ttype)
    segments = []
    start = 0
    for i in range(1, n + 1):
        boundary = (
            i == n
            or ttype[i] != ttype[i - 1]
            or dt[i] > GAP_TOL_S
        )
        if boundary:
            t = ttype[start]
            if 0 <= t <= 3 and (i - start) >= MIN_PRIMITIVE_FRAMES:
                segments.append((start, i, int(t)))
            start = i
    return segments


def resample_contour(path_cents: np.ndarray, start: int, end: int, k: int = PRIMITIVE_RESAMPLE_POINTS) -> np.ndarray:
    """Relative-to-segment-start contour, linearly resampled to k points --
    a fixed-size feature from a variable-length primitive, no learned
    representation."""
    seg = path_cents[start:end] - path_cents[start]
    src_x = np.linspace(0.0, 1.0, num=len(seg))
    dst_x = np.linspace(0.0, 1.0, num=k)
    return np.interp(dst_x, src_x, seg)
