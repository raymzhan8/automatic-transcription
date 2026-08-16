"""Step 14: the relative-pitch feature function phi(), shared identically by
estimated (Fused+D3) and oracle (GT) pitch sources -- spec section 6's
requirement that condition C and D differ only in the source pitch path,
never in the feature function itself.

phi(cents, valid) -> [T, 4]: octave-unwrapped tonic-relative-cents deltas at
4 offsets (10/50/100/200ms at the native 10ms hop), zeroed wherever either
endpoint of a given offset is not a trusted valid_target frame. No absolute
pitch is ever exposed as a feature -- only differences.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

PITCH_OFFSETS_FRAMES = (1, 5, 10, 20)  # ~10ms, 50ms, 100ms, 200ms at the 10ms native hop
PITCH_FEATURE_DIM = len(PITCH_OFFSETS_FRAMES)

DENSE_ESTIMATED_PITCH_PATH = (
    Path(__file__).resolve().parents[1]
    / "output" / "pitch_diagnostics" / "relative_pitch" / "dense_fused_d3_log2hz.pkl"
)


def load_dense_estimated_pitch(path: Path = DENSE_ESTIMATED_PITCH_PATH) -> dict[str, np.ndarray]:
    """Step 13's frozen Fused+D3 path (training/pitch_diagnostics/relative_pitch/
    dense_pitch_path.py), log2-Hz, one dense (every native frame) array per
    recording -- same units/shape as RecordingLaneIndex's pitch_log2_hz."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


def octave_unwrap(delta_cents: np.ndarray) -> np.ndarray:
    """Remove octave-equivalent (integer multiples of 1200c) jumps while
    preserving local movement: delta - 1200*round(delta/1200)."""
    return delta_cents - 1200.0 * np.round(delta_cents / 1200.0)


def compute_phi(
    cents: np.ndarray,
    valid: np.ndarray,
    offsets: tuple[int, ...] = PITCH_OFFSETS_FRAMES,
) -> np.ndarray:
    """cents: [T] float, NaN allowed at untrusted positions. valid: [T] bool
    (the SAME valid_target mask regardless of pitch source -- estimated
    pitch is defined everywhere, but its feature is still zeroed off wherever
    GT doesn't trust the frame, for a fair, aligned C-vs-D comparison).
    Returns [T, len(offsets)] float32, octave-unwrapped, valid-gated deltas;
    0.0 for offsets reaching before frame 0 or across an invalid boundary.
    """
    cents = np.asarray(cents, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    T = len(cents)
    feats = np.zeros((T, len(offsets)), dtype=np.float32)
    for j, k in enumerate(offsets):
        if k >= T:
            continue
        raw = cents[k:] - cents[:-k]
        raw = octave_unwrap(raw)
        both_valid = valid[k:] & valid[:-k]
        raw = np.where(both_valid, raw, 0.0)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        feats[k:, j] = raw.astype(np.float32)
    return feats


def phi_stats(phi_arrays: list[np.ndarray], valid_arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-offset-channel mean/std over valid, non-zero-gated positions only
    (i.e. only where BOTH endpoints of the delta were trusted -- excluding
    the zero-filled gap/boundary positions from skewing the statistics
    toward zero). Call with TRAIN recordings only (spec section 10)."""
    SIGMA_FLOOR = 1e-3
    dim = phi_arrays[0].shape[1]
    mu = np.zeros(dim, dtype=np.float64)
    sigma = np.ones(dim, dtype=np.float64)
    for j, k in enumerate(PITCH_OFFSETS_FRAMES):
        vals: list[np.ndarray] = []
        for phi, valid in zip(phi_arrays, valid_arrays):
            if k >= len(valid):
                continue
            both_valid = valid[k:] & valid[:-k]
            vals.append(phi[k:, j][both_valid])
        if vals:
            v = np.concatenate(vals)
        else:
            v = np.array([])
        if len(v) > 0:
            mu[j] = float(v.mean())
            sigma[j] = max(float(v.std()), SIGMA_FLOOR)
    return mu.astype(np.float32), sigma.astype(np.float32)


def standardize_phi(phi: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return ((phi - mu[None, :]) / sigma[None, :]).astype(np.float32)
