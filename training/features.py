"""CQT log-magnitude features aligned to the canonical 10 ms target grid."""

from __future__ import annotations

import librosa
import numpy as np

from dataset.canonical.frames import frame_centers
from dataset.canonical.schema import HOP_S

SR = 22050
CQT_HOP = 220
FMIN = 75
N_BINS = 360
BINS_PER_OCTAVE = 72
ALIGN_TOLERANCE_S = 1e-6


def cqt_log_magnitude(y: np.ndarray, sr: int = SR, *, filter_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (native_frame_times_s, log10_cqt) for librosa default framing.

    `filter_scale` defaults to librosa's own default (1.0) -- every trained
    model in this repo up to and including Step 19 used this default
    implicitly. Step 20 Phase B is the first caller to pass a non-default
    value (0.5, the Phase A challenger `A1a_cqt_fs0.5`)."""
    mag = np.abs(
        librosa.cqt(
            y,
            sr=sr,
            hop_length=CQT_HOP,
            fmin=FMIN,
            n_bins=N_BINS,
            bins_per_octave=BINS_PER_OCTAVE,
            filter_scale=filter_scale,
        )
    )
    log_mag = np.log10(np.maximum(mag, 1e-10))
    native_times = librosa.frames_to_time(
        np.arange(log_mag.shape[1]), sr=sr, hop_length=CQT_HOP
    )
    return native_times, log_mag


def interpolate_cqt_to_target_grid(
    native_times: np.ndarray,
    log_cqt: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate each frequency bin onto target center times."""
    out = np.empty((log_cqt.shape[0], len(target_times)), dtype=np.float32)
    for b in range(log_cqt.shape[0]):
        out[b] = np.interp(target_times, native_times, log_cqt[b])
    return out


def extract_features_at_target_centers(
    y: np.ndarray,
    duration_s: float,
    *,
    sr: int = SR,
) -> tuple[np.ndarray, np.ndarray]:
    """One feature vector per target frame center (Step 6.5 convention)."""
    target_times = frame_centers(duration_s, hop_s=HOP_S)
    native_times, log_cqt = cqt_log_magnitude(y, sr=sr)
    features = interpolate_cqt_to_target_grid(native_times, log_cqt, target_times)
    return target_times, features


def feature_metadata(*, sr: int = SR) -> dict[str, float | int]:
    return {
        "sr": sr,
        "hop_s": HOP_S,
        "cqt_hop": CQT_HOP,
        "fmin": FMIN,
        "n_bins": N_BINS,
        "bins_per_octave": BINS_PER_OCTAVE,
    }
