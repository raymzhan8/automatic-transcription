"""Step 20: acoustic-frontend definitions for the Phase A bake-off.

Every frontend is a deterministic, untrained time-frequency transform of raw
audio: A0 (current production CQT, filter_scale=1), A1a/A1b (shorter-context
CQT via filter_scale<1, SAME log-frequency bin grid so nominal cents/bin
never changes -- only the wavelet's time support and therefore its true
frequency-discriminating power), A2a/b/c (fixed-window STFT at three window
lengths, linear-frequency bins), and A3 (an untrained, simple max-fusion of
a short- and a long-window STFT sharing one FFT bin grid).

All frontends share the canonical 10 ms hop (`CQT_HOP` samples) so their
outputs land on the identical frame grid after interpolation onto the
recording's `frame_centers()` target times -- see `align_to_canonical_grid`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import librosa
import numpy as np

from training.features import BINS_PER_OCTAVE, CQT_HOP, FMIN, N_BINS, SR

# Physical candidate band shared by every frontend, matching the frozen
# Step 11 candidate range (34, 244) on the production CQT's 72-bins/octave
# grid: hz_from_bin(34)=104.04Hz, hz_from_bin(243)=778.10Hz (Step 19 §2).
CAND_LO_HZ = float(FMIN * 2.0 ** (34 / BINS_PER_OCTAVE))
CAND_HI_HZ = float(FMIN * 2.0 ** (243 / BINS_PER_OCTAVE))
MEDIAN_HZ = 254.7  # corpus median pitch, Step 19 §2


@dataclass(frozen=True)
class FrontendSpec:
    name: str
    kind: str  # "cqt" | "stft" | "multires"
    params: dict = field(default_factory=dict)


FRONTENDS: dict[str, FrontendSpec] = {
    "A0_cqt_fs1": FrontendSpec("A0_cqt_fs1", "cqt", {"filter_scale": 1.0}),
    "A1a_cqt_fs0.5": FrontendSpec("A1a_cqt_fs0.5", "cqt", {"filter_scale": 0.5}),
    "A1b_cqt_fs0.25": FrontendSpec("A1b_cqt_fs0.25", "cqt", {"filter_scale": 0.25}),
    "A2a_stft_46ms": FrontendSpec("A2a_stft_46ms", "stft", {"win_length": 1024}),
    "A2b_stft_93ms": FrontendSpec("A2b_stft_93ms", "stft", {"win_length": 2048}),
    "A2c_stft_186ms": FrontendSpec("A2c_stft_186ms", "stft", {"win_length": 4096}),
    "A3_multires_46_186ms": FrontendSpec(
        "A3_multires_46_186ms", "multires", {"win_short": 1024, "win_long": 4096}
    ),
}


def cqt_bin_hz(n_bins: int = N_BINS) -> np.ndarray:
    return FMIN * 2.0 ** (np.arange(n_bins) / BINS_PER_OCTAVE)


def cqt_hz_to_bin(hz: np.ndarray) -> np.ndarray:
    hz = np.asarray(hz, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return BINS_PER_OCTAVE * np.log2(np.maximum(hz, 1e-12) / FMIN)


def stft_bin_hz(n_fft: int, sr: int = SR) -> np.ndarray:
    return np.arange(n_fft // 2 + 1) * sr / n_fft


def stft_hz_to_bin(hz: np.ndarray, n_fft: int, sr: int = SR) -> np.ndarray:
    return np.asarray(hz, dtype=np.float64) * n_fft / sr


def cqt_wavelet_length_s(filter_scale: float, n_bins: int = N_BINS, sr: int = SR) -> np.ndarray:
    """Actual per-bin analysis-window duration (seconds), NOT inferred from
    hop/array shape -- librosa's own wavelet-length calculation."""
    freqs = librosa.cqt_frequencies(n_bins, fmin=FMIN, bins_per_octave=BINS_PER_OCTAVE)
    lengths, _ = librosa.filters.wavelet_lengths(freqs=freqs, sr=sr, filter_scale=filter_scale)
    return lengths / sr


def effective_window_ms(spec: FrontendSpec, hz: float) -> float | tuple[float, float]:
    """Effective analysis-window duration in ms at frequency `hz`."""
    if spec.kind == "cqt":
        lengths_s = cqt_wavelet_length_s(spec.params["filter_scale"])
        bin_idx = int(round(float(cqt_hz_to_bin(np.array([hz])))))
        bin_idx = int(np.clip(bin_idx, 0, N_BINS - 1))
        return 1000.0 * float(lengths_s[bin_idx])
    if spec.kind == "stft":
        return 1000.0 * spec.params["win_length"] / SR
    if spec.kind == "multires":
        return (1000.0 * spec.params["win_short"] / SR, 1000.0 * spec.params["win_long"] / SR)
    raise ValueError(spec.kind)


def compute_frontend_native(y: np.ndarray, spec: FrontendSpec, sr: int = SR) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (native_times[s], mag_linear[n_bins, n_native_frames], hz_of_bin[n_bins])."""
    if spec.kind == "cqt":
        mag = np.abs(librosa.cqt(
            y, sr=sr, hop_length=CQT_HOP, fmin=FMIN, n_bins=N_BINS,
            bins_per_octave=BINS_PER_OCTAVE, filter_scale=spec.params["filter_scale"],
        ))
        times = librosa.frames_to_time(np.arange(mag.shape[1]), sr=sr, hop_length=CQT_HOP)
        return times, mag, cqt_bin_hz()

    if spec.kind == "stft":
        win = spec.params["win_length"]
        n_fft = 1 << (win - 1).bit_length()  # next pow2 >= win, minimal zero-pad for a valid n_fft
        n_fft = max(n_fft, win)
        mag = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=CQT_HOP, win_length=win, window="hann", center=True))
        times = librosa.frames_to_time(np.arange(mag.shape[1]), sr=sr, hop_length=CQT_HOP)
        return times, mag, stft_bin_hz(n_fft, sr)

    if spec.kind == "multires":
        win_s, win_l = spec.params["win_short"], spec.params["win_long"]
        n_fft = win_l  # share the long window's FFT size/bin grid; short window zero-padded up to it
        mag_s = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=CQT_HOP, win_length=win_s, window="hann", center=True))
        mag_l = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=CQT_HOP, win_length=win_l, window="hann", center=True))
        n = min(mag_s.shape[1], mag_l.shape[1])
        mag_s, mag_l = mag_s[:, :n], mag_l[:, :n]
        norm_s = mag_s / np.maximum(mag_s.max(axis=0, keepdims=True), 1e-12)
        norm_l = mag_l / np.maximum(mag_l.max(axis=0, keepdims=True), 1e-12)
        mag = np.maximum(norm_s, norm_l)  # untrained per-frame max fusion, no learned weights
        times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=CQT_HOP)
        return times, mag, stft_bin_hz(n_fft, sr)

    raise ValueError(spec.kind)


def hz_to_bin(spec: FrontendSpec, hz: np.ndarray) -> np.ndarray:
    if spec.kind == "cqt":
        return cqt_hz_to_bin(hz)
    if spec.kind == "stft":
        win = spec.params["win_length"]
        n_fft = max(1 << (win - 1).bit_length(), win)
    elif spec.kind == "multires":
        n_fft = spec.params["win_long"]
    else:
        raise ValueError(spec.kind)
    return stft_hz_to_bin(hz, n_fft)


def align_to_canonical_grid(native_times: np.ndarray, mag: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Linear interpolation of each bin (linear-magnitude domain) onto the
    recording's canonical frame_centers() grid -- mirrors
    training/features.py::interpolate_cqt_to_target_grid, applied in linear
    magnitude space to match how every prior pitch_diagnostics step uses
    `linear_mag(cqt_log)` before any rank/contrast computation."""
    out = np.empty((mag.shape[0], len(target_times)), dtype=np.float32)
    for b in range(mag.shape[0]):
        out[b] = np.interp(target_times, native_times, mag[b])
    return out


def candidate_band_mask(hz_of_bin: np.ndarray) -> np.ndarray:
    return (hz_of_bin >= CAND_LO_HZ) & (hz_of_bin <= CAND_HI_HZ)
