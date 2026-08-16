"""Shared candidate-frequency, harmonic-offset, and soft-target helpers for Step 11.

All bin/Hz/cents geometry reuses ``training.pitch_diagnostics.common`` — this
module only adds the pieces specific to a *candidate-frequency* salience
representation (a restricted, harmonic-aware sub-range of the full 360-bin
CQT axis), which Step 10 baselines/diagnostics did not need.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from dataset.canonical.schema import recordings_dir
from training.features import BINS_PER_OCTAVE, N_BINS
from training.framewise_dataset import RecordingLaneIndex
from training.pitch_diagnostics.common import PRIMARY_LANE, bin_from_hz, hz_from_bin

CANDIDATE_RANGE_MARGIN_CENTS = 200.0
# Soft-target Gaussian width. 1 CQT bin = 1200/72 = 16.667 cents; sigma=30 cents
# is ~1.8 bins, wide enough to spread mass onto immediate bin neighbors (so the
# softmax target is not degenerate for a network whose candidate grid is at
# CQT-bin resolution) while still being tight relative to a semitone (100
# cents) so the target stays a genuinely *local* pitch estimate, not a vague
# region estimate.
SALIENCE_SIGMA_CENTS = 30.0


def harmonic_bin_offset(k: int) -> int:
    """Bin offset from fundamental bin to the k-th harmonic bin.

    On the log-frequency CQT grid this is an exact consequence of the grid
    (not an assumption): bin(k*f) - bin(f) = BINS_PER_OCTAVE * log2(k) for
    any f, because bin(hz) = BINS_PER_OCTAVE * log2(hz/FMIN) is linear in
    log2(hz). We still compute it via the real bin_from_hz/hz_from_bin maps
    (round-tripped through frequency) rather than hardcoding the formula, so
    a change to FMIN/BINS_PER_OCTAVE upstream would be picked up.
    """
    probe_hz = 220.0
    b0 = float(bin_from_hz(probe_hz))
    b1 = float(bin_from_hz(probe_hz * k))
    return int(round(b1 - b0))


def compute_candidate_range(
    index: RecordingLaneIndex,
    *,
    margin_cents: float = CANDIDATE_RANGE_MARGIN_CENTS,
    q_lo: float = 0.005,
    q_hi: float = 0.995,
) -> dict[str, Any]:
    """Corpus-wide (not test-fold-specific) candidate fundamental-bin range.

    Uses annotated pitch_log2_hz across every recording with frame targets to
    pick a coarse, disclosed, dataset-level physical range — not a per-fold or
    per-recording oracle crop (see plan decision #3).
    """
    all_hz: list[np.ndarray] = []
    per_recording: dict[str, dict[str, float]] = {}
    for lane in index.lanes:
        frames = index._frames[(lane.recording_id, lane.lane_id)]
        valid = frames["valid_target"]
        if not np.any(valid):
            continue
        log2_hz = frames["pitch_log2_hz"][valid].astype(np.float64)
        hz = np.exp2(log2_hz)
        all_hz.append(hz)
        per_recording[lane.recording_id] = {
            "n_valid": int(valid.sum()),
            "min_hz": float(hz.min()),
            "max_hz": float(hz.max()),
        }
    corpus_hz = np.concatenate(all_hz)
    quantile_levels = [0.001, q_lo, 0.05, 0.25, 0.5, 0.75, 0.95, q_hi, 0.999]
    quantiles_hz = np.quantile(corpus_hz, quantile_levels)
    lo_hz_raw = float(quantiles_hz[quantile_levels.index(q_lo)])
    hi_hz_raw = float(quantiles_hz[quantile_levels.index(q_hi)])

    margin_ratio = 2.0 ** (margin_cents / 1200.0)
    lo_hz = lo_hz_raw / margin_ratio
    hi_hz = hi_hz_raw * margin_ratio

    lo_bin = int(np.clip(int(np.floor(bin_from_hz(lo_hz))), 0, N_BINS - 1))
    hi_bin = int(np.clip(int(np.ceil(bin_from_hz(hi_hz))), lo_bin + 1, N_BINS))

    return {
        "target_min_hz": float(corpus_hz.min()),
        "target_max_hz": float(corpus_hz.max()),
        "target_quantiles_hz": {
            f"q{int(round(lvl * 1000))}": float(v)
            for lvl, v in zip(quantile_levels, quantiles_hz)
        },
        "q_lo": q_lo,
        "q_hi": q_hi,
        "margin_cents": margin_cents,
        "candidate_lo_hz": float(hz_from_bin(lo_bin)),
        "candidate_hi_hz": float(hz_from_bin(hi_bin - 1)),
        "candidate_lo_bin": lo_bin,
        "candidate_hi_bin": hi_bin,  # exclusive
        "n_candidate_bins": hi_bin - lo_bin,
        "full_cqt_bins": N_BINS,
        "cents_per_bin": 1200.0 / BINS_PER_OCTAVE,
        "per_recording": per_recording,
        "n_recordings": len(per_recording),
        "n_valid_frames_total": int(sum(v["n_valid"] for v in per_recording.values())),
    }


def load_or_compute_candidate_range(
    repo_root: Path,
    index: RecordingLaneIndex,
    *,
    out_path: Path | None = None,
) -> dict[str, Any]:
    from training.pitch_diagnostics.common import OUT_DIR, write_json

    out_path = out_path or (OUT_DIR / "harmonic_salience_candidate_range.json")
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    result = compute_candidate_range(index)
    write_json(out_path, result)
    return result


def candidate_bin_centers(lo_bin: int, hi_bin: int) -> np.ndarray:
    """Absolute CQT bin index for each candidate position (float, np.arange)."""
    return np.arange(lo_bin, hi_bin, dtype=np.float64)


def candidate_hz(lo_bin: int, hi_bin: int) -> np.ndarray:
    return np.asarray(hz_from_bin(candidate_bin_centers(lo_bin, hi_bin)))


def gaussian_soft_target_cents(
    target_cents: np.ndarray,
    candidate_cents: np.ndarray,
    *,
    sigma_cents: float = SALIENCE_SIGMA_CENTS,
) -> np.ndarray:
    """Y(f,t) ∝ exp(-(cents(f)-target_cents(t))^2 / 2sigma^2), normalized to sum 1.

    target_cents: [T] (or [B,T])
    candidate_cents: [F_cand]
    returns: [..., F_cand, T] soft target distribution, normalized over F_cand.
    """
    target = np.asarray(target_cents, dtype=np.float64)
    cand = np.asarray(candidate_cents, dtype=np.float64)
    diff = cand[:, None] - target[None, :]  # [F_cand, T]
    logits = -0.5 * (diff / sigma_cents) ** 2
    logits -= logits.max(axis=0, keepdims=True)
    w = np.exp(logits)
    w_sum = w.sum(axis=0, keepdims=True)
    w_sum = np.maximum(w_sum, 1e-12)
    return w / w_sum


def gaussian_soft_target_log2hz_torch(
    target_log2_hz: "torch.Tensor",
    candidate_log2_hz: "torch.Tensor",
    *,
    sigma_cents: float = SALIENCE_SIGMA_CENTS,
) -> "torch.Tensor":
    """Batched torch version for training. Both args are ABSOLUTE log2(Hz) —
    tonic cancels out of a cents *distance*, so no per-example tonic needed:
    distance_cents(f, p_t) = 1200*(candidate_log2_hz - target_log2_hz).

    target_log2_hz: [B,T]; candidate_log2_hz: [F_cand]
    returns: [B,F_cand,T] soft target distribution, normalized over F_cand.
    """
    import torch

    cand = candidate_log2_hz.view(1, -1, 1)  # [1,F,1]
    target = target_log2_hz.unsqueeze(1)  # [B,1,T]
    diff_cents = 1200.0 * (cand - target)
    logits = -0.5 * (diff_cents / sigma_cents) ** 2
    return torch.softmax(logits, dim=1)


def distribution_entropy(probs: np.ndarray, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """Shannon entropy in nats along `axis` of a probability array that sums to 1."""
    p = np.clip(probs, eps, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def target_rank_in_distribution(
    probs: np.ndarray,
    target_cents: np.ndarray,
    candidate_cents: np.ndarray,
    *,
    tolerance_cents: float = SALIENCE_SIGMA_CENTS,
) -> np.ndarray:
    """Rank (1=best) of the highest-scoring candidate within `tolerance_cents`
    of the true target, among all candidates sorted by score (descending).

    probs: [F_cand, T]
    target_cents, returns: [T]
    """
    f_cand, n_t = probs.shape
    cand = np.asarray(candidate_cents, dtype=np.float64)
    diff = np.abs(cand[:, None] - np.asarray(target_cents, dtype=np.float64)[None, :])  # [F,T]
    near = diff <= tolerance_cents
    rank_matrix = np.argsort(np.argsort(-probs, axis=0), axis=0) + 1  # [F,T], 1=best
    masked = np.where(near, rank_matrix, f_cand + 1)
    ranks = masked.min(axis=0)
    ranks[~near.any(axis=0)] = f_cand + 1
    return ranks.astype(np.int64)


def fundamental_hz_for_recording(recording_id: str, repo_root: Path) -> float:
    doc = json.loads((recordings_dir(repo_root) / f"{recording_id}.json").read_text(encoding="utf-8"))
    return float(doc["raga"]["fundamental_hz"])


__all__ = [
    "CANDIDATE_RANGE_MARGIN_CENTS",
    "SALIENCE_SIGMA_CENTS",
    "PRIMARY_LANE",
    "harmonic_bin_offset",
    "compute_candidate_range",
    "load_or_compute_candidate_range",
    "candidate_bin_centers",
    "candidate_hz",
    "gaussian_soft_target_cents",
    "gaussian_soft_target_log2hz_torch",
    "distribution_entropy",
    "target_rank_in_distribution",
    "fundamental_hz_for_recording",
]
