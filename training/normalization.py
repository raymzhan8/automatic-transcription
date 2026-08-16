"""Fold-specific CQT and pitch normalization statistics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dataset.canonical.schema import (
    features_dir,
    frames_dir,
    frames_npz_name,
    normalization_dir,
    recordings_dir,
)
from training.features import N_BINS

SIGMA_FLOOR = 1e-3
PRIMARY_LANE = "0:0"
PITCH_UNIT = "cents"


def log2_hz_to_cents(
    pitch_log2_hz: np.ndarray | float,
    fundamental_hz: float,
) -> np.ndarray | float:
    """Tonic-relative cents: 1200 * log2(f / tonic)."""
    tonic_log2 = np.log2(float(fundamental_hz))
    return 1200.0 * (pitch_log2_hz - tonic_log2)


def standardize_pitch(
    pitch_cents: np.ndarray,
    mu: float | np.ndarray,
    sigma: float | np.ndarray,
) -> np.ndarray:
    return ((pitch_cents - float(mu)) / float(sigma)).astype(np.float32)


def denormalize_pitch(
    pitch_standardized: np.ndarray,
    mu: float | np.ndarray,
    sigma: float | np.ndarray,
) -> np.ndarray:
    return (pitch_standardized * float(sigma) + float(mu)).astype(np.float32)


def compute_cqt_stats(
    recording_ids: list[str],
    repo_root: Path,
) -> dict[str, np.ndarray | int]:
    """Per-bin mean/std over all frames of train recordings."""
    sums = np.zeros(N_BINS, dtype=np.float64)
    sumsq = np.zeros(N_BINS, dtype=np.float64)
    count = 0
    for recording_id in recording_ids:
        feat_path = features_dir(repo_root) / f"{recording_id}.npz"
        if not feat_path.exists():
            raise FileNotFoundError(f"missing features for {recording_id}")
        data = np.load(feat_path)
        cqt = data["cqt_log"].astype(np.float64)
        sums += cqt.sum(axis=1)
        sumsq += (cqt ** 2).sum(axis=1)
        count += cqt.shape[1]

    mu = sums / max(count, 1)
    var = sumsq / max(count, 1) - mu ** 2
    sigma = np.sqrt(np.maximum(var, 0.0))
    sigma = np.maximum(sigma, SIGMA_FLOOR)
    return {"mu": mu.astype(np.float32), "sigma": sigma.astype(np.float32), "count": count}


def _fundamental_hz(recording_id: str, repo_root: Path) -> float:
    rec_path = recordings_dir(repo_root) / f"{recording_id}.json"
    doc = json.loads(rec_path.read_text(encoding="utf-8"))
    return float(doc["raga"]["fundamental_hz"])


def compute_pitch_stats(
    recording_ids: list[str],
    repo_root: Path,
    *,
    lane: str = PRIMARY_LANE,
) -> dict[str, np.ndarray | int | str]:
    """Mean/std of tonic-relative cents on train valid_target frames only."""
    values: list[float] = []
    for recording_id in recording_ids:
        fname = frames_npz_name(recording_id, lane)
        path = frames_dir(repo_root) / fname
        if not path.exists():
            continue
        data = np.load(path)
        valid = data["valid_target"]
        pitch_log2 = data["pitch_log2_hz"][valid].astype(np.float64)
        fund = _fundamental_hz(recording_id, repo_root)
        cents = log2_hz_to_cents(pitch_log2, fund)
        values.extend(np.asarray(cents, dtype=np.float64).tolist())

    if not values:
        return {
            "mu": np.array(0.0, dtype=np.float32),
            "sigma": np.array(1.0, dtype=np.float32),
            "count": 0,
            "unit": PITCH_UNIT,
        }
    arr = np.array(values, dtype=np.float64)
    mu = float(arr.mean())
    sigma = max(float(arr.std()), SIGMA_FLOOR)
    return {
        "mu": np.array(mu, dtype=np.float32),
        "sigma": np.array(sigma, dtype=np.float32),
        "count": len(values),
        "unit": PITCH_UNIT,
    }


def normalize_cqt(cqt_log: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return ((cqt_log - mu[:, None]) / sigma[:, None]).astype(np.float32)


def save_fold_stats(
    fold_index: int,
    cqt_stats: dict[str, np.ndarray | int],
    pitch_stats: dict[str, np.ndarray | int | str],
    repo_root: Path,
) -> tuple[Path, Path]:
    out_dir = normalization_dir(repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    cqt_path = out_dir / f"fold_{fold_index}_cqt_stats.npz"
    pitch_path = out_dir / f"fold_{fold_index}_pitch_stats.npz"
    np.savez_compressed(cqt_path, **cqt_stats)
    np.savez_compressed(pitch_path, **pitch_stats)
    return cqt_path, pitch_path


def load_fold_cqt_stats(fold_index: int, repo_root: Path) -> tuple[np.ndarray, np.ndarray]:
    path = normalization_dir(repo_root) / f"fold_{fold_index}_cqt_stats.npz"
    data = np.load(path)
    return data["mu"], data["sigma"]


def load_fold_pitch_stats(fold_index: int, repo_root: Path) -> tuple[float, float]:
    path = normalization_dir(repo_root) / f"fold_{fold_index}_pitch_stats.npz"
    data = np.load(path)
    return float(data["mu"]), float(data["sigma"])
