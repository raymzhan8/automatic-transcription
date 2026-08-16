"""Shared CQT geometry, paths, and JSON helpers for Step 10."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from training.features import BINS_PER_OCTAVE, FMIN, N_BINS

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "output" / "pitch_diagnostics"
B1_RUN = REPO_ROOT / "output" / "framewise_runs" / "b1_tcn_type_pitch"
C_RUN = REPO_ROOT / "output" / "framewise_runs" / "c_bigru_type_pitch"
PRIMARY_LANE = "0:0"
CENTS_PER_BIN = 1200.0 / BINS_PER_OCTAVE
REF_TONIC_BIN = BINS_PER_OCTAVE  # 1 octave above fmin (150 Hz)
OCTAVE_KS = (-2, -1, 0, 1, 2)
DRONE_CENTS = (0.0, 1200.0, -1200.0, 700.0, -700.0, 2400.0, -2400.0)


def hz_from_bin(bin_index: np.ndarray | float) -> np.ndarray | float:
    return FMIN * (2.0 ** (np.asarray(bin_index) / BINS_PER_OCTAVE))


def bin_from_hz(hz: np.ndarray | float) -> np.ndarray | float:
    hz_arr = np.asarray(hz, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        bins = BINS_PER_OCTAVE * np.log2(np.maximum(hz_arr, 1e-12) / FMIN)
    return bins


def hz_from_cents(cents: np.ndarray | float, tonic_hz: float) -> np.ndarray | float:
    return float(tonic_hz) * (2.0 ** (np.asarray(cents) / 1200.0))


def cents_from_hz(hz: np.ndarray | float, tonic_hz: float) -> np.ndarray | float:
    hz_arr = np.asarray(hz, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1200.0 * np.log2(np.maximum(hz_arr, 1e-12) / float(tonic_hz))


def linear_mag(cqt_log: np.ndarray) -> np.ndarray:
    """Invert log10 magnitude used in precomputed features."""
    return np.power(10.0, cqt_log.astype(np.float64))


def clip_bin(bin_f: np.ndarray | float) -> np.ndarray:
    return np.clip(np.rint(np.asarray(bin_f)), 0, N_BINS - 1).astype(np.int64)


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2) + "\n", encoding="utf-8")


def summarize_array(values: np.ndarray) -> dict[str, float | int]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"n": 0}
    qs = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": int(len(v)),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "min": float(v.min()),
        "max": float(v.max()),
        "p05": float(qs[0]),
        "p25": float(qs[1]),
        "p50": float(qs[2]),
        "p75": float(qs[3]),
        "p95": float(qs[4]),
    }


def octave_adjusted_error(pred: np.ndarray, true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    diffs = np.stack([np.abs(pred - true - 1200.0 * k) for k in OCTAVE_KS], axis=0)
    best_i = diffs.argmin(axis=0)
    return diffs.min(axis=0), np.array(OCTAVE_KS)[best_i]


def tonic_align_cqt(cqt: np.ndarray, tonic_hz: float, ref_bin: int = REF_TONIC_BIN) -> np.ndarray:
    """Shift frequency axis so tonic maps to ref_bin. Pad/crop, no wrap."""
    src = float(bin_from_hz(tonic_hz))
    shift = int(round(ref_bin - src))
    f_bins, t_bins = cqt.shape
    out = np.zeros_like(cqt)
    if shift >= 0:
        n = f_bins - shift
        if n > 0:
            out[shift:, :] = cqt[:n, :]
    else:
        n = f_bins + shift
        if n > 0:
            out[:n, :] = cqt[-shift:, :]
    return out
