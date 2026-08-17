"""Step 16: shared per-recording data bundle for the fine-contour acoustic
pitch audit. No new model training, no new decoding -- reuses Step 13's
frozen dense Fused+D3 path (log2Hz, every native frame) and GT annotations
from RecordingLaneIndex directly.

Pitch sources audited: GT parametric pitch and Fused+D3 estimated pitch
(spec section 1's "at minimum"). HPS+D3 is not included -- no dense
per-frame HPS+D3 cache exists (Steps 12/12.5 only cached aggregated
metrics for it), and the spec only asks for it "if the required cached
outputs already exist."
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.folds import build_fold_split, load_kfold_manifest  # noqa: E402
from training.framewise_dataset import PRIMARY_LANE, RecordingLaneIndex  # noqa: E402
from training.normalization import log2_hz_to_cents  # noqa: E402
from training.pitch_diagnostics.common import OUT_DIR, write_json  # noqa: E402
from training.relative_pitch_features import load_dense_estimated_pitch, octave_unwrap  # noqa: E402

AUDIT_DIR = OUT_DIR / "pitch_audit"
NATIVE_HOP_S = 0.01
LARGE_FAILURE_RECORDINGS = ["6417585554a0bfbd8de2d3ff", "6824de49abc4705438ce918b"]
OUTLIER_RECORDING = "692ed7e6213d07041b95a80d"


def recording_fold_map(repo_root: Path = REPO_ROOT) -> dict[str, int]:
    manifest = load_kfold_manifest(repo_root)
    out = {}
    for fold in range(5):
        split = build_fold_split(manifest, fold, seed=42)
        for rid in split.test_recording_ids:
            out[rid] = fold
    return out


def _load_estimated_pitch_variant(variant: str) -> dict[str, np.ndarray]:
    """variant='D1' (default): Step 13's frozen Fused+D3 Viterbi decode.
    variant='D0': Step 17's framewise-independent argmax over the SAME
    frozen fused salience -- no temporal transition cost. Both built from
    identical per-fold checkpoints/fusion hyperparameters; D0 differs only
    in skipping the Viterbi call (spec section 2)."""
    if variant == "D1":
        return load_dense_estimated_pitch()
    if variant == "D0":
        from training.pitch_diagnostics.relative_pitch.dense_framewise_argmax_path import build as build_d0
        return build_d0()
    if variant in ("0.25x", "0.5x"):
        from training.pitch_diagnostics.relative_pitch.dense_lambda_sweep_path import build as build_sweep
        return build_sweep(float(variant.rstrip("x")))
    if variant == "CREPE":
        from training.pitch_diagnostics.relative_pitch.dense_crepe_path import build as build_crepe
        return build_crepe()
    raise ValueError(f"unknown pitch-path variant {variant!r}")


def build_bundles(repo_root: Path = REPO_ROOT, variant: str = "D1") -> list[dict[str, Any]]:
    """One bundle per recording, full native grid (not valid-only), so
    boundary/phase/lag analyses have real temporal continuity to work with.
    `variant` selects the estimated-pitch source (D0 or D1); GT and every
    other field are identical regardless."""
    index = RecordingLaneIndex.build(repo_root)
    estimated_pitch = _load_estimated_pitch_variant(variant)
    fold_map = recording_fold_map(repo_root)

    bundles = []
    for lane in index.lanes:
        rid = lane.recording_id
        frames = index._frames[(rid, PRIMARY_LANE)]
        n = min(len(estimated_pitch[rid]), lane.n_frames, len(frames["frame_time_s"]))
        times = frames["frame_time_s"][:n]
        valid = frames["valid_target"][:n] & (times < lane.duration_s)
        tonic_hz = lane.fundamental_hz

        gt_log2 = frames["pitch_log2_hz"][:n].astype(np.float64)
        gt_cents = log2_hz_to_cents(gt_log2, tonic_hz)
        est_log2 = estimated_pitch[rid][:n].astype(np.float64)
        est_cents = log2_hz_to_cents(est_log2, tonic_hz)

        bundles.append({
            "recording_id": rid,
            "variant": variant,
            "fold": fold_map.get(rid),
            "n": n,
            "times": times,
            "valid": valid,
            "gt_cents": gt_cents,
            "est_cents": est_cents,
            "trajectory_type": frames["trajectory_type"][:n],
            "dp_dt_cents_s": frames["dp_dt_log2_hz_per_s"][:n].astype(np.float64) * 1200.0,
            "tonic_hz": tonic_hz,
            "duration_s": lane.duration_s,
            "primitives": index.primitives_for_recording(rid),
            "boundary_start_frame_indices": frames["boundary_start_frame_indices"],
            "boundary_end_frame_indices": frames["boundary_end_frame_indices"],
        })
    return bundles


def delta_at_offset(cents: np.ndarray, k: int) -> np.ndarray:
    """[n] -> [n], delta[t] = octave_unwrap(cents[t]-cents[t-k]) for t>=k, else NaN."""
    n = len(cents)
    out = np.full(n, np.nan)
    if k < n:
        raw = cents[k:] - cents[:-k]
        out[k:] = octave_unwrap(raw)
    return out


def both_valid_mask(valid: np.ndarray, k: int) -> np.ndarray:
    n = len(valid)
    out = np.zeros(n, dtype=bool)
    out[k:] = valid[k:] & valid[:-k]
    return out


if __name__ == "__main__":
    bundles = build_bundles()
    total_valid = sum(int(b["valid"].sum()) for b in bundles)
    total_frames = sum(b["n"] for b in bundles)
    print(f"{len(bundles)} recordings, {total_valid} valid frames, {total_frames} total frames")
