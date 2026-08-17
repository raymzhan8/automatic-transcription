"""Step 15 P2: register-invariant windowed relative-salience cache.

For every native-hop frame, extracts a FIXED window of the fused
HPS+learned salience distribution S(f,t) (native 34-244 range, same frozen
checkpoints/hyperparameters as Step 12/12.5/13/14 -- no retraining, no new
decoder), recentered on the frame's own Fused+D3 decoded bin (from the
already-cached dense_fused_d3_log2hz.pkl). Unlike the P1/P0 pitch path,
this does NOT collapse S(f,t) to a single scalar first -- it keeps the
local shape of the distribution (secondary peaks, uncertainty), just
expressed in RELATIVE bin-offset coordinates instead of absolute frequency.

Window: W_BINS = 73 (+/-36 bins, ~600c each side at 16.67c/bin) -- wide
enough to hold nearby competing pitch-class candidates (Step 12's median
GT rank was in the single digits when the octave is correct, i.e. competing
candidates are typically close in frequency, not full octaves away), narrow
enough to stay small and interpretable per spec section 7.

Register invariance argument (spec section 7): a sustained absolute-pitch
offset (correct or wrong-by-a-constant-octave) shifts BOTH the true
candidate distribution AND the recentering reference by the same amount,
so their difference -- the windowed relative view -- is unchanged. This is
exactly the differencing-cancels-sustained-octave-error mechanism Step 13
established for the 1-D decoded path, applied here to the 2-D distribution
instead of a single scalar.

Cache: {recording_id: float32 array [W_BINS, n_frames]}, one probability
column (renormalized within the window) per native frame -- large
(~530 MB total across the corpus) but computed once.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.folds import build_fold_split, load_kfold_manifest  # noqa: E402
from training.framewise_dataset import RecordingLaneIndex  # noqa: E402
from training.normalization import load_fold_cqt_stats, normalize_cqt  # noqa: E402
from training.pitch_diagnostics.common import OUT_DIR, PRIMARY_LANE, bin_from_hz, linear_mag  # noqa: E402
from training.pitch_diagnostics.hps_salience import hps_salience_probs  # noqa: E402
from training.pitch_diagnostics.register_resolution.common import load_learned_model, native_range  # noqa: E402
from training.relative_pitch_features import load_dense_estimated_pitch  # noqa: E402
from training.pitch_diagnostics.relative_pitch.path_cache import REL_DIR, _load_fixed_hyperparams  # noqa: E402

CACHE_PATH = REL_DIR / "dense_relative_salience.pkl"
W_BINS = 73
HALF_W = W_BINS // 2


def _fused_probs(hps_probs: np.ndarray, learned_probs: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    score = alpha * np.log(np.maximum(learned_probs, 1e-12)) + beta * np.log(np.maximum(hps_probs, 1e-12))
    score = score - score.max(axis=0, keepdims=True)
    probs = np.exp(score)
    probs /= np.maximum(probs.sum(axis=0, keepdims=True), 1e-12)
    return probs


def _window_extract(probs: np.ndarray, lo_bin: int, ref_bin: np.ndarray) -> np.ndarray:
    """probs: [F, T] over native bins [lo_bin, lo_bin+F). ref_bin: [T]
    absolute CQT bin (float). Returns [W_BINS, T], zero outside the native
    range. NOT renormalized to sum to 1 -- the window's total captured mass
    (<1 when salience spills outside +/-600c or the reference sits near the
    native-range edge) is itself an uncertainty signal, deliberately kept
    rather than normalized away."""
    F, T = probs.shape
    hi_bin = lo_bin + F
    ref = np.rint(ref_bin).astype(np.int64)
    out = np.zeros((W_BINS, T), dtype=np.float32)
    offsets = np.arange(-HALF_W, -HALF_W + W_BINS)
    for j, off in enumerate(offsets):
        abs_bin = ref + off
        valid = (abs_bin >= lo_bin) & (abs_bin < hi_bin)
        rel_bin = np.clip(abs_bin - lo_bin, 0, F - 1)
        col = probs[rel_bin, np.arange(T)]
        out[j] = np.where(valid, col, 0.0)
    return out


def build(force: bool = False) -> dict[str, np.ndarray]:
    if CACHE_PATH.exists() and not force:
        print(f"Loading cached relative salience from {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as fh:
            return pickle.load(fh)

    hp = _load_fixed_hyperparams()
    index = RecordingLaneIndex.build(REPO_ROOT)
    manifest = load_kfold_manifest(REPO_ROOT)
    lrn_lo, lrn_hi = native_range()
    dense_pitch = load_dense_estimated_pitch()

    out: dict[str, np.ndarray] = {}
    for fold in range(5):
        split = build_fold_split(manifest, fold, seed=42)
        model, _ckpt = load_learned_model("harmonic", fold, lrn_lo, lrn_hi)
        mu, sigma = load_fold_cqt_stats(fold, REPO_ROOT)
        h = hp[fold]

        for rec_id in split.test_recording_ids:
            lane = next(x for x in index.lanes if x.recording_id == rec_id)
            cqt_log = index._features[rec_id]["cqt_log"]
            n = min(cqt_log.shape[1], lane.n_frames)
            cqt_log = cqt_log[:, :n]

            mag = linear_mag(cqt_log)
            hps_probs = hps_salience_probs(mag, lrn_lo, lrn_hi)
            spec = normalize_cqt(cqt_log, mu, sigma).astype(np.float32)
            spec_t = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = model(spec_t)[0].numpy()
            logits = logits - logits.max(axis=0, keepdims=True)
            learned_probs = np.exp(logits)
            learned_probs /= np.maximum(learned_probs.sum(axis=0, keepdims=True), 1e-12)

            fused_probs = _fused_probs(hps_probs, learned_probs, *h["fusion_ratio"])
            ref_log2hz = dense_pitch[rec_id][:n]
            ref_bin = bin_from_hz(np.exp2(ref_log2hz))
            windowed = _window_extract(fused_probs, lrn_lo, ref_bin)
            out[rec_id] = windowed
            print(f"  fold {fold} {rec_id}: shape={windowed.shape}")

    REL_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    print(f"Saved relative salience for {len(out)} recordings to {CACHE_PATH}")
    return out


if __name__ == "__main__":
    paths = build(force=True)
    total = sum(v.shape[1] for v in paths.values())
    print(f"Total recordings: {len(paths)}, total frames: {total}")
