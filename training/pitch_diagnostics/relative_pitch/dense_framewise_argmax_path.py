"""Step 17 D0: framewise-independent decode from the SAME frozen fused
salience evidence D1 (Fused+D3, dense_pitch_path.py) uses -- argmax per
frame, no temporal transition cost, no future-frame information. Isolates
temporal (Viterbi) decoding as the only difference between D0 and D1: both
are built from the identical per-fold fused_probs (same checkpoints, same
fusion hyperparameters, same candidate range) computed by the identical
forward pass; D0 skips the `viterbi_decode` call D1 makes and takes a plain
per-column argmax instead (spec section 2's "D0b = fused framewise
candidate argmax").
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
from training.pitch_diagnostics.common import OUT_DIR, PRIMARY_LANE, linear_mag  # noqa: E402
from training.pitch_diagnostics.hps_salience import hps_salience_probs  # noqa: E402
from training.pitch_diagnostics.register_resolution.common import candidate_hz, load_learned_model, native_range  # noqa: E402
from training.pitch_diagnostics.relative_pitch.path_cache import REL_DIR, _load_fixed_hyperparams  # noqa: E402

DENSE_CACHE_PATH = REL_DIR / "dense_framewise_argmax_log2hz.pkl"


def _fused_probs(hps_probs: np.ndarray, learned_probs: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    score = alpha * np.log(np.maximum(learned_probs, 1e-12)) + beta * np.log(np.maximum(hps_probs, 1e-12))
    score = score - score.max(axis=0, keepdims=True)
    probs = np.exp(score)
    probs /= np.maximum(probs.sum(axis=0, keepdims=True), 1e-12)
    return probs


def build(force: bool = False) -> dict[str, np.ndarray]:
    if DENSE_CACHE_PATH.exists() and not force:
        print(f"Loading cached D0 framewise-argmax paths from {DENSE_CACHE_PATH}")
        with open(DENSE_CACHE_PATH, "rb") as fh:
            return pickle.load(fh)

    hp = _load_fixed_hyperparams()
    index = RecordingLaneIndex.build(REPO_ROOT)
    manifest = load_kfold_manifest(REPO_ROOT)
    lrn_lo, lrn_hi = native_range()
    cand_cents_learned = 1200.0 * np.log2(candidate_hz(lrn_lo, lrn_hi))

    out: dict[str, np.ndarray] = {}
    for fold in range(5):
        split = build_fold_split(manifest, fold, seed=42)
        model, _ckpt = load_learned_model("harmonic", fold, lrn_lo, lrn_hi)
        mu, sigma = load_fold_cqt_stats(fold, REPO_ROOT)
        h = hp[fold]

        for rec_id in split.test_recording_ids:
            lane = next(x for x in index.lanes if x.recording_id == rec_id)
            frames = index._frames[(rec_id, PRIMARY_LANE)]
            cqt_log = index._features[rec_id]["cqt_log"]
            n = min(cqt_log.shape[1], lane.n_frames)
            cqt_log = cqt_log[:, :n]

            mag = linear_mag(cqt_log)
            hps_probs_shared = hps_salience_probs(mag, lrn_lo, lrn_hi)
            spec = normalize_cqt(cqt_log, mu, sigma).astype(np.float32)
            spec_t = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = model(spec_t)[0].numpy()
            logits = logits - logits.max(axis=0, keepdims=True)
            learned_probs = np.exp(logits)
            learned_probs /= np.maximum(learned_probs.sum(axis=0, keepdims=True), 1e-12)

            fused_probs = _fused_probs(hps_probs_shared, learned_probs, *h["fusion_ratio"])
            idx = fused_probs.argmax(axis=0)
            path_abs_cents = cand_cents_learned[idx]
            path_log2_hz = (path_abs_cents / 1200.0).astype(np.float32)

            if n < frames["frame_time_s"].shape[0]:
                pad = frames["frame_time_s"].shape[0] - n
                path_log2_hz = np.pad(path_log2_hz, (0, pad), mode="edge")
            out[rec_id] = path_log2_hz
            print(f"  fold {fold} {rec_id}: n_frames={len(path_log2_hz)}")

    REL_DIR.mkdir(parents=True, exist_ok=True)
    with open(DENSE_CACHE_PATH, "wb") as fh:
        pickle.dump(out, fh)
    print(f"Saved D0 framewise-argmax paths for {len(out)} recordings to {DENSE_CACHE_PATH}")
    return out


if __name__ == "__main__":
    paths = build(force=True)
    total = sum(len(v) for v in paths.values())
    print(f"Total recordings: {len(paths)}, total frames: {total}")
