"""Step 14: continuous (every native 10ms frame, not just valid-target
frames) Fused+D3 log2-Hz path per recording, for use as a trajectory-model
input feature. Step 12/12.5's Viterbi decode was deliberately restricted to
valid-target frames (a documented simplification -- see decoders.py's
docstring); here the SAME frozen model checkpoints, salience computation,
and Viterbi code (`_states_for_frame`/`viterbi_decode`) are applied to the
FULL frame grid instead, using the exact per-fold (alpha, beta, lambda_t)
already validation-selected in Step 12/12.5 -- no new grid search, no
decoder change. Over the full grid every consecutive pair is exactly one
native 10ms hop apart, so dt_steps is uniformly 1.0 (simpler than the
valid-only, time-gap-aware case, not a new mechanism).

Leakage: each recording is decoded using the checkpoint from the register-
resolution fold that held it out as TEST (same grouped-kfold manifest/seed
as the trajectory classifier uses), so this single cache is reusable,
leakage-safe, as an input feature regardless of which trajectory-training
fold later treats that recording as train, val, or test.
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
from training.pitch_diagnostics.register_resolution.common import (  # noqa: E402
    candidate_hz, load_learned_model, native_range,
)
from training.pitch_diagnostics.register_resolution.decoders import _states_for_frame, viterbi_decode  # noqa: E402
from training.pitch_diagnostics.relative_pitch.path_cache import REL_DIR, _load_fixed_hyperparams  # noqa: E402

DENSE_CACHE_PATH = REL_DIR / "dense_fused_d3_log2hz.pkl"


def _fused_probs(hps_probs: np.ndarray, learned_probs: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    score = alpha * np.log(np.maximum(learned_probs, 1e-12)) + beta * np.log(np.maximum(hps_probs, 1e-12))
    score = score - score.max(axis=0, keepdims=True)
    probs = np.exp(score)
    probs /= np.maximum(probs.sum(axis=0, keepdims=True), 1e-12)
    return probs


def build(force: bool = False) -> dict[str, np.ndarray]:
    if DENSE_CACHE_PATH.exists() and not force:
        print(f"Loading cached dense pitch paths from {DENSE_CACHE_PATH}")
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
            dt_steps = np.ones(n, dtype=np.float64)
            states = [_states_for_frame(fused_probs[:, t], cand_cents_learned, lrn_lo) for t in range(n)]
            path_abs_cents = viterbi_decode(states, dt_steps, h["fused_lambda_t"], 0.0)
            path_log2_hz = (path_abs_cents / 1200.0).astype(np.float32)

            if n < frames["frame_time_s"].shape[0]:
                pad = frames["frame_time_s"].shape[0] - n
                path_log2_hz = np.pad(path_log2_hz, (0, pad), mode="edge")
            out[rec_id] = path_log2_hz
            print(f"  fold {fold} {rec_id}: n_frames={len(path_log2_hz)}")

    REL_DIR.mkdir(parents=True, exist_ok=True)
    with open(DENSE_CACHE_PATH, "wb") as fh:
        pickle.dump(out, fh)
    print(f"Saved dense pitch paths for {len(out)} recordings to {DENSE_CACHE_PATH}")
    return out


if __name__ == "__main__":
    paths = build(force=True)
    total = sum(len(v) for v in paths.values())
    print(f"Total recordings: {len(paths)}, total frames: {total}")
