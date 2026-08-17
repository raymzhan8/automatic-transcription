"""Step 17 section 17: minimal diagnostic sweep over the Viterbi movement-
cost weight lambda_t (`training/pitch_diagnostics/register_resolution/
decoders.py::viterbi_decode`'s `trans = lambda_t * move`, move = capped
|delta_cents|/dt_steps -- see decoders.py). Builds dense paths at a SMALL
set of multipliers of each fold's already-validation-selected fused_lambda_t
(Step 12.5) -- 0x is D0 (already built, no Viterbi), 1x is D1 (already
built, current system); this script adds ONLY the intermediate 0.25x/0.5x
points. Not a hyperparameter search -- lambda_t is scaled uniformly, not
re-optimized per fold.
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
from training.pitch_diagnostics.register_resolution.decoders import _states_for_frame, viterbi_decode  # noqa: E402
from training.pitch_diagnostics.relative_pitch.path_cache import REL_DIR, _load_fixed_hyperparams  # noqa: E402

MULTIPLIERS = (0.25, 0.5)


def _fused_probs(hps_probs, learned_probs, alpha, beta):
    score = alpha * np.log(np.maximum(learned_probs, 1e-12)) + beta * np.log(np.maximum(hps_probs, 1e-12))
    score = score - score.max(axis=0, keepdims=True)
    probs = np.exp(score)
    probs /= np.maximum(probs.sum(axis=0, keepdims=True), 1e-12)
    return probs


def cache_path(mult: float) -> Path:
    return REL_DIR / f"dense_lambda_{mult}x_log2hz.pkl"


def build(mult: float, force: bool = False) -> dict[str, np.ndarray]:
    path = cache_path(mult)
    if path.exists() and not force:
        with open(path, "rb") as fh:
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
        lambda_t = h["fused_lambda_t"] * mult

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
            path_abs_cents = viterbi_decode(states, dt_steps, lambda_t, 0.0)
            path_log2_hz = (path_abs_cents / 1200.0).astype(np.float32)

            if n < frames["frame_time_s"].shape[0]:
                pad = frames["frame_time_s"].shape[0] - n
                path_log2_hz = np.pad(path_log2_hz, (0, pad), mode="edge")
            out[rec_id] = path_log2_hz
        print(f"  fold {fold} done (lambda_t x{mult})")

    REL_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(out, fh)
    print(f"Saved {mult}x paths to {path}")
    return out


if __name__ == "__main__":
    for m in MULTIPLIERS:
        build(m, force=True)
