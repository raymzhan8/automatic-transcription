"""Step 19: one shared forward pass per recording extracting everything the
predecoder audit needs at every VALID frame -- raw CQT-based acoustic
evidence (stage A), fused salience (stage S=C, since this pipeline's
"framewise candidate score" and "salience" are the identical array -- see
docs/step_19 section 1), and D0's already-cached decoded selection.
No new model, no new decoding: reuses the exact same per-fold checkpoints,
fusion hyperparameters, and candidate range as every prior pitch_diagnostics
step.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.folds import build_fold_split, load_kfold_manifest  # noqa: E402
from training.framewise_dataset import PRIMARY_LANE, RecordingLaneIndex  # noqa: E402
from training.normalization import load_fold_cqt_stats, log2_hz_to_cents, normalize_cqt  # noqa: E402
from training.pitch_diagnostics.common import bin_from_hz, linear_mag  # noqa: E402
from training.pitch_diagnostics.hps_salience import hps_salience_probs  # noqa: E402
from training.pitch_diagnostics.register_resolution.common import candidate_hz, load_learned_model, native_range  # noqa: E402
from training.pitch_diagnostics.relative_pitch.dense_framewise_argmax_path import build as build_d0  # noqa: E402
from training.pitch_diagnostics.relative_pitch.path_cache import _load_fixed_hyperparams  # noqa: E402
from training.relative_pitch_features import octave_unwrap  # noqa: E402

CENTS_PER_BIN = 16.6666667
LRN_LO, LRN_HI = native_range()  # (34, 244) -- the shared candidate window for S=C throughout


def _fused_probs(hps_probs, learned_probs, alpha, beta):
    score = alpha * np.log(np.maximum(learned_probs, 1e-12)) + beta * np.log(np.maximum(hps_probs, 1e-12))
    score = score - score.max(axis=0, keepdims=True)
    probs = np.exp(score)
    probs /= np.maximum(probs.sum(axis=0, keepdims=True), 1e-12)
    return probs


def iter_recording_records(repo_root: Path = REPO_ROOT):
    """Generator, one record per recording (full native grid, so
    both_valid_mask/delta_at_offset's time-gap-aware NaN gating stays
    correct -- matches every prior step's convention). Heavy 2D arrays
    (mag, fused_probs) are ~200-500MB for the largest recordings; yielding
    one at a time keeps peak memory bounded to a single recording instead
    of holding all 17 simultaneously."""
    hp = _load_fixed_hyperparams()
    index = RecordingLaneIndex.build(repo_root)
    manifest = load_kfold_manifest(repo_root)
    d0_pitch = build_d0()
    cand_cents = 1200.0 * np.log2(candidate_hz(LRN_LO, LRN_HI))

    for fold in range(5):
        split = build_fold_split(manifest, fold, seed=42)
        model, _ckpt = load_learned_model("harmonic", fold, LRN_LO, LRN_HI)
        mu, sigma = load_fold_cqt_stats(fold, repo_root)
        h = hp[fold]

        for rid in split.test_recording_ids:
            lane = next(x for x in index.lanes if x.recording_id == rid)
            frames = index._frames[(rid, PRIMARY_LANE)]
            cqt_log = index._features[rid]["cqt_log"]
            n = min(cqt_log.shape[1], lane.n_frames, len(d0_pitch[rid]))
            cqt_log_n = cqt_log[:, :n]
            times = frames["frame_time_s"][:n]
            valid = frames["valid_target"][:n] & (times < lane.duration_s)
            if not np.any(valid):
                continue
            tonic_hz = lane.fundamental_hz
            tonic_term = 1200.0 * np.log2(tonic_hz)

            mag = linear_mag(cqt_log_n)  # [360, n], linear CQT magnitude (stage-A raw evidence)
            hps_probs = hps_salience_probs(mag, LRN_LO, LRN_HI)  # [210, n]
            spec = normalize_cqt(cqt_log_n, mu, sigma).astype(np.float32)
            spec_t = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = model(spec_t)[0].numpy()
            logits = logits - logits.max(axis=0, keepdims=True)
            learned_probs = np.exp(logits)
            learned_probs /= np.maximum(learned_probs.sum(axis=0, keepdims=True), 1e-12)
            fused_probs = _fused_probs(hps_probs, learned_probs, *h["fusion_ratio"])  # [210, n] = S(t,f) = C(t,f)

            gt_log2 = frames["pitch_log2_hz"][:n].astype(np.float64)
            gt_cents_rel = log2_hz_to_cents(gt_log2, tonic_hz)
            gt_bin_abs = bin_from_hz(np.exp2(gt_log2))  # absolute CQT bin index, float

            d0_log2 = d0_pitch[rid][:n].astype(np.float64)
            d0_cents_rel = log2_hz_to_cents(d0_log2, tonic_hz)

            yield {
                "recording_id": rid, "fold": fold, "n": n, "tonic_hz": tonic_hz,
                "times": times, "valid": valid,
                "trajectory_type": frames["trajectory_type"][:n],
                "dp_dt_cents_s": frames["dp_dt_log2_hz_per_s"][:n].astype(np.float64) * 1200.0,
                "gt_cents_rel": gt_cents_rel, "gt_bin_abs": gt_bin_abs,
                "d0_cents_rel": d0_cents_rel,
                "mag": mag,  # [360, n] linear CQT magnitude, full range
                "fused_probs": fused_probs,  # [210, n] over [LRN_LO, LRN_HI)
                "cand_cents": cand_cents,  # [210]
                "primitives": index.primitives_for_recording(rid),
            }
        print(f"  fold {fold} predecoder data built")
