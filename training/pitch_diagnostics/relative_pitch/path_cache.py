"""Step 13: build per-recording decoded absolute-pitch-cent paths for the
five frontends compared in this step (GT is just true_cents, computed for
free). No new decoder machinery, no new grid search -- every Viterbi
lambda_t and every fusion (alpha, beta) below is loaded verbatim from the
already-validated Step 12 / 12.5 artifacts (`decoder_ablation.json`,
`fusion_viterbi_result.json`); this script only re-runs the existing
forward passes (HPS salience product, learned-model forward, existing
`_states_for_frame`/`viterbi_decode`) once each to materialize the raw
per-frame cent sequences those artifacts only ever stored aggregated
metrics for.

Ranges (unchanged from Step 12/12.5, kept because they're each frontend's
valid operating domain -- see register_resolution/common.py::native_range):
 - HPS / HPS+D3: full 0-360 CQT range (this is Step 12's headline HPS
   number, 322.1/317.5c -- the "strongest practical absolute-pitch
   frontend" this step's brief refers to).
 - Learned / Learned+D3: native trained range (34-244).
 - Fused+D3: shared 34-244 window (the only range fusion is defined over).

D4 (octave-jump penalty) is not included here -- Step 12/12.5 already
established it adds ~nothing over D3 for every method, and the brief
explicitly asks for D3, not another decoder variant.
"""

from __future__ import annotations

import json
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
    FULL_HI_BIN, FULL_LO_BIN, REG_DIR, candidate_hz, load_learned_model, native_range,
)
from training.pitch_diagnostics.register_resolution.decoders import _states_for_frame, viterbi_decode  # noqa: E402

REL_DIR = OUT_DIR / "relative_pitch"
CACHE_PATH = REL_DIR / "path_cache.pkl"
NATIVE_HOP_S = 0.01


def _fused_probs(hps_probs: np.ndarray, learned_probs: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    score = alpha * np.log(np.maximum(learned_probs, 1e-12)) + beta * np.log(np.maximum(hps_probs, 1e-12))
    score = score - score.max(axis=0, keepdims=True)
    probs = np.exp(score)
    probs /= np.maximum(probs.sum(axis=0, keepdims=True), 1e-12)
    return probs


def _decode(probs: np.ndarray, cand_cents: np.ndarray, lo: int, dt_steps: np.ndarray, tonic_term: float, lambda_t: float) -> np.ndarray:
    T = probs.shape[1]
    states = [_states_for_frame(probs[:, t], cand_cents, lo) for t in range(T)]
    path_abs = viterbi_decode(states, dt_steps, lambda_t, 0.0)
    return path_abs - tonic_term


def _load_fixed_hyperparams() -> dict:
    decoder_ablation = json.loads((REG_DIR / "decoder_ablation.json").read_text(encoding="utf-8"))
    fusion_vit = json.loads((REG_DIR / "fusion_viterbi_result.json").read_text(encoding="utf-8"))
    by_fold: dict[int, dict] = {}
    for entry in decoder_ablation["per_fold"]:
        f = entry["fold"]
        by_fold.setdefault(f, {})["hps_lambda_t"] = entry["best"]["hps"]["lambda_t"]
        by_fold[f]["learned_lambda_t"] = entry["best"]["learned"]["lambda_t"]
    for entry in fusion_vit["per_fold"]:
        f = entry["fold"]
        by_fold.setdefault(f, {})["fusion_ratio"] = tuple(entry["best_ratio"])
        by_fold[f]["fused_lambda_t"] = entry["fused_best"]["lambda_t"]
    return by_fold


def build(force: bool = False) -> list[dict]:
    if CACHE_PATH.exists() and not force:
        print(f"Loading cached decoded paths from {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as fh:
            return pickle.load(fh)

    hp = _load_fixed_hyperparams()
    index = RecordingLaneIndex.build(REPO_ROOT)
    manifest = load_kfold_manifest(REPO_ROOT)

    lrn_lo, lrn_hi = native_range()
    cand_cents_hps_full = 1200.0 * np.log2(candidate_hz(FULL_LO_BIN, FULL_HI_BIN))
    cand_cents_learned = 1200.0 * np.log2(candidate_hz(lrn_lo, lrn_hi))

    records = []
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
            times = frames["frame_time_s"][:n]
            valid = frames["valid_target"][:n] & (times < lane.duration_s)
            if not np.any(valid):
                continue
            tonic_term = 1200.0 * np.log2(lane.fundamental_hz)
            true_rel = (1200.0 * frames["pitch_log2_hz"][:n].astype(np.float64) - tonic_term)[valid]

            mag = linear_mag(cqt_log)
            hps_probs_full = hps_salience_probs(mag, FULL_LO_BIN, FULL_HI_BIN)[:, valid]
            hps_probs_shared = hps_salience_probs(mag, lrn_lo, lrn_hi)[:, valid]
            spec = normalize_cqt(cqt_log, mu, sigma).astype(np.float32)
            spec_t = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = model(spec_t)[0].numpy()
            logits = logits - logits.max(axis=0, keepdims=True)
            learned_probs = np.exp(logits)
            learned_probs /= np.maximum(learned_probs.sum(axis=0, keepdims=True), 1e-12)
            learned_probs = learned_probs[:, valid]

            t_valid = times[valid]
            dt_steps = np.empty(len(t_valid))
            dt_steps[0] = 1.0
            dt_steps[1:] = np.maximum((t_valid[1:] - t_valid[:-1]) / NATIVE_HOP_S, 1.0)
            dt_seconds = np.empty(len(t_valid))
            dt_seconds[0] = np.nan
            dt_seconds[1:] = t_valid[1:] - t_valid[:-1]

            fused_probs = _fused_probs(hps_probs_shared, learned_probs, *h["fusion_ratio"])

            hps_argmax_cents = cand_cents_hps_full[hps_probs_full.argmax(axis=0)] - tonic_term
            hps_d3_cents = _decode(hps_probs_full, cand_cents_hps_full, FULL_LO_BIN, dt_steps, tonic_term, h["hps_lambda_t"])
            learned_argmax_cents = cand_cents_learned[learned_probs.argmax(axis=0)] - tonic_term
            learned_d3_cents = _decode(learned_probs, cand_cents_learned, lrn_lo, dt_steps, tonic_term, h["learned_lambda_t"])
            fused_d3_cents = _decode(fused_probs, cand_cents_learned, lrn_lo, dt_steps, tonic_term, h["fused_lambda_t"])

            records.append({
                "recording_id": rec_id, "fold": fold, "n_valid": int(valid.sum()),
                "times": t_valid, "dt_seconds": dt_seconds,
                "true_cents": true_rel,
                "trajectory_type": frames["trajectory_type"][:n][valid],
                "dp_dt_cents_s": frames["dp_dt_log2_hz_per_s"][:n][valid] * 1200.0,
                "hps_argmax_cents": hps_argmax_cents, "hps_d3_cents": hps_d3_cents,
                "learned_argmax_cents": learned_argmax_cents, "learned_d3_cents": learned_d3_cents,
                "fused_d3_cents": fused_d3_cents,
            })
            print(f"  fold {fold} {rec_id}: n_valid={int(valid.sum())}")

    REL_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as fh:
        pickle.dump(records, fh)
    print(f"Saved {len(records)} recording paths to {CACHE_PATH}")
    return records


if __name__ == "__main__":
    recs = build(force=True)
    print(f"Total recordings: {len(recs)}, total valid frames: {sum(r['n_valid'] for r in recs)}")
