"""Step 12 spec §11: training-only static register prior. Per fold, estimate
P_train(f) (absolute-Hz pitch-bin histogram, smoothed, over the full 360-bin
CQT axis) from TRAIN recordings only, select lambda on VALIDATION only from a
small grid, then apply to TEST. Decode: argmax_f [ log S(f,t) + lambda *
log P_train(f) ]. HPS is scored over the full range; the learned model is
scored over its native trained range (34-244, see common.py::native_range /
candidate_range_fixed.py for why) -- the prior itself is always computed over
the full 360 bins and sliced to whichever sub-range a given method uses.
"""

from __future__ import annotations

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
from training.pitch_diagnostics.common import OUT_DIR, PRIMARY_LANE, bin_from_hz, linear_mag, write_json  # noqa: E402
from training.pitch_diagnostics.hps_salience import hps_salience_probs  # noqa: E402
from training.pitch_diagnostics.register_resolution.common import (  # noqa: E402
    FULL_HI_BIN, FULL_LO_BIN, REG_DIR, candidate_hz, extended_pitch_metrics, load_learned_model, native_range,
)

LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0)
SMOOTH_BINS = 3  # simple moving-average smoothing half-width for the histogram prior


def train_prior_full(train_ids: list[str], index: RecordingLaneIndex) -> np.ndarray:
    """Always over the full 360-bin axis -- a physical fact about where
    pitches occur, independent of any model's candidate window."""
    counts = np.zeros(FULL_HI_BIN - FULL_LO_BIN, dtype=np.float64)
    for rid in train_ids:
        frames = index._frames[(rid, PRIMARY_LANE)]
        valid = frames["valid_target"]
        if not np.any(valid):
            continue
        hz = np.exp2(frames["pitch_log2_hz"][valid].astype(np.float64))
        bins = np.clip(np.rint(np.asarray(bin_from_hz(hz))), FULL_LO_BIN, FULL_HI_BIN - 1).astype(np.int64)
        np.add.at(counts, bins - FULL_LO_BIN, 1.0)
    kernel = np.ones(2 * SMOOTH_BINS + 1) / (2 * SMOOTH_BINS + 1)
    smoothed = np.convolve(counts, kernel, mode="same") + 1.0  # additive smoothing, avoid log(0)
    return smoothed / smoothed.sum()


def _recording_probs(rec_id: str, index: RecordingLaneIndex, model, mu, sigma, lrn_lo: int, lrn_hi: int):
    lane = next(x for x in index.lanes if x.recording_id == rec_id)
    frames = index._frames[(rec_id, PRIMARY_LANE)]
    cqt_log = index._features[rec_id]["cqt_log"]
    n = min(cqt_log.shape[1], lane.n_frames)
    cqt_log = cqt_log[:, :n]
    times = frames["frame_time_s"][:n]
    valid = frames["valid_target"][:n] & (times < lane.duration_s)
    tonic_term = 1200.0 * np.log2(lane.fundamental_hz)
    true_rel = (1200.0 * frames["pitch_log2_hz"][:n].astype(np.float64) - tonic_term)[valid]

    mag = linear_mag(cqt_log)
    hps_probs = hps_salience_probs(mag, FULL_LO_BIN, FULL_HI_BIN)[:, valid]
    spec = normalize_cqt(cqt_log, mu, sigma).astype(np.float32)
    spec_t = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(spec_t)[0].numpy()
    logits -= logits.max(axis=0, keepdims=True)
    learned_probs = np.exp(logits)
    learned_probs /= np.maximum(learned_probs.sum(axis=0, keepdims=True), 1e-12)
    learned_probs = learned_probs[:, valid]
    return hps_probs, learned_probs, true_rel, tonic_term


def rescored_argmax(probs: np.ndarray, cand_cents_abs: np.ndarray, log_prior_slice: np.ndarray, lam: float, tonic_term: float) -> np.ndarray:
    score = np.log(np.maximum(probs, 1e-12)) + lam * log_prior_slice[:, None]
    idx = score.argmax(axis=0)
    return cand_cents_abs[idx] - tonic_term


def main() -> None:
    index = RecordingLaneIndex.build(REPO_ROOT)
    manifest = load_kfold_manifest(REPO_ROOT)
    cand_cents_hps = 1200.0 * np.log2(candidate_hz(FULL_LO_BIN, FULL_HI_BIN))
    lrn_lo, lrn_hi = native_range()
    cand_cents_lrn = 1200.0 * np.log2(candidate_hz(lrn_lo, lrn_hi))

    out = {
        "lambdas_tried": list(LAMBDAS),
        "per_fold": [],
        "method": "absolute-Hz pitch-bin histogram over TRAIN frames (full 360-bin axis), +1 additive smoothing, 3-bin moving average",
        "ranges": {"hps": [FULL_LO_BIN, FULL_HI_BIN], "learned": [lrn_lo, lrn_hi]},
    }
    pooled = {m: {"raw": {"pred": [], "true": []}, "prior": {"pred": [], "true": []}} for m in ("hps", "learned")}

    for fold in range(5):
        split = build_fold_split(manifest, fold, seed=42)
        prior_full = train_prior_full(split.train_recording_ids, index)
        log_prior_full = np.log(prior_full)
        log_prior_hps = log_prior_full  # full range, no slicing needed
        log_prior_lrn = log_prior_full[lrn_lo:lrn_hi]
        model, _ckpt = load_learned_model("harmonic", fold, lrn_lo, lrn_hi)
        mu, sigma = load_fold_cqt_stats(fold, REPO_ROOT)

        fold_entry = {"fold": fold, "n_train": len(split.train_recording_ids)}
        for method, cand_cents, log_prior in (("hps", cand_cents_hps, log_prior_hps), ("learned", cand_cents_lrn, log_prior_lrn)):
            val_scores = {lam: {"pred": [], "true": []} for lam in LAMBDAS}
            for rid in split.val_recording_ids:
                hps_p, learned_p, true_rel, tonic = _recording_probs(rid, index, model, mu, sigma, lrn_lo, lrn_hi)
                probs = hps_p if method == "hps" else learned_p
                for lam in LAMBDAS:
                    pred = rescored_argmax(probs, cand_cents, log_prior, lam, tonic)
                    val_scores[lam]["pred"].append(pred); val_scores[lam]["true"].append(true_rel)
            best_lam, best_mae = 0.0, float("inf")
            val_mae_by_lambda = {}
            for lam in LAMBDAS:
                if not val_scores[lam]["pred"]:
                    continue
                p = np.concatenate(val_scores[lam]["pred"]); t = np.concatenate(val_scores[lam]["true"])
                mae = float(np.abs(p - t).mean())
                val_mae_by_lambda[lam] = mae
                if mae < best_mae:
                    best_mae, best_lam = mae, lam
            fold_entry[f"{method}_val_mae_by_lambda"] = val_mae_by_lambda
            fold_entry[f"{method}_best_lambda"] = best_lam

            for rid in split.test_recording_ids:
                hps_p, learned_p, true_rel, tonic = _recording_probs(rid, index, model, mu, sigma, lrn_lo, lrn_hi)
                probs = hps_p if method == "hps" else learned_p
                raw_pred = cand_cents[probs.argmax(axis=0)] - tonic
                prior_pred = rescored_argmax(probs, cand_cents, log_prior, best_lam, tonic)
                pooled[method]["raw"]["pred"].append(raw_pred); pooled[method]["raw"]["true"].append(true_rel)
                pooled[method]["prior"]["pred"].append(prior_pred); pooled[method]["prior"]["true"].append(true_rel)
        out["per_fold"].append(fold_entry)
        print(f"fold {fold} done, best lambdas: hps={fold_entry['hps_best_lambda']} learned={fold_entry['learned_best_lambda']}")

    summary = {}
    for method in ("hps", "learned"):
        raw_p = np.concatenate(pooled[method]["raw"]["pred"]); raw_t = np.concatenate(pooled[method]["raw"]["true"])
        prior_p = np.concatenate(pooled[method]["prior"]["pred"]); prior_t = np.concatenate(pooled[method]["prior"]["true"])
        summary[method] = {"raw_argmax": extended_pitch_metrics(raw_p, raw_t), "static_prior_decode": extended_pitch_metrics(prior_p, prior_t)}
    out["test_summary"] = summary
    write_json(REG_DIR / "static_register_prior.json", out)
    print("=== static prior summary (test, pooled) ===")
    for m, v in summary.items():
        print(m, "raw MAE", round(v["raw_argmax"]["mae_cents"], 1), "prior MAE", round(v["static_prior_decode"]["mae_cents"], 1))


if __name__ == "__main__":
    main()
