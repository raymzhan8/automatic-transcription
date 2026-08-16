"""Step 12 spec sections 9-10: decoder-free upper bounds. Oracle top-k (§9)
uses the pooled cache (top5 candidates already stored). Octave-oracle (§10)
needs a fresh full-probs pass restricted to a +/-600c band around the true
pitch, since that's not part of the cached top-5.
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
from training.pitch_diagnostics.common import OUT_DIR, PRIMARY_LANE, linear_mag, octave_adjusted_error, write_json  # noqa: E402
from training.pitch_diagnostics.hps_salience import hps_salience_probs  # noqa: E402
from training.pitch_diagnostics.register_resolution.collect import build  # noqa: E402
from training.pitch_diagnostics.register_resolution.common import (  # noqa: E402
    FULL_HI_BIN, FULL_LO_BIN, REG_DIR, candidate_hz, extended_pitch_metrics, load_learned_model, native_range,
)


def oracle_topk_from_cache() -> dict:
    records = build()
    out = {}
    for method in ("hps", "learned"):
        argmax_pred, argmax_true = [], []
        oracle_preds = {2: [], 3: [], 5: []}
        for rec in records:
            true = rec["true_cents"]
            argmax_pred.append(rec[method]["argmax_cents"]); argmax_true.append(true)
            top5c = rec[method]["top5_cents"]  # [5,T]
            for k in (2, 3, 5):
                cand = top5c[:k]  # [k,T]
                diffs = np.abs(cand - true[None, :])
                best = np.argmin(diffs, axis=0)
                oracle_preds[k].append(cand[best, np.arange(len(true))])
        ap, at = np.concatenate(argmax_pred), np.concatenate(argmax_true)
        entry = {"argmax": extended_pitch_metrics(ap, at)}
        for k in (2, 3, 5):
            op = np.concatenate(oracle_preds[k])
            entry[f"oracle_top{k}"] = extended_pitch_metrics(op, at)
        out[method] = entry
    return out


def octave_oracle() -> dict:
    """If the correct octave were known, best candidate salience picks within
    that +/-600c band. This is a genuinely different (usually better) upper
    bound than octave_adjusted_error, which just re-scores the SAME argmax
    pick rather than re-searching within the true octave."""
    index = RecordingLaneIndex.build(REPO_ROOT)
    manifest = load_kfold_manifest(REPO_ROOT)
    cand_cents_hps = 1200.0 * np.log2(candidate_hz(FULL_LO_BIN, FULL_HI_BIN))
    lrn_lo, lrn_hi = native_range()
    cand_cents_lrn = 1200.0 * np.log2(candidate_hz(lrn_lo, lrn_hi))

    results = {"hps": [], "learned": []}
    for fold in range(5):
        split = build_fold_split(manifest, fold, seed=42)
        model, _ckpt = load_learned_model("harmonic", fold, lrn_lo, lrn_hi)
        mu, sigma = load_fold_cqt_stats(fold, REPO_ROOT)
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
            true_cents_abs = (1200.0 * frames["pitch_log2_hz"][:n].astype(np.float64))[valid]
            true_rel = true_cents_abs - tonic_term

            mag = linear_mag(cqt_log)
            hps_probs = hps_salience_probs(mag, FULL_LO_BIN, FULL_HI_BIN)[:, valid]  # [360,Tv]
            spec = normalize_cqt(cqt_log, mu, sigma).astype(np.float32)
            spec_t = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = model(spec_t)[0].numpy()
            logits -= logits.max(axis=0, keepdims=True)
            learned_probs = np.exp(logits)
            learned_probs /= np.maximum(learned_probs.sum(axis=0, keepdims=True), 1e-12)
            learned_probs = learned_probs[:, valid]

            for method, probs, cand_cents in (("hps", hps_probs, cand_cents_hps), ("learned", learned_probs, cand_cents_lrn)):
                band = np.abs(cand_cents[:, None] - true_cents_abs[None, :]) <= 600.0
                masked = np.where(band, probs, -1.0)
                if not band.any(axis=0).all():
                    # true octave band falls entirely outside this method's candidate range for some frames
                    masked[:, ~band.any(axis=0)] = probs[:, ~band.any(axis=0)]
                idx = masked.argmax(axis=0)
                pred_abs = cand_cents[idx]
                pred_rel = pred_abs - tonic_term
                results[method].append((pred_rel, true_rel))

    out = {}
    for method in ("hps", "learned"):
        preds = np.concatenate([p for p, _ in results[method]])
        trues = np.concatenate([t for _, t in results[method]])
        oct_err, _ = octave_adjusted_error(preds, trues)
        out[method] = extended_pitch_metrics(preds, trues)
        out[method]["note"] = "candidate search restricted to +/-600c band around true pitch (oracle octave), diagnostic only"
        out[method]["octave_adjusted_mae_of_oracle"] = float(oct_err.mean())
    return out


def main() -> None:
    topk = oracle_topk_from_cache()
    write_json(REG_DIR / "oracle_topk.json", topk)
    print("oracle_topk done:", {m: round(topk[m]["argmax"]["mae_cents"], 1) for m in topk})

    oo = octave_oracle()
    write_json(REG_DIR / "octave_oracle.json", oo)
    print("octave_oracle done:", {m: round(oo[m]["mae_cents"], 1) for m in oo})


if __name__ == "__main__":
    main()
