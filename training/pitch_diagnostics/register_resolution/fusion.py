"""Step 12 spec §12/§21 D2: frame-independent HPS/learned-salience fusion.
Justified by Phase A's disagreement_analysis.json (complementary errors: both
HPS-correct/learned-wrong and learned-correct/HPS-wrong are non-trivial
fractions -- see phase_a_synthesis.json's D2_fusion_justified=True).

R(f,t) = alpha*log S_learned(f,t) + beta*log S_HPS(f,t)   (gamma*log P_train
dropped -- static_prior.py already found the register prior gives a negative
result, D1 not justified, so no register-prior term here). Both maps are
restricted to the learned model's native trained candidate window (34-244,
see common.py::native_range) since that's the only range the learned model is
valid over (per the §2 reconciliation finding) -- HPS is simply sliced to the
same window for a fair frame-by-frame comparison at each candidate bin.

alpha/beta are selected via a SMALL grid (5 mixing ratios) per fold on
VALIDATION only, matching static_prior.py's pattern exactly.
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
from training.pitch_diagnostics.common import PRIMARY_LANE, linear_mag, write_json  # noqa: E402
from training.pitch_diagnostics.hps_salience import hps_salience_probs  # noqa: E402
from training.pitch_diagnostics.register_resolution.common import (  # noqa: E402
    REG_DIR, candidate_hz, extended_pitch_metrics, load_learned_model, native_range,
)

# (alpha, beta) mixing ratios -- alpha weights learned salience, beta weights HPS.
RATIOS = ((1.0, 0.0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.0, 1.0))


def _recording_probs_shared(rec_id, index, model, mu, sigma, lo, hi):
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
    hps_probs = hps_salience_probs(mag, lo, hi)[:, valid]  # sliced to shared window
    spec = normalize_cqt(cqt_log, mu, sigma).astype(np.float32)
    spec_t = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(spec_t)[0].numpy()
    logits -= logits.max(axis=0, keepdims=True)
    learned_probs = np.exp(logits)
    learned_probs /= np.maximum(learned_probs.sum(axis=0, keepdims=True), 1e-12)
    learned_probs = learned_probs[:, valid]
    return hps_probs, learned_probs, true_rel, tonic_term


def fused_argmax(hps_p, learned_p, cand_cents_abs, alpha, beta, tonic_term):
    score = alpha * np.log(np.maximum(learned_p, 1e-12)) + beta * np.log(np.maximum(hps_p, 1e-12))
    idx = score.argmax(axis=0)
    return cand_cents_abs[idx] - tonic_term


def main() -> None:
    index = RecordingLaneIndex.build(REPO_ROOT)
    manifest = load_kfold_manifest(REPO_ROOT)
    lo, hi = native_range()
    cand_cents = 1200.0 * np.log2(candidate_hz(lo, hi))

    out = {"ratios_tried": list(RATIOS), "shared_range_bins": [lo, hi], "per_fold": []}
    pooled = {"hps_only": {"pred": [], "true": []}, "learned_only": {"pred": [], "true": []}, "fused": {"pred": [], "true": []}}

    for fold in range(5):
        split = build_fold_split(manifest, fold, seed=42)
        model, _ckpt = load_learned_model("harmonic", fold, lo, hi)
        mu, sigma = load_fold_cqt_stats(fold, REPO_ROOT)

        val_scores = {r: {"pred": [], "true": []} for r in RATIOS}
        for rid in split.val_recording_ids:
            hps_p, learned_p, true_rel, tonic = _recording_probs_shared(rid, index, model, mu, sigma, lo, hi)
            for (a, b) in RATIOS:
                pred = fused_argmax(hps_p, learned_p, cand_cents, a, b, tonic)
                val_scores[(a, b)]["pred"].append(pred); val_scores[(a, b)]["true"].append(true_rel)
        best_ratio, best_mae, val_mae_by_ratio = (1.0, 0.0), float("inf"), {}
        for r in RATIOS:
            if not val_scores[r]["pred"]:
                continue
            p = np.concatenate(val_scores[r]["pred"]); t = np.concatenate(val_scores[r]["true"])
            mae = float(np.abs(p - t).mean())
            val_mae_by_ratio[str(r)] = mae
            if mae < best_mae:
                best_mae, best_ratio = mae, r

        fold_entry = {"fold": fold, "val_mae_by_ratio": val_mae_by_ratio, "best_ratio": list(best_ratio)}
        for rid in split.test_recording_ids:
            hps_p, learned_p, true_rel, tonic = _recording_probs_shared(rid, index, model, mu, sigma, lo, hi)
            hps_pred = cand_cents[hps_p.argmax(axis=0)] - tonic
            learned_pred = cand_cents[learned_p.argmax(axis=0)] - tonic
            fused_pred = fused_argmax(hps_p, learned_p, cand_cents, best_ratio[0], best_ratio[1], tonic)
            pooled["hps_only"]["pred"].append(hps_pred); pooled["hps_only"]["true"].append(true_rel)
            pooled["learned_only"]["pred"].append(learned_pred); pooled["learned_only"]["true"].append(true_rel)
            pooled["fused"]["pred"].append(fused_pred); pooled["fused"]["true"].append(true_rel)
        out["per_fold"].append(fold_entry)
        print(f"fold {fold} done, best ratio (alpha_learned,beta_hps)={best_ratio}")

    summary = {}
    for k in pooled:
        p = np.concatenate(pooled[k]["pred"]); t = np.concatenate(pooled[k]["true"])
        summary[k] = extended_pitch_metrics(p, t)
    out["test_summary_pooled"] = summary
    write_json(REG_DIR / "fusion_result.json", out)
    print("=== fusion summary (test, pooled, shared 34-244 window) ===")
    for k, v in summary.items():
        print(k, "MAE", round(v["mae_cents"], 1), "octave_adj", round(v["octave_adjusted_mae"], 1))


if __name__ == "__main__":
    main()
