"""Step 12 spec §14-19, §21-22: D3 (movement-cost Viterbi) and D4
(movement-cost + explicit octave-jump-penalty Viterbi), run over BOTH the
HPS and learned salience maps as emission sources.

Design decisions (see plan file + Phase A synthesis for justification):
 - Candidate states per frame = top-K salience candidates (K=5, chosen from
   the rank data already computed in Phase A -- GT-in-top5 recall is already
   known per gt_topk_given_wrong_octave.json/oracle_topk.json's monotone
   top2/3/5 improvement -- rather than recomputing a separate recall sweep,
   which would require re-materializing full [F,T] probs arrays a second
   time; top-5 is the same K already validated as giving large oracle
   headroom in Phase A) unioned with each state's /2 and *2 octave-shifted
   bin (looked up in the same probability map, clipped to its valid range).
 - Decoding runs over the sequence of VALID frames only (not literally every
   raw 10ms frame including invalid-target gaps) -- a deliberate, documented
   simplification of spec §19's "decode continuously through the recording":
   the transition cost is made TIME-GAP-AWARE (normalized by elapsed time
   between consecutive valid frames, in units of one native 10ms hop) so a
   large gap doesn't get penalized as harshly as the same cents jump between
   two adjacent frames -- this preserves the spirit of "don't break the path
   at invalid regions" (no artificial reset, cost still evaluated across the
   gap) without requiring dense per-frame salience over silence/unsupervised
   regions this fork didn't have time to build. Flagged explicitly in the
   Step 12 report as a documented scope simplification, not silently done.
 - D3 transition cost: piecewise-capped time-normalized movement cost only.
 - D4 = D3 + an additive penalty when |delta_cents| is close to a nonzero
   integer multiple of 1200c (an octave jump), so the decoder needs stronger
   emission evidence to choose that transition than an equivalent same-size
   non-octave jump would need.
 - lambda_transition / lambda_octave / K are selected on VALIDATION only per
   fold, from a small fixed grid (spec explicitly says do not grid-search
   heavily).
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
from training.pitch_diagnostics.common import PRIMARY_LANE, bin_from_hz, linear_mag, write_json  # noqa: E402
from training.pitch_diagnostics.hps_salience import hps_salience_probs  # noqa: E402
from training.pitch_diagnostics.register_resolution.common import (  # noqa: E402
    FULL_HI_BIN, FULL_LO_BIN, REG_DIR, candidate_hz, extended_pitch_metrics, load_learned_model, native_range,
)

K_TOP = 5
CAP_CENTS = 1200.0          # movement cost saturates at one octave-per-native-hop
OCTAVE_TOL_CENTS = 50.0     # how close to an exact 1200c multiple counts as "octave jump"
LAMBDA_T_GRID = (0.0005, 0.002, 0.008)
LAMBDA_OCT_GRID = (0.0, 5.0, 15.0)
NATIVE_HOP_S = 0.01


def _states_for_frame(probs_col: np.ndarray, cand_cents: np.ndarray, lo_bin: int) -> tuple[np.ndarray, np.ndarray]:
    """probs_col: [F] over bins [lo_bin, lo_bin+F). Returns (cents[k'], logscore[k'])
    for the top-K states unioned with their /2, *2 octave-shifted variants,
    deduped by rounded bin index."""
    n = probs_col.shape[0]
    k = min(K_TOP, n)
    top_idx = np.argpartition(-probs_col, k - 1)[:k]
    seen: dict[int, tuple[float, float]] = {}
    for i in top_idx:
        seen[int(i)] = (float(cand_cents[i]), float(probs_col[i]))
        for mult in (0.5, 2.0):
            shifted_hz = np.exp2(cand_cents[i] / 1200.0) * mult
            b = int(np.clip(round(float(bin_from_hz(shifted_hz))), lo_bin, lo_bin + n - 1)) - lo_bin
            if b not in seen:
                seen[b] = (float(cand_cents[b]), float(probs_col[b]))
    cents = np.array([v[0] for v in seen.values()])
    logscore = np.log(np.maximum(np.array([v[1] for v in seen.values()]), 1e-12))
    return cents, logscore


def _movement_cost(delta_cents: np.ndarray, dt_steps: np.ndarray) -> np.ndarray:
    normalized = np.abs(delta_cents) / np.maximum(dt_steps, 1.0)
    return np.minimum(normalized, CAP_CENTS)


def _octave_penalty(delta_cents: np.ndarray) -> np.ndarray:
    nearest_mult = np.round(delta_cents / 1200.0)
    resid = np.abs(delta_cents - nearest_mult * 1200.0)
    is_octave_jump = (nearest_mult != 0) & (resid < OCTAVE_TOL_CENTS)
    return is_octave_jump.astype(np.float64)


def viterbi_decode(states: list[tuple[np.ndarray, np.ndarray]], dt_steps: np.ndarray, lambda_t: float, lambda_oct: float) -> np.ndarray:
    """states[t] = (cents[k], logscore[k]) for frame t. dt_steps[t] = elapsed
    native-hops between frame t-1 and t (dt_steps[0] unused). Returns decoded
    cents path, length T."""
    T = len(states)
    cents0, log0 = states[0]
    cost = -log0.copy()  # cumulative min-cost to reach each state at t=0
    backptr: list[np.ndarray] = [np.full(len(cents0), -1, dtype=np.int64)]
    for t in range(1, T):
        prev_cents, _ = states[t - 1]
        cur_cents, cur_log = states[t]
        delta = cur_cents[None, :] - prev_cents[:, None]  # [n_prev, n_cur]
        move = _movement_cost(delta, dt_steps[t])
        trans = lambda_t * move
        if lambda_oct > 0:
            trans = trans + lambda_oct * _octave_penalty(delta)
        total = cost[:, None] + trans - cur_log[None, :]
        best_prev = np.argmin(total, axis=0)
        cost = total[best_prev, np.arange(len(cur_cents))]
        backptr.append(best_prev)
        states_t_cents = cur_cents
    # backtrace
    path_idx = np.zeros(T, dtype=np.int64)
    path_idx[-1] = int(np.argmin(cost))
    for t in range(T - 1, 0, -1):
        path_idx[t - 1] = backptr[t][path_idx[t]]
    return np.array([states[t][0][path_idx[t]] for t in range(T)])


def _recording_bundle(rec_id, index, model, mu, sigma, lrn_lo, lrn_hi):
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

    t_valid = times[valid]
    dt_steps = np.empty(len(t_valid))
    dt_steps[0] = 1.0
    dt_steps[1:] = np.maximum((t_valid[1:] - t_valid[:-1]) / NATIVE_HOP_S, 1.0)

    return {
        "hps_probs": hps_probs, "learned_probs": learned_probs,
        "true_cents": true_rel, "tonic_term": tonic_term, "dt_steps": dt_steps,
        "trajectory_type": frames["trajectory_type"][:n][valid],
        "dp_dt": frames["dp_dt_log2_hz_per_s"][:n][valid],
        "times": t_valid,
    }


def decode_recording(probs: np.ndarray, cand_cents_abs: np.ndarray, lo_bin: int, dt_steps: np.ndarray, tonic_term: float, lambda_t: float, lambda_oct: float) -> np.ndarray:
    T = probs.shape[1]
    states = [_states_for_frame(probs[:, t], cand_cents_abs, lo_bin) for t in range(T)]
    path_abs = viterbi_decode(states, dt_steps, lambda_t, lambda_oct)
    return path_abs - tonic_term


def main() -> None:
    index = RecordingLaneIndex.build(REPO_ROOT)
    manifest = load_kfold_manifest(REPO_ROOT)
    cand_cents_hps = 1200.0 * np.log2(candidate_hz(FULL_LO_BIN, FULL_HI_BIN))
    lrn_lo, lrn_hi = native_range()
    cand_cents_lrn = 1200.0 * np.log2(candidate_hz(lrn_lo, lrn_hi))

    out = {"K_TOP": K_TOP, "cap_cents": CAP_CENTS, "octave_tol_cents": OCTAVE_TOL_CENTS,
           "lambda_t_grid": list(LAMBDA_T_GRID), "lambda_oct_grid": list(LAMBDA_OCT_GRID), "per_fold": []}
    pooled = {f"{m}_{d}": {"pred": [], "true": []} for m in ("hps", "learned") for d in ("d3", "d4")}
    per_recording: dict[str, dict] = {}
    by_type: dict[str, dict[int, dict]] = {f"{m}_{d}": {} for m in ("hps", "learned") for d in ("d3", "d4")}

    for fold in range(5):
        split = build_fold_split(manifest, fold, seed=42)
        model, _ckpt = load_learned_model("harmonic", fold, lrn_lo, lrn_hi)
        mu, sigma = load_fold_cqt_stats(fold, REPO_ROOT)

        val_bundles = [_recording_bundle(rid, index, model, mu, sigma, lrn_lo, lrn_hi) for rid in split.val_recording_ids]
        fold_entry = {"fold": fold}
        best = {}
        for method, probs_key, cand_cents, lo in (("hps", "hps_probs", cand_cents_hps, FULL_LO_BIN), ("learned", "learned_probs", cand_cents_lrn, lrn_lo)):
            val_mae_d3, val_mae_d4 = {}, {}
            for lt in LAMBDA_T_GRID:
                preds, trues = [], []
                for b in val_bundles:
                    pred = decode_recording(b[probs_key], cand_cents, lo, b["dt_steps"], b["tonic_term"], lt, 0.0)
                    preds.append(pred); trues.append(b["true_cents"])
                mae = float(np.abs(np.concatenate(preds) - np.concatenate(trues)).mean())
                val_mae_d3[lt] = mae
            best_lt = min(val_mae_d3, key=val_mae_d3.get)
            for lo_ct in LAMBDA_OCT_GRID:
                preds, trues = [], []
                for b in val_bundles:
                    pred = decode_recording(b[probs_key], cand_cents, lo, b["dt_steps"], b["tonic_term"], best_lt, lo_ct)
                    preds.append(pred); trues.append(b["true_cents"])
                mae = float(np.abs(np.concatenate(preds) - np.concatenate(trues)).mean())
                val_mae_d4[lo_ct] = mae
            best_oct = min(val_mae_d4, key=val_mae_d4.get)
            best[method] = {"lambda_t": best_lt, "lambda_oct": best_oct, "val_mae_d3": val_mae_d3, "val_mae_d4": val_mae_d4}
        fold_entry["best"] = best

        for rid in split.test_recording_ids:
            b = _recording_bundle(rid, index, model, mu, sigma, lrn_lo, lrn_hi)
            for method, probs_key, cand_cents, lo in (("hps", "hps_probs", cand_cents_hps, FULL_LO_BIN), ("learned", "learned_probs", cand_cents_lrn, lrn_lo)):
                lt = best[method]["lambda_t"]; loct = best[method]["lambda_oct"]
                pred_d3 = decode_recording(b[probs_key], cand_cents, lo, b["dt_steps"], b["tonic_term"], lt, 0.0)
                pred_d4 = decode_recording(b[probs_key], cand_cents, lo, b["dt_steps"], b["tonic_term"], lt, loct)
                for tag, pred in (("d3", pred_d3), ("d4", pred_d4)):
                    key = f"{method}_{tag}"
                    pooled[key]["pred"].append(pred); pooled[key]["true"].append(b["true_cents"])
                    m = extended_pitch_metrics(pred, b["true_cents"])
                    per_recording.setdefault(rid, {})[key] = {"fold": fold, "support": len(pred), **m}
                    for ttype in np.unique(b["trajectory_type"]):
                        if ttype < 0:
                            continue
                        mask = b["trajectory_type"] == ttype
                        by_type[key].setdefault(int(ttype), {"pred": [], "true": []})
                        by_type[key][int(ttype)]["pred"].append(pred[mask]); by_type[key][int(ttype)]["true"].append(b["true_cents"][mask])
        out["per_fold"].append(fold_entry)
        print(f"fold {fold} done: hps best={best['hps']}, learned best={best['learned']}")

    summary = {k: extended_pitch_metrics(np.concatenate(v["pred"]), np.concatenate(v["true"])) for k, v in pooled.items()}
    out["test_summary_pooled"] = summary
    out["per_recording"] = per_recording
    type_summary = {}
    for key, types in by_type.items():
        type_summary[key] = {}
        for tt, d in types.items():
            if d["pred"]:
                type_summary[key][tt] = extended_pitch_metrics(np.concatenate(d["pred"]), np.concatenate(d["true"]))
    out["per_type"] = type_summary
    write_json(REG_DIR / "decoder_ablation.json", out)
    print("=== D3/D4 summary (pooled) ===")
    for k, v in summary.items():
        print(k, "MAE", round(v["mae_cents"], 1), "octave_adj", round(v["octave_adjusted_mae"], 1), "correct_octave", round(v["correct_octave"], 3))


if __name__ == "__main__":
    main()
