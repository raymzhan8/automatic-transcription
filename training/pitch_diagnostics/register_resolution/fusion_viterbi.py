"""Step 12.5: the one cheap D2+D3/D4 combination Step 12 left untested --
Viterbi decoding (movement-cost, +/- octave penalty) run on the FUSED
HPS+learned salience distribution, rather than on either source alone.

Reuses Step 12's own implementations directly, no new decoder/transition
logic:
 - fusion.py::RATIOS, fused_argmax, _recording_probs_shared (log-linear
   fusion, alpha/beta selected on validation only, shared 34-244 window)
 - decoders.py::K_TOP, CAP_CENTS, OCTAVE_TOL_CENTS, LAMBDA_T_GRID,
   LAMBDA_OCT_GRID, _states_for_frame, viterbi_decode (same small grids,
   same top-K-plus-octave-shift state construction, same time-gap-aware
   movement cost)

Fused emissions for decoding: S_fused(f,t) = softmax_f[ alpha*log
S_learned(f,t) + beta*log S_HPS(f,t) ] -- the natural normalized probability
interpretation of fusion.py's fused_argmax score, needed here because the
Viterbi state/emission construction (_states_for_frame) consumes a proper
per-frame probability column, not just an argmax.

All comparisons in the main table share the 34-244 candidate window (the
only range the fused distribution is defined over, since the learned model
is only valid there -- see common.py::native_range). The "fair HPS 34-244"
and "learned 34-244" rows are computed here (same window, same recordings,
same code path) rather than reused from decoder_ablation.json (full 0-360
range for HPS there). The original full-range HPS numbers are pulled
unmodified from oracle_topk.json / decoder_ablation.json as a reference
column only, not recomputed.

trajectory_type and dp_dt are carried through as EVALUATION METADATA ONLY
(never used for decoding) to check for oversmoothing (spec section 6).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.folds import build_fold_split, load_kfold_manifest  # noqa: E402
from training.framewise_dataset import RecordingLaneIndex  # noqa: E402
from training.metrics import TYPE_NAMES  # noqa: E402
from training.normalization import load_fold_cqt_stats, normalize_cqt  # noqa: E402
from training.pitch_diagnostics.common import PRIMARY_LANE, linear_mag, write_json  # noqa: E402
from training.pitch_diagnostics.hps_salience import hps_salience_probs  # noqa: E402
from training.pitch_diagnostics.register_resolution.common import (  # noqa: E402
    REG_DIR, candidate_hz, extended_pitch_metrics, load_learned_model, native_range,
)
from training.pitch_diagnostics.register_resolution.decoders import (  # noqa: E402
    CAP_CENTS, K_TOP, LAMBDA_OCT_GRID, LAMBDA_T_GRID, OCTAVE_TOL_CENTS,
    _states_for_frame, viterbi_decode,
)
from training.pitch_diagnostics.register_resolution.fusion import RATIOS, fused_argmax  # noqa: E402

NATIVE_HOP_S = 0.01
LARGE_FAILURE_RECORDINGS = ["6417585554a0bfbd8de2d3ff", "6824de49abc4705438ce918b"]
DPDT_BUCKET_EDGES_CENTS_S = (0.0, 100.0, 400.0, 1000.0, np.inf)
DPDT_BUCKET_NAMES = ("0-100c/s", "100-400c/s", "400-1000c/s", ">1000c/s")


def _bundle(rec_id: str, index: RecordingLaneIndex, model, mu, sigma, lo: int, hi: int) -> dict:
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
    hps_probs = hps_salience_probs(mag, lo, hi)[:, valid]
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

    return {
        "recording_id": rec_id, "fold": None, "hps_probs": hps_probs, "learned_probs": learned_probs,
        "true_cents": true_rel, "tonic_term": tonic_term, "dt_steps": dt_steps,
        "trajectory_type": frames["trajectory_type"][:n][valid],
        "dp_dt_cents_s": frames["dp_dt_log2_hz_per_s"][:n][valid] * 1200.0,
        "times": t_valid,
    }


def _fused_probs(hps_probs: np.ndarray, learned_probs: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    score = alpha * np.log(np.maximum(learned_probs, 1e-12)) + beta * np.log(np.maximum(hps_probs, 1e-12))
    score = score - score.max(axis=0, keepdims=True)
    probs = np.exp(score)
    probs /= np.maximum(probs.sum(axis=0, keepdims=True), 1e-12)
    return probs


def _select_lambda_t(bundles: list[dict], probs_key_or_array_fn, cand_cents: np.ndarray, lo: int) -> tuple[float, dict]:
    val_mae = {}
    for lt in LAMBDA_T_GRID:
        preds, trues = [], []
        for b in bundles:
            probs = probs_key_or_array_fn(b)
            T = probs.shape[1]
            states = [_states_for_frame(probs[:, t], cand_cents, lo) for t in range(T)]
            path_abs = viterbi_decode(states, b["dt_steps"], lt, 0.0)
            preds.append(path_abs - b["tonic_term"]); trues.append(b["true_cents"])
        val_mae[lt] = float(np.abs(np.concatenate(preds) - np.concatenate(trues)).mean())
    best_lt = min(val_mae, key=val_mae.get)
    return best_lt, val_mae


def _select_lambda_oct(bundles: list[dict], probs_key_or_array_fn, cand_cents: np.ndarray, lo: int, lambda_t: float) -> tuple[float, dict]:
    val_mae = {}
    for lo_ct in LAMBDA_OCT_GRID:
        preds, trues = [], []
        for b in bundles:
            probs = probs_key_or_array_fn(b)
            T = probs.shape[1]
            states = [_states_for_frame(probs[:, t], cand_cents, lo) for t in range(T)]
            path_abs = viterbi_decode(states, b["dt_steps"], lambda_t, lo_ct)
            preds.append(path_abs - b["tonic_term"]); trues.append(b["true_cents"])
        val_mae[lo_ct] = float(np.abs(np.concatenate(preds) - np.concatenate(trues)).mean())
    best_oct = min(val_mae, key=val_mae.get)
    return best_oct, val_mae


def _decode(probs: np.ndarray, cand_cents: np.ndarray, lo: int, dt_steps: np.ndarray, tonic_term: float, lambda_t: float, lambda_oct: float) -> np.ndarray:
    T = probs.shape[1]
    states = [_states_for_frame(probs[:, t], cand_cents, lo) for t in range(T)]
    path_abs = viterbi_decode(states, dt_steps, lambda_t, lambda_oct)
    return path_abs - tonic_term


def main() -> None:
    index = RecordingLaneIndex.build(REPO_ROOT)
    manifest = load_kfold_manifest(REPO_ROOT)
    lo, hi = native_range()
    cand_cents = 1200.0 * np.log2(candidate_hz(lo, hi))

    methods = ("hps_argmax", "hps_d3", "hps_d4", "learned_argmax", "learned_d3",
               "fused_argmax", "fused_d3", "fused_d4")
    pooled = {m: {"pred": [], "true": []} for m in methods}
    per_fold_pooled: dict[int, dict[str, dict]] = {}
    per_recording: dict[str, dict] = {}
    per_type: dict[str, dict[int, dict]] = {m: {} for m in methods}
    dpdt_bucket: dict[str, dict[str, dict]] = {m: {name: {"pred": [], "true": []} for name in DPDT_BUCKET_NAMES} for m in methods}
    per_fold_log: list[dict] = []

    for fold in range(5):
        split = build_fold_split(manifest, fold, seed=42)
        model, _ckpt = load_learned_model("harmonic", fold, lo, hi)
        mu, sigma = load_fold_cqt_stats(fold, REPO_ROOT)

        val_bundles = [_bundle(rid, index, model, mu, sigma, lo, hi) for rid in split.val_recording_ids]

        # ---- select alpha/beta on validation (fusion.py's own procedure) ----
        val_scores = {r: {"pred": [], "true": []} for r in RATIOS}
        for b in val_bundles:
            for (a, bw) in RATIOS:
                pred = fused_argmax(b["hps_probs"], b["learned_probs"], cand_cents, a, bw, b["tonic_term"])
                val_scores[(a, bw)]["pred"].append(pred); val_scores[(a, bw)]["true"].append(b["true_cents"])
        best_ratio, best_ratio_mae, val_mae_by_ratio = (1.0, 0.0), float("inf"), {}
        for r in RATIOS:
            p = np.concatenate(val_scores[r]["pred"]); t = np.concatenate(val_scores[r]["true"])
            mae = float(np.abs(p - t).mean())
            val_mae_by_ratio[str(r)] = mae
            if mae < best_ratio_mae:
                best_ratio_mae, best_ratio = mae, r
        alpha, beta = best_ratio

        for b in val_bundles:
            b["fused_probs"] = _fused_probs(b["hps_probs"], b["learned_probs"], alpha, beta)

        # ---- select lambda_t / lambda_oct on validation, separately for HPS, learned, fused ----
        best = {}
        for key, fn in (
            ("hps", lambda b: b["hps_probs"]),
            ("learned", lambda b: b["learned_probs"]),
            ("fused", lambda b: b["fused_probs"]),
        ):
            lt, val_mae_d3 = _select_lambda_t(val_bundles, fn, cand_cents, lo)
            loct, val_mae_d4 = _select_lambda_oct(val_bundles, fn, cand_cents, lo, lt)
            best[key] = {"lambda_t": lt, "lambda_oct": loct, "val_mae_d3": val_mae_d3, "val_mae_d4": val_mae_d4}

        fold_entry = {
            "fold": fold, "best_ratio": list(best_ratio), "val_mae_by_ratio": val_mae_by_ratio,
            "hps_best": best["hps"], "learned_best": best["learned"], "fused_best": best["fused"],
        }
        per_fold_log.append(fold_entry)
        per_fold_pooled[fold] = {m: {"pred": [], "true": []} for m in methods}
        print(f"fold {fold}: ratio={best_ratio} hps(lt={best['hps']['lambda_t']},loct={best['hps']['lambda_oct']}) "
              f"learned(lt={best['learned']['lambda_t']},loct={best['learned']['lambda_oct']}) "
              f"fused(lt={best['fused']['lambda_t']},loct={best['fused']['lambda_oct']})")

        for rid in split.test_recording_ids:
            b = _bundle(rid, index, model, mu, sigma, lo, hi)
            fused_probs = _fused_probs(b["hps_probs"], b["learned_probs"], alpha, beta)

            preds = {
                "hps_argmax": cand_cents[b["hps_probs"].argmax(axis=0)] - b["tonic_term"],
                "learned_argmax": cand_cents[b["learned_probs"].argmax(axis=0)] - b["tonic_term"],
                "fused_argmax": fused_argmax(b["hps_probs"], b["learned_probs"], cand_cents, alpha, beta, b["tonic_term"]),
                "hps_d3": _decode(b["hps_probs"], cand_cents, lo, b["dt_steps"], b["tonic_term"], best["hps"]["lambda_t"], 0.0),
                "hps_d4": _decode(b["hps_probs"], cand_cents, lo, b["dt_steps"], b["tonic_term"], best["hps"]["lambda_t"], best["hps"]["lambda_oct"]),
                "learned_d3": _decode(b["learned_probs"], cand_cents, lo, b["dt_steps"], b["tonic_term"], best["learned"]["lambda_t"], 0.0),
                "fused_d3": _decode(fused_probs, cand_cents, lo, b["dt_steps"], b["tonic_term"], best["fused"]["lambda_t"], 0.0),
                "fused_d4": _decode(fused_probs, cand_cents, lo, b["dt_steps"], b["tonic_term"], best["fused"]["lambda_t"], best["fused"]["lambda_oct"]),
            }
            true = b["true_cents"]
            per_recording.setdefault(rid, {"fold": fold, "support": len(true)})
            for m, pred in preds.items():
                pooled[m]["pred"].append(pred); pooled[m]["true"].append(true)
                per_fold_pooled[fold][m]["pred"].append(pred); per_fold_pooled[fold][m]["true"].append(true)
                per_recording[rid][m] = extended_pitch_metrics(pred, true)
                for ttype in np.unique(b["trajectory_type"]):
                    if ttype < 0:
                        continue
                    mask = b["trajectory_type"] == ttype
                    per_type[m].setdefault(int(ttype), {"pred": [], "true": []})
                    per_type[m][int(ttype)]["pred"].append(pred[mask]); per_type[m][int(ttype)]["true"].append(true[mask])
                abs_dpdt = np.abs(b["dp_dt_cents_s"])
                for name, lo_e, hi_e in zip(DPDT_BUCKET_NAMES, DPDT_BUCKET_EDGES_CENTS_S[:-1], DPDT_BUCKET_EDGES_CENTS_S[1:]):
                    mask = (abs_dpdt >= lo_e) & (abs_dpdt < hi_e)
                    if not np.any(mask):
                        continue
                    dpdt_bucket[m][name]["pred"].append(pred[mask]); dpdt_bucket[m][name]["true"].append(true[mask])

    summary_pooled = {m: extended_pitch_metrics(np.concatenate(v["pred"]), np.concatenate(v["true"])) for m, v in pooled.items()}
    summary_per_fold = {
        fold: {m: extended_pitch_metrics(np.concatenate(v["pred"]), np.concatenate(v["true"])) for m, v in d.items()}
        for fold, d in per_fold_pooled.items()
    }
    summary_per_type = {}
    for m, types in per_type.items():
        summary_per_type[m] = {}
        for tt, d in types.items():
            if d["pred"]:
                summary_per_type[m][TYPE_NAMES[tt]] = extended_pitch_metrics(np.concatenate(d["pred"]), np.concatenate(d["true"]))
    summary_dpdt = {}
    for m, buckets in dpdt_bucket.items():
        summary_dpdt[m] = {}
        for name, d in buckets.items():
            if d["pred"]:
                summary_dpdt[m][name] = extended_pitch_metrics(np.concatenate(d["pred"]), np.concatenate(d["true"]))

    # ---- reference-only: original full-range HPS numbers, unmodified, from existing Step 12 artifacts ----
    oracle_topk = json.loads((REG_DIR / "oracle_topk.json").read_text(encoding="utf-8"))
    decoder_ablation = json.loads((REG_DIR / "decoder_ablation.json").read_text(encoding="utf-8"))
    reference_full_range_hps = {
        "argmax": oracle_topk["hps"]["argmax"],
        "d3": decoder_ablation["test_summary_pooled"]["hps_d3"],
        "d4": decoder_ablation["test_summary_pooled"]["hps_d4"],
    }

    large_failure = {rid: per_recording[rid] for rid in LARGE_FAILURE_RECORDINGS if rid in per_recording}

    out = {
        "shared_range_bins": [lo, hi],
        "ratios_tried": list(RATIOS),
        "lambda_t_grid": list(LAMBDA_T_GRID),
        "lambda_oct_grid": list(LAMBDA_OCT_GRID),
        "k_top": K_TOP, "cap_cents": CAP_CENTS, "octave_tol_cents": OCTAVE_TOL_CENTS,
        "dpdt_bucket_edges_cents_s": list(DPDT_BUCKET_EDGES_CENTS_S),
        "per_fold": per_fold_log,
        "test_summary_pooled": summary_pooled,
        "test_summary_per_fold": summary_per_fold,
        "per_recording": per_recording,
        "per_type": summary_per_type,
        "per_dpdt_bucket": summary_dpdt,
        "reference_full_range_hps": reference_full_range_hps,
        "large_failure_recordings": large_failure,
    }
    write_json(REG_DIR / "fusion_viterbi_result.json", out)

    print("\n=== Step 12.5 pooled summary (34-244 shared window) ===")
    for m, v in summary_pooled.items():
        print(f"{m:16s} MAE {v['mae_cents']:7.1f}  med {v['median_ae_cents']:6.1f}  "
              f"+-25c {v['pct_within_25']*100:5.1f}%  +-50c {v['pct_within_50']*100:5.1f}%  "
              f"oct_adj {v['octave_adjusted_mae']:6.1f}  correct_oct {v['correct_octave']*100:5.1f}%")
    print("\nreference full-range HPS: argmax MAE", round(reference_full_range_hps["argmax"]["mae_cents"], 1),
          "d3 MAE", round(reference_full_range_hps["d3"]["mae_cents"], 1),
          "d4 MAE", round(reference_full_range_hps["d4"]["mae_cents"], 1))


if __name__ == "__main__":
    main()
