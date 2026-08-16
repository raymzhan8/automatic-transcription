"""Step 11 §8, §13, §19-27: full-recording evaluation of the trained salience
models against HPS and every Step 10 baseline.

Evaluation runs on FULL recordings (not excerpts) per fold's test split, one
forward pass per recording through the trained model — mirrors how
baselines_b.py/hps_salience.py already evaluate HPS, so results are directly
comparable. "Grouped mean" MAE is computed the same way Step 10 reports it:
the mean of the 5 per-fold overall MAEs (not a pooled/weighted mean), so the
harmonic-vs-HPS comparison uses an identical aggregation rule.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.folds import build_fold_split, load_kfold_manifest  # noqa: E402
from training.framewise_dataset import RecordingLaneIndex  # noqa: E402
from training.metrics import pitch_error_metrics, pitch_metrics_by_type  # noqa: E402
from training.normalization import load_fold_cqt_stats, log2_hz_to_cents, normalize_cqt  # noqa: E402
from training.pitch_diagnostics.baselines_b import harmonic_product_cents  # noqa: E402
from training.pitch_diagnostics.common import OUT_DIR, PRIMARY_LANE, linear_mag, octave_adjusted_error, write_json  # noqa: E402
from training.pitch_diagnostics.salience_common import (  # noqa: E402
    SALIENCE_SIGMA_CENTS,
    candidate_hz,
    distribution_entropy,
    load_or_compute_candidate_range,
    target_rank_in_distribution,
)
from training.pitch_diagnostics.salience_models import HarmonicSalienceModel  # noqa: E402
from training.pitch_diagnostics.train_salience import VARIANT_CONFIG  # noqa: E402

RUN_NAMES = {"local": "local_salience_abs", "harmonic": "harmonic_salience_abs"}


def extended_pitch_metrics(pred_cents: np.ndarray, true_cents: np.ndarray) -> dict[str, Any]:
    base = pitch_error_metrics(pred_cents, true_cents)
    if len(true_cents) == 0:
        base.update({"pct_within_100": 0.0, "octave_adjusted_mae": 0.0, "octave_adjusted_median_ae": 0.0})
        return base
    err = np.abs(pred_cents - true_cents)
    oct_err, _ = octave_adjusted_error(pred_cents, true_cents)
    base["pct_within_100"] = float((err <= 100).mean())
    base["octave_adjusted_mae"] = float(oct_err.mean())
    base["octave_adjusted_median_ae"] = float(np.median(oct_err))
    return base


def load_model(variant: str, fold: int, lo_bin: int, hi_bin: int) -> HarmonicSalienceModel:
    cfg = VARIANT_CONFIG[variant]
    model = HarmonicSalienceModel(
        candidate_lo_bin=lo_bin, candidate_hi_bin=hi_bin,
        harmonic_ks=cfg["harmonic_ks"], hidden=cfg["hidden"],
    )
    ckpt = torch.load(OUT_DIR / "runs" / RUN_NAMES[variant] / f"fold_{fold}" / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def evaluate_recording(
    model: HarmonicSalienceModel,
    mu: np.ndarray,
    sigma: np.ndarray,
    cand_cents_abs: np.ndarray,
    rec_id: str,
    index: RecordingLaneIndex,
    *,
    ablate_ks: frozenset[int] = frozenset(),
) -> dict[str, np.ndarray]:
    lane = next(x for x in index.lanes if x.recording_id == rec_id)
    frames = index._frames[(rec_id, PRIMARY_LANE)]
    cqt_log = index._features[rec_id]["cqt_log"]
    n = min(cqt_log.shape[1], lane.n_frames)
    cqt_log = cqt_log[:, :n]
    times = frames["frame_time_s"][:n]
    valid = frames["valid_target"][:n] & (times < lane.duration_s)

    spec = normalize_cqt(cqt_log, mu, sigma).astype(np.float32)
    spec_t = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)  # [1,1,360,T]
    with torch.no_grad():
        logits = model(spec_t, ablate_ks=ablate_ks)  # [1,F_cand,T]
        probs_t = torch.softmax(logits, dim=1)
        idx = logits.argmax(dim=1)[0].numpy()  # [T]
        probs = probs_t[0].numpy()  # [F_cand,T]

    argmax_cents_abs = cand_cents_abs[idx]
    expected_cents_abs = (probs * cand_cents_abs[:, None]).sum(axis=0)

    true_log2 = frames["pitch_log2_hz"][:n].astype(np.float64)
    true_cents_abs = 1200.0 * true_log2
    tonic_term = 1200.0 * np.log2(lane.fundamental_hz)

    argmax_rel = (argmax_cents_abs - tonic_term)[valid]
    expected_rel = (expected_cents_abs - tonic_term)[valid]
    true_rel = (true_cents_abs - tonic_term)[valid]
    target_cents_abs_valid = true_cents_abs[valid]

    rank = target_rank_in_distribution(probs[:, valid], target_cents_abs_valid, cand_cents_abs, tolerance_cents=SALIENCE_SIGMA_CENTS)
    ent = distribution_entropy(probs[:, valid], axis=0)
    traj = frames["trajectory_type"][:n][valid]

    return {
        "argmax_cents": argmax_rel, "expected_cents": expected_rel, "true_cents": true_rel,
        "rank": rank, "entropy": ent, "trajectory_type": traj,
        "n_valid": int(valid.sum()), "fold_recording": rec_id,
    }


def run_full_eval() -> None:
    repo_root = REPO_ROOT
    index = RecordingLaneIndex.build(repo_root)
    manifest = load_kfold_manifest(repo_root)
    cr = load_or_compute_candidate_range(repo_root, index)
    lo_bin, hi_bin = cr["candidate_lo_bin"], cr["candidate_hi_bin"]
    cand_hz_arr = candidate_hz(lo_bin, hi_bin)
    cand_cents_abs = 1200.0 * np.log2(cand_hz_arr)

    per_variant_fold: dict[str, list[dict[str, Any]]] = {"local": [], "harmonic": []}
    per_variant_pooled: dict[str, dict[str, list[np.ndarray]]] = {
        v: {"argmax": [], "expected": [], "true": [], "rank": [], "entropy": [], "type": [], "rec": [], "fold": []}
        for v in ("local", "harmonic")
    }
    per_recording_by_variant: dict[str, dict[str, Any]] = {"local": {}, "harmonic": {}}
    hps_per_recording: dict[str, Any] = {}
    hps_pooled = {"pred": [], "true": [], "rec": [], "fold": []}

    for variant in ("local", "harmonic"):
        for fold in range(5):
            split = build_fold_split(manifest, fold, seed=42)
            mu, sigma = load_fold_cqt_stats(fold, repo_root)
            model, ckpt = load_model(variant, fold, lo_bin, hi_bin)

            fold_argmax, fold_expected, fold_true = [], [], []
            for rec_id in split.test_recording_ids:
                out = evaluate_recording(model, mu, sigma, cand_cents_abs, rec_id, index)
                if out["n_valid"] == 0:
                    continue
                fold_argmax.append(out["argmax_cents"])
                fold_expected.append(out["expected_cents"])
                fold_true.append(out["true_cents"])
                for key, arr in (("argmax", out["argmax_cents"]), ("expected", out["expected_cents"]),
                                  ("true", out["true_cents"]), ("rank", out["rank"]), ("entropy", out["entropy"]),
                                  ("type", out["trajectory_type"])):
                    per_variant_pooled[variant][key].append(arr)
                per_variant_pooled[variant]["rec"].append(np.full(out["n_valid"], rec_id, dtype=object))
                per_variant_pooled[variant]["fold"].append(np.full(out["n_valid"], fold, dtype=np.int32))

                rec_metrics_argmax = extended_pitch_metrics(out["argmax_cents"], out["true_cents"])
                rec_metrics_expected = extended_pitch_metrics(out["expected_cents"], out["true_cents"])
                per_recording_by_variant[variant][rec_id] = {
                    "fold": fold, "n_valid": out["n_valid"],
                    "argmax": rec_metrics_argmax, "expected": rec_metrics_expected,
                    "median_rank": float(np.median(out["rank"])), "mean_rank": float(np.mean(out["rank"])),
                    "mean_entropy": float(np.mean(out["entropy"])),
                }

                if variant == "harmonic" and rec_id not in hps_per_recording:
                    lane = next(x for x in index.lanes if x.recording_id == rec_id)
                    frames = index._frames[(rec_id, PRIMARY_LANE)]
                    n = min(index._features[rec_id]["cqt_log"].shape[1], lane.n_frames)
                    mag = linear_mag(index._features[rec_id]["cqt_log"][:, :n])
                    times = frames["frame_time_s"][:n]
                    valid = frames["valid_target"][:n] & (times < lane.duration_s)
                    true_c = np.asarray(log2_hz_to_cents(frames["pitch_log2_hz"][:n], lane.fundamental_hz))
                    hps_pred = harmonic_product_cents(mag, lane.fundamental_hz)
                    pv, tv = hps_pred[valid], true_c[valid]
                    hps_per_recording[rec_id] = {"fold": fold, "n_valid": int(valid.sum()), **extended_pitch_metrics(pv, tv)}
                    hps_pooled["pred"].append(pv)
                    hps_pooled["true"].append(tv)
                    hps_pooled["rec"].append(np.full(len(pv), rec_id, dtype=object))
                    hps_pooled["fold"].append(np.full(len(pv), fold, dtype=np.int32))

            fa, fe, ft = np.concatenate(fold_argmax), np.concatenate(fold_expected), np.concatenate(fold_true)
            per_variant_fold[variant].append({
                "fold": fold, "n_params": ckpt["n_params"], "best_epoch": ckpt["best_epoch"],
                "argmax": extended_pitch_metrics(fa, ft), "expected": extended_pitch_metrics(fe, ft),
            })
            print(f"{variant} fold {fold}: argmax MAE={extended_pitch_metrics(fa, ft)['mae_cents']:.1f} "
                  f"expected MAE={extended_pitch_metrics(fe, ft)['mae_cents']:.1f}")

    # ---- decode comparison + primary metrics + baseline table ----
    decode_comparison = {}
    primary = {}
    for variant in ("local", "harmonic"):
        argmax_maes = [f["argmax"]["mae_cents"] for f in per_variant_fold[variant]]
        expected_maes = [f["expected"]["mae_cents"] for f in per_variant_fold[variant]]
        primary_decode = "argmax" if np.mean(argmax_maes) <= np.mean(expected_maes) else "expected"
        decode_comparison[variant] = {
            "argmax_mean_mae": float(np.mean(argmax_maes)), "argmax_per_fold": argmax_maes,
            "expected_mean_mae": float(np.mean(expected_maes)), "expected_per_fold": expected_maes,
            "primary_decode": primary_decode,
        }
        primary[variant] = {
            "per_fold": [f[primary_decode] for f in per_variant_fold[variant]],
            "mean_mae": float(np.mean([f[primary_decode]["mae_cents"] for f in per_variant_fold[variant]])),
            "mean_median_ae": float(np.mean([f[primary_decode]["median_ae_cents"] for f in per_variant_fold[variant]])),
            "mean_pct_within_50": float(np.mean([f[primary_decode]["pct_within_50"] for f in per_variant_fold[variant]])),
            "mean_octave_adjusted_mae": float(np.mean([f[primary_decode]["octave_adjusted_mae"] for f in per_variant_fold[variant]])),
            "n_params": per_variant_fold[variant][0]["n_params"],
        }

    hps_repro = OUT_DIR / "harmonic_salience_hps_reproduction.json"
    import json as _json
    hps_repro_data = _json.loads(hps_repro.read_text()) if hps_repro.exists() else None

    report_numbers = {
        "candidate_range": {"lo_bin": lo_bin, "hi_bin": hi_bin, "n_candidates": hi_bin - lo_bin},
        "decode_comparison": decode_comparison,
        "primary_metrics": primary,
        "hps_reproduction": hps_repro_data,
        "step10_baselines_mean_mae": {
            "train_mean_baseline_scalar_abs_fold_mean": None,  # filled below if scalar_abs report present
            "cqt_argmax": 903.76, "tonic_neighborhood": 822.61, "harmonic_product_hps": 279.04,
            "pyin": 454.32, "step10_scalar_cnn_abs": 527.63,
        },
    }
    write_json(OUT_DIR / "harmonic_salience_report_numbers.json", report_numbers)

    # ---- per-class (§21) ----
    per_class = {}
    for variant in ("local", "harmonic"):
        decode_key = decode_comparison[variant]["primary_decode"]
        pred_all = np.concatenate(per_variant_pooled[variant][decode_key])
        true_all = np.concatenate(per_variant_pooled[variant]["true"])
        type_all = np.concatenate(per_variant_pooled[variant]["type"])
        per_class[variant] = pitch_metrics_by_type(pred_all, true_all, type_all)
    hps_pred_all = np.concatenate(hps_pooled["pred"])
    hps_true_all = np.concatenate(hps_pooled["true"])
    # HPS per-class needs trajectory_type too; reuse harmonic variant's type array
    # (same frames/order not guaranteed identical order to hps_pooled, so recompute per-type via per-recording loop)
    hps_type_all = np.concatenate(per_variant_pooled["harmonic"]["type"])
    per_class["hps"] = pitch_metrics_by_type(hps_pred_all, hps_true_all, hps_type_all) if len(hps_pred_all) == len(hps_type_all) else {"note": "length mismatch, see per_recording instead"}
    write_json(OUT_DIR / "harmonic_salience_per_class.json", per_class)

    # ---- per-recording (§22) ----
    per_recording_report = {}
    all_rec_ids = sorted(set(list(per_recording_by_variant["harmonic"].keys()) + list(hps_per_recording.keys())))
    for rid in all_rec_ids:
        h = per_recording_by_variant["harmonic"].get(rid)
        l = per_recording_by_variant["local"].get(rid)
        hps_r = hps_per_recording.get(rid)
        if h is None or hps_r is None:
            continue
        decode_key = decode_comparison["harmonic"]["primary_decode"]
        harm_mae = h[decode_key]["mae_cents"]
        entry = {
            "fold": h["fold"], "support": h["n_valid"],
            "hps_mae": hps_r["mae_cents"], "harmonic_mae": harm_mae,
            "delta_mae": hps_r["mae_cents"] - harm_mae,
            "hps_octave_adjusted_mae": hps_r["octave_adjusted_mae"],
            "harmonic_octave_adjusted_mae": h[decode_key]["octave_adjusted_mae"],
            "delta_octave_adjusted_mae": hps_r["octave_adjusted_mae"] - h[decode_key]["octave_adjusted_mae"],
        }
        if l is not None:
            entry["local_mae"] = l[decode_comparison["local"]["primary_decode"]]["mae_cents"]
        per_recording_report[rid] = entry
    per_recording_sorted = dict(sorted(per_recording_report.items(), key=lambda kv: -kv[1]["delta_mae"]))
    write_json(OUT_DIR / "harmonic_salience_per_recording.json", per_recording_sorted)

    # ---- rank/entropy (§23-24) ----
    rank_entropy = {}
    for variant in ("local", "harmonic"):
        rank_all = np.concatenate(per_variant_pooled[variant]["rank"])
        ent_all = np.concatenate(per_variant_pooled[variant]["entropy"])
        decode_key = decode_comparison[variant]["primary_decode"]
        pred_all = np.concatenate(per_variant_pooled[variant][decode_key])
        true_all = np.concatenate(per_variant_pooled[variant]["true"])
        err_all = np.abs(pred_all - true_all)
        corr = float(np.corrcoef(ent_all, err_all)[0, 1]) if len(ent_all) > 1 else None
        rank_entropy[variant] = {
            "n": int(len(rank_all)),
            "fraction_gt_in_top1": float((rank_all <= 1).mean()),
            "fraction_gt_in_top3": float((rank_all <= 3).mean()),
            "fraction_gt_in_top5": float((rank_all <= 5).mean()),
            "median_rank": float(np.median(rank_all)),
            "mean_rank": float(np.mean(rank_all)),
            "mean_entropy_nats": float(np.mean(ent_all)),
            "median_entropy_nats": float(np.median(ent_all)),
            "entropy_vs_abs_error_correlation": corr,
        }
    write_json(OUT_DIR / "harmonic_salience_rank_entropy.json", rank_entropy)

    # ---- octave confusion for learned models (§25) ----
    octave_learned = {}
    for variant in ("local", "harmonic"):
        decode_key = decode_comparison[variant]["primary_decode"]
        pred_all = np.concatenate(per_variant_pooled[variant][decode_key])
        true_all = np.concatenate(per_variant_pooled[variant]["true"])
        rank_all = np.concatenate(per_variant_pooled[variant]["rank"])
        oct_err, best_k = octave_adjusted_error(pred_all, true_all)
        octave_learned[variant] = {
            "n": int(len(pred_all)),
            "fraction_top1_correct_octave": float((best_k == 0).mean()),
            "fraction_top1_within_1_octave": float(np.isin(best_k, [-1, 0, 1]).mean()),
            "fraction_correct_pitch_in_top3": float((rank_all <= 3).mean()),
            "raw_mae": float(np.abs(pred_all - true_all).mean()),
            "octave_adjusted_mae": float(oct_err.mean()),
        }
    write_json(OUT_DIR / "harmonic_salience_octave_confusion_learned.json", octave_learned)

    # ---- harmonic-channel ablation (§27), harmonic variant only ----
    ablation = {}
    ablate_sets = {"ablate_2f": frozenset({2}), "ablate_3f": frozenset({3}), "ablate_4f": frozenset({4}),
                   "fundamental_only_ablate_234": frozenset({2, 3, 4})}
    for name, ks in ablate_sets.items():
        fold_maes = []
        for fold in range(5):
            split = build_fold_split(manifest, fold, seed=42)
            mu, sigma = load_fold_cqt_stats(fold, repo_root)
            model, _ckpt = load_model("harmonic", fold, lo_bin, hi_bin)
            preds, trues = [], []
            for rec_id in split.test_recording_ids:
                out = evaluate_recording(model, mu, sigma, cand_cents_abs, rec_id, index, ablate_ks=ks)
                if out["n_valid"] == 0:
                    continue
                decode_key = decode_comparison["harmonic"]["primary_decode"]
                preds.append(out[f"{decode_key}_cents"])
                trues.append(out["true_cents"])
            if preds:
                fold_maes.append(extended_pitch_metrics(np.concatenate(preds), np.concatenate(trues))["mae_cents"])
        ablation[name] = {"per_fold_mae": fold_maes, "mean_mae": float(np.mean(fold_maes))}
    ablation["no_ablation_baseline_mean_mae"] = primary["harmonic"]["mean_mae"]
    write_json(OUT_DIR / "harmonic_salience_ablation.json", ablation)

    print("\n=== Summary ===")
    print("HPS mean MAE (frozen):", round(hps_repro_data["mean_mae"], 1) if hps_repro_data else "?")
    for variant in ("local", "harmonic"):
        print(f"{variant} mean MAE ({decode_comparison[variant]['primary_decode']}):", round(primary[variant]["mean_mae"], 1))


if __name__ == "__main__":
    run_full_eval()
