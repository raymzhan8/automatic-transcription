"""Step 11 §3-5,7: reproduce frozen HPS baseline, generalize to a full salience
map, and characterize HPS failures before any learned model is built.

Frozen benchmark (k=2,3, matches Step 10 exactly): reuses
``baselines_b.harmonic_product_cents`` unmodified. The salience-map / failure
analysis functions in this file generalize that SAME k=2,3 product into a
full [F_cand, T] distribution (not just its argmax) — the learned model's
input features (salience_features.py) separately extend to k=1..4, which is
a deliberate, documented difference (see plan §"Decisions on spec
ambiguities" #1), tested via the harmonic-channel ablation later, not a
change to this frozen benchmark.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.features import N_BINS  # noqa: E402
from training.folds import build_fold_split, load_kfold_manifest  # noqa: E402
from training.framewise_dataset import RecordingLaneIndex  # noqa: E402
from training.metrics import NUM_TYPES, TYPE_NAMES, pitch_error_metrics  # noqa: E402
from training.normalization import log2_hz_to_cents  # noqa: E402
from training.pitch_diagnostics.analyze_a import NEIGHBOR, _frame_energy_stats  # noqa: E402
from training.pitch_diagnostics.baselines_b import (  # noqa: E402
    cqt_argmax_cents,
    eval_fold,
    harmonic_product_cents,
)
from training.pitch_diagnostics.common import (  # noqa: E402
    OUT_DIR,
    PRIMARY_LANE,
    linear_mag,
    octave_adjusted_error,
    summarize_array,
    write_json,
)
from training.pitch_diagnostics.salience_common import harmonic_bin_offset  # noqa: E402

HPS_HARMONICS = (2, 3)  # frozen, matches Step 10 exactly


def hps_product_array(mag: np.ndarray, k_list: tuple[int, ...] = HPS_HARMONICS) -> np.ndarray:
    """Full [F,T] harmonic-product array (generalization of harmonic_product_cents
    before its final argmax). Identical math/constants to baselines_b.harmonic_product_cents.
    """
    f_bins, n_t = mag.shape
    hps = mag.copy()
    for k in k_list:
        shift = harmonic_bin_offset(k)
        padded = np.zeros_like(mag)
        if shift < f_bins:
            padded[: f_bins - shift] = mag[shift:]
        hps = hps * np.maximum(padded, 1e-12)
    return hps


def hps_salience_probs(
    mag: np.ndarray,
    lo_bin: int,
    hi_bin: int,
    *,
    k_list: tuple[int, ...] = HPS_HARMONICS,
) -> np.ndarray:
    """Deterministic salience distribution S_HPS(f,t) over candidate bins [lo_bin,hi_bin)."""
    full = hps_product_array(mag, k_list)
    cand = full[lo_bin:hi_bin, :]
    s = cand.sum(axis=0, keepdims=True)
    s = np.maximum(s, 1e-12)
    return cand / s


def reproduce_frozen_hps_baseline(index: RecordingLaneIndex, manifest: dict[str, Any]) -> dict[str, Any]:
    """Exactly mirrors baselines_b.main()'s 'harmonic_product' loop; sanity gate."""
    folds = []
    for i in range(5):
        split = build_fold_split(manifest, i, seed=42)
        folds.append({"fold": i, **eval_fold(index, split.test_recording_ids, harmonic_product_cents)})
    maes = [f["overall"]["mae_cents"] for f in folds]
    return {
        "folds": folds,
        "fold_maes": [round(float(m), 1) for m in maes],
        "mean_mae": float(np.mean(maes)),
        "expected_fold_maes_step10": [210, 264, 409, 354, 158],
        "expected_mean_step10": 279,
    }


def _duration_bucket(dur_s: float) -> str:
    if not np.isfinite(dur_s):
        return "unknown"
    if dur_s < 0.1:
        return "<0.1s"
    if dur_s < 0.25:
        return "0.1-0.25s"
    if dur_s < 0.5:
        return "0.25-0.5s"
    if dur_s < 1.0:
        return "0.5-1.0s"
    if dur_s < 2.0:
        return "1.0-2.0s"
    return ">=2.0s"


def _rank_bucket(rank: float) -> str:
    r = int(rank)
    if r <= 1:
        return "rank_1"
    if r <= 5:
        return "rank_2_5"
    if r <= 10:
        return "rank_6_10"
    if r <= 25:
        return "rank_11_25"
    return "rank_gt_25"


def _register_bucket(cents: float) -> str:
    # 200-cent-wide buckets over tonic-relative cents, labeled by lower edge.
    if not np.isfinite(cents):
        return "unknown"
    edge = int(np.floor(cents / 200.0) * 200)
    return f"[{edge},{edge + 200})"


def hps_failure_analysis(
    index: RecordingLaneIndex,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Per-frame HPS error characterized by type/rank/recording/fold/register/duration,
    plus octave-confusion. Runs BEFORE any learned model, per spec ordering.
    """
    fold_of: dict[str, str] = dict(manifest["assignments"])

    all_pred, all_true, all_type, all_rank, all_dur, all_fold, all_rec = ([] for _ in range(7))
    per_recording: dict[str, Any] = {}

    for lane in index.lanes:
        rid = lane.recording_id
        frames = index._frames[(rid, PRIMARY_LANE)]
        cqt = index._features[rid]["cqt_log"]
        n = min(cqt.shape[1], lane.n_frames)
        cqt = cqt[:, :n]
        times = frames["frame_time_s"][:n]
        valid = frames["valid_target"][:n] & (times < lane.duration_s)
        if not np.any(valid):
            continue
        mag = linear_mag(cqt)
        true_cents = np.asarray(log2_hz_to_cents(frames["pitch_log2_hz"][:n], lane.fundamental_hz), dtype=np.float64)
        pred_cents = harmonic_product_cents(mag, lane.fundamental_hz)
        traj = frames["trajectory_type"][:n]
        prim_ids = frames["primitive_id"][:n]

        target_hz = np.exp2(frames["pitch_log2_hz"][:n].astype(np.float64))
        vis = _frame_energy_stats(mag[:, valid], target_hz[valid], lane.fundamental_hz)
        rank_v = vis["rank"]

        dur_v = np.full(int(valid.sum()), np.nan, dtype=np.float64)
        valid_prim_ids = prim_ids[valid]
        cache: dict[str, float] = {}
        for i, pid in enumerate(valid_prim_ids):
            pid = str(pid)
            if not pid:
                continue
            if pid not in cache:
                prim = index.get_primitive(rid, pid)
                cache[pid] = float(prim["end_s"] - prim["start_s"]) if prim else np.nan
            dur_v[i] = cache[pid]

        pv = pred_cents[valid]
        tv = true_cents[valid]
        typev = traj[valid]
        fold_name = fold_of.get(rid, "unknown")

        oct_err, _best_k_rec = octave_adjusted_error(pv, tv)

        per_recording[rid] = {
            "fold": fold_name,
            "n_valid": int(valid.sum()),
            "raw_metrics": pitch_error_metrics(pv, tv),
            "octave_adjusted_mae": float(oct_err.mean()),
            "octave_adjusted_median_ae": float(np.median(oct_err)),
        }

        all_pred.append(pv)
        all_true.append(tv)
        all_type.append(typev)
        all_rank.append(rank_v)
        all_dur.append(dur_v)
        all_fold.append(np.full(pv.shape, fold_name, dtype=object))
        all_rec.append(np.full(pv.shape, rid, dtype=object))

    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    typ = np.concatenate(all_type)
    rank = np.concatenate(all_rank)
    dur = np.concatenate(all_dur)
    fold_arr = np.concatenate(all_fold)
    rec_arr = np.concatenate(all_rec)

    raw_err = np.abs(pred - true)
    oct_err, best_k = octave_adjusted_error(pred, true)

    def bucket_metrics(mask: np.ndarray) -> dict[str, Any]:
        return {
            "n": int(mask.sum()),
            "raw": pitch_error_metrics(pred[mask], true[mask]),
            "octave_adjusted_mae": float(oct_err[mask].mean()) if mask.any() else None,
            "octave_adjusted_median_ae": float(np.median(oct_err[mask])) if mask.any() else None,
        }

    by_type = {TYPE_NAMES[t]: bucket_metrics(typ == t) for t in range(NUM_TYPES)}

    rank_buckets = ["rank_1", "rank_2_5", "rank_6_10", "rank_11_25", "rank_gt_25"]
    rank_labels = np.array([_rank_bucket(r) for r in rank], dtype=object)
    by_rank = {b: bucket_metrics(rank_labels == b) for b in rank_buckets}

    by_fold = {f"fold_{i}": bucket_metrics(fold_arr == f"fold_{i}") for i in range(5)}

    reg_labels = np.array([_register_bucket(c) for c in true], dtype=object)
    reg_buckets = sorted(set(reg_labels.tolist()), key=lambda s: (s == "unknown", s))
    by_register = {b: bucket_metrics(reg_labels == b) for b in reg_buckets}

    dur_buckets = ["<0.1s", "0.1-0.25s", "0.25-0.5s", "0.5-1.0s", "1.0-2.0s", ">=2.0s", "unknown"]
    dur_labels = np.array([_duration_bucket(d) for d in dur], dtype=object)
    by_duration = {b: bucket_metrics(dur_labels == b) for b in dur_buckets}

    by_recording = {
        rid: {**block, "delta_raw_vs_octave_mae": block["raw_metrics"]["mae_cents"] - block["octave_adjusted_mae"]}
        for rid, block in per_recording.items()
    }

    failure_analysis = {
        "n_valid_frames_total": int(len(pred)),
        "overall_raw": pitch_error_metrics(pred, true),
        "overall_octave_adjusted_mae": float(oct_err.mean()),
        "overall_octave_adjusted_median_ae": float(np.median(oct_err)),
        "by_type": by_type,
        "by_spectral_rank": by_rank,
        "by_fold": by_fold,
        "by_register_tonic_relative_cents": by_register,
        "by_primitive_duration": by_duration,
        "by_recording": by_recording,
        "rank_bucket_definition": "rank_1 (spectral peak), rank_2_5, rank_6_10, rank_11_25, rank_gt_25 — rank of target's CQT bin among all 360 bins, 1=loudest (same methodology as Step 10 visibility.json / analyze_a._frame_energy_stats)",
        "register_bucket_definition": "200-cent-wide buckets of tonic-relative target cents, labeled by lower edge",
        "duration_bucket_definition": "primitive duration_s = end_s - start_s from dataset/canonical primitives doc, looked up per frame's primitive_id",
    }

    k_counts = {int(k): int((best_k == k).sum()) for k in sorted(set(best_k.tolist()))}
    octave_confusion = {
        "n_valid_frames_total": int(len(pred)),
        "raw_mae": float(raw_err.mean()),
        "octave_adjusted_mae": float(oct_err.mean()),
        "improvement_from_octave_adjustment_cents": float(raw_err.mean() - oct_err.mean()),
        "best_octave_shift_k_counts": k_counts,
        "best_octave_shift_k_fractions": {k: v / len(pred) for k, v in k_counts.items()},
        "fraction_top1_correct_octave": float((best_k == 0).mean()),
        "fraction_top1_within_1_octave": float(np.isin(best_k, [-1, 0, 1]).mean()),
        "by_fold_octave_shift_k_fractions": {
            f"fold_{i}": {
                int(k): float((best_k[fold_arr == f"fold_{i}"] == k).mean())
                for k in sorted(set(best_k[fold_arr == f"fold_{i}"].tolist()))
            }
            for i in range(5)
            if (fold_arr == f"fold_{i}").any()
        },
        "note": "best_octave_shift_k: pred - true - 1200*k minimizes |error|; k=0 means already correct octave.",
    }

    return failure_analysis, octave_confusion


def main() -> None:
    index = RecordingLaneIndex.build(REPO_ROOT)
    manifest = load_kfold_manifest(REPO_ROOT)

    repro = reproduce_frozen_hps_baseline(index, manifest)
    print("HPS reproduction: fold MAEs", repro["fold_maes"], "mean", round(repro["mean_mae"], 1))
    print("Step 10 expected:", repro["expected_fold_maes_step10"], "mean", repro["expected_mean_step10"])
    write_json(OUT_DIR / "harmonic_salience_hps_reproduction.json", repro)

    failure_analysis, octave_confusion = hps_failure_analysis(index, manifest)
    write_json(OUT_DIR / "harmonic_salience_hps_failure_analysis.json", failure_analysis)
    write_json(OUT_DIR / "harmonic_salience_octave_confusion.json", octave_confusion)

    print("\nBy type (raw MAE / oct-adj MAE):")
    for t, block in failure_analysis["by_type"].items():
        print(f"  {t}: n={block['n']} raw={block['raw']['mae_cents']:.1f} oct_adj={block['octave_adjusted_mae']}")
    print("\nOctave confusion: fraction top1 correct octave =", round(octave_confusion["fraction_top1_correct_octave"], 3))
    print("k fractions:", {k: round(v, 3) for k, v in octave_confusion["best_octave_shift_k_fractions"].items()})


if __name__ == "__main__":
    main()
