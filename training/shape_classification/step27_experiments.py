"""Step 27: does one small nonlinear audio-pitch interaction layer (L1)
recover Sloped-start without sacrificing Step 26's Cosine improvement over
linear fusion (L0/A2)? Loads A0 and L0 from Step 26's cached results
(reused, not retrained -- byte-identical protocol), trains L1, then runs
the full comparison battery.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.shape_classification.dataset import CLASS_NAMES, OUT_DIR, build  # noqa: E402
from training.shape_classification.duration_span_analysis import DURATION_BUCKET_NAMES, DURATION_BUCKETS_S  # noqa: E402
from training.shape_classification.metrics_utils import eval_metrics, prediction_frequency  # noqa: E402
from training.shape_classification.step25_experiments import fold_consistency, recording_consistency  # noqa: E402
from training.shape_classification.step27_train import FOUR_CLASS_NAMES  # noqa: E402

STEP26_DIR = OUT_DIR / "step26"
STEP27_DIR = OUT_DIR / "step27"
N_FOLDS = 5


def load_cached(path: Path) -> dict:
    d = json.loads(path.read_text())
    d["per_fold_macro_f1"] = {int(k): v for k, v in d["per_fold_macro_f1"].items()}
    for fold in d["folds"]:
        fold["fold"] = int(fold["fold"])
    return d


def pooled_pred_true(result: dict) -> tuple[np.ndarray, np.ndarray]:
    pred = np.concatenate([np.array(f["test_pred"]) for f in result["folds"]])
    true = np.concatenate([np.array(f["test_true"]) for f in result["folds"]])
    return pred, true


def _duration_lookup(records: list[dict]) -> dict[str, float]:
    return {r["primitive_id"]: r["duration_s"] for r in records}


def bucket_by_duration(result: dict, lookup: dict[str, float]) -> dict[str, dict]:
    by_pid_pred, by_pid_true = {}, {}
    for fold in result["folds"]:
        for pid, p, t in zip(fold["test_primitive_id"], fold["test_pred"], fold["test_true"]):
            by_pid_pred[pid] = p; by_pid_true[pid] = t
    out = {}
    for name, lo, hi in zip(DURATION_BUCKET_NAMES, DURATION_BUCKETS_S[:-1], DURATION_BUCKETS_S[1:]):
        pids = [pid for pid, v in lookup.items() if pid in by_pid_pred and lo <= v < hi]
        if not pids:
            out[name] = {"n": 0}; continue
        pred = np.array([by_pid_pred[pid] for pid in pids]); true = np.array([by_pid_true[pid] for pid in pids])
        m = eval_metrics(pred, true, FOUR_CLASS_NAMES)
        out[name] = {"n": len(pids), "macro_f1": m["macro_f1"],
                      "sloped_start_f1": m["per_class"]["Sloped-start"]["f1"],
                      "sloped_start_recall": m["per_class"]["Sloped-start"]["recall"]}
    return out


def fusion_usage_diagnostic(l1: dict) -> dict:
    pooled_true = np.concatenate([np.array(f["test_true"]) for f in l1["folds"]])
    normal_pred = np.concatenate([np.array(f["test_pred"]) for f in l1["folds"]])
    audio_zeroed_pred = np.concatenate([np.array(f["audio_zeroed_test_pred"]) for f in l1["folds"]])
    pitch_zeroed_pred = np.concatenate([np.array(f["pitch_zeroed_test_pred"]) for f in l1["folds"]])
    normal_f1 = eval_metrics(normal_pred, pooled_true, FOUR_CLASS_NAMES)["macro_f1"]
    audio_zeroed_f1 = eval_metrics(audio_zeroed_pred, pooled_true, FOUR_CLASS_NAMES)["macro_f1"]
    pitch_zeroed_f1 = eval_metrics(pitch_zeroed_pred, pooled_true, FOUR_CLASS_NAMES)["macro_f1"]
    return {"l1_normal_macro_f1": normal_f1, "l1_audio_zeroed_macro_f1": audio_zeroed_f1,
            "l1_pitch_zeroed_macro_f1": pitch_zeroed_f1,
            "audio_contributes": normal_f1 > audio_zeroed_f1, "pitch_contributes": normal_f1 > pitch_zeroed_f1}


def interaction_diagnostic(l1: dict) -> dict:
    """Section 16: class-wise standardized mean difference of the hidden
    fusion activation z, true Cosine vs true Sloped-start, held-out test
    examples pooled across folds."""
    z_all, y_all = [], []
    for fold in l1["folds"]:
        z_all.extend(fold["test_z"]); y_all.extend(fold["test_true"])
    z = np.array(z_all); y = np.array(y_all)
    z_cosine = z[y == 1]; z_sloped_start = z[y == 2]
    mean_c, mean_s = z_cosine.mean(axis=0), z_sloped_start.mean(axis=0)
    pooled_std = np.sqrt((z_cosine.var(axis=0) * len(z_cosine) + z_sloped_start.var(axis=0) * len(z_sloped_start))
                          / (len(z_cosine) + len(z_sloped_start) - 2))
    pooled_std = np.maximum(pooled_std, 1e-8)
    smd = (mean_c - mean_s) / pooled_std
    return {"n_cosine": int(len(z_cosine)), "n_sloped_start": int(len(z_sloped_start)),
            "mean_abs_standardized_diff": float(np.mean(np.abs(smd))),
            "n_dims_abs_diff_over_0.5": int((np.abs(smd) > 0.5).sum()),
            "n_dims_abs_diff_over_1.0": int((np.abs(smd) > 1.0).sum()),
            "per_dim_standardized_diff": smd.tolist()}


def recovery_breakage_analysis(records: list[dict], l0: dict, l1: dict, n: int = 4) -> dict[str, Any]:
    rec_by_pid = {r["primitive_id"]: r for r in records}
    l0_by_pid, l1_by_pid = {}, {}
    for fold in l0["folds"]:
        for pid, p, t in zip(fold["test_primitive_id"], fold["test_pred"], fold["test_true"]):
            l0_by_pid[pid] = (p, t)
    for fold in l1["folds"]:
        for pid, p, t, probs in zip(fold["test_primitive_id"], fold["test_pred"], fold["test_true"], fold["test_probs"]):
            l1_by_pid[pid] = (p, t, probs)

    common = sorted(set(l0_by_pid) & set(l1_by_pid))
    # Set A: L0 predicts Cosine(1), true Sloped-start(2) -> does L1 recover (predict 2)?
    set_a = [pid for pid in common if l0_by_pid[pid][0] == 1 and l0_by_pid[pid][1] == 2]
    recovered = [pid for pid in set_a if l1_by_pid[pid][0] == 2]
    # Set B: L0 correctly predicts Cosine (pred=true=1) -> does L1 break it (pred != 1)?
    set_b = [pid for pid in common if l0_by_pid[pid][0] == 1 and l0_by_pid[pid][1] == 1]
    broken = [pid for pid in set_b if l1_by_pid[pid][0] != 1]

    def describe(pid: str) -> dict:
        r = rec_by_pid[pid]
        return {"primitive_id": pid, "recording_id": r["recording_id"], "duration_s": round(r["duration_s"], 3),
                "true": CLASS_NAMES[r["canonical_type"]], "L0_pred": CLASS_NAMES[l0_by_pid[pid][0]],
                "L1_pred": CLASS_NAMES[l1_by_pid[pid][0]], "L1_confidence": round(max(l1_by_pid[pid][2]), 3)}

    recovered_sorted = sorted(recovered, key=lambda pid: max(l1_by_pid[pid][2]), reverse=True)
    broken_sorted = sorted(broken, key=lambda pid: max(l1_by_pid[pid][2]), reverse=True)
    return {
        "n_set_A_L0_cosine_true_sloped_start": len(set_a), "n_recovered_by_L1": len(recovered),
        "n_set_B_L0_correct_cosine": len(set_b), "n_broken_by_L1": len(broken),
        "top_recovered": [describe(pid) for pid in recovered_sorted[:n]],
        "top_broken": [describe(pid) for pid in broken_sorted[:n]],
    }


def fold0_class_breakdown(a0: dict, l0: dict, l1: dict) -> dict:
    out = {}
    for tag, res in (("A0", a0), ("L0", l0), ("L1", l1)):
        fold0 = next(f for f in res["folds"] if f["fold"] == 0)
        out[tag] = {"macro_f1": fold0["test_metrics"]["macro_f1"],
                     "per_class_f1": {k: v["f1"] for k, v in fold0["test_metrics"]["per_class"].items()}}
    return out


def main() -> None:
    STEP27_DIR.mkdir(parents=True, exist_ok=True)
    records = build()

    a0 = load_cached(STEP26_DIR / "a0_full.json")
    l0 = load_cached(STEP26_DIR / "a2_full.json")  # Step 26's linear fusion, reused unchanged
    l1 = load_cached(STEP27_DIR / "l1_full.json")

    print("=== §2 L0 reproduction check ===")
    print(f"L0 pooled macro_f1={l0['pooled']['macro_f1']:.4f} (Step 26 A2 reference: 0.3668)")
    print(f"L0 grouped mean={l0['grouped_mean_macro_f1']:.4f}+/-{l0['grouped_std_macro_f1']:.4f} (reference: 0.3500+/-0.0770)")

    print("\n=== §5 parameter counts ===")
    print(f"L0 n_params={l0['n_params']}  L1 n_params={l1['n_params']}  delta={l1['n_params']-l0['n_params']}")

    print("\n=== §7 primary result table ===")
    for tag, res in (("A0", a0), ("L0", l0), ("L1", l1)):
        p = res["pooled"]
        row = " ".join(f"{p['per_class'][n]['f1']:.3f}" for n in FOUR_CLASS_NAMES)
        print(f"{tag:4s} macro_f1={p['macro_f1']:.4f} acc={p['accuracy']:.4f} [{','.join(FOUR_CLASS_NAMES)}]={row} "
              f"grouped_mean={res['grouped_mean_macro_f1']:.4f}+/-{res['grouped_std_macro_f1']:.4f}")

    # §9 fold consistency
    fc = fold_consistency(l0, l1)
    print("\n=== §9 fold consistency (L1 vs L0) ===", fc)

    # §10 recording consistency
    rc = recording_consistency(l0, l1)
    print("\n=== §10 recording consistency (L1 vs L0) ===", {k: v for k, v in rc.items() if k != "per_recording_delta"})

    # §11-12 per-class + Cosine/Sloped-start table
    per_class = {tag: res["pooled"]["per_class"] for tag, res in (("L0", l0), ("L1", l1))}
    print("\n=== §12 Cosine <-> Sloped-start table ===")
    for cls in ("Cosine", "Sloped-start"):
        for metric in ("precision", "recall", "f1"):
            v0, v1 = per_class["L0"][cls][metric], per_class["L1"][cls][metric]
            print(f"  {cls} {metric}: L0={v0:.3f} L1={v1:.3f} delta={v1-v0:+.3f}")

    # §13 confusion matrices
    confusion = {tag: res["pooled"]["confusion_matrix"] for tag, res in (("L0", l0), ("L1", l1))}
    print("\n=== §13 confusion matrices ===")
    for tag in ("L0", "L1"):
        print(tag)
        for row in confusion[tag]:
            print(" ", row)

    # §14 prediction frequency
    _, true_pooled = pooled_pred_true(l0)
    pred_freq = {"true": prediction_frequency(true_pooled, FOUR_CLASS_NAMES)}
    for tag, res in (("L0", l0), ("L1", l1)):
        pred, _ = pooled_pred_true(res)
        pred_freq[tag] = prediction_frequency(pred, FOUR_CLASS_NAMES)
    print("\n=== §14 prediction frequency ===")
    for k, v in pred_freq.items():
        print(k, {kk: round(vv, 3) for kk, vv in v.items()})

    # §15 modality-zeroing
    fusion_diag = fusion_usage_diagnostic(l1)
    print("\n=== §15 modality-zeroing sanity check (L1) ===", fusion_diag)

    # §16 interaction diagnostic
    interaction = interaction_diagnostic(l1)
    print("\n=== §16 interaction diagnostic (z, Cosine vs Sloped-start) ===")
    print(f"  mean|standardized diff|={interaction['mean_abs_standardized_diff']:.3f}  "
          f"dims>0.5={interaction['n_dims_abs_diff_over_0.5']}/16  dims>1.0={interaction['n_dims_abs_diff_over_1.0']}/16")

    # §17 recovery/breakage
    recovery = recovery_breakage_analysis(records, l0, l1)
    print(f"\n=== §17 recovery/breakage === Set A (L0 Cosine/true SlS)={recovery['n_set_A_L0_cosine_true_sloped_start']} "
          f"recovered={recovery['n_recovered_by_L1']}  Set B (L0 correct Cosine)={recovery['n_set_B_L0_correct_cosine']} "
          f"broken={recovery['n_broken_by_L1']}")

    # §18 duration buckets
    dur_lookup = _duration_lookup(records)
    duration_report = {tag: bucket_by_duration(res, dur_lookup) for tag, res in (("L0", l0), ("L1", l1))}
    print("\n=== §18 duration buckets (Sloped-start F1/recall) ===")
    for name in DURATION_BUCKET_NAMES:
        d0, d1 = duration_report["L0"][name], duration_report["L1"][name]
        print(f"  {name:12s} L0 n={d0.get('n')} SlS_F1={d0.get('sloped_start_f1')}  "
              f"L1 n={d1.get('n')} SlS_F1={d1.get('sloped_start_f1')}")

    # §19 fold 0
    fold0 = fold0_class_breakdown(a0, l0, l1)
    print("\n=== §19 fold 0 explicit breakdown ===")
    for tag, d in fold0.items():
        print(f"  {tag}: macro_f1={d['macro_f1']:.4f}  per_class_f1={d['per_class_f1']}")

    out = {
        "L0_reproduction_check": {"pooled_macro_f1": l0["pooled"]["macro_f1"], "grouped_mean": l0["grouped_mean_macro_f1"]},
        "param_counts": {"L0": l0["n_params"], "L1": l1["n_params"], "delta": l1["n_params"] - l0["n_params"]},
        "A0": {k: v for k, v in a0.items() if k != "folds"},
        "L0": {k: v for k, v in l0.items() if k != "folds"},
        "L1": {k: v for k, v in l1.items() if k != "folds"},
        "fold_consistency_L1_vs_L0": fc, "recording_consistency_L1_vs_L0": rc,
        "per_class": per_class, "confusion_matrices": confusion, "prediction_frequency": pred_freq,
        "fusion_usage_diagnostic": fusion_diag, "interaction_diagnostic": interaction,
        "recovery_breakage": recovery, "duration_buckets": duration_report, "fold0_breakdown": fold0,
    }
    (STEP27_DIR / "results.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(f"\nsaved to {STEP27_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
