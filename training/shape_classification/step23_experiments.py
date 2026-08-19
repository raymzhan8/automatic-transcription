"""Step 23: is Step 22's CREPE Sloped-start/Sloped-end zero-F1 collapse a
class-prior/objective problem, or a class-overlap/noise problem? Trains
exactly three four-class conditions (B0 unweighted / B1 balanced sampler /
B2 weighted CE) plus a binary T2-vs-T3 diagnostic and a natural-vs-balanced
3-way bend-only diagnostic (M0/M1), all on the frozen Step 22 CREPE
q(x)+dq/dx representation and frozen ContourCNN. Writes every result JSON
consumed by docs/step_23_balanced_shape_classification.md.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.shape_classification.dataset import OUT_DIR, build  # noqa: E402
from training.shape_classification.metrics_utils import eval_metrics, prediction_frequency  # noqa: E402
from training.shape_classification.step23_train import run_condition  # noqa: E402

STEP23_DIR = OUT_DIR / "step23"
SOURCE = "crepe"
INPUT_MODE = "shape_velocity"  # Step 22's best CREPE representation, frozen

FOUR_CLASS_IDS = (0, 1, 2, 3)
FOUR_CLASS_NAMES = ("Fixed", "Cosine", "Sloped-start", "Sloped-end")
T2T3_CLASS_IDS = (2, 3)
T2T3_NAMES = ("Sloped-start", "Sloped-end")
BEND_CLASS_IDS = (1, 2, 3)
BEND_NAMES = ("Cosine", "Sloped-start", "Sloped-end")

SIGN_TEST_T2T3_ACCURACY = 0.768  # Step 22 §18, reused for comparison only


def per_recording_macro_f1(result: dict, label_names: tuple[str, ...]) -> dict[str, float]:
    by_rid = defaultdict(lambda: ([], []))
    for fold in result["folds"]:
        for rid, p, t in zip(fold["test_recording_ids"], fold["test_pred"], fold["test_true"]):
            by_rid[rid][0].append(p)
            by_rid[rid][1].append(t)
    out = {}
    for rid, (preds, trues) in by_rid.items():
        out[rid] = eval_metrics(np.array(preds), np.array(trues), label_names)["macro_f1"]
    return out


def confidence_analysis(result: dict, label_names: tuple[str, ...], true_class: str, distractor_class: str) -> dict:
    ti = label_names.index(true_class)
    di = label_names.index(distractor_class)
    p_true, p_distractor = [], []
    for fold in result["folds"]:
        probs = np.array(fold["test_probs"])
        true = np.array(fold["test_true"])
        mask = true == ti
        if mask.sum() == 0:
            continue
        p_true.append(probs[mask, ti])
        p_distractor.append(probs[mask, di])
    if not p_true:
        return {"n": 0}
    p_true = np.concatenate(p_true); p_distractor = np.concatenate(p_distractor)
    return {
        "n": int(len(p_true)),
        "mean_P_true_class": float(p_true.mean()), "median_P_true_class": float(np.median(p_true)),
        f"mean_P_{distractor_class}": float(p_distractor.mean()), f"median_P_{distractor_class}": float(np.median(p_distractor)),
    }


def fold_consistency(b0: dict, other: dict) -> dict:
    deltas = {f: other["per_fold_macro_f1"][f] - b0["per_fold_macro_f1"][f] for f in b0["per_fold_macro_f1"]}
    vals = list(deltas.values())
    return {"per_fold_delta": deltas, "n_improved": sum(1 for v in vals if v > 0),
            "n_worsened": sum(1 for v in vals if v < 0), "median_delta": float(np.median(vals))}


def recording_consistency(b0: dict, other: dict, label_names: tuple[str, ...]) -> dict:
    r0 = per_recording_macro_f1(b0, label_names)
    r1 = per_recording_macro_f1(other, label_names)
    common = sorted(set(r0) & set(r1))
    deltas = {rid: r1[rid] - r0[rid] for rid in common}
    vals = list(deltas.values())
    return {"per_recording_delta": deltas, "n_improved": sum(1 for v in vals if v > 0),
            "n_worsened": sum(1 for v in vals if v < 0),
            "median_delta": float(np.median(vals)) if vals else None}


def main() -> None:
    records = build()
    STEP23_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    print("=== Step 23 §3-4: B0/B1/B2 four-class ===")
    for tag, balancing in (("B0", "none"), ("B1", "sampler"), ("B2", "weighted_ce")):
        print(f"training {tag} (balancing={balancing}) ...")
        results[tag] = run_condition(records, SOURCE, INPUT_MODE, FOUR_CLASS_IDS, FOUR_CLASS_NAMES,
                                      balancing=balancing, save_root=STEP23_DIR / tag.lower())
        p = results[tag]["pooled"]
        row = " ".join(f"{p['per_class'][n]['f1']:.3f}" for n in FOUR_CLASS_NAMES)
        print(f"{tag}: macro_f1={p['macro_f1']:.4f} [{','.join(FOUR_CLASS_NAMES)}]={row} "
              f"grouped_mean={results[tag]['grouped_mean_macro_f1']:.4f}")

    print("\n=== Step 23 §9: binary T2 vs T3 ===")
    results["binary_t2_t3"] = run_condition(records, SOURCE, INPUT_MODE, T2T3_CLASS_IDS, T2T3_NAMES,
                                             balancing="none", save_root=STEP23_DIR / "binary_t2_t3")
    p = results["binary_t2_t3"]["pooled"]
    print(f"binary T2/T3: acc={p['accuracy']:.4f} macro_f1={p['macro_f1']:.4f} "
          f"T2_F1={p['per_class']['Sloped-start']['f1']:.4f} T3_F1={p['per_class']['Sloped-end']['f1']:.4f} "
          f"(vs. sign-test baseline {SIGN_TEST_T2T3_ACCURACY:.3f} accuracy)")

    print("\n=== Step 23 §10: 3-way bend-only (Cosine/Sloped-start/Sloped-end), natural (M0) vs balanced (M1) ===")
    results["M0_natural"] = run_condition(records, SOURCE, INPUT_MODE, BEND_CLASS_IDS, BEND_NAMES,
                                           balancing="none", save_root=STEP23_DIR / "m0_natural")
    results["M1_balanced"] = run_condition(records, SOURCE, INPUT_MODE, BEND_CLASS_IDS, BEND_NAMES,
                                            balancing="sampler", save_root=STEP23_DIR / "m1_balanced")
    for tag in ("M0_natural", "M1_balanced"):
        p = results[tag]["pooled"]
        row = " ".join(f"{p['per_class'][n]['f1']:.3f}" for n in BEND_NAMES)
        print(f"{tag}: macro_f1={p['macro_f1']:.4f} [{','.join(BEND_NAMES)}]={row}")

    # §7 prediction-frequency diagnostic (pooled, four-class conditions)
    print("\n=== Step 23 §7: prediction frequency vs. true distribution (four-class) ===")
    true_pooled = np.concatenate([np.array(f["test_true"]) for f in results["B0"]["folds"]])
    true_dist = prediction_frequency(true_pooled, FOUR_CLASS_NAMES)
    pred_freq = {"true_distribution": true_dist}
    print("true dist:", {k: round(v, 3) for k, v in true_dist.items()})
    for tag in ("B0", "B1", "B2"):
        pooled_pred = np.concatenate([np.array(f["test_pred"]) for f in results[tag]["folds"]])
        pred_freq[tag] = prediction_frequency(pooled_pred, FOUR_CLASS_NAMES)
        print(f"{tag} pred freq:", {k: round(v, 3) for k, v in pred_freq[tag].items()})

    # §12 fold consistency (B1/B2 vs B0)
    fold_cons = {"B1_vs_B0": fold_consistency(results["B0"], results["B1"]),
                 "B2_vs_B0": fold_consistency(results["B0"], results["B2"])}

    # §13 per-recording consistency
    rec_cons = {"B1_vs_B0": recording_consistency(results["B0"], results["B1"], FOUR_CLASS_NAMES),
                "B2_vs_B0": recording_consistency(results["B0"], results["B2"], FOUR_CLASS_NAMES)}

    # §14 confidence analysis for true T2/T3 (four-class conditions)
    print("\n=== Step 23 §14: confidence analysis (true T2/T3 examples) ===")
    confidence = {}
    for tag in ("B0", "B1", "B2"):
        confidence[tag] = {
            "true_T2": confidence_analysis(results[tag], FOUR_CLASS_NAMES, "Sloped-start", "Cosine"),
            "true_T3": confidence_analysis(results[tag], FOUR_CLASS_NAMES, "Sloped-end", "Cosine"),
        }
        for cls in ("true_T2", "true_T3"):
            d = confidence[tag][cls]
            if d["n"] == 0:
                continue
            print(f"{tag} {cls}: n={d['n']} mean_P(true)={d['mean_P_true_class']:.4f} "
                  f"mean_P(Cosine)={d['mean_P_Cosine']:.4f}")

    out = {
        "B0": {k: v for k, v in results["B0"].items() if k != "folds"},
        "B1": {k: v for k, v in results["B1"].items() if k != "folds"},
        "B2": {k: v for k, v in results["B2"].items() if k != "folds"},
        "binary_t2_t3": {k: v for k, v in results["binary_t2_t3"].items() if k != "folds"},
        "M0_natural": {k: v for k, v in results["M0_natural"].items() if k != "folds"},
        "M1_balanced": {k: v for k, v in results["M1_balanced"].items() if k != "folds"},
        "prediction_frequency": pred_freq,
        "fold_consistency": fold_cons,
        "recording_consistency": rec_cons,
        "confidence_analysis": confidence,
        "sign_test_t2_t3_baseline_accuracy": SIGN_TEST_T2T3_ACCURACY,
    }
    (STEP23_DIR / "results.json").write_text(json.dumps(out, indent=2) + "\n")

    full = {k: v for k, v in results.items()}
    with open(STEP23_DIR / "results_full.json", "w") as fh:
        json.dump(full, fh, indent=2)

    print(f"\nsaved to {STEP23_DIR / 'results.json'} and results_full.json")


if __name__ == "__main__":
    main()
