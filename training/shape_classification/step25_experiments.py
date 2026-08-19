"""Step 25: do Step 24's normalized canonical-template residuals (z) add
information the balanced ContourCNN doesn't already extract from
q(x)+dq/dx alone? Trains F0 (Step 23 B1, reused unchanged), F1 (template
evidence only, `Linear(4->4)`), and F2 (ContourCNN + z fused before the
final layer), then compares macro F1, fold/recording consistency,
per-class effects, prediction frequency, confusion matrices, and a
template-feature-use sanity check (weight magnitude + z-zeroed-at-test-time
ablation, no retraining).
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

from training.shape_classification.dataset import CLASS_NAMES, OUT_DIR, build  # noqa: E402
from training.shape_classification.metrics_utils import eval_metrics, prediction_frequency  # noqa: E402
from training.shape_classification.step23_train import run_condition as run_f0_condition  # noqa: E402
from training.shape_classification.step25_features import build_z_lookup  # noqa: E402
from training.shape_classification.step25_train import FOUR_CLASS_NAMES, run_condition  # noqa: E402
from training.shape_classification.templates import template_errors  # noqa: E402

STEP25_DIR = OUT_DIR / "step25"


def per_recording_macro_f1(result: dict, names: tuple[str, ...] = FOUR_CLASS_NAMES) -> dict[str, float]:
    by_rid = defaultdict(lambda: ([], []))
    for fold in result["folds"]:
        for rid, p, t in zip(fold["test_recording_ids"], fold["test_pred"], fold["test_true"]):
            by_rid[rid][0].append(p)
            by_rid[rid][1].append(t)
    return {rid: eval_metrics(np.array(p), np.array(t), names)["macro_f1"] for rid, (p, t) in by_rid.items()}


def fold_consistency(base: dict, other: dict) -> dict:
    deltas = {f: other["per_fold_macro_f1"][f] - base["per_fold_macro_f1"][f] for f in base["per_fold_macro_f1"]}
    vals = list(deltas.values())
    return {"per_fold_delta": deltas, "n_improved": sum(1 for v in vals if v > 0),
            "n_worsened": sum(1 for v in vals if v < 0), "median_delta": float(np.median(vals))}


def recording_consistency(base: dict, other: dict) -> dict:
    r0 = per_recording_macro_f1(base); r1 = per_recording_macro_f1(other)
    common = sorted(set(r0) & set(r1))
    deltas = {rid: r1[rid] - r0[rid] for rid in common}
    vals = list(deltas.values())
    return {"per_recording_delta": deltas, "n_improved": sum(1 for v in vals if v > 0),
            "n_worsened": sum(1 for v in vals if v < 0),
            "median_delta": float(np.median(vals)) if vals else None,
            "mean_delta": float(np.mean(vals)) if vals else None}


def weight_use_diagnostic(f2_result: dict) -> dict:
    per_fold = []
    for fold in f2_result["folds"]:
        w = np.array(fold["head_weight"])  # [4, hidden+4]
        contour_w = np.abs(w[:, :-4]).mean()
        z_w = np.abs(w[:, -4:]).mean()
        per_fold.append({"fold": fold["fold"], "mean_abs_contour_weight": float(contour_w),
                          "mean_abs_z_weight": float(z_w), "ratio_z_to_contour": float(z_w / contour_w)})
    # z-zeroed ablation, same trained weights, no retraining
    pooled_pred = np.concatenate([np.array(f["test_pred"]) for f in f2_result["folds"]])
    pooled_true = np.concatenate([np.array(f["test_true"]) for f in f2_result["folds"]])
    zeroed_pred = np.concatenate([np.array(f["z_zeroed_test_pred"]) for f in f2_result["folds"]])
    normal_metrics = eval_metrics(pooled_pred, pooled_true, FOUR_CLASS_NAMES)
    zeroed_metrics = eval_metrics(zeroed_pred, pooled_true, FOUR_CLASS_NAMES)
    return {"per_fold_weights": per_fold, "f2_normal_macro_f1": normal_metrics["macro_f1"],
            "f2_z_zeroed_macro_f1": zeroed_metrics["macro_f1"]}


def representative_changed_decisions(records: list[dict], z_lookup: dict, f0: dict, f2: dict, n: int = 4) -> dict[str, Any]:
    rec_by_pid = {r["primitive_id"]: r for r in records}
    f0_by_pid, f2_by_pid = {}, {}
    for fold in f0["folds"]:
        for pid, p, t in zip(fold["test_primitive_id"], fold["test_pred"], fold["test_true"]):
            f0_by_pid[pid] = (p, t)
    for fold in f2["folds"]:
        for pid, p, t in zip(fold["test_primitive_id"], fold["test_pred"], fold["test_true"]):
            f2_by_pid[pid] = (p, t)

    common = sorted(set(f0_by_pid) & set(f2_by_pid) & set(z_lookup))
    f0_wrong_f2_right = [pid for pid in common if f0_by_pid[pid][0] != f0_by_pid[pid][1] and f2_by_pid[pid][0] == f2_by_pid[pid][1]]
    f0_right_f2_wrong = [pid for pid in common if f0_by_pid[pid][0] == f0_by_pid[pid][1] and f2_by_pid[pid][0] != f2_by_pid[pid][1]]

    def describe(pid: str) -> dict:
        r = rec_by_pid[pid]
        z = z_lookup[pid]
        errs = template_errors(r["crepe"]["r"], r["crepe"]["span_cents"], robust=False)
        return {
            "primitive_id": pid, "recording_id": r["recording_id"], "true": CLASS_NAMES[r["canonical_type"]],
            "F0_pred": CLASS_NAMES[f0_by_pid[pid][0]], "F2_pred": CLASS_NAMES[f2_by_pid[pid][0]],
            "z": z, "template_argmin": CLASS_NAMES[int(np.argmin(errs))],
            "M_start_vs_cosine": errs[1] - errs[2], "M_end_vs_cosine": errs[1] - errs[3],
        }

    out = {"F0_wrong_F2_correct": [describe(pid) for pid in f0_wrong_f2_right[:n]],
           "F0_correct_F2_wrong": [describe(pid) for pid in f0_right_f2_wrong[:n]],
           "n_F0_wrong_F2_correct_total": len(f0_wrong_f2_right),
           "n_F0_correct_F2_wrong_total": len(f0_right_f2_wrong)}
    return out


def _print_row(tag: str, result: dict, names=FOUR_CLASS_NAMES) -> None:
    p = result["pooled"]
    row = " ".join(f"{p['per_class'][n]['f1']:.3f}" for n in names)
    print(f"{tag:20s} macro_f1={p['macro_f1']:.4f} acc={p['accuracy']:.4f} [{','.join(names)}]={row} "
          f"grouped_mean={result['grouped_mean_macro_f1']:.4f}+/-{result['grouped_std_macro_f1']:.4f} "
          f"n_params={result.get('n_params')}")


def main() -> None:
    STEP25_DIR.mkdir(parents=True, exist_ok=True)
    records = build()
    z_lookup = build_z_lookup(records, "crepe")

    print("=== Step 25 §6: F0/F1/F2 ===")
    f0 = run_f0_condition(records, "crepe", "shape_velocity", (0, 1, 2, 3), FOUR_CLASS_NAMES,
                           balancing="sampler", save_root=STEP25_DIR / "f0")
    _print_row("F0 (contour)", f0)
    f1 = run_condition(records, z_lookup, "template_linear", save_root=STEP25_DIR / "f1")
    _print_row("F1 (template only)", f1)
    f2 = run_condition(records, z_lookup, "fusion", save_root=STEP25_DIR / "f2")
    _print_row("F2 (contour+template)", f2)

    print(f"\nparam count: F0={f0['folds'][0]['n_params']} F2={f2['n_params']} "
          f"delta={f2['n_params'] - f0['folds'][0]['n_params']}")

    # §10 fold consistency
    fc = {"F1_vs_F0": fold_consistency(f0, f1), "F2_vs_F0": fold_consistency(f0, f2)}
    print("\n=== §10 fold consistency (F2 vs F0) ===", fc["F2_vs_F0"])

    # §11 recording consistency
    rc = {"F1_vs_F0": recording_consistency(f0, f1), "F2_vs_F0": recording_consistency(f0, f2)}
    print("=== §11 recording consistency (F2 vs F0) ===",
          {k: v for k, v in rc["F2_vs_F0"].items() if k != "per_recording_delta"})

    # §13 prediction frequency
    true_pooled = np.concatenate([np.array(f["test_true"]) for f in f0["folds"]])
    pred_freq = {"true": prediction_frequency(true_pooled, FOUR_CLASS_NAMES)}
    for tag, res in (("F0", f0), ("F1", f1), ("F2", f2)):
        pooled_pred = np.concatenate([np.array(f["test_pred"]) for f in res["folds"]])
        pred_freq[tag] = prediction_frequency(pooled_pred, FOUR_CLASS_NAMES)
    print("\n=== §13 prediction frequency ===")
    for k, v in pred_freq.items():
        print(k, {kk: round(vv, 3) for kk, vv in v.items()})

    # §15 template-feature-use diagnostic
    weight_diag = weight_use_diagnostic(f2)
    print("\n=== §15 template-feature-use diagnostic ===")
    print(f"F2 normal macro_f1={weight_diag['f2_normal_macro_f1']:.4f}  "
          f"F2 z-zeroed macro_f1={weight_diag['f2_z_zeroed_macro_f1']:.4f}  "
          f"F0 macro_f1={f0['pooled']['macro_f1']:.4f}")
    for row in weight_diag["per_fold_weights"]:
        print(f"  fold {row['fold']}: mean|contour_w|={row['mean_abs_contour_weight']:.4f} "
              f"mean|z_w|={row['mean_abs_z_weight']:.4f} ratio={row['ratio_z_to_contour']:.3f}")

    # §16 representative changed decisions
    changed = representative_changed_decisions(records, z_lookup, f0, f2)
    print(f"\n=== §16 changed decisions === F0-wrong/F2-correct: {changed['n_F0_wrong_F2_correct_total']}  "
          f"F0-correct/F2-wrong: {changed['n_F0_correct_F2_wrong_total']}")

    # §18 optional oracle control (secondary; does not determine the CREPE outcome)
    print("\n=== §18 optional oracle control ===")
    z_lookup_oracle = build_z_lookup(records, "oracle")
    f0_o = run_f0_condition(records, "oracle", "shape_velocity", (0, 1, 2, 3), FOUR_CLASS_NAMES,
                             balancing="sampler", save_root=STEP25_DIR / "f0_oracle")
    _print_row("F0-oracle", f0_o)
    f1_o = run_condition(records, z_lookup_oracle, "template_linear", save_root=STEP25_DIR / "f1_oracle", source_key="oracle")
    _print_row("F1-oracle", f1_o)
    f2_o = run_condition(records, z_lookup_oracle, "fusion", save_root=STEP25_DIR / "f2_oracle", source_key="oracle")
    _print_row("F2-oracle", f2_o)
    oracle_control = {
        "F0_oracle": {k: v for k, v in f0_o.items() if k != "folds"},
        "F1_oracle": {k: v for k, v in f1_o.items() if k != "folds"},
        "F2_oracle": {k: v for k, v in f2_o.items() if k != "folds"},
    }

    out = {
        "F0": {k: v for k, v in f0.items() if k != "folds"},
        "F1": {k: v for k, v in f1.items() if k != "folds"},
        "F2": {k: v for k, v in f2.items() if k != "folds"},
        "param_count": {"F0": f0["folds"][0]["n_params"], "F2": f2["n_params"],
                         "delta": f2["n_params"] - f0["folds"][0]["n_params"]},
        "fold_consistency": fc, "recording_consistency": rc, "prediction_frequency": pred_freq,
        "weight_use_diagnostic": weight_diag, "representative_changed_decisions": changed,
        "oracle_control": oracle_control,
    }
    (STEP25_DIR / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nsaved to {STEP25_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
