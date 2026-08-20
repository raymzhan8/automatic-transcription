"""Step 29: does an order-aware BiGRU over [e_prev,e_center,e_next] use ±1
trajectory context better than Step 28's linear concatenation (S1/C1)? Loads
S0 (=Step26 L0/Step28 C0) and S1 (=Step28 C1) from cache unchanged, loads
the two trained S2 conditions, and runs the full comparison battery.
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
from training.shape_classification.metrics_utils import eval_metrics  # noqa: E402
from training.shape_classification.step25_experiments import fold_consistency, recording_consistency  # noqa: E402
from training.shape_classification.step29_train import FOUR_CLASS_NAMES  # noqa: E402

STEP26_DIR = OUT_DIR / "step26"
STEP28_DIR = OUT_DIR / "step28"
STEP29_DIR = OUT_DIR / "step29"


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


def t2_recovery_breakage(records: list[dict], s1: dict, s2: dict, n: int = 4) -> dict[str, Any]:
    rec_by_pid = {r["primitive_id"]: r for r in records}
    s1_by_pid, s2_by_pid = {}, {}
    for fold in s1["folds"]:
        for pid, p, t in zip(fold["test_primitive_id"], fold["test_pred"], fold["test_true"]):
            s1_by_pid[pid] = (p, t)
    for fold in s2["folds"]:
        for pid, p, t, probs in zip(fold["test_primitive_id"], fold["test_pred"], fold["test_true"], fold["test_probs"]):
            s2_by_pid[pid] = (p, t, probs)
    common = sorted(set(s1_by_pid) & set(s2_by_pid))
    set_a = [pid for pid in common if s1_by_pid[pid][0] == 1 and s1_by_pid[pid][1] == 2]
    recovered = [pid for pid in set_a if s2_by_pid[pid][0] == 2]
    set_b = [pid for pid in common if s1_by_pid[pid][0] == 1 and s1_by_pid[pid][1] == 1]
    broken = [pid for pid in set_b if s2_by_pid[pid][0] != 1]

    def describe(pid: str) -> dict:
        r = rec_by_pid[pid]
        return {"primitive_id": pid, "recording_id": r["recording_id"], "true": CLASS_NAMES[r["canonical_type"]],
                "S1_pred": CLASS_NAMES[s1_by_pid[pid][0]], "S2_pred": CLASS_NAMES[s2_by_pid[pid][0]],
                "S2_confidence": round(max(s2_by_pid[pid][2]), 3)}

    recovered_sorted = sorted(recovered, key=lambda pid: max(s2_by_pid[pid][2]), reverse=True)
    broken_sorted = sorted(broken, key=lambda pid: max(s2_by_pid[pid][2]), reverse=True)
    return {"n_set_A": len(set_a), "n_recovered": len(recovered), "n_set_B": len(set_b), "n_broken": len(broken),
            "top_recovered": [describe(pid) for pid in recovered_sorted[:n]],
            "top_broken": [describe(pid) for pid in broken_sorted[:n]]}


def swap_diagnostic(s2_context: dict) -> dict:
    pooled_true = np.concatenate([np.array(f["test_true"]) for f in s2_context["folds"]])
    normal = np.concatenate([np.array(f["test_pred"]) for f in s2_context["folds"]])
    swapped = np.concatenate([np.array(f["swap_test_pred"]) for f in s2_context["folds"]])
    normal_m = eval_metrics(normal, pooled_true, FOUR_CLASS_NAMES)
    swapped_m = eval_metrics(swapped, pooled_true, FOUR_CLASS_NAMES)
    return {
        "normal_macro_f1": normal_m["macro_f1"], "swapped_macro_f1": swapped_m["macro_f1"],
        "normal_sloped_start_f1": normal_m["per_class"]["Sloped-start"]["f1"],
        "swapped_sloped_start_f1": swapped_m["per_class"]["Sloped-start"]["f1"],
        "normal_sloped_end_f1": normal_m["per_class"]["Sloped-end"]["f1"],
        "swapped_sloped_end_f1": swapped_m["per_class"]["Sloped-end"]["f1"],
    }


def main() -> None:
    records = build()

    s0 = load_cached(STEP26_DIR / "a2_full.json")
    s1 = load_cached(STEP28_DIR / "c1_full.json")
    s2c = load_cached(STEP29_DIR / "s2_center_only_full.json")
    s2x = load_cached(STEP29_DIR / "s2_context_full.json")
    o_context = json.loads((STEP28_DIR / "results.json").read_text())["oracle_neighbor_ceiling"]

    print("=== §6/§10 architecture + parameter counts ===")
    for tag, res in (("S0", s0), ("S1", s1), ("S2-center-only", s2c), ("S2-context", s2x)):
        print(f"{tag}: n_params={res.get('n_params')}")

    print("\n=== §12 primary result table ===")
    for tag, res in (("S0", s0), ("S1", s1), ("S2-center-only", s2c), ("S2-context", s2x)):
        p = res["pooled"]
        row = " ".join(f"{p['per_class'][n]['f1']:.3f}" for n in FOUR_CLASS_NAMES)
        print(f"{tag:16s} macro_f1={p['macro_f1']:.4f} acc={p['accuracy']:.4f} [{','.join(FOUR_CLASS_NAMES)}]={row} "
              f"grouped_mean={res['grouped_mean_macro_f1']:.4f}+/-{res['grouped_std_macro_f1']:.4f}")
    print(f"O-context (Step28 oracle ceiling, non-deployable): pooled={o_context['pooled']['macro_f1']:.4f} "
          f"grouped_mean={o_context['grouped_mean_macro_f1']:.4f}")

    # §17 fold consistency (S2-context vs S1, and vs S2-center-only)
    fc_vs_s1 = fold_consistency(s1, s2x)
    fc_vs_center = fold_consistency(s2c, s2x)
    print("\n=== §17 fold consistency: S2-context vs S1 (Comparison A) ===", fc_vs_s1)
    print("=== §17 fold consistency: S2-context vs S2-center-only (Comparison B) ===", fc_vs_center)

    print("\n=== §17 fold table ===")
    print(f"{'fold':>4s} {'S0':>8s} {'S1':>8s} {'S2-ctr':>8s} {'S2-ctx':>8s} {'S2x-S1':>8s}")
    for f in range(5):
        print(f"{f:>4d} {s0['per_fold_macro_f1'][f]:>8.4f} {s1['per_fold_macro_f1'][f]:>8.4f} "
              f"{s2c['per_fold_macro_f1'][f]:>8.4f} {s2x['per_fold_macro_f1'][f]:>8.4f} "
              f"{s2x['per_fold_macro_f1'][f]-s1['per_fold_macro_f1'][f]:>+8.4f}")

    # §18 recording consistency
    rc_vs_s1 = recording_consistency(s1, s2x)
    rc_vs_center = recording_consistency(s2c, s2x)
    print("\n=== §18 recording consistency: S2-context vs S1 ===", {k: v for k, v in rc_vs_s1.items() if k != "per_recording_delta"})
    print("=== §18 recording consistency: S2-context vs S2-center-only ===", {k: v for k, v in rc_vs_center.items() if k != "per_recording_delta"})

    # §14-15 per-class + confusion
    per_class = {tag: res["pooled"]["per_class"] for tag, res in (("S1", s1), ("S2x", s2x))}
    print("\n=== §15 Cosine <-> T2 table (S1 vs S2-context) ===")
    for cls in ("Cosine", "Sloped-start"):
        for metric in ("precision", "recall", "f1"):
            v0, v1 = per_class["S1"][cls][metric], per_class["S2x"][cls][metric]
            print(f"  {cls} {metric}: S1={v0:.3f} S2={v1:.3f} delta={v1-v0:+.3f}")

    confusion = {tag: res["pooled"]["confusion_matrix"] for tag, res in (("S1", s1), ("S2-center", s2c), ("S2-context", s2x))}
    print("\n=== §16 confusion matrices ===")
    for tag in ("S1", "S2-center", "S2-context"):
        print(tag)
        for row in confusion[tag]:
            print(" ", row)

    # §16 T2 recovery/breakage (S1 -> S2-context)
    recovery = t2_recovery_breakage(records, s1, s2x)
    print(f"\n=== §16 T2 recovery/breakage (S2-context vs S1) === set_A={recovery['n_set_A']} "
          f"recovered={recovery['n_recovered']}  set_B={recovery['n_set_B']} broken={recovery['n_broken']}")

    # §19 swap diagnostic
    swap = swap_diagnostic(s2x)
    print("\n=== §19 temporal-order swap diagnostic (S2-context, no retraining) ===")
    print(f"  normal macro_f1={swap['normal_macro_f1']:.4f}  swapped macro_f1={swap['swapped_macro_f1']:.4f}")
    print(f"  Sloped-start F1: normal={swap['normal_sloped_start_f1']:.3f} swapped={swap['swapped_sloped_start_f1']:.3f}")
    print(f"  Sloped-end F1:   normal={swap['normal_sloped_end_f1']:.3f} swapped={swap['swapped_sloped_end_f1']:.3f}")

    # §20 T2 vs T3
    print("\n=== §20 T2 vs T3 ===")
    for tag, res in (("S1", s1), ("S2-context", s2x)):
        p = res["pooled"]["per_class"]
        print(f"  {tag}: Sloped-start F1={p['Sloped-start']['f1']:.3f}  Sloped-end F1={p['Sloped-end']['f1']:.3f}")

    out = {
        "S0": {k: v for k, v in s0.items() if k != "folds"}, "S1": {k: v for k, v in s1.items() if k != "folds"},
        "S2_center_only": {k: v for k, v in s2c.items() if k != "folds"}, "S2_context": {k: v for k, v in s2x.items() if k != "folds"},
        "O_context_ceiling": o_context,
        "fold_consistency_vs_S1": fc_vs_s1, "fold_consistency_vs_S2center": fc_vs_center,
        "recording_consistency_vs_S1": rc_vs_s1, "recording_consistency_vs_S2center": rc_vs_center,
        "per_class": per_class, "confusion_matrices": confusion,
        "t2_recovery_breakage": recovery, "swap_diagnostic": swap,
    }
    STEP29_DIR.mkdir(parents=True, exist_ok=True)
    (STEP29_DIR / "results.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(f"\nsaved to {STEP29_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
