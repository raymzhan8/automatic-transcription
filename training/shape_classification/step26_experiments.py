"""Step 26: does audio contain trajectory-type information beyond frozen
CREPE pitch contours, under oracle (GT) trajectory boundaries? Runs A0
(CREPE contour only, reused unchanged from Step 23 B1 / Step 25 F0), A1
(audio only), A2 (CREPE + audio, single linear fusion head), loads A3
(oracle-contour reference from Step 25, not retrained), and A4 (oracle +
audio, optional secondary control) -- then the full battery of consistency,
attribution, and sanity checks the spec (docs/step_26_audio_complementarity.md's
source instructions) lays out.

The primary comparison is A2 vs A0.
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

from training.folds import assert_no_split_leakage, build_fold_split, load_kfold_manifest  # noqa: E402
from training.shape_classification.dataset import CLASS_NAMES, OUT_DIR, build  # noqa: E402
from training.shape_classification.duration_span_analysis import (  # noqa: E402
    DURATION_BUCKET_NAMES, DURATION_BUCKETS_S, SPAN_BUCKET_NAMES, SPAN_BUCKETS_CENTS,
)
from training.shape_classification.metrics_utils import eval_metrics, prediction_frequency  # noqa: E402
from training.shape_classification.step23_train import run_condition as run_a0_condition  # noqa: E402
from training.shape_classification.step25_experiments import fold_consistency, recording_consistency  # noqa: E402
from training.shape_classification.step26_features import build as build_audio, frontend_metadata  # noqa: E402
from training.shape_classification.step26_model import (  # noqa: E402
    AUDIO_HIDDEN, AudioOnlyModel, FusionModel, PITCH_HIDDEN, count_params,
)
from training.shape_classification.step26_train import (  # noqa: E402
    FOUR_CLASS_NAMES, run_condition_audio_only, run_condition_fusion,
)
from training.shape_classification.templates import template_errors  # noqa: E402

STEP26_DIR = OUT_DIR / "step26"
STEP25_RESULTS = OUT_DIR / "step25" / "results.json"
N_FOLDS = 5


# ---------------------------------------------------------------- sections 12-13


def pooled_pred_true(result: dict) -> tuple[np.ndarray, np.ndarray]:
    pred = np.concatenate([np.array(f["test_pred"]) for f in result["folds"]])
    true = np.concatenate([np.array(f["test_true"]) for f in result["folds"]])
    return pred, true


# ---------------------------------------------------------------- section 17: CREPE-ambiguity strata


def crepe_ambiguity_margin(records: list[dict]) -> dict[str, float]:
    """margin = (second-lowest Step-24 template error) - (lowest) on the
    primitive's OWN crepe contour; small margin = ambiguous template fit."""
    out = {}
    for r in records:
        d = r["crepe"]
        if d is None:
            continue
        errs = sorted(template_errors(d["r"], d["span_cents"], robust=False))
        out[r["primitive_id"]] = float(errs[1] - errs[0])
    return out


def ambiguity_strata(margins: dict[str, float]) -> dict[str, tuple[float, float]]:
    """Global quartiles, fixed once over the whole corpus -- diagnostic
    bucket edges, not tuned to any Step 26 result."""
    vals = np.array(list(margins.values()))
    q1, q2, q3 = np.percentile(vals, [25, 50, 75])
    return {
        "Q1_most_ambiguous": (-np.inf, q1),
        "Q2": (q1, q2),
        "Q3": (q2, q3),
        "Q4_least_ambiguous": (q3, np.inf),
    }


def bucket_by_margin(
    result: dict, margins: dict[str, float], strata: dict[str, tuple[float, float]],
) -> dict[str, dict]:
    by_pid_pred, by_pid_true = {}, {}
    for fold in result["folds"]:
        for pid, p, t in zip(fold["test_primitive_id"], fold["test_pred"], fold["test_true"]):
            by_pid_pred[pid] = p
            by_pid_true[pid] = t
    out = {}
    for name, (lo, hi) in strata.items():
        pids = [pid for pid, m in margins.items() if lo <= m < hi and pid in by_pid_pred]
        if not pids:
            out[name] = {"n": 0}
            continue
        pred = np.array([by_pid_pred[pid] for pid in pids])
        true = np.array([by_pid_true[pid] for pid in pids])
        m = eval_metrics(pred, true, FOUR_CLASS_NAMES)
        out[name] = {"n": len(pids), "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]}
    return out


# ---------------------------------------------------------------- sections 18-19: duration / span buckets


def _duration_span_lookup(records: list[dict]) -> dict[str, tuple[float, float]]:
    return {r["primitive_id"]: (r["duration_s"], abs(r["crepe"]["span_cents"]) if r["crepe"] else 0.0)
            for r in records}


def bucket_by(
    result: dict, lookup: dict[str, tuple[float, float]], field: int, edges: tuple, names: tuple,
    *, exclude_fixed: bool = False,
) -> dict[str, dict]:
    by_pid_pred, by_pid_true = {}, {}
    for fold in result["folds"]:
        for pid, p, t in zip(fold["test_primitive_id"], fold["test_pred"], fold["test_true"]):
            by_pid_pred[pid] = p
            by_pid_true[pid] = t
    out = {}
    for name, lo, hi in zip(names, edges[:-1], edges[1:]):
        pids = [pid for pid, v in lookup.items()
                if pid in by_pid_pred and lo <= v[field] < hi
                and not (exclude_fixed and by_pid_true[pid] == 0)]
        if not pids:
            out[name] = {"n": 0}
            continue
        pred = np.array([by_pid_pred[pid] for pid in pids])
        true = np.array([by_pid_true[pid] for pid in pids])
        m = eval_metrics(pred, true, FOUR_CLASS_NAMES)
        out[name] = {"n": len(pids), "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]}
    return out


# ---------------------------------------------------------------- section 20: fusion-usage sanity check


def fusion_usage_diagnostic(a2: dict) -> dict:
    pooled_true = np.concatenate([np.array(f["test_true"]) for f in a2["folds"]])
    normal_pred = np.concatenate([np.array(f["test_pred"]) for f in a2["folds"]])
    audio_zeroed_pred = np.concatenate([np.array(f["audio_zeroed_test_pred"]) for f in a2["folds"]])
    pitch_zeroed_pred = np.concatenate([np.array(f["pitch_zeroed_test_pred"]) for f in a2["folds"]])
    normal_f1 = eval_metrics(normal_pred, pooled_true, FOUR_CLASS_NAMES)["macro_f1"]
    audio_zeroed_f1 = eval_metrics(audio_zeroed_pred, pooled_true, FOUR_CLASS_NAMES)["macro_f1"]
    pitch_zeroed_f1 = eval_metrics(pitch_zeroed_pred, pooled_true, FOUR_CLASS_NAMES)["macro_f1"]

    per_fold_weights = []
    for fold in a2["folds"]:
        w = np.array(fold["head_weight"])  # [4, PITCH_HIDDEN+AUDIO_HIDDEN]
        pitch_w = np.abs(w[:, :PITCH_HIDDEN]).mean()
        audio_w = np.abs(w[:, PITCH_HIDDEN:PITCH_HIDDEN + AUDIO_HIDDEN]).mean()
        per_fold_weights.append({"fold": fold["fold"], "mean_abs_pitch_weight": float(pitch_w),
                                  "mean_abs_audio_weight": float(audio_w), "ratio_audio_to_pitch": float(audio_w / pitch_w)})
    return {
        "a2_normal_macro_f1": normal_f1, "a2_audio_zeroed_macro_f1": audio_zeroed_f1,
        "a2_pitch_zeroed_macro_f1": pitch_zeroed_f1, "per_fold_weights": per_fold_weights,
        "audio_contributes": normal_f1 > audio_zeroed_f1, "pitch_contributes": normal_f1 > pitch_zeroed_f1,
    }


# ---------------------------------------------------------------- section 21: representative changed decisions


def representative_changed_decisions(records: list[dict], a0: dict, a2: dict, n: int = 4) -> dict[str, Any]:
    rec_by_pid = {r["primitive_id"]: r for r in records}
    a0_by_pid, a2_by_pid = {}, {}
    for fold in a0["folds"]:
        for pid, p, t in zip(fold["test_primitive_id"], fold["test_pred"], fold["test_true"]):
            a0_by_pid[pid] = (p, t)
    for fold in a2["folds"]:
        for pid, p, t, probs in zip(fold["test_primitive_id"], fold["test_pred"], fold["test_true"], fold["test_probs"]):
            a2_by_pid[pid] = (p, t, probs)

    common = sorted(set(a0_by_pid) & set(a2_by_pid))
    a0_wrong_a2_right = [pid for pid in common if a0_by_pid[pid][0] != a0_by_pid[pid][1] and a2_by_pid[pid][0] == a2_by_pid[pid][1]]
    a0_right_a2_wrong = [pid for pid in common if a0_by_pid[pid][0] == a0_by_pid[pid][1] and a2_by_pid[pid][0] != a2_by_pid[pid][1]]

    def confidence(pid: str) -> float:
        probs = a2_by_pid[pid][2]
        return max(probs)

    a0_wrong_a2_right.sort(key=confidence, reverse=True)
    a0_right_a2_wrong.sort(key=confidence, reverse=True)

    def describe(pid: str) -> dict:
        r = rec_by_pid[pid]
        c = r["crepe"]
        return {
            "primitive_id": pid, "recording_id": r["recording_id"], "duration_s": r["duration_s"],
            "true": CLASS_NAMES[r["canonical_type"]],
            "A0_pred": CLASS_NAMES[a0_by_pid[pid][0]], "A2_pred": CLASS_NAMES[a2_by_pid[pid][0]],
            "A2_confidence": round(confidence(pid), 3),
            "crepe_span_cents": round(c["span_cents"], 1) if c else None,
            "crepe_q_at_half": round(float(np.interp(0.5, np.linspace(0, 1, len(c["q"])), c["q"])), 3) if c else None,
        }

    return {
        "A0_wrong_A2_correct": [describe(pid) for pid in a0_wrong_a2_right[:n]],
        "A0_correct_A2_wrong": [describe(pid) for pid in a0_right_a2_wrong[:n]],
        "n_A0_wrong_A2_correct_total": len(a0_wrong_a2_right),
        "n_A0_correct_A2_wrong_total": len(a0_right_a2_wrong),
    }


# ---------------------------------------------------------------- section 23: leakage check


def leakage_and_distribution_check(records: list[dict]) -> dict[str, Any]:
    manifest = load_kfold_manifest(REPO_ROOT)
    fold_reports = []
    for f in range(N_FOLDS):
        split = build_fold_split(manifest, f, seed=42)
        leakage = assert_no_split_leakage(split, REPO_ROOT)
        fold_reports.append({"fold": f, "leakage_assertions": leakage})

    by_recording = defaultdict(lambda: [0, 0, 0, 0])
    for r in records:
        by_recording[r["recording_id"]][r["canonical_type"]] += 1
    class_dist_by_recording = {rid: dict(zip(FOUR_CLASS_NAMES, counts)) for rid, counts in by_recording.items()}
    return {"per_fold_leakage": fold_reports, "class_distribution_by_recording": class_dist_by_recording}


# ---------------------------------------------------------------- reporting


def _print_row(tag: str, result: dict) -> None:
    p = result["pooled"]
    row = " ".join(f"{p['per_class'][n]['f1']:.3f}" for n in FOUR_CLASS_NAMES)
    print(f"{tag:22s} macro_f1={p['macro_f1']:.4f} acc={p['accuracy']:.4f} "
          f"[{','.join(FOUR_CLASS_NAMES)}]={row} "
          f"grouped_mean={result['grouped_mean_macro_f1']:.4f}+/-{result['grouped_std_macro_f1']:.4f} "
          f"n_params={result.get('n_params', result.get('folds', [{}])[0].get('n_params'))}")


CONDITION_FULL_PATH = {c: STEP26_DIR / f"{c}_full.json" for c in ("a0", "a1", "a2", "a4")}


def train_condition(name: str) -> None:
    """Train exactly one condition and save its FULL result (incl. per-fold
    test predictions) to its own json, so a long combined run is never
    required and a killed process loses at most one condition's work."""
    STEP26_DIR.mkdir(parents=True, exist_ok=True)
    records = build()
    audio_lookup = build_audio() if name != "a0" else None

    if name == "a0":
        print("=== A0: CREPE contour only (reused, Step 23 B1 / Step 25 F0) ===", flush=True)
        result = run_a0_condition(records, "crepe", "shape_velocity", (0, 1, 2, 3), FOUR_CLASS_NAMES,
                                   balancing="sampler", save_root=STEP26_DIR / "a0")
    elif name == "a1":
        print("=== A1: audio only ===", flush=True)
        result = run_condition_audio_only(records, audio_lookup, save_root=STEP26_DIR / "a1")
    elif name == "a2":
        print("=== A2: CREPE + audio fusion ===", flush=True)
        result = run_condition_fusion(records, audio_lookup, contour_source="crepe", save_root=STEP26_DIR / "a2")
    elif name == "a4":
        print("=== A4 (optional): oracle contour + audio fusion ===", flush=True)
        result = run_condition_fusion(records, audio_lookup, contour_source="oracle", save_root=STEP26_DIR / "a4")
    else:
        raise ValueError(name)

    _print_row(name.upper(), result)
    CONDITION_FULL_PATH[name].write_text(json.dumps(result, indent=2) + "\n")
    print(f"saved to {CONDITION_FULL_PATH[name]}", flush=True)


def analyze() -> None:
    """Load all pre-trained conditions (no training here) and run every
    downstream analysis section (12-23), then write results.json."""
    STEP26_DIR.mkdir(parents=True, exist_ok=True)
    records = build()

    a0 = json.loads(CONDITION_FULL_PATH["a0"].read_text())
    a1 = json.loads(CONDITION_FULL_PATH["a1"].read_text())
    a2 = json.loads(CONDITION_FULL_PATH["a2"].read_text())
    a4 = json.loads(CONDITION_FULL_PATH["a4"].read_text()) if CONDITION_FULL_PATH["a4"].exists() else None
    # json roundtrip turns the folds' int-keyed per_fold_macro_f1 dict keys into strings; normalize back
    for res in (a0, a1, a2, *([a4] if a4 else [])):
        res["per_fold_macro_f1"] = {int(k): v for k, v in res["per_fold_macro_f1"].items()}
        for fold in res["folds"]:
            fold["fold"] = int(fold["fold"])

    a1_params = count_params(AudioOnlyModel())
    a2_params = count_params(FusionModel())
    print("=== Step 26 acoustic frontend ===")
    print(json.dumps(frontend_metadata(), indent=2))
    print("\n=== Step 26 §6/§10 architecture + parameter counts ===")
    print(f"A1 AudioOnlyModel: {a1_params} params")
    print(f"A2 FusionModel:    {a2_params} params  (PITCH_HIDDEN={PITCH_HIDDEN}, AUDIO_HIDDEN={AUDIO_HIDDEN})")
    _print_row("A0 (CREPE)", a0)
    _print_row("A1 (audio)", a1)
    _print_row("A2 (CREPE+audio)", a2)
    if a4:
        _print_row("A4 (oracle+audio)", a4)

    print("\n=== A3: oracle-contour reference (loaded from Step 25, not retrained) ===")
    step25 = json.loads(STEP25_RESULTS.read_text())
    a3 = step25["oracle_control"]["F0_oracle"]
    print(f"A3 (oracle contour, Step 25 F0-oracle) macro_f1={a3['pooled']['macro_f1']:.4f} "
          f"grouped_mean={a3['grouped_mean_macro_f1']:.4f}+/-{a3['grouped_std_macro_f1']:.4f}")

    # section 12: fold consistency, primary comparison A2 vs A0
    print("\n=== §12 fold consistency ===")
    fc = {"A1_vs_A0": fold_consistency(a0, a1), "A2_vs_A0": fold_consistency(a0, a2)}
    print("A2 vs A0:", fc["A2_vs_A0"])

    # section 13: recording consistency
    print("\n=== §13 recording consistency ===")
    rc = {"A1_vs_A0": recording_consistency(a0, a1), "A2_vs_A0": recording_consistency(a0, a2)}
    print("A2 vs A0:", {k: v for k, v in rc["A2_vs_A0"].items() if k != "per_recording_delta"})

    # section 14: per-class attribution (already in eval_metrics' per_class)
    per_class = {tag: res["pooled"]["per_class"] for tag, res in (("A0", a0), ("A1", a1), ("A2", a2))}

    # section 15: confusion matrices (already computed by eval_metrics)
    confusion = {tag: res["pooled"]["confusion_matrix"] for tag, res in (("A0", a0), ("A1", a1), ("A2", a2))}

    # section 16: prediction frequencies
    _, true_pooled = pooled_pred_true(a0)
    pred_freq = {"true": prediction_frequency(true_pooled, FOUR_CLASS_NAMES)}
    for tag, res in (("A0", a0), ("A1", a1), ("A2", a2)):
        pred, _ = pooled_pred_true(res)
        pred_freq[tag] = prediction_frequency(pred, FOUR_CLASS_NAMES)
    print("\n=== §16 prediction frequency ===")
    for k, v in pred_freq.items():
        print(k, {kk: round(vv, 3) for kk, vv in v.items()})

    # section 17: CREPE-ambiguity strata
    margins = crepe_ambiguity_margin(records)
    strata = ambiguity_strata(margins)
    ambiguity_a0 = bucket_by_margin(a0, margins, strata)
    ambiguity_a2 = bucket_by_margin(a2, margins, strata)
    print("\n=== §17 CREPE-ambiguity strata (A0 vs A2 macro F1) ===")
    for name in strata:
        n0, n2 = ambiguity_a0[name].get("n", 0), ambiguity_a2[name].get("n", 0)
        f0 = ambiguity_a0[name].get("macro_f1")
        f2 = ambiguity_a2[name].get("macro_f1")
        print(f"  {name:20s} n_A0={n0:4d} n_A2={n2:4d}  A0={f0}  A2={f2}")

    # sections 18-19: duration / pitch-span buckets
    ds_lookup = _duration_span_lookup(records)
    duration_report = {tag: bucket_by(res, ds_lookup, 0, DURATION_BUCKETS_S, DURATION_BUCKET_NAMES)
                        for tag, res in (("A0", a0), ("A1", a1), ("A2", a2))}
    span_report = {tag: bucket_by(res, ds_lookup, 1, SPAN_BUCKETS_CENTS, SPAN_BUCKET_NAMES, exclude_fixed=True)
                   for tag, res in (("A0", a0), ("A1", a1), ("A2", a2))}
    print("\n=== §18 duration buckets (macro F1) ===")
    for name in DURATION_BUCKET_NAMES:
        print(f"  {name:12s} A0={duration_report['A0'][name].get('macro_f1')}  "
              f"A1={duration_report['A1'][name].get('macro_f1')}  A2={duration_report['A2'][name].get('macro_f1')}")
    print("\n=== §19 pitch-span buckets (macro F1, moving primitives only) ===")
    for name in SPAN_BUCKET_NAMES:
        print(f"  {name:10s} A0={span_report['A0'][name].get('macro_f1')}  "
              f"A1={span_report['A1'][name].get('macro_f1')}  A2={span_report['A2'][name].get('macro_f1')}")

    # section 20: fusion-usage sanity check
    fusion_diag = fusion_usage_diagnostic(a2)
    print("\n=== §20 fusion-usage sanity check ===")
    print(f"A2 normal={fusion_diag['a2_normal_macro_f1']:.4f}  "
          f"audio-zeroed={fusion_diag['a2_audio_zeroed_macro_f1']:.4f}  "
          f"pitch-zeroed={fusion_diag['a2_pitch_zeroed_macro_f1']:.4f}")

    # section 21: representative changed decisions
    changed = representative_changed_decisions(records, a0, a2)
    print(f"\n=== §21 changed decisions === A0-wrong/A2-correct: {changed['n_A0_wrong_A2_correct_total']}  "
          f"A0-correct/A2-wrong: {changed['n_A0_correct_A2_wrong_total']}")

    # section 23: leakage + distribution check
    leakage_check = leakage_and_distribution_check(records)
    print("\n=== §23 leakage check ===")
    for row in leakage_check["per_fold_leakage"]:
        print(f"  fold {row['fold']}: {row['leakage_assertions']}")

    out = {
        "frontend": frontend_metadata(),
        "param_counts": {"A0": a0["folds"][0]["n_params"], "A1": a1_params, "A2": a2_params,
                          "A4": a4["n_params"] if a4 else None},
        "A0": {k: v for k, v in a0.items() if k != "folds"},
        "A1": {k: v for k, v in a1.items() if k != "folds"},
        "A2": {k: v for k, v in a2.items() if k != "folds"},
        "A3_oracle_reference": {k: v for k, v in a3.items() if k != "folds"},
        "A4_optional": ({k: v for k, v in a4.items() if k != "folds"} if a4 else None),
        "fold_consistency": fc, "recording_consistency": {k: v for k, v in rc.items()},
        "per_class": per_class, "confusion_matrices": confusion, "prediction_frequency": pred_freq,
        "crepe_ambiguity_strata_edges": {k: list(v) for k, v in strata.items()},
        "crepe_ambiguity_A0": ambiguity_a0, "crepe_ambiguity_A2": ambiguity_a2,
        "duration_buckets": duration_report, "pitch_span_buckets": span_report,
        "fusion_usage_diagnostic": fusion_diag, "representative_changed_decisions": changed,
        "leakage_check": leakage_check,
    }
    (STEP26_DIR / "results.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(f"\nsaved to {STEP26_DIR / 'results.json'}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("a0", "a1", "a2", "a4"):
        train_condition(sys.argv[1])
    else:
        analyze()
