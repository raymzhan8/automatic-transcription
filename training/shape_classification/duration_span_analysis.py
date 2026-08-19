"""Step 22 sections 15-16: does phase normalization actually remove
duration-dependence (bucket CREPE CNN test performance by primitive
duration), and are small pitch-span trajectories harder even after shape
normalization (bucket by total pitch displacement)? Uses the CREPE
shape_only CNN's pooled test predictions from cnn_model.py (per-primitive
duration_s/abs_span_cents were attached at prediction time -- no model
change here).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.metrics import frame_metrics  # noqa: E402
from training.shape_classification.dataset import OUT_DIR  # noqa: E402

DURATION_BUCKETS_S = (0.0, 0.1, 0.25, 0.5, 1.0, np.inf)
DURATION_BUCKET_NAMES = ("<100ms", "100-250ms", "250-500ms", "500ms-1s", ">1s")
# Pitch-span buckets (cents) for the moving-trajectory-only slice.
SPAN_BUCKETS_CENTS = (0.0, 50.0, 100.0, 200.0, 400.0, np.inf)
SPAN_BUCKET_NAMES = ("<50c", "50-100c", "100-200c", "200-400c", ">400c")


def _pool_condition(full: dict, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred, true, dur, span = [], [], [], []
    for fold in full[key]["folds"]:
        pred.append(np.array(fold["test_pred"]))
        true.append(np.array(fold["test_true"]))
        dur.append(np.array(fold["test_duration_s"]))
        span.append(np.array(fold["test_abs_span_cents"]))
    return np.concatenate(pred), np.concatenate(true), np.concatenate(dur), np.concatenate(span)


def _bucket_report(pred, true, values, edges, names) -> dict:
    out = {}
    for name, lo, hi in zip(names, edges[:-1], edges[1:]):
        mask = (values >= lo) & (values < hi)
        if mask.sum() == 0:
            out[name] = {"n": 0}
            continue
        m = frame_metrics(pred[mask], true[mask])
        out[name] = {"n": int(mask.sum()), "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                      "per_class_f1": {k: v["f1"] for k, v in m["per_class"].items()}}
    return out


def main() -> None:
    full_path = OUT_DIR / "cnn_results_full.json"
    full = json.loads(full_path.read_text())

    result = {}
    print("=== Step 22 §15 duration analysis (CREPE shape_only, all classes) ===")
    for key in ("oracle_shape_only", "crepe_shape_only"):
        pred, true, dur, span = _pool_condition(full, key)
        dur_report = _bucket_report(pred, true, dur, DURATION_BUCKETS_S, DURATION_BUCKET_NAMES)
        result[f"{key}_by_duration"] = dur_report
        print(f"\n-- {key} --")
        for name in DURATION_BUCKET_NAMES:
            d = dur_report[name]
            if d["n"] == 0:
                print(f"{name:12s} n=0"); continue
            print(f"{name:12s} n={d['n']:5d}  macro_f1={d['macro_f1']:.4f}  acc={d['accuracy']:.4f}")

        # Pitch-span bucket restricted to genuinely-moving GT primitives
        # (true class != Fixed) -- span is undefined/~0 for Fixed by construction.
        moving_mask = true != 0
        span_report = _bucket_report(pred[moving_mask], true[moving_mask], span[moving_mask],
                                      SPAN_BUCKETS_CENTS, SPAN_BUCKET_NAMES)
        result[f"{key}_by_pitch_span"] = span_report

    print("\n=== Step 22 §16 pitch-span analysis (moving primitives only) ===")
    for key in ("oracle_shape_only", "crepe_shape_only"):
        print(f"\n-- {key} --")
        for name in SPAN_BUCKET_NAMES:
            d = result[f"{key}_by_pitch_span"][name]
            if d["n"] == 0:
                print(f"{name:10s} n=0"); continue
            print(f"{name:10s} n={d['n']:5d}  macro_f1={d['macro_f1']:.4f}  acc={d['accuracy']:.4f}")

    (OUT_DIR / "duration_span_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nsaved to {OUT_DIR / 'duration_span_analysis.json'}")


if __name__ == "__main__":
    main()
