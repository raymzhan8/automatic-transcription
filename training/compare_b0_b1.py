"""Paired B0 vs B1 comparison from eval_summary.json files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _delta(b1: float | None, b0: float | None) -> float | None:
    if b1 is None or b0 is None:
        return None
    return float(b1) - float(b0)


def fold_delta(b0: dict[str, Any], b1: dict[str, Any]) -> dict[str, Any]:
    fm0, fm1 = b0["frame_metrics"], b1["frame_metrics"]
    tm0 = b0.get("trajectory_metrics") or {}
    tm1 = b1.get("trajectory_metrics") or {}
    per_class = {}
    for cls in ("T0", "T1", "T2", "T3"):
        c0 = fm0.get("per_class", {}).get(cls, {})
        c1 = fm1.get("per_class", {}).get(cls, {})
        per_class[cls] = {
            "delta_f1": _delta(c1.get("f1"), c0.get("f1")),
            "delta_recall": _delta(c1.get("recall"), c0.get("recall")),
            "b0_f1": c0.get("f1"),
            "b1_f1": c1.get("f1"),
            "b0_recall": c0.get("recall"),
            "b1_recall": c1.get("recall"),
        }
    pairs = {}
    p0 = b0.get("confusion_pairs") or {}
    p1 = b1.get("confusion_pairs") or {}
    for k in sorted(set(p0) | set(p1)):
        pairs[k] = {"b0": p0.get(k, 0), "b1": p1.get(k, 0), "delta": p1.get(k, 0) - p0.get(k, 0)}
    t1 = {}
    for bucket in ("raw_t1", "t4_decomposition", "t5_decomposition", "t6_decomposition"):
        a = (b0.get("t1_provenance") or {}).get(bucket, {})
        b = (b1.get("t1_provenance") or {}).get(bucket, {})
        t1[bucket] = {
            "b0_recall": a.get("recall"),
            "b1_recall": b.get("recall"),
            "delta_recall": _delta(b.get("recall"), a.get("recall")),
            "b0_count": a.get("count"),
            "b1_count": b.get("count"),
        }
    return {
        "fold_index": b0["fold_index"],
        "frame_accuracy": {
            "b0": fm0.get("accuracy"),
            "b1": fm1.get("accuracy"),
            "delta": _delta(fm1.get("accuracy"), fm0.get("accuracy")),
        },
        "frame_macro_f1": {
            "b0": fm0.get("macro_f1"),
            "b1": fm1.get("macro_f1"),
            "delta": _delta(fm1.get("macro_f1"), fm0.get("macro_f1")),
        },
        "trajectory_accuracy": {
            "b0": tm0.get("accuracy"),
            "b1": tm1.get("accuracy"),
            "delta": _delta(tm1.get("accuracy"), tm0.get("accuracy")),
        },
        "trajectory_macro_f1": {
            "b0": tm0.get("macro_f1"),
            "b1": tm1.get("macro_f1"),
            "delta": _delta(tm1.get("macro_f1"), tm0.get("macro_f1")),
        },
        "per_class": per_class,
        "confusion_pairs": pairs,
        "t1_provenance": t1,
        "pitch": b1.get("pitch", {}).get("overall"),
    }


def summarize(deltas: list[dict[str, Any]]) -> dict[str, Any]:
    def collect(path: tuple[str, ...]) -> list[float]:
        vals = []
        for d in deltas:
            cur: Any = d
            for k in path:
                cur = cur.get(k) if isinstance(cur, dict) else None
            if isinstance(cur, (int, float)):
                vals.append(float(cur))
        return vals

    keys = {
        "delta_frame_macro_f1": ("frame_macro_f1", "delta"),
        "delta_traj_macro_f1": ("trajectory_macro_f1", "delta"),
        "delta_T2_recall": ("per_class", "T2", "delta_recall"),
        "delta_T3_recall": ("per_class", "T3", "delta_recall"),
    }
    out: dict[str, Any] = {}
    for name, path in keys.items():
        vals = collect(path)
        out[name] = {
            "values": vals,
            "mean": float(np.mean(vals)) if vals else None,
            "median": float(np.median(vals)) if vals else None,
            "n_folds_improved": int(sum(v > 0 for v in vals)),
            "n_folds": len(vals),
        }
    try:
        from scipy.stats import wilcoxon

        f1 = collect(("frame_macro_f1", "delta"))
        if len(f1) >= 5 and any(v != 0 for v in f1):
            stat = wilcoxon(f1)
            out["wilcoxon_frame_macro_f1_delta"] = {
                "statistic": float(stat.statistic),
                "pvalue": float(stat.pvalue),
                "note": "exploratory only; n=5",
            }
    except Exception:
        pass
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0-dir", type=Path, default=REPO_ROOT / "output" / "framewise_runs" / "b0_tcn_type_only")
    parser.add_argument("--b1-dir", type=Path, default=REPO_ROOT / "output" / "framewise_runs" / "b1_tcn_type_pitch")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    deltas = []
    for i in range(5):
        p0 = args.b0_dir / f"fold_{i}" / "eval" / "eval_summary.json"
        p1 = args.b1_dir / f"fold_{i}" / "eval" / "eval_summary.json"
        if not p0.exists() or not p1.exists():
            print(f"skip fold {i}: missing {p0 if not p0.exists() else p1}", file=sys.stderr)
            continue
        deltas.append(fold_delta(_load(p0), _load(p1)))

    report = {"folds": deltas, "summary": summarize(deltas)}
    out = args.output or (args.b1_dir / "b0_vs_b1.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
