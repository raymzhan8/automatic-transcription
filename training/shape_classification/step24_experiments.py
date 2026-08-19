"""Step 24: does explicit fitting against the four KNOWN canonical
trajectory templates (recovered from idtap's own Trajectory formulas,
`templates.py`) classify CREPE-normalized contours better than Step 23's
learned CNN had to, without any class prior, class balancing, or learned
parameter at all? Deterministic argmin-of-template-error classification,
oracle sanity gate, MSE vs. one robust (Huber) scorer, margin analysis,
T2-vs-T3 and 3-way-bend sub-diagnostics, endpoint-error and duration/span
slicing, and (gated) one smoothing control. Reuses Step 22's corpus
(`dataset.build()`) directly -- no new pitch extraction.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.pitch_diagnostics.pitch_audit.common import recording_fold_map  # noqa: E402
from training.shape_classification.contours import X_GRID  # noqa: E402
from training.shape_classification.dataset import CLASS_NAMES, OUT_DIR, build  # noqa: E402
from training.shape_classification.duration_span_analysis import (  # noqa: E402
    DURATION_BUCKET_NAMES, DURATION_BUCKETS_S, SPAN_BUCKET_NAMES, SPAN_BUCKETS_CENTS,
)
from training.shape_classification.metrics_utils import eval_metrics, prediction_frequency  # noqa: E402
from training.shape_classification.templates import (  # noqa: E402
    HUBER_DELTA_CENTS, TEMPLATE_NAMES, template_errors,
)

STEP24_DIR = OUT_DIR / "step24"
FIG_DIR = STEP24_DIR / "figures"


# --------------------------------------------------------------- scoring ----

def score_corpus(records: list[dict], source_key: str, *, robust: bool, fold_map: dict[str, int]) -> list[dict[str, Any]]:
    out = []
    for r in records:
        d = r[source_key]
        if d is None:
            continue
        errs = template_errors(d["r"], d["span_cents"], robust=robust)
        out.append({
            "primitive_id": r["primitive_id"], "recording_id": r["recording_id"],
            "fold": fold_map.get(r["recording_id"]), "true": r["canonical_type"],
            "pred": int(np.argmin(errs)), "errors": errs,
            "duration_s": r["duration_s"], "span_cents": abs(d["span_cents"]),
        })
    return out


def pooled_metrics(scored: list[dict], names: tuple[str, ...] = TEMPLATE_NAMES) -> dict:
    pred = np.array([s["pred"] for s in scored])
    true = np.array([s["true"] for s in scored])
    return eval_metrics(pred, true, names)


def grouped_fold_stats(scored: list[dict], names: tuple[str, ...] = TEMPLATE_NAMES) -> dict:
    by_fold = defaultdict(list)
    for s in scored:
        by_fold[s["fold"]].append(s)
    per_fold_f1 = {}
    for fold, items in by_fold.items():
        pred = np.array([s["pred"] for s in items]); true = np.array([s["true"] for s in items])
        per_fold_f1[fold] = eval_metrics(pred, true, names)["macro_f1"]
    return {"per_fold_macro_f1": per_fold_f1,
            "grouped_mean_macro_f1": float(np.mean(list(per_fold_f1.values()))),
            "grouped_std_macro_f1": float(np.std(list(per_fold_f1.values())))}


def restricted_metrics(scored: list[dict], class_ids: tuple[int, ...], names: tuple[str, ...]) -> dict:
    id_to_new = {c: i for i, c in enumerate(class_ids)}
    pred, true = [], []
    for s in scored:
        if s["true"] not in id_to_new:
            continue
        # re-argmin restricted to the allowed template subset
        sub_errs = [s["errors"][c] for c in class_ids]
        pred.append(id_to_new[class_ids[int(np.argmin(sub_errs))]])
        true.append(id_to_new[s["true"]])
    return eval_metrics(np.array(pred), np.array(true), names)


def bucket_report(scored: list[dict], key: str, edges: tuple[float, ...], names: tuple[str, ...],
                   restrict_moving: bool = False) -> dict:
    out = {}
    items = [s for s in scored if (not restrict_moving or s["true"] != 0)]
    for name, lo, hi in zip(names, edges[:-1], edges[1:]):
        pool = [s for s in items if lo <= s[key] < hi]
        if not pool:
            out[name] = {"n": 0}
            continue
        m = pooled_metrics(pool)
        out[name] = {"n": len(pool), "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]}
    return out


def _margin(s: dict) -> float:
    errs = s["errors"]
    others = [errs[k] for k in range(4) if k != s["true"]]
    if s["pred"] == s["true"]:
        return min(others) - errs[s["true"]]  # positive: how much true beats best competitor
    return errs[s["true"]] - errs[s["pred"]]  # positive: how much wrong pred beats true


def plot_representative_segments(records: list[dict], scored: list[dict]) -> None:
    """Section 22: deterministic example selection (highest-confidence
    correct / lowest-margin correct / largest-margin error) per class, no
    cherry-picking beyond the stated criteria."""
    from training.shape_classification.templates import template_curves

    rec_by_pid = {r["primitive_id"]: r for r in records}
    curves = template_curves()
    classes = (1, 2, 3)  # Cosine, Sloped-start, Sloped-end -- the classes under dispute
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True)
    col_titles = ("highest-confidence correct", "lowest-margin correct", "largest-margin error")

    for row, t in enumerate(classes):
        pool = [s for s in scored if s["true"] == t]
        correct = [s for s in pool if s["pred"] == t]
        incorrect = [s for s in pool if s["pred"] != t]
        picks = [
            max(correct, key=_margin) if correct else None,
            min(correct, key=_margin) if correct else None,
            max(incorrect, key=_margin) if incorrect else None,
        ]
        for col, s in enumerate(picks):
            ax = axes[row, col]
            if s is None:
                ax.set_axis_off()
                continue
            rec = rec_by_pid[s["primitive_id"]]
            r_obs = rec["crepe"]["r"]
            span = rec["crepe"]["span_cents"]
            ax.plot(X_GRID, r_obs, color="black", linewidth=1.2, label="observed CREPE r(x)")
            ax.plot(X_GRID, span * curves[t], color="tab:green", linestyle="--", label=f"true template ({CLASS_NAMES[t]})")
            if s["pred"] != t:
                ax.plot(X_GRID, span * curves[s["pred"]], color="tab:red", linestyle=":",
                        label=f"predicted template ({CLASS_NAMES[s['pred']]})")
            errs_str = ", ".join(f"{CLASS_NAMES[k]}={s['errors'][k]:.0f}" for k in range(4))
            ax.set_title(f"{col_titles[col]}\n{errs_str}", fontsize=7)
            ax.legend(fontsize=6)
            if col == 0:
                ax.set_ylabel(f"true {CLASS_NAMES[t]}\nr(x) [cents]")
    fig.suptitle("Step 24 §22: representative CREPE MSE template fits")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "representative_segments.png", dpi=130)
    plt.close(fig)
    print(f"saved {FIG_DIR / 'representative_segments.png'}")


# ---------------------------------------------------------------- report ----

def main() -> None:
    STEP24_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    records = build()
    fold_map = recording_fold_map(REPO_ROOT)

    results: dict[str, Any] = {}

    print("=== Step 24 §4-6, §10: 4-way template classification ===")
    for tag, source, robust in (
        ("oracle_mse", "oracle", False), ("oracle_robust", "oracle", True),
        ("crepe_mse", "crepe", False), ("crepe_robust", "crepe", True),
    ):
        scored = score_corpus(records, source, robust=robust, fold_map=fold_map)
        pooled = pooled_metrics(scored)
        grouped = grouped_fold_stats(scored)
        results[tag] = {"pooled": pooled, **grouped, "scored": scored}
        row = " ".join(f"{pooled['per_class'][n]['f1']:.3f}" for n in TEMPLATE_NAMES)
        print(f"{tag:14s} macro_f1={pooled['macro_f1']:.4f} acc={pooled['accuracy']:.4f} "
              f"[{','.join(TEMPLATE_NAMES)}]={row} grouped_mean={grouped['grouped_mean_macro_f1']:.4f}")

    # §7 prediction frequency (crepe_mse, for direct Step 23 comparison)
    crepe_mse_scored = results["crepe_mse"]["scored"]
    pred_freq = prediction_frequency(np.array([s["pred"] for s in crepe_mse_scored]), TEMPLATE_NAMES)
    true_freq = prediction_frequency(np.array([s["true"] for s in crepe_mse_scored]), TEMPLATE_NAMES)
    print("\nprediction frequency (crepe_mse):", {k: round(v, 3) for k, v in pred_freq.items()})
    print("true distribution:               ", {k: round(v, 3) for k, v in true_freq.items()})

    # §12 T2-vs-T3 template diagnostic
    print("\n=== Step 24 §12: T2-vs-T3 template diagnostic ===")
    t2t3 = {}
    for tag in ("oracle_mse", "oracle_robust", "crepe_mse", "crepe_robust"):
        m = restricted_metrics(results[tag]["scored"], (2, 3), ("Sloped-start", "Sloped-end"))
        t2t3[tag] = m
        print(f"{tag:14s} acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} "
              f"T2_F1={m['per_class']['Sloped-start']['f1']:.4f} T3_F1={m['per_class']['Sloped-end']['f1']:.4f}")

    # §13 3-way bend-only template diagnostic
    print("\n=== Step 24 §13: 3-way bend-only template diagnostic ===")
    bend3 = {}
    for tag in ("oracle_mse", "oracle_robust", "crepe_mse", "crepe_robust"):
        m = restricted_metrics(results[tag]["scored"], (1, 2, 3), ("Cosine", "Sloped-start", "Sloped-end"))
        bend3[tag] = m
        row = " ".join(f"{m['per_class'][n]['f1']:.3f}" for n in ("Cosine", "Sloped-start", "Sloped-end"))
        print(f"{tag:14s} macro_f1={m['macro_f1']:.4f} [Cos,SlS,SlE]={row}")

    # §11 Cosine-vs-Sloped margin analysis (crepe_mse and oracle_mse)
    print("\n=== Step 24 §11: Cosine-vs-Sloped margin analysis ===")
    margins = {}
    for tag in ("oracle_mse", "crepe_mse"):
        scored = results[tag]["scored"]
        m_start = {t: [] for t in (1, 2, 3)}
        m_end = {t: [] for t in (1, 2, 3)}
        for s in scored:
            if s["true"] not in (1, 2, 3):
                continue
            e = s["errors"]
            m_start[s["true"]].append(e[1] - e[2])  # E_cosine - E_sloped_start
            m_end[s["true"]].append(e[1] - e[3])    # E_cosine - E_sloped_end
        margins[tag] = {
            "M_start_by_true": {CLASS_NAMES[t]: {"mean": float(np.mean(v)), "median": float(np.median(v)),
                                                   "frac_positive": float(np.mean(np.array(v) > 0)), "n": len(v)}
                                 for t, v in m_start.items()},
            "M_end_by_true": {CLASS_NAMES[t]: {"mean": float(np.mean(v)), "median": float(np.median(v)),
                                                 "frac_positive": float(np.mean(np.array(v) > 0)), "n": len(v)}
                               for t, v in m_end.items()},
        }
        print(f"-- {tag} --")
        for t in (1, 2, 3):
            print(f"  true={CLASS_NAMES[t]:14s} M_start(median)={margins[tag]['M_start_by_true'][CLASS_NAMES[t]]['median']:+.1f} "
                  f"frac>0={margins[tag]['M_start_by_true'][CLASS_NAMES[t]]['frac_positive']:.3f}  "
                  f"M_end(median)={margins[tag]['M_end_by_true'][CLASS_NAMES[t]]['median']:+.1f} "
                  f"frac>0={margins[tag]['M_end_by_true'][CLASS_NAMES[t]]['frac_positive']:.3f}")

        if tag == "crepe_mse":
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
            for ax, m_dict, title in ((axes[0], m_start, "M_start = E_cosine - E_sloped_start"),
                                        (axes[1], m_end, "M_end = E_cosine - E_sloped_end")):
                # Fixed +-3000 cents^2 window (sqrt(3000)~=55c, close to MIN_SPAN_CENTS) --
                # the decision-relevant region; a handful of extreme-noise outliers otherwise
                # stretch the histogram range until the near-zero region is unreadable.
                lo, hi = -3000.0, 3000.0
                bins = np.linspace(lo, hi, 41)
                for t, color in ((1, "tab:blue"), (2, "tab:green"), (3, "tab:red")):
                    ax.hist(np.clip(m_dict[t], lo, hi), bins=bins, alpha=0.5, label=f"true {CLASS_NAMES[t]}", color=color, density=True)
                ax.axvline(0.0, color="black", linewidth=1)
                ax.set_title(title + "\n(clipped to ±3000 cents²)")
                ax.set_xlabel("margin (cents²)")
                ax.legend(fontsize=8)
            fig.suptitle(f"Step 24 §11: Cosine-vs-Sloped margins ({tag})")
            fig.tight_layout()
            fig.savefig(FIG_DIR / "margins_crepe_mse.png", dpi=130)
            plt.close(fig)
            print(f"saved {FIG_DIR / 'margins_crepe_mse.png'}")

    # §14 endpoint-error analysis
    print("\n=== Step 24 §14: endpoint-error analysis ===")
    endpoint = []
    for r in records:
        o, c = r["oracle"], r["crepe"]
        if o is None or c is None:
            continue
        start_err = 1200.0 * (c["log2"][0] - o["log2"][0])
        end_err = 1200.0 * (c["log2"][-1] - o["log2"][-1])
        span_err = c["span_cents"] - o["span_cents"]
        endpoint.append({"primitive_id": r["primitive_id"], "true": r["canonical_type"],
                          "start_err_cents": start_err, "end_err_cents": end_err, "span_err_cents": span_err})
    ep_by_pid = {e["primitive_id"]: e for e in endpoint}
    correct_start_err, incorrect_start_err = [], []
    correct_end_err, incorrect_end_err = [], []
    for s in crepe_mse_scored:
        e = ep_by_pid.get(s["primitive_id"])
        if e is None:
            continue
        is_correct = s["pred"] == s["true"]
        (correct_start_err if is_correct else incorrect_start_err).append(abs(e["start_err_cents"]))
        (correct_end_err if is_correct else incorrect_end_err).append(abs(e["end_err_cents"]))
    endpoint_summary = {
        "mean_abs_start_err_correct": float(np.mean(correct_start_err)) if correct_start_err else None,
        "mean_abs_start_err_incorrect": float(np.mean(incorrect_start_err)) if incorrect_start_err else None,
        "mean_abs_end_err_correct": float(np.mean(correct_end_err)) if correct_end_err else None,
        "mean_abs_end_err_incorrect": float(np.mean(incorrect_end_err)) if incorrect_end_err else None,
        "corr_abs_start_err_vs_correct": float(np.corrcoef(
            [abs(ep_by_pid[s["primitive_id"]]["start_err_cents"]) for s in crepe_mse_scored if s["primitive_id"] in ep_by_pid],
            [int(s["pred"] == s["true"]) for s in crepe_mse_scored if s["primitive_id"] in ep_by_pid])[0, 1]),
    }
    print(json.dumps(endpoint_summary, indent=2))

    # §22 representative segment plots (deterministic selection, crepe_mse)
    plot_representative_segments(records, crepe_mse_scored)

    # §15-16 duration / pitch-span analysis (crepe_mse, crepe_robust)
    print("\n=== Step 24 §15-16: duration / pitch-span analysis ===")
    duration_span = {}
    for tag in ("crepe_mse", "crepe_robust", "oracle_mse"):
        scored = results[tag]["scored"]
        duration_span[f"{tag}_by_duration"] = bucket_report(scored, "duration_s", DURATION_BUCKETS_S, DURATION_BUCKET_NAMES)
        duration_span[f"{tag}_by_span"] = bucket_report(scored, "span_cents", SPAN_BUCKETS_CENTS, SPAN_BUCKET_NAMES, restrict_moving=True)
    for name in DURATION_BUCKET_NAMES:
        d = duration_span["crepe_mse_by_duration"][name]
        print(f"duration {name:12s} n={d.get('n',0):5d}  macro_f1={d.get('macro_f1', float('nan')):.4f}" if d.get("n") else f"duration {name:12s} n=0")

    # §17 smoothing gate check: does robust scoring meaningfully help CREPE?
    mse_f1 = results["crepe_mse"]["pooled"]["macro_f1"]
    robust_f1 = results["crepe_robust"]["pooled"]["macro_f1"]
    gate_met = (robust_f1 - mse_f1) > 0.02  # robust clearly helps -> jitter/outliers are a real factor
    print(f"\n§17 smoothing gate: robust_f1={robust_f1:.4f} vs mse_f1={mse_f1:.4f} -> gate_met={gate_met}")
    smoothing_result = None
    if gate_met:
        from training.shape_classification.step24_smoothing import run_smoothing_control
        smoothing_result = run_smoothing_control(records, fold_map)
        smoothing_result.update(grouped_fold_stats(smoothing_result["scored"]))
        results["crepe_smoothed_mse"] = smoothing_result
        p = smoothing_result["pooled"]
        row = " ".join(f"{p['per_class'][n]['f1']:.3f}" for n in TEMPLATE_NAMES)
        print(f"crepe_smoothed_mse macro_f1={p['macro_f1']:.4f} [{','.join(TEMPLATE_NAMES)}]={row}")

    # save
    out = {
        "four_way": {tag: {k: v for k, v in results[tag].items() if k != "scored"}
                     for tag in ("oracle_mse", "oracle_robust", "crepe_mse", "crepe_robust")},
        "prediction_frequency": {"crepe_mse": pred_freq, "true": true_freq},
        "t2_vs_t3": t2t3, "three_way_bend": bend3, "margins": margins,
        "endpoint_error": endpoint_summary, "duration_span": duration_span,
        "smoothing_gate_met": gate_met,
        "smoothing_result": ({k: v for k, v in smoothing_result.items() if k != "scored"}
                              if smoothing_result is not None else None),
        "huber_delta_cents": HUBER_DELTA_CENTS,
    }
    (STEP24_DIR / "results.json").write_text(json.dumps(out, indent=2) + "\n")

    scored_out = {tag: results[tag]["scored"] for tag in ("oracle_mse", "oracle_robust", "crepe_mse", "crepe_robust")}
    if smoothing_result is not None:
        scored_out["crepe_smoothed_mse"] = smoothing_result["scored"]
    with open(STEP24_DIR / "scored_primitives.json", "w") as fh:
        json.dump(scored_out, fh, indent=2)
    endpoint_path = STEP24_DIR / "endpoint_errors.json"
    endpoint_path.write_text(json.dumps(endpoint, indent=2) + "\n")

    print(f"\nsaved to {STEP24_DIR / 'results.json'}, scored_primitives.json, endpoint_errors.json")


if __name__ == "__main__":
    main()
