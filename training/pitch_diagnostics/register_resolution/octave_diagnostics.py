"""Step 12 spec sections 3-8: octave-error temporal structure, GT availability
given wrong-octave argmax, salience margins, GT/2 vs GT vs 2*GT harmonic-pair
analysis, and the two-large-failure-recordings deep dive. Consumes the pooled
per-recording cache built by collect.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.pitch_diagnostics.common import summarize_array, write_json  # noqa: E402
from training.pitch_diagnostics.register_resolution.collect import build  # noqa: E402
from training.pitch_diagnostics.register_resolution.common import (  # noqa: E402
    LARGE_FAILURE_RECORDINGS,
    REG_DIR,
)

HOP_S = 0.01
DUR_BUCKETS = [("<30ms", 0, 0.03), ("30-100ms", 0.03, 0.1), ("100-250ms", 0.1, 0.25),
               ("250-500ms", 0.25, 0.5), ("500ms-1s", 0.5, 1.0), (">1s", 1.0, float("inf"))]
OCT_STATES = (-2, -1, 0, 1, 2)


def _bucket(dur: float) -> str:
    for name, lo, hi in DUR_BUCKETS:
        if lo <= dur < hi:
            return name
    return ">1s"


def extract_runs(rec: dict, method: str) -> list[dict]:
    times = rec["times"]
    k = np.asarray(rec[method]["octave_k"])
    runs = []
    i = 0
    n = len(k)
    while i < n:
        if k[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and k[j + 1] == k[i]:
            j += 1
        dur = float(times[j] - times[i] + HOP_S)
        runs.append({
            "recording_id": rec["recording_id"], "fold": rec["fold"], "k": int(k[i]),
            "n_frames": j - i + 1, "duration_s": dur, "start_time_s": float(times[i]),
            "mean_entropy": float(rec[method]["entropy"][i:j + 1].mean()),
        })
        i = j + 1
    return runs


def transition_matrix(rec: dict, method: str) -> np.ndarray:
    """Counts[i,j] = #(k_{t-1}=OCT_STATES[i] -> k_t=OCT_STATES[j]) for TRUE
    temporally-adjacent valid frames (times differ by ~hop_s)."""
    times = rec["times"]
    k = np.clip(np.asarray(rec[method]["octave_k"]), -2, 2)
    counts = np.zeros((5, 5), dtype=np.int64)
    idx = {v: i for i, v in enumerate(OCT_STATES)}
    for t in range(len(times) - 1):
        if abs(times[t + 1] - times[t] - HOP_S) < 1.5 * HOP_S:
            counts[idx[int(k[t])], idx[int(k[t + 1])]] += 1
    return counts


def run_duration_stats(runs: list[dict]) -> dict:
    if not runs:
        return {"n_runs": 0}
    durs = np.array([r["duration_s"] for r in runs])
    qs = np.quantile(durs, [0.5, 0.75, 0.9, 0.95])
    bucket_counts = {name: 0 for name, _, _ in DUR_BUCKETS}
    for d in durs:
        bucket_counts[_bucket(d)] += 1
    return {
        "n_runs": len(runs), "median_s": float(qs[0]), "p75_s": float(qs[1]),
        "p90_s": float(qs[2]), "p95_s": float(qs[3]), "max_s": float(durs.max()),
        "mean_s": float(durs.mean()), "bucket_counts": bucket_counts,
        "bucket_fractions": {k: v / len(runs) for k, v in bucket_counts.items()},
        "k_distribution": {str(kv): int(sum(1 for r in runs if r["k"] == kv)) for kv in (-2, -1, 1, 2)},
    }


def gt_topk_given_wrong_octave(records: list[dict], method: str) -> dict:
    ranks_wrong, ranks_correct = [], []
    for rec in records:
        k = np.asarray(rec[method]["octave_k"])
        rank = np.asarray(rec[method]["rank"])
        ranks_wrong.append(rank[k != 0])
        ranks_correct.append(rank[k == 0])
    rw = np.concatenate(ranks_wrong) if ranks_wrong else np.array([])
    rc = np.concatenate(ranks_correct) if ranks_correct else np.array([])
    out = {"n_wrong_octave": int(len(rw)), "n_correct_octave": int(len(rc)), "tolerance_cents": 30.0}
    for k_ in (2, 3, 5, 10):
        out[f"p_gt_in_top{k_}_given_wrong_octave"] = float((rw <= k_).mean()) if len(rw) else None
        out[f"p_gt_in_top{k_}_given_correct_octave"] = float((rc <= k_).mean()) if len(rc) else None
    out["median_rank_wrong_octave"] = float(np.median(rw)) if len(rw) else None
    out["median_rank_correct_octave"] = float(np.median(rc)) if len(rc) else None
    return out


def salience_margins(records: list[dict], method: str) -> dict:
    m_correct, m_wrong, e_correct, e_wrong = [], [], [], []
    for rec in records:
        k = np.asarray(rec[method]["octave_k"])
        m = np.asarray(rec[method]["margin12"])
        e = np.asarray(rec[method]["entropy"])
        m_correct.append(m[k == 0]); m_wrong.append(m[k != 0])
        e_correct.append(e[k == 0]); e_wrong.append(e[k != 0])
    return {
        "margin_correct_octave": summarize_array(np.concatenate(m_correct)),
        "margin_wrong_octave": summarize_array(np.concatenate(m_wrong)),
        "entropy_correct_octave": summarize_array(np.concatenate(e_correct)),
        "entropy_wrong_octave": summarize_array(np.concatenate(e_wrong)),
    }


def harmonic_pair_analysis(records: list[dict], method: str) -> dict:
    by_k: dict[str, dict] = {}
    for kv, name in ((0, "correct_octave"), (1, "plus1"), (-1, "minus1")):
        gt2, gt, twogt = [], [], []
        for rec in records:
            k = np.asarray(rec[method]["octave_k"])
            mask = k == kv
            gt2.append(rec[method]["gt2_score"][mask])
            gt.append(rec[method]["gt_score"][mask])
            twogt.append(rec[method]["2gt_score"][mask])
        gt2c, gtc, twogtc = np.concatenate(gt2), np.concatenate(gt), np.concatenate(twogt)
        if len(gtc) == 0:
            by_k[name] = {"n": 0}
            continue
        norm = gt2c + gtc + twogtc + 1e-12
        by_k[name] = {
            "n": int(len(gtc)),
            "mean_S_GTover2": float(gt2c.mean()), "mean_S_GT": float(gtc.mean()), "mean_S_2GT": float(twogtc.mean()),
            "mean_normalized_S_GTover2": float((gt2c / norm).mean()),
            "mean_normalized_S_GT": float((gtc / norm).mean()),
            "mean_normalized_S_2GT": float((twogtc / norm).mean()),
            "frac_2GT_gt_GT": float((twogtc > gtc).mean()),
            "frac_GTover2_gt_GT": float((gt2c > gtc).mean()),
        }
    return by_k


def large_recordings_deep_dive(records: list[dict]) -> dict:
    by_rec = {r["recording_id"]: r for r in records}
    others_sorted = sorted(
        [r for r in records if r["recording_id"] not in LARGE_FAILURE_RECORDINGS],
        key=lambda r: np.abs(r["learned"]["argmax_cents"] - r["true_cents"]).mean()
        - np.abs(r["hps"]["argmax_cents"] - r["true_cents"]).mean(),
    )
    strong_improve = others_sorted[:3]  # most negative delta = learned much better than HPS

    def _summary(rec: dict) -> dict:
        hps_err = np.abs(rec["hps"]["argmax_cents"] - rec["true_cents"])
        learned_err = np.abs(rec["learned"]["argmax_cents"] - rec["true_cents"])
        hps_k = np.asarray(rec["hps"]["octave_k"]); learned_k = np.asarray(rec["learned"]["octave_k"])
        return {
            "n_valid": rec["n_valid"], "fold": rec["fold"],
            "hps_raw_mae": float(hps_err.mean()), "learned_raw_mae": float(learned_err.mean()),
            "hps_octave_adjusted_mae": float(np.asarray(rec["hps"]["octave_err"]).mean()),
            "learned_octave_adjusted_mae": float(np.asarray(rec["learned"]["octave_err"]).mean()),
            "hps_correct_octave_rate": float((hps_k == 0).mean()), "learned_correct_octave_rate": float((learned_k == 0).mean()),
            "hps_plus1_rate": float((hps_k == 1).mean()), "hps_minus1_rate": float((hps_k == -1).mean()),
            "learned_plus1_rate": float((learned_k == 1).mean()), "learned_minus1_rate": float((learned_k == -1).mean()),
            "hps_mean_entropy": float(np.mean(rec["hps"]["entropy"])), "learned_mean_entropy": float(np.mean(rec["learned"]["entropy"])),
            "hps_gt_top3_rate": float((np.asarray(rec["hps"]["rank"]) <= 3).mean()),
            "learned_gt_top3_rate": float((np.asarray(rec["learned"]["rank"]) <= 3).mean()),
        }

    return {
        "large_failure_recordings": {rid: _summary(by_rec[rid]) for rid in LARGE_FAILURE_RECORDINGS if rid in by_rec},
        "contrast_strong_improvement_recordings": {r["recording_id"]: _summary(r) for r in strong_improve},
    }


def main() -> None:
    records = build()
    out: dict = {}

    for method in ("hps", "learned"):
        all_runs = []
        trans = np.zeros((5, 5), dtype=np.int64)
        for rec in records:
            all_runs.extend(extract_runs(rec, method))
            trans += transition_matrix(rec, method)
        out.setdefault("run_duration_stats", {})[method] = run_duration_stats(all_runs)
        row_sums = trans.sum(axis=1, keepdims=True)
        probs = np.divide(trans, np.maximum(row_sums, 1), where=row_sums > 0)
        out.setdefault("transition_matrix", {})[method] = {
            "states": list(OCT_STATES), "counts": trans.tolist(), "probabilities": probs.tolist(),
            "P_k0_given_k0": float(probs[2, 2]), "P_kplus1_given_k0": float(probs[2, 3]),
            "P_kplus1_given_kplus1": float(probs[3, 3]), "P_k0_given_kplus1": float(probs[3, 2]),
            "P_kminus1_given_k0": float(probs[2, 1]), "P_kminus1_given_kminus1": float(probs[1, 1]),
            "P_k0_given_kminus1": float(probs[1, 2]),
        }

    write_json(REG_DIR / "octave_run_durations.json", out["run_duration_stats"])
    write_json(REG_DIR / "octave_transition_matrix.json", out["transition_matrix"])

    gt_topk = {m: gt_topk_given_wrong_octave(records, m) for m in ("hps", "learned")}
    write_json(REG_DIR / "gt_topk_given_wrong_octave.json", gt_topk)

    margins = {m: salience_margins(records, m) for m in ("hps", "learned")}
    write_json(REG_DIR / "salience_margins.json", margins)

    pairs = {m: harmonic_pair_analysis(records, m) for m in ("hps", "learned")}
    write_json(REG_DIR / "harmonic_pair_analysis.json", pairs)

    deep_dive = large_recordings_deep_dive(records)
    write_json(REG_DIR / "large_failure_recordings_analysis.json", deep_dive)

    print("=== octave_diagnostics summary ===")
    for m in ("hps", "learned"):
        print(m, "n_runs", out["run_duration_stats"][m].get("n_runs"),
              "median_run_s", out["run_duration_stats"][m].get("median_s"))
        print(m, "GT top3 | wrong octave:", gt_topk[m]["p_gt_in_top3_given_wrong_octave"])
        print(m, "P(k=0|k_prev=0):", out["transition_matrix"][m]["P_k0_given_k0"])


if __name__ == "__main__":
    main()
