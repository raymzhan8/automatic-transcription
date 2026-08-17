"""Step 16 sections 11-13: octave-transition contribution to motion error
(closing the register question again, briefly), salience GT-rank/top-k
evidence audit, and harmonic/drone error-association audit.
"""

from __future__ import annotations

import numpy as np

from training.metrics import TYPE_NAMES
from training.pitch_diagnostics.common import octave_adjusted_error, write_json
from training.pitch_diagnostics.pitch_audit.common import AUDIT_DIR, build_bundles, delta_at_offset
from training.pitch_diagnostics.relative_pitch.dense_relative_salience import HALF_W, W_BINS
from training.pitch_diagnostics.relative_pitch.dense_relative_salience import build as build_relative_salience
from training.pitch_diagnostics.common import bin_from_hz

TRANSITION_WINDOWS_MS = (50, 100, 200)
LARGE_DELTA_ERROR_CENTS = 100.0  # "large motion error" threshold at the 50ms (k=5) scale
CENTS_PER_BIN = 16.6666667

HARMONIC_TARGETS = {
    "octave_+1": 1200.0, "octave_-1": -1200.0, "octave+fifth_+1": 1901.96, "octave+fifth_-1": -1901.96,
}
HARMONIC_TOL_CENTS = 50.0
LARGE_ABS_ERROR_CENTS = 200.0


def octave_transition_contribution(bundles: list[dict]) -> dict:
    out_by_window = {}
    total_large_error = 0
    total_large_error_near_transition = 0
    frac_near = {w: [] for w in TRANSITION_WINDOWS_MS}

    for b in bundles:
        valid = b["valid"]
        _err, k = octave_adjusted_error(b["est_cents"], b["gt_cents"])
        k = np.where(valid, k, np.nan)
        transition = np.zeros(len(k), dtype=bool)
        transition[1:] = valid[1:] & valid[:-1] & (k[1:] != k[:-1])
        trans_idx = np.flatnonzero(transition)

        d_est = delta_at_offset(b["est_cents"], 5)
        d_gt = delta_at_offset(b["gt_cents"], 5)
        err50 = np.abs(d_est - d_gt)
        large = valid & np.isfinite(err50) & (err50 > LARGE_DELTA_ERROR_CENTS)
        total_large_error += int(large.sum())

        for w_ms in TRANSITION_WINDOWS_MS:
            w_frames = w_ms // 10
            near = np.zeros(len(k), dtype=bool)
            for ti in trans_idx:
                lo, hi = max(0, ti - w_frames), min(len(k), ti + w_frames + 1)
                near[lo:hi] = True
            near = near & valid
            frac_near[w_ms].append(float(near.sum()) / max(int(valid.sum()), 1))
            if w_ms == 100:
                total_large_error_near_transition += int((large & near).sum())

    for w_ms in TRANSITION_WINDOWS_MS:
        out_by_window[f"{w_ms}ms"] = {"mean_frac_valid_frames_near_transition": float(np.mean(frac_near[w_ms]))}

    return {
        "by_window": out_by_window,
        "large_error_threshold_cents_at_50ms_scale": LARGE_DELTA_ERROR_CENTS,
        "total_large_error_frames": total_large_error,
        "large_error_frames_within_100ms_of_transition": total_large_error_near_transition,
        "fraction_of_large_error_attributable_to_transitions_100ms": (
            total_large_error_near_transition / total_large_error if total_large_error else None
        ),
    }


def salience_evidence_audit(bundles: list[dict]) -> dict:
    rel_sal = build_relative_salience()
    dpdt_edges = (0.0, 100.0, 400.0, 1000.0, np.inf)
    dpdt_names = ("0-100c/s", "100-400c/s", "400-1000c/s", ">1000c/s")

    rows_type = {t: {"gt_rank": [], "sal_at_gt": [], "top1_dist_cents": [], "top1_2_margin": [], "entropy": [],
                      "cov25": [], "cov50": [], "cov100": []} for t in range(4)}
    rows_dpdt = {name: {"gt_rank": [], "top1_dist_cents": []} for name in dpdt_names}

    rng = np.random.default_rng(0)
    for b in bundles:
        sal = rel_sal[b["recording_id"]]
        n = min(sal.shape[1], b["n"])
        valid = b["valid"][:n]
        idxs = np.flatnonzero(valid)
        if len(idxs) > 6000:
            idxs = rng.choice(idxs, size=6000, replace=False)
        gt_bin = bin_from_hz(np.exp2((b["gt_cents"][:n] + 1200.0 * np.log2(b["tonic_hz"])) / 1200.0))
        est_bin_ref = bin_from_hz(np.exp2((b["est_cents"][:n] + 1200.0 * np.log2(b["tonic_hz"])) / 1200.0))
        gt_offset = np.rint(gt_bin - est_bin_ref).astype(np.int64) + HALF_W

        for t in idxs:
            col = sal[:, t]
            gob = int(gt_offset[t])
            tt = int(b["trajectory_type"][t])
            dpdt = abs(b["dp_dt_cents_s"][t]) if np.isfinite(b["dp_dt_cents_s"][t]) else None

            if gob < 0 or gob >= W_BINS:
                rank, sal_at_gt, dist = W_BINS + 1, 0.0, None
            else:
                order = np.argsort(-col)
                rank = int(np.where(order == gob)[0][0]) + 1
                sal_at_gt = float(col[gob])
                dist = 0.0  # by construction the window is 0 at the GT bin itself; use top1 distance instead
            top1_bin = int(np.argmax(col))
            top1_dist = abs(top1_bin - gob) * CENTS_PER_BIN if 0 <= gob < W_BINS else None
            sorted_vals = np.sort(col)[::-1]
            margin = float(sorted_vals[0] - sorted_vals[1])
            p = np.maximum(col, 1e-12); p = p / p.sum()
            entropy = float(-(p * np.log(p)).sum())
            cov25 = float(col[max(0, gob - 2):gob + 3].sum()) if 0 <= gob < W_BINS else 0.0
            cov50 = float(col[max(0, gob - 3):gob + 4].sum()) if 0 <= gob < W_BINS else 0.0
            cov100 = float(col[max(0, gob - 6):gob + 7].sum()) if 0 <= gob < W_BINS else 0.0

            if tt in range(4):
                d = rows_type[tt]
                d["gt_rank"].append(rank); d["sal_at_gt"].append(sal_at_gt)
                if top1_dist is not None:
                    d["top1_dist_cents"].append(top1_dist)
                d["top1_2_margin"].append(margin); d["entropy"].append(entropy)
                d["cov25"].append(cov25); d["cov50"].append(cov50); d["cov100"].append(cov100)
            if dpdt is not None:
                for name, lo, hi in zip(dpdt_names, dpdt_edges[:-1], dpdt_edges[1:]):
                    if lo <= dpdt < hi:
                        rows_dpdt[name]["gt_rank"].append(rank)
                        if top1_dist is not None:
                            rows_dpdt[name]["top1_dist_cents"].append(top1_dist)
                        break

    def summ(lst):
        a = np.array([x for x in lst if x is not None])
        if len(a) == 0:
            return {"n": 0}
        return {"n": len(a), "mean": float(np.mean(a)), "median": float(np.median(a))}

    by_type = {}
    for t in range(4):
        d = rows_type[t]
        by_type[TYPE_NAMES[t]] = {
            "gt_rank": summ(d["gt_rank"]), "salience_at_gt": summ(d["sal_at_gt"]),
            "top1_distance_from_gt_cents": summ(d["top1_dist_cents"]),
            "top1_top2_margin": summ(d["top1_2_margin"]), "entropy": summ(d["entropy"]),
            "mean_coverage_25c": float(np.mean(d["cov25"])) if d["cov25"] else None,
            "mean_coverage_50c": float(np.mean(d["cov50"])) if d["cov50"] else None,
            "mean_coverage_100c": float(np.mean(d["cov100"])) if d["cov100"] else None,
            "n": len(d["gt_rank"]),
        }
    by_dpdt = {name: {"gt_rank": summ(d["gt_rank"]), "top1_distance_from_gt_cents": summ(d["top1_dist_cents"])}
               for name, d in rows_dpdt.items()}
    return {"by_type": by_type, "by_dpdt_bucket": by_dpdt}


def harmonic_drone_audit(bundles: list[dict]) -> dict:
    counts_by_type = {t: {"n_large_error": 0, **{k: 0 for k in HARMONIC_TARGETS}, "tonic": 0, "other": 0} for t in range(4)}
    for b in bundles:
        valid = b["valid"]
        err = b["est_cents"] - b["gt_cents"]
        large = valid & np.isfinite(err) & (np.abs(err) > LARGE_ABS_ERROR_CENTS)
        idxs = np.flatnonzero(large)
        est_abs_cents = b["est_cents"]  # tonic-relative; "near tonic" means est_abs_cents near 0 mod 1200
        for i in idxs:
            tt = int(b["trajectory_type"][i])
            if tt not in range(4):
                continue
            counts_by_type[tt]["n_large_error"] += 1
            e = err[i]
            matched = False
            for name, target in HARMONIC_TARGETS.items():
                if abs(e - target) < HARMONIC_TOL_CENTS:
                    counts_by_type[tt][name] += 1
                    matched = True
                    break
            if not matched:
                tonic_mod = ((est_abs_cents[i] + 600.0) % 1200.0) - 600.0
                if abs(tonic_mod) < HARMONIC_TOL_CENTS:
                    counts_by_type[tt]["tonic"] += 1
                else:
                    counts_by_type[tt]["other"] += 1

    out = {}
    for t, d in counts_by_type.items():
        n = max(d["n_large_error"], 1)
        out[TYPE_NAMES[t]] = {"n_large_error": d["n_large_error"],
                               "fractions": {k: d[k] / n for k in list(HARMONIC_TARGETS) + ["tonic", "other"]}}
    return out


def main() -> None:
    bundles = build_bundles()
    out = {
        "octave_transition_contribution": octave_transition_contribution(bundles),
        "salience_evidence": salience_evidence_audit(bundles),
        "harmonic_drone": harmonic_drone_audit(bundles),
    }
    write_json(AUDIT_DIR / "salience_harmonics_audit.json", out)

    print("=== octave transition contribution ===")
    print(out["octave_transition_contribution"])
    print("\n=== salience evidence by type ===")
    for k, v in out["salience_evidence"]["by_type"].items():
        print(k, v)
    print("\n=== harmonic/drone by type ===")
    for k, v in out["harmonic_drone"].items():
        print(k, v)


if __name__ == "__main__":
    main()
