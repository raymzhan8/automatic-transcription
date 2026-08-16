"""Step 13 sections 2-3, 5: octave-invariant relative-pitch-movement metrics.

For each frontend (HPS argmax, HPS+D3, learned argmax, learned+D3,
fused+D3) and each consecutive VALID-frame pair within a recording, compare
the frontend's frame-to-frame delta/velocity against the oracle
(GT-annotation-derived) delta/velocity for the same pair. Also isolates
pairs by each frontend's own octave-correctness (spec section 2's core
question) and breaks results down by trajectory type / |dp/dt| bucket
(section 3) and for the two Step 11/12 large-failure recordings (section 5).

Deadband for the flat/rising/falling direction class: 100 cents/s, reusing
Step 12.5's own first |dp/dt| bucket boundary (0-100 c/s) rather than
inventing a new threshold (spec section 2 instruction).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.metrics import TYPE_NAMES  # noqa: E402
from training.pitch_diagnostics.common import octave_adjusted_error, write_json  # noqa: E402
from training.pitch_diagnostics.relative_pitch.path_cache import REL_DIR, build  # noqa: E402
from training.pitch_diagnostics.relative_pitch.windows import eligible_centers, relative_contour  # noqa: E402

METHODS = ("hps_argmax", "hps_d3", "learned_argmax", "learned_d3", "fused_d3")
DIRECTION_DEADBAND_CENTS_S = 100.0
DPDT_BUCKET_EDGES = (0.0, 100.0, 400.0, 1000.0, np.inf)
DPDT_BUCKET_NAMES = ("0-100c/s", "100-400c/s", "400-1000c/s", ">1000c/s")
LARGE_FAILURE_RECORDINGS = ["6417585554a0bfbd8de2d3ff", "6824de49abc4705438ce918b"]


def _direction(vel: np.ndarray) -> np.ndarray:
    d = np.zeros(len(vel), dtype=np.int8)  # 0=flat, 1=rising, -1=falling
    d[vel > DIRECTION_DEADBAND_CENTS_S] = 1
    d[vel < -DIRECTION_DEADBAND_CENTS_S] = -1
    return d


def _pair_arrays(record: dict, method: str) -> dict:
    """Per-recording consecutive-pair arrays: gt/method delta & velocity,
    method octave_k at the landing frame (t) and at t-1, trajectory_type[t],
    dp_dt_cents_s[t]."""
    path = record[f"{method}_cents"]
    true = record["true_cents"]
    dt_s = record["dt_seconds"]
    n = len(true)
    if n < 2:
        return None
    gt_delta = true[1:] - true[:-1]
    m_delta = path[1:] - path[:-1]
    dts = dt_s[1:]
    gt_vel = gt_delta / dts
    m_vel = m_delta / dts
    oct_err, oct_k = octave_adjusted_error(path, true)
    return {
        "gt_delta": gt_delta, "m_delta": m_delta, "gt_vel": gt_vel, "m_vel": m_vel,
        "delta_err": np.abs(m_delta - gt_delta), "vel_err": np.abs(m_vel - gt_vel),
        "gt_dir": _direction(gt_vel), "m_dir": _direction(m_vel),
        "k_t": oct_k[1:], "k_tm1": oct_k[:-1],
        "trajectory_type": record["trajectory_type"][1:],
        "dp_dt_cents_s": record["dp_dt_cents_s"][1:],
        "recording_id": record["recording_id"], "fold": record["fold"],
    }


def _summ(delta_err, vel_err, gt_dir, m_dir) -> dict:
    n = len(delta_err)
    if n == 0:
        return {"n": 0}
    return {
        "n": int(n),
        "delta_mae_cents": float(np.mean(delta_err)),
        "delta_median_ae_cents": float(np.median(delta_err)),
        "velocity_mae_cents_s": float(np.mean(vel_err)),
        "velocity_median_ae_cents_s": float(np.median(vel_err)),
        "direction_accuracy": float(np.mean(gt_dir == m_dir)),
        "direction_accuracy_by_class": {
            cls_name: float(np.mean(m_dir[gt_dir == cls] == cls)) if np.any(gt_dir == cls) else None
            for cls, cls_name in ((1, "rising"), (-1, "falling"), (0, "flat"))
        },
    }


def main() -> None:
    records = build()

    pooled = {m: {"delta_err": [], "vel_err": [], "gt_dir": [], "m_dir": []} for m in METHODS}
    octave_cond = {m: {"both_correct": {"delta_err": [], "vel_err": [], "gt_dir": [], "m_dir": []},
                        "both_wrong_same_k": {"delta_err": [], "vel_err": [], "gt_dir": [], "m_dir": []},
                        "transition": {"delta_err": [], "vel_err": [], "gt_dir": [], "m_dir": []}}
                   for m in METHODS}
    by_type = {m: {t: {"delta_err": [], "vel_err": [], "gt_dir": [], "m_dir": []} for t in range(4)} for m in METHODS}
    by_dpdt = {m: {b: {"delta_err": [], "vel_err": [], "gt_dir": [], "m_dir": []} for b in DPDT_BUCKET_NAMES} for m in METHODS}
    per_recording = {}
    absolute_vs_relative = {}

    for record in records:
        rid = record["recording_id"]
        per_recording[rid] = {"fold": record["fold"], "n_valid": record["n_valid"]}
        for m in METHODS:
            p = _pair_arrays(record, m)
            if p is None:
                continue
            pooled[m]["delta_err"].append(p["delta_err"]); pooled[m]["vel_err"].append(p["vel_err"])
            pooled[m]["gt_dir"].append(p["gt_dir"]); pooled[m]["m_dir"].append(p["m_dir"])

            same_k = p["k_t"] == p["k_tm1"]
            both_correct = same_k & (p["k_t"] == 0)
            both_wrong_same = same_k & (p["k_t"] != 0)
            transition = ~same_k
            for cat_name, mask in (("both_correct", both_correct), ("both_wrong_same_k", both_wrong_same), ("transition", transition)):
                d = octave_cond[m][cat_name]
                d["delta_err"].append(p["delta_err"][mask]); d["vel_err"].append(p["vel_err"][mask])
                d["gt_dir"].append(p["gt_dir"][mask]); d["m_dir"].append(p["m_dir"][mask])

            for t in range(4):
                mask = p["trajectory_type"] == t
                d = by_type[m][t]
                d["delta_err"].append(p["delta_err"][mask]); d["vel_err"].append(p["vel_err"][mask])
                d["gt_dir"].append(p["gt_dir"][mask]); d["m_dir"].append(p["m_dir"][mask])

            abs_dpdt = np.abs(p["dp_dt_cents_s"])
            for name, lo_e, hi_e in zip(DPDT_BUCKET_NAMES, DPDT_BUCKET_EDGES[:-1], DPDT_BUCKET_EDGES[1:]):
                mask = (abs_dpdt >= lo_e) & (abs_dpdt < hi_e)
                d = by_dpdt[m][name]
                d["delta_err"].append(p["delta_err"][mask]); d["vel_err"].append(p["vel_err"][mask])
                d["gt_dir"].append(p["gt_dir"][mask]); d["m_dir"].append(p["m_dir"][mask])

            per_recording[rid][m] = {
                "abs_mae_cents": float(np.mean(np.abs(record[f"{m}_cents"] - record["true_cents"]))),
                **_summ(p["delta_err"], p["vel_err"], p["gt_dir"], p["m_dir"]),
            }

        # ---- contour MAE (windowed, spec section 1C) ----
        centers = eligible_centers(record)
        if len(centers) > 0:
            gt_path = record["true_cents"]
            for m in METHODS:
                m_path = record[f"{m}_cents"]
                errs = [np.mean(np.abs(relative_contour(m_path, c) - relative_contour(gt_path, c))) for c in centers]
                per_recording[rid].setdefault(m, {})["contour_mae_cents"] = float(np.mean(errs))

    def pool(d):
        return _summ(np.concatenate(d["delta_err"]) if d["delta_err"] else np.array([]),
                     np.concatenate(d["vel_err"]) if d["vel_err"] else np.array([]),
                     np.concatenate(d["gt_dir"]) if d["gt_dir"] else np.array([]),
                     np.concatenate(d["m_dir"]) if d["m_dir"] else np.array([]))

    pooled_summary = {m: pool(pooled[m]) for m in METHODS}
    octave_cond_summary = {m: {cat: pool(octave_cond[m][cat]) for cat in octave_cond[m]} for m in METHODS}
    by_type_summary = {m: {TYPE_NAMES[t]: pool(by_type[m][t]) for t in range(4)} for m in METHODS}
    by_dpdt_summary = {m: {b: pool(by_dpdt[m][b]) for b in DPDT_BUCKET_NAMES} for m in METHODS}

    large_failure = {rid: per_recording[rid] for rid in LARGE_FAILURE_RECORDINGS if rid in per_recording}

    out = {
        "direction_deadband_cents_s": DIRECTION_DEADBAND_CENTS_S,
        "dpdt_bucket_edges_cents_s": list(DPDT_BUCKET_EDGES),
        "window_frames": 21,
        "pooled": pooled_summary,
        "octave_conditional": octave_cond_summary,
        "by_type": by_type_summary,
        "by_dpdt_bucket": by_dpdt_summary,
        "per_recording": per_recording,
        "large_failure_recordings": large_failure,
    }
    write_json(REL_DIR / "signals_result.json", out)

    print("=== Step 13 pooled relative-motion summary ===")
    for m, v in pooled_summary.items():
        print(f"{m:16s} delta_mae={v['delta_mae_cents']:6.1f}c  vel_mae={v['velocity_mae_cents_s']:7.1f}c/s  dir_acc={v['direction_accuracy']*100:5.1f}%")
    print("\n=== octave-conditional delta MAE (both_correct vs both_wrong_same_k vs transition) ===")
    for m in METHODS:
        c = octave_cond_summary[m]
        print(f"{m:16s}", {k: (round(v['delta_mae_cents'], 1) if v['n'] > 0 else None) for k, v in c.items()})
    print("\n=== large-failure recordings: absolute vs relative (hps_d3) ===")
    for rid, rec in large_failure.items():
        if "hps_d3" in rec:
            v = rec["hps_d3"]
            print(rid, "abs_mae", round(v["abs_mae_cents"], 1), "delta_mae", round(v["delta_mae_cents"], 1),
                  "vel_mae", round(v["velocity_mae_cents_s"], 1), "dir_acc", round(v["direction_accuracy"] * 100, 1))


if __name__ == "__main__":
    main()
