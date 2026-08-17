"""Step 16 sections 2-6: absolute vs relative-motion error, multi-timescale
delta-error, smoothing/attenuation ratio, temporal-lag analysis, and
velocity/slope fidelity -- estimated (Fused+D3) vs GT parametric pitch, all
by trajectory type.
"""

from __future__ import annotations

import numpy as np

from training.metrics import TYPE_NAMES
from training.pitch_diagnostics.pitch_audit.common import (
    AUDIT_DIR, NATIVE_HOP_S, both_valid_mask, build_bundles, delta_at_offset,
)

OFFSETS = (1, 2, 5, 10, 20)  # 10, 20, 50, 100, 200 ms
OFFSET_MS = {1: 10, 2: 20, 5: 50, 10: 100, 20: 200}
DIRECTION_DEADBAND_CENTS_S = 100.0
DPDT_BUCKET_EDGES = (0.0, 100.0, 400.0, 1000.0, np.inf)
DPDT_BUCKET_NAMES = ("0-100c/s", "100-400c/s", "400-1000c/s", ">1000c/s")
LAG_RANGE_FRAMES = range(-10, 11)  # -100ms..+100ms in 10ms steps


def _direction(vel: np.ndarray) -> np.ndarray:
    d = np.zeros(len(vel), dtype=np.int8)
    d[vel > DIRECTION_DEADBAND_CENTS_S] = 1
    d[vel < -DIRECTION_DEADBAND_CENTS_S] = -1
    return d


def _summ(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    return {
        "n": int(len(x)), "mae": float(np.mean(np.abs(x))), "median_ae": float(np.median(np.abs(x))),
        "p25_ae": float(np.percentile(np.abs(x), 25)), "p75_ae": float(np.percentile(np.abs(x), 75)),
        "p95_ae": float(np.percentile(np.abs(x), 95)),
    }


def absolute_error(bundles: list[dict]) -> dict:
    all_e, all_type = [], []
    per_rec = {}
    for b in bundles:
        mask = b["valid"]
        e = b["est_cents"][mask] - b["gt_cents"][mask]
        all_e.append(e); all_type.append(b["trajectory_type"][mask])
        per_rec[b["recording_id"]] = _summ(e)
    e = np.concatenate(all_e); t = np.concatenate(all_type)
    out = {"overall": _summ(e), "by_type": {}, "per_recording": per_rec}
    for tt in range(4):
        out["by_type"][TYPE_NAMES[tt]] = _summ(e[t == tt])
    return out


def multiscale_delta_error(bundles: list[dict]) -> dict:
    out = {"by_offset": {}, "by_offset_by_type": {}}
    for k in OFFSETS:
        errs, types, dir_agree_num, dir_agree_den = [], [], 0, 0
        for b in bundles:
            d_est = delta_at_offset(b["est_cents"], k)
            d_gt = delta_at_offset(b["gt_cents"], k)
            m = both_valid_mask(b["valid"], k) & np.isfinite(d_est) & np.isfinite(d_gt)
            errs.append((d_est - d_gt)[m])
            types.append(b["trajectory_type"][m])
            vel_est = d_est[m] / (k * NATIVE_HOP_S)
            vel_gt = d_gt[m] / (k * NATIVE_HOP_S)
            dir_agree_num += int(np.sum(_direction(vel_est) == _direction(vel_gt)))
            dir_agree_den += int(m.sum())
        e = np.concatenate(errs); t = np.concatenate(types)
        key = f"{OFFSET_MS[k]}ms"
        out["by_offset"][key] = {**_summ(e), "direction_agreement": dir_agree_num / max(dir_agree_den, 1)}
        out["by_offset_by_type"][key] = {}
        for tt in range(4):
            out["by_offset_by_type"][key][TYPE_NAMES[tt]] = _summ(e[t == tt])
    return out


def attenuation_ratio(bundles: list[dict]) -> dict:
    """R_k = median(|est|)/median(|gt|). Reported pooled (all frames) AND
    restricted to GT-moving frames (|delta_gt| > 1e-6) -- pooled R is
    degenerate (median GT delta is exactly 0 at short offsets, since most
    native-hop frame pairs show no annotated movement at all, matching
    Steps 13-15's own finding), so the GT-moving-only version is the
    "statistically meaningful" one spec section 4 asks for."""
    out = {"by_offset": {}, "by_offset_by_type": {}}
    for k in OFFSETS:
        abs_est, abs_gt, types = [], [], []
        for b in bundles:
            d_est = delta_at_offset(b["est_cents"], k)
            d_gt = delta_at_offset(b["gt_cents"], k)
            m = both_valid_mask(b["valid"], k) & np.isfinite(d_est) & np.isfinite(d_gt)
            abs_est.append(np.abs(d_est[m])); abs_gt.append(np.abs(d_gt[m])); types.append(b["trajectory_type"][m])
        ae = np.concatenate(abs_est); ag = np.concatenate(abs_gt); t = np.concatenate(types)
        key = f"{OFFSET_MS[k]}ms"
        moving = ag > 1e-6

        def _r(mask):
            n = int(mask.sum())
            if n == 0:
                return {"n": 0}
            mg = float(np.median(ag[mask]))
            return {
                "n": n, "median_abs_est": float(np.median(ae[mask])), "median_abs_gt": mg,
                "R": float(np.median(ae[mask]) / mg) if mg > 0 else None,
                "frac_est_exactly_zero": float(np.mean(ae[mask] < 1e-6)),
            }

        out["by_offset"][key] = {"pooled": _r(np.ones(len(ag), dtype=bool)), "gt_moving_only": _r(moving)}
        out["by_offset_by_type"][key] = {}
        for tt in range(4):
            mm = t == tt
            out["by_offset_by_type"][key][TYPE_NAMES[tt]] = {
                "pooled": _r(mm), "gt_moving_only": _r(mm & moving),
            }
    return out


def temporal_lag(bundles: list[dict], align_k: int = 2) -> dict:
    """Cross-correlation-style lag search on the k=2 (20ms) octave-unwrapped
    delta series (a middle ground between raw 10ms noise and over-smoothed
    longer offsets), per recording per type. Diagnostic only."""
    per_recording: dict[str, dict] = {}
    agg_by_type: dict[str, list] = {TYPE_NAMES[t]: [] for t in range(4)}
    for b in bundles:
        d_est_full = delta_at_offset(b["est_cents"], align_k)
        d_gt_full = delta_at_offset(b["gt_cents"], align_k)
        rec_out = {}
        for tt in range(4):
            type_mask = b["trajectory_type"] == tt
            lag_mae = {}
            for lag in LAG_RANGE_FRAMES:
                if lag >= 0:
                    est_shift = d_est_full[lag:]
                    gt_ref = d_gt_full[: len(d_gt_full) - lag] if lag > 0 else d_gt_full
                    mask_shift = (both_valid_mask(b["valid"], align_k) & type_mask)[lag:]
                    mask_ref = (both_valid_mask(b["valid"], align_k) & type_mask)[: len(d_gt_full) - lag] if lag > 0 else (both_valid_mask(b["valid"], align_k) & type_mask)
                else:
                    L = -lag
                    est_shift = d_est_full[: len(d_est_full) - L]
                    gt_ref = d_gt_full[L:]
                    mask_shift = (both_valid_mask(b["valid"], align_k) & type_mask)[: len(d_est_full) - L]
                    mask_ref = (both_valid_mask(b["valid"], align_k) & type_mask)[L:]
                m = mask_shift & mask_ref & np.isfinite(est_shift) & np.isfinite(gt_ref)
                if m.sum() < 30:
                    continue
                lag_mae[lag] = float(np.mean(np.abs(est_shift[m] - gt_ref[m])))
            if not lag_mae:
                continue
            best_lag = min(lag_mae, key=lag_mae.get)
            entry = {
                "best_lag_ms": best_lag * 10, "mae_at_lag0": lag_mae.get(0),
                "mae_at_best_lag": lag_mae[best_lag], "n_lags_evaluated": len(lag_mae),
            }
            rec_out[TYPE_NAMES[tt]] = entry
            agg_by_type[TYPE_NAMES[tt]].append(entry)
        per_recording[b["recording_id"]] = rec_out

    by_type_summary = {}
    for name, entries in agg_by_type.items():
        if not entries:
            continue
        best_lags = [e["best_lag_ms"] for e in entries]
        mae0 = [e["mae_at_lag0"] for e in entries if e["mae_at_lag0"] is not None]
        maeb = [e["mae_at_best_lag"] for e in entries]
        by_type_summary[name] = {
            "n_recordings": len(entries), "median_best_lag_ms": float(np.median(best_lags)),
            "mean_mae_at_lag0": float(np.mean(mae0)) if mae0 else None,
            "mean_mae_at_best_lag": float(np.mean(maeb)),
            "mean_improvement": float(np.mean(mae0) - np.mean(maeb)) if mae0 else None,
        }
    return {"align_offset_ms": OFFSET_MS[align_k], "per_recording": per_recording, "by_type_summary": by_type_summary}


def velocity_fidelity(bundles: list[dict]) -> dict:
    gt_vel_all, est_vel_all, dpdt_all, type_all = [], [], [], []
    for b in bundles:
        m = b["valid"] & np.isfinite(b["dp_dt_cents_s"])
        gt_vel = b["dp_dt_cents_s"][m]
        d_est = delta_at_offset(b["est_cents"], 1)
        est_vel = (d_est / NATIVE_HOP_S)[m]
        both = np.isfinite(est_vel)
        gt_vel_all.append(gt_vel[both]); est_vel_all.append(est_vel[both])
        dpdt_all.append(np.abs(gt_vel[both])); type_all.append(b["trajectory_type"][m][both])

    gt_vel = np.concatenate(gt_vel_all); est_vel = np.concatenate(est_vel_all)
    dpdt = np.concatenate(dpdt_all); ttype = np.concatenate(type_all)

    def block(mask):
        if mask.sum() < 2:
            return {"n": int(mask.sum())}
        g, e = gt_vel[mask], est_vel[mask]
        corr = float(np.corrcoef(g, e)[0, 1]) if np.std(g) > 0 and np.std(e) > 0 else None
        sign_agree = float(np.mean(_direction(e) == _direction(g)))
        med_g = float(np.median(np.abs(g)))
        fast_gt = np.abs(g) > 100.0
        return {
            "n": int(mask.sum()), "correlation": corr, "mae_cents_s": float(np.mean(np.abs(e - g))),
            "sign_agreement": sign_agree, "median_abs_gt_vel": med_g,
            "median_abs_est_vel": float(np.median(np.abs(e))),
            "magnitude_ratio": float(np.median(np.abs(e)) / med_g) if med_g > 0 else None,
            "frac_est_delta_exactly_zero": float(np.mean(np.abs(e) < 1e-6)),
            "frac_est_delta_exactly_zero_when_gt_fast": float(np.mean(np.abs(e[fast_gt]) < 1e-6)) if fast_gt.any() else None,
            "n_gt_fast": int(fast_gt.sum()),
        }

    out = {"overall": block(np.ones(len(gt_vel), dtype=bool)), "by_type": {}, "by_dpdt_bucket": {}}
    for tt in range(4):
        out["by_type"][TYPE_NAMES[tt]] = block(ttype == tt)
    for name, lo, hi in zip(DPDT_BUCKET_NAMES, DPDT_BUCKET_EDGES[:-1], DPDT_BUCKET_EDGES[1:]):
        out["by_dpdt_bucket"][name] = block((dpdt >= lo) & (dpdt < hi))
    return out


def main() -> None:
    bundles = build_bundles()
    out = {
        "absolute_error": absolute_error(bundles),
        "multiscale_delta_error": multiscale_delta_error(bundles),
        "attenuation_ratio": attenuation_ratio(bundles),
        "temporal_lag": temporal_lag(bundles),
        "velocity_fidelity": velocity_fidelity(bundles),
    }
    from training.pitch_diagnostics.common import write_json
    write_json(AUDIT_DIR / "motion_audit.json", out)

    print("=== absolute error (cents) ===", out["absolute_error"]["overall"])
    print("\n=== delta-error by offset (cents) ===")
    for k, v in out["multiscale_delta_error"]["by_offset"].items():
        print(k, "mae", round(v["mae"], 1), "median", round(v["median_ae"], 1), "dir_agree", round(v["direction_agreement"], 3))
    print("\n=== attenuation ratio R_k by offset (GT-moving-only) ===")
    for k, v in out["attenuation_ratio"]["by_offset"].items():
        gm = v["gt_moving_only"]
        print(k, "R=", round(gm["R"], 3) if gm.get("R") else None, "n=", gm.get("n"),
              "frac_est_exactly_zero=", round(gm.get("frac_est_exactly_zero", 0), 3))
    print("\n=== velocity fidelity overall ===", out["velocity_fidelity"]["overall"])
    print("\n=== velocity fidelity by type ===")
    for tt, v in out["velocity_fidelity"]["by_type"].items():
        print(tt, v)
    print("\n=== temporal lag by type ===")
    for tt, v in out["temporal_lag"]["by_type_summary"].items():
        print(tt, v)


if __name__ == "__main__":
    main()
