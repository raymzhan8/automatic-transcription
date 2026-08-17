"""Step 16 sections 16-17: motion error by normalized primitive phase /
boundary proximity, and a targeted per-recording investigation of
692ed7e6... (Step 14/15's persistent oracle-pitch outlier).
"""

from __future__ import annotations

import numpy as np

from training.pitch_diagnostics.common import write_json
from training.pitch_diagnostics.pitch_audit.common import (
    AUDIT_DIR, NATIVE_HOP_S, OUTLIER_RECORDING, both_valid_mask, build_bundles, delta_at_offset,
)

PHASE_BINS = ("beginning", "early_middle", "middle", "late_middle", "end")
BOUNDARY_WINDOWS_MS = (50, 100)


def primitive_phase_localization(bundles: list[dict]) -> dict:
    phase_err = {p: [] for p in PHASE_BINS}
    boundary_err = {w: [] for w in BOUNDARY_WINDOWS_MS}
    non_boundary_err = []

    for b in bundles:
        d_est = delta_at_offset(b["est_cents"], 5)  # 50ms scale
        d_gt = delta_at_offset(b["gt_cents"], 5)
        err = np.abs(d_est - d_gt)
        m = both_valid_mask(b["valid"], 5) & np.isfinite(err)
        times = b["times"]

        near_boundary = {w: np.zeros(len(times), dtype=bool) for w in BOUNDARY_WINDOWS_MS}
        for prim in b["primitives"]:
            start_s, end_s = prim["start_s"], prim["end_s"]
            dur = end_s - start_s
            if dur <= 0:
                continue
            in_prim = m & (times >= start_s) & (times < end_s)
            phase = np.clip((times[in_prim] - start_s) / dur, 0.0, 0.999999)
            bins = np.minimum((phase * 5).astype(int), 4)
            e_in = err[in_prim]
            for i, name in enumerate(PHASE_BINS):
                phase_err[name].extend(e_in[bins == i].tolist())
            for w_ms in BOUNDARY_WINDOWS_MS:
                w_s = w_ms / 1000.0
                near_boundary[w_ms] |= m & (np.abs(times - start_s) <= w_s)
                near_boundary[w_ms] |= m & (np.abs(times - end_s) <= w_s)

        for w_ms in BOUNDARY_WINDOWS_MS:
            boundary_err[w_ms].extend(err[near_boundary[w_ms] & m].tolist())
        far = m & ~near_boundary[max(BOUNDARY_WINDOWS_MS)]
        non_boundary_err.extend(err[far].tolist())

    def summ(lst):
        a = np.array(lst)
        if len(a) == 0:
            return {"n": 0}
        return {"n": len(a), "mae": float(np.mean(a)), "median_ae": float(np.median(a))}

    return {
        "by_phase": {p: summ(v) for p, v in phase_err.items()},
        "by_boundary_window": {f"{w}ms": summ(v) for w, v in boundary_err.items()},
        "non_boundary": summ(non_boundary_err),
        "delta_scale_ms": 50,
    }


def outlier_recording_investigation(bundles: list[dict]) -> dict:
    b = next((x for x in bundles if x["recording_id"] == OUTLIER_RECORDING), None)
    if b is None:
        return {"error": "recording not found"}

    valid = b["valid"]
    type_counts = {int(t): int((b["trajectory_type"][valid] == t).sum()) for t in range(4)}
    total = max(int(valid.sum()), 1)
    type_fractions = {t: c / total for t, c in type_counts.items()}

    abs_err = np.abs(b["est_cents"][valid] - b["gt_cents"][valid])
    d_est = delta_at_offset(b["est_cents"], 5); d_gt = delta_at_offset(b["gt_cents"], 5)
    m = both_valid_mask(valid, 5) & np.isfinite(d_est) & np.isfinite(d_gt)
    motion_err_50ms = np.abs(d_est - d_gt)[m]

    gt_vel = b["dp_dt_cents_s"]
    dpdt_valid = valid & np.isfinite(gt_vel)
    median_abs_dpdt = float(np.median(np.abs(gt_vel[dpdt_valid]))) if dpdt_valid.any() else None

    durations = [p["end_s"] - p["start_s"] for p in b["primitives"]]

    return {
        "recording_id": OUTLIER_RECORDING, "fold": b["fold"], "n_valid": int(valid.sum()),
        "duration_s": b["duration_s"], "n_primitives": len(b["primitives"]),
        "type_fractions": {f"T{t}": v for t, v in type_fractions.items()},
        "absolute_pitch_error_mae": float(np.mean(abs_err)), "absolute_pitch_error_median": float(np.median(abs_err)),
        "motion_error_50ms_mae": float(np.mean(motion_err_50ms)) if len(motion_err_50ms) else None,
        "median_abs_gt_dpdt_cents_s": median_abs_dpdt,
        "median_primitive_duration_s": float(np.median(durations)) if durations else None,
        "mean_primitive_duration_s": float(np.mean(durations)) if durations else None,
    }


def main() -> None:
    bundles = build_bundles()
    out = {
        "primitive_phase_localization": primitive_phase_localization(bundles),
        "outlier_recording": outlier_recording_investigation(bundles),
    }
    write_json(AUDIT_DIR / "phase_and_recording_audit.json", out)
    print("=== phase localization (50ms-scale motion error) ===")
    for p, v in out["primitive_phase_localization"]["by_phase"].items():
        print(p, v)
    print("boundary windows:", out["primitive_phase_localization"]["by_boundary_window"])
    print("non-boundary:", out["primitive_phase_localization"]["non_boundary"])
    print("\n=== outlier recording (692ed7e6...) ===")
    print(out["outlier_recording"])


if __name__ == "__main__":
    main()
