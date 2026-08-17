"""Step 17: run Step 16's exact diagnostics on D0 (framewise-independent
argmax, no Viterbi) and D1 (Step 13's frozen Fused+D3 Viterbi decode) side
by side, from the IDENTICAL frozen per-fold fused salience -- isolating
temporal decoding as the only difference. Plus the new candidate-switch
analysis (section 11) and the central D0/D1/D2 comparison table (section 20).
"""

from __future__ import annotations

import numpy as np

from training.metrics import TYPE_NAMES
from training.pitch_diagnostics.common import write_json
from training.pitch_diagnostics.pitch_audit.common import (
    AUDIT_DIR, NATIVE_HOP_S, build_bundles, delta_at_offset,
)
from training.pitch_diagnostics.pitch_audit.motion import (
    absolute_error, attenuation_ratio, multiscale_delta_error, velocity_fidelity,
)
from training.pitch_diagnostics.pitch_audit.phase_and_recording import (
    outlier_recording_investigation, primitive_phase_localization,
)
from training.pitch_diagnostics.pitch_audit.salience_and_harmonics import (
    harmonic_drone_audit, octave_transition_contribution,
)
from training.pitch_diagnostics.pitch_audit.shape import (
    jitter_audit, shape_confusion, staircase_runs, turning_points,
)

CENTS_PER_BIN = 16.6666667


def candidate_switch_analysis(bundles: list[dict]) -> dict:
    """D0-specific: how often does the framewise argmax candidate change
    between consecutive frames, and does the switch move the estimate
    toward or away from GT?"""
    out_by_type = {t: {"n_pairs": 0, "n_switch": 0, "n_toward": 0, "n_away": 0, "jump_cents": []} for t in range(4)}
    for b in bundles:
        valid = b["valid"]
        est = b["est_cents"]
        gt = b["gt_cents"]
        n = len(est)
        switched = np.zeros(n, dtype=bool)
        switched[1:] = np.abs(est[1:] - est[:-1]) > (CENTS_PER_BIN / 2)
        both_valid = np.zeros(n, dtype=bool)
        both_valid[1:] = valid[1:] & valid[:-1]
        idx = np.flatnonzero(both_valid)
        for t in idx:
            tt = int(b["trajectory_type"][t])
            if tt not in range(4):
                continue
            d = out_by_type[tt]
            d["n_pairs"] += 1
            if switched[t]:
                d["n_switch"] += 1
                jump = est[t] - est[t - 1]
                d["jump_cents"].append(float(jump))
                dist_before = abs(est[t - 1] - gt[t]) if np.isfinite(gt[t]) else None
                dist_after = abs(est[t] - gt[t]) if np.isfinite(gt[t]) else None
                if dist_before is not None:
                    if dist_after < dist_before:
                        d["n_toward"] += 1
                    elif dist_after > dist_before:
                        d["n_away"] += 1
    summary = {}
    for t, d in out_by_type.items():
        n_switch = max(d["n_switch"], 1)
        jumps = np.array(d["jump_cents"]) if d["jump_cents"] else np.array([0.0])
        summary[TYPE_NAMES[t]] = {
            "n_pairs": d["n_pairs"], "n_switch": d["n_switch"],
            "switch_rate": d["n_switch"] / max(d["n_pairs"], 1),
            "frac_toward_gt": d["n_toward"] / n_switch, "frac_away_from_gt": d["n_away"] / n_switch,
            "jump_median_abs_cents": float(np.median(np.abs(jumps))), "jump_p90_abs_cents": float(np.percentile(np.abs(jumps), 90)),
        }
    return summary


def run_all(variant: str) -> dict:
    bundles = build_bundles(variant=variant)
    return {
        "absolute_error": absolute_error(bundles),
        "multiscale_delta_error": multiscale_delta_error(bundles),
        "attenuation_ratio": attenuation_ratio(bundles),
        "velocity_fidelity": velocity_fidelity(bundles),
        "turning_points": turning_points(bundles),
        "shape_confusion": shape_confusion(bundles),
        "staircase_runs": staircase_runs(bundles),
        "jitter": jitter_audit(bundles),
        "octave_transition_contribution": octave_transition_contribution(bundles),
        "harmonic_drone": harmonic_drone_audit(bundles),
        "primitive_phase_localization": primitive_phase_localization(bundles),
        "outlier_recording": outlier_recording_investigation(bundles),
        "candidate_switch": candidate_switch_analysis(bundles) if variant == "D0" else None,
    }


def central_table(d0: dict, d1: dict) -> dict:
    def gtmoving_R(res, k_key):
        return res["attenuation_ratio"]["by_offset"][k_key]["gt_moving_only"].get("R")

    def zero_delta_fast(res):
        return res["velocity_fidelity"]["overall"].get("frac_est_delta_exactly_zero_when_gt_fast")

    def turn_recall_50(res):
        vals = [res["turning_points"]["by_type"][t]["recall_by_tolerance"]["50ms"]
                for t in ("T1", "T2", "T3") if t in res["turning_points"]["by_type"]]
        return float(np.mean(vals)) if vals else None

    def shape_recall(res, cls_idx):
        m = res["shape_confusion"]["overall"]["row_normalized"]
        return m[cls_idx][cls_idx]

    def boundary_mae(res, key):
        return res["primitive_phase_localization"]["by_boundary_window"][key]["mae"]

    def moving_run(res):
        vals = [res["staircase_runs"]["est_by_type"][t]["median_frames"]
                for t in ("T1", "T2", "T3") if res["staircase_runs"]["est_by_type"][t].get("n", 0) > 0]
        return float(np.mean(vals)) if vals else None

    return {
        "absolute_pitch_mae": {"D0": d0["absolute_error"]["overall"]["mae"], "D1": d1["absolute_error"]["overall"]["mae"]},
        "delta50_mae": {"D0": d0["multiscale_delta_error"]["by_offset"]["50ms"]["mae"], "D1": d1["multiscale_delta_error"]["by_offset"]["50ms"]["mae"]},
        "R50": {"D0": gtmoving_R(d0, "50ms"), "D1": gtmoving_R(d1, "50ms")},
        "R100": {"D0": gtmoving_R(d0, "100ms"), "D1": gtmoving_R(d1, "100ms")},
        "zero_delta_during_gt_motion": {"D0": zero_delta_fast(d0), "D1": zero_delta_fast(d1)},
        "velocity_correlation": {"D0": d0["velocity_fidelity"]["overall"].get("correlation"), "D1": d1["velocity_fidelity"]["overall"].get("correlation")},
        "turning_recall_50ms_T1T2T3_mean": {"D0": turn_recall_50(d0), "D1": turn_recall_50(d1)},
        "rise_fall_recall": {"D0": shape_recall(d0, 3), "D1": shape_recall(d1, 3)},
        "fall_rise_recall": {"D0": shape_recall(d0, 4), "D1": shape_recall(d1, 4)},
        "boundary_50ms_mae": {"D0": boundary_mae(d0, "50ms"), "D1": boundary_mae(d1, "50ms")},
        "moving_region_run_length_frames": {"D0": moving_run(d0), "D1": moving_run(d1)},
    }


def main() -> None:
    print("=== running D0 (framewise argmax) diagnostics ===")
    d0 = run_all("D0")
    print("=== running D1 (Viterbi) diagnostics ===")
    d1 = run_all("D1")
    table = central_table(d0, d1)

    out = {"D0": d0, "D1": d1, "central_table": table}
    write_json(AUDIT_DIR / "decoder_ablation_D0_vs_D1.json", out)

    print("\n=== central comparison table (D0 vs D1) ===")
    for metric, vals in table.items():
        print(f"{metric:35s} D0={vals['D0']!r:>10}  D1={vals['D1']!r:>10}")

    print("\n=== candidate-switch analysis (D0) ===")
    for t, v in d0["candidate_switch"].items():
        print(t, v)


if __name__ == "__main__":
    main()
