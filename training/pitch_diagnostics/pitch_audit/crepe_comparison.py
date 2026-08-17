"""Side experiment (post-Step 20): does a pretrained, generic monophonic
pitch tracker (CREPE, on the vocals stem) beat the current CQT/salience/
Viterbi pipeline outright? Reuses the exact Step 16-17 diagnostics
(training/pitch_diagnostics/pitch_audit/{motion,shape}.py) against D1
(current system, frozen Fused+D3 Viterbi decode) and GT oracle, so results
are directly comparable to every number already reported in Steps 16-19.
No training; CREPE's public pretrained weights only.
"""

from __future__ import annotations

import numpy as np

from training.metrics import TYPE_NAMES
from training.pitch_diagnostics.common import write_json
from training.pitch_diagnostics.pitch_audit.common import AUDIT_DIR, build_bundles
from training.pitch_diagnostics.pitch_audit.motion import (
    absolute_error, attenuation_ratio, multiscale_delta_error, velocity_fidelity,
)
from training.pitch_diagnostics.pitch_audit.shape import shape_confusion, staircase_runs, turning_points


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
    }


def central_table(d1: dict, crepe: dict) -> dict:
    def gtmoving_R(res, k_key):
        return res["attenuation_ratio"]["by_offset"][k_key]["gt_moving_only"].get("R")

    def zero_delta_fast(res):
        return res["velocity_fidelity"]["overall"].get("frac_est_delta_exactly_zero_when_gt_fast")

    def turn_recall_50(res):
        vals = [res["turning_points"]["by_type"][t]["recall_by_tolerance"]["50ms"]
                for t in ("T1", "T2", "T3") if t in res["turning_points"]["by_type"]]
        return float(np.mean(vals)) if vals else None

    def moving_run(res):
        vals = [res["staircase_runs"]["est_by_type"][t]["median_frames"]
                for t in ("T1", "T2", "T3") if res["staircase_runs"]["est_by_type"][t].get("n", 0) > 0]
        return float(np.mean(vals)) if vals else None

    return {
        "absolute_pitch_mae": {"D1": d1["absolute_error"]["overall"]["mae"], "CREPE": crepe["absolute_error"]["overall"]["mae"]},
        "absolute_pitch_mae_by_type": {
            t: {"D1": d1["absolute_error"]["by_type"][t]["mae"], "CREPE": crepe["absolute_error"]["by_type"][t]["mae"]}
            for t in TYPE_NAMES.values()
        },
        "delta50_mae": {"D1": d1["multiscale_delta_error"]["by_offset"]["50ms"]["mae"], "CREPE": crepe["multiscale_delta_error"]["by_offset"]["50ms"]["mae"]},
        "R50": {"D1": gtmoving_R(d1, "50ms"), "CREPE": gtmoving_R(crepe, "50ms")},
        "R100": {"D1": gtmoving_R(d1, "100ms"), "CREPE": gtmoving_R(crepe, "100ms")},
        "R200": {"D1": gtmoving_R(d1, "200ms"), "CREPE": gtmoving_R(crepe, "200ms")},
        "zero_delta_during_gt_motion": {"D1": zero_delta_fast(d1), "CREPE": zero_delta_fast(crepe)},
        "velocity_correlation": {"D1": d1["velocity_fidelity"]["overall"].get("correlation"), "CREPE": crepe["velocity_fidelity"]["overall"].get("correlation")},
        "turning_recall_50ms_T1T2T3_mean": {"D1": turn_recall_50(d1), "CREPE": turn_recall_50(crepe)},
        "moving_region_run_length_frames": {"D1": moving_run(d1), "CREPE": moving_run(crepe)},
    }


def main() -> None:
    print("=== running D1 (current system) diagnostics ===")
    d1 = run_all("D1")
    print("=== running CREPE diagnostics ===")
    crepe = run_all("CREPE")
    table = central_table(d1, crepe)
    write_json(AUDIT_DIR / "crepe_comparison.json", {"central_table": table, "D1": d1, "CREPE": crepe})
    print("\n=== central table ===")
    for k, v in table.items():
        print(k, v)


if __name__ == "__main__":
    main()
