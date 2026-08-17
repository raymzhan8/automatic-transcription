"""Step 17 sections 17-18: diagnostic-only Viterbi movement-cost sweep
(0x=D0, 0.25x, 0.5x, 1.0x=D1) -- core motion-fidelity metrics only, reusing
Step 16/17's existing analysis functions unmodified. Not a hyperparameter
search: lambda_t is scaled by a fixed small multiplier set, not re-tuned.
"""

from __future__ import annotations

import numpy as np

from training.pitch_diagnostics.common import write_json
from training.pitch_diagnostics.pitch_audit.common import AUDIT_DIR, build_bundles
from training.pitch_diagnostics.pitch_audit.motion import absolute_error, attenuation_ratio, multiscale_delta_error
from training.pitch_diagnostics.pitch_audit.phase_and_recording import primitive_phase_localization
from training.pitch_diagnostics.pitch_audit.shape import turning_points

VARIANTS = ("0.25x", "0.5x")  # 0x and 1.0x already computed as D0/D1


def core_metrics(bundles: list[dict]) -> dict:
    ae = absolute_error(bundles)
    mde = multiscale_delta_error(bundles)
    att = attenuation_ratio(bundles)
    tp = turning_points(bundles)
    bnd = primitive_phase_localization(bundles)

    turn_recall_50 = np.mean([
        tp["by_type"][t]["recall_by_tolerance"]["50ms"] for t in ("T1", "T2", "T3") if t in tp["by_type"]
    ])
    return {
        "absolute_mae": ae["overall"]["mae"],
        "R50_gt_moving": att["by_offset"]["50ms"]["gt_moving_only"].get("R"),
        "R100_gt_moving": att["by_offset"]["100ms"]["gt_moving_only"].get("R"),
        "zero_delta_frac_gt_moving_50ms": att["by_offset"]["50ms"]["gt_moving_only"].get("frac_est_exactly_zero"),
        "turning_recall_50ms_mean_T1T2T3": float(turn_recall_50),
        "boundary_50ms_mae": bnd["by_boundary_window"]["50ms"]["mae"],
        "delta50_mae": mde["by_offset"]["50ms"]["mae"],
    }


def main() -> None:
    results = {}
    for v in VARIANTS:
        print(f"=== {v} ===")
        bundles = build_bundles(variant=v)
        results[v] = core_metrics(bundles)
        print(results[v])

    write_json(AUDIT_DIR / "lambda_sweep_diagnostics.json", results)
    print("\n=== summary ===")
    for v, m in results.items():
        print(v, m)


if __name__ == "__main__":
    main()
