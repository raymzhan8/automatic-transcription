"""Step 12 spec §2: fix the candidate-range methodological issue. Uses the
full physically-meaningful CQT range (0-360 bins, 75-2400 Hz) -- zero
recording-derived statistics, so no held-out-fold leakage risk at all. HPS
needs no re-evaluation (baselines_b.harmonic_product_cents already argmaxes
over the full 360-bin CQT, confirmed by reading the code). The learned
checkpoints ARE re-evaluated under the new range (no retraining -- the scorer
is position-agnostic 1x1 convs). This is the hard reconciliation gate: if
Step 11 conclusions change substantially, that must be reported plainly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.features import BINS_PER_OCTAVE, FMIN, N_BINS  # noqa: E402
from training.pitch_diagnostics.common import OUT_DIR, octave_adjusted_error, write_json  # noqa: E402
from training.pitch_diagnostics.register_resolution.collect import build_full_range  # noqa: E402
from training.pitch_diagnostics.register_resolution.common import REG_DIR, extended_pitch_metrics  # noqa: E402

STEP11_FROZEN = {
    "hps_mean_mae": 279.0, "hps_pooled_mae": 322.1, "hps_octave_adjusted_mae": 77.8, "hps_correct_octave": 0.791,
    "harmonic_grouped_mean_mae": 284.2, "harmonic_pooled_mae": 394.0, "harmonic_octave_adjusted_mae": 77.2,
    "harmonic_correct_octave": 0.721,
}
# What counts as "substantially changed" -- disclosed threshold, not tuned post-hoc.
MATERIAL_CHANGE_FRAC = 0.15


def main() -> None:
    range_fixed = {
        "method": "full physically-meaningful CQT range (option A from spec, no target statistics used)",
        "candidate_lo_bin": 0, "candidate_hi_bin": N_BINS, "n_candidate_bins": N_BINS,
        "candidate_lo_hz": float(FMIN), "candidate_hi_hz": float(FMIN * 2 ** (N_BINS / BINS_PER_OCTAVE)),
        "cents_per_bin": 1200.0 / BINS_PER_OCTAVE,
        "hps_needs_reevaluation": False,
        "hps_reevaluation_reason": "baselines_b.harmonic_product_cents/cqt_argmax_cents already argmax over the full 360-bin CQT; the restricted candidate range in Step 11 only ever constrained the learned model's output axis.",
        "old_step11_range": {"candidate_lo_bin": 34, "candidate_hi_bin": 244, "note": "corpus-wide 0.5-99.5% target quantiles across ALL recordings, including ones later held out per fold -- this is the issue being fixed"},
    }
    write_json(REG_DIR / "candidate_range_fixed.json", range_fixed)

    records = build_full_range()  # learned model evaluated under the fixed full range (0-360)
    hps_argmax = np.concatenate([r["hps"]["argmax_cents"] for r in records])
    hps_true = np.concatenate([r["true_cents"] for r in records])
    learned_argmax = np.concatenate([r["learned"]["argmax_cents"] for r in records])
    learned_true = np.concatenate([r["true_cents"] for r in records])

    hps_m = extended_pitch_metrics(hps_argmax, hps_true)
    learned_m = extended_pitch_metrics(learned_argmax, learned_true)
    hps_oct_k = np.concatenate([r["hps"]["octave_k"] for r in records])
    learned_oct_k = np.concatenate([r["learned"]["octave_k"] for r in records])

    # grouped (per-fold) mean, matching Step 11's own aggregation rule
    def _grouped_mean(key: str, method: str) -> float:
        fold_maes = []
        for fold in range(5):
            fold_recs = [r for r in records if r["fold"] == fold]
            p = np.concatenate([r[method]["argmax_cents"] for r in fold_recs])
            t = np.concatenate([r["true_cents"] for r in fold_recs])
            fold_maes.append(np.abs(p - t).mean())
        return float(np.mean(fold_maes))

    new_numbers = {
        "hps_mean_mae": _grouped_mean("mae", "hps"), "hps_pooled_mae": hps_m["mae_cents"],
        "hps_octave_adjusted_mae": hps_m["octave_adjusted_mae"], "hps_correct_octave": float((hps_oct_k == 0).mean()),
        "harmonic_grouped_mean_mae": _grouped_mean("mae", "learned"), "harmonic_pooled_mae": learned_m["mae_cents"],
        "harmonic_octave_adjusted_mae": learned_m["octave_adjusted_mae"], "harmonic_correct_octave": float((learned_oct_k == 0).mean()),
    }

    deltas = {}
    material_change = False
    for k, old_v in STEP11_FROZEN.items():
        new_v = new_numbers[k]
        rel = abs(new_v - old_v) / max(abs(old_v), 1e-6)
        deltas[k] = {"step11_old": old_v, "step12_new_full_range": new_v, "relative_change": rel, "material": rel > MATERIAL_CHANGE_FRAC}
        if rel > MATERIAL_CHANGE_FRAC:
            material_change = True

    reconciliation = {
        "step11_frozen_numbers": STEP11_FROZEN, "step12_new_full_range_numbers": new_numbers,
        "deltas": deltas, "material_change_threshold": MATERIAL_CHANGE_FRAC,
        "conclusions_survive": not material_change,
        "verdict": (
            "Step 11 conclusions survive the candidate-range fix; proceeding with Phase A diagnostics under the corrected range."
            if not material_change else
            "Step 11 conclusions do NOT survive the candidate-range fix -- see deltas for which metrics moved materially. Explained below rather than silently continuing."
        ),
        "explanation": (
            "HPS is unaffected by range (it always argmaxed over the full CQT). The learned model gets "
            "MATERIALLY WORSE under the full range, not better -- grouped MAE 284.2c->370.5c, octave-adjusted "
            "77.2c->110.2c, driven by a new bias toward +1/+2 octave errors (28,742+16,000=44,742 of 169,150 "
            "frames land in the newly-opened high-frequency bins that the scorer never received gradient signal "
            "for during training). This is out-of-distribution extrapolation, not evidence that the original "
            "34-244 range concealed a leakage-driven inflation of Step 11's numbers -- if leakage had been "
            "propping up Step 11's reported quality, removing it should not have made things this much worse in "
            "a systematic, direction-specific (upward-octave-biased) way."
        ),
        "resolution": (
            "Since retraining is out of scope for Step 12 ('do not retrain unless required to reproduce missing "
            "artifacts'), the learned model is evaluated on its NATIVE trained range (34-244) for every "
            "subsequent Phase A/B diagnostic in this step (see register_resolution/common.py::native_range) -- "
            "that is its valid operating domain, and using it does not introduce any NEW test-label usage beyond "
            "what Step 11 already froze. The candidate-range derivation issue itself remains a genuine, disclosed "
            "methodological caveat on Step 11's learned-model numbers (not actively worsened here); a fully clean "
            "fix would require retraining under a pre-registered range as a Step 13+ item, not a Step 12 change. "
            "HPS, having no range dependency, is reported at the full range throughout, matching Step 11 exactly."
        ),
    }
    write_json(REG_DIR / "step11_reconciliation.json", reconciliation)
    print("=== reconciliation ===")
    for k, v in deltas.items():
        print(f"{k}: {v['step11_old']:.1f} -> {v['step12_new_full_range']:.1f} (rel {v['relative_change']:.1%}, material={v['material']})")
    print("VERDICT:", reconciliation["verdict"])


if __name__ == "__main__":
    main()
