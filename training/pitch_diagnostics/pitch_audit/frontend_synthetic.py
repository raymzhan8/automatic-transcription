"""Step 20 sections 7-8: the exact Step 19 synthetic no-learning benchmark
(flat / slow ramp / fast ramp / rise-then-fall), re-run through EVERY
candidate frontend, at the corpus median pitch (255Hz, primary benchmark)
and at three additional low-register stress frequencies (110/150/500Hz,
section 8's mandatory low-frequency stress test). Adds turning-point- and
endpoint-localized error on top of Step 19's global MAE/median, since Step
19 showed average ramp error can look fine while transient error is large.
No learning anywhere in this script -- pure deterministic signal synthesis
and deterministic feature extraction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.features import SR  # noqa: E402
from training.pitch_diagnostics.common import write_json  # noqa: E402
from training.pitch_diagnostics.pitch_audit.common import AUDIT_DIR  # noqa: E402
from training.pitch_diagnostics.pitch_audit.frontends import FRONTENDS, compute_frontend_native  # noqa: E402

DURATION_S = 2.0
N_HARMONICS = 4
PRIMARY_HZ = 254.7
STRESS_HZ = (110.0, 150.0, 254.7, 500.0)


def synth_tone(f_of_t, sr: int = SR, duration: float = DURATION_S) -> np.ndarray:
    t = np.arange(int(sr * duration)) / sr
    phase = 2 * np.pi * np.cumsum(f_of_t(t)) / sr
    y = np.zeros_like(t)
    for h in range(1, N_HARMONICS + 1):
        y += (1.0 / h) * np.sin(h * phase)
    y *= 0.3 / np.max(np.abs(y))
    return y.astype(np.float32)


def make_contours(f0: float) -> dict:
    return {
        "flat": lambda t: np.full_like(t, f0),
        "ramp_slow": lambda t: f0 * 2 ** (0.5 * (t - DURATION_S / 2) / DURATION_S),
        "ramp_fast": lambda t: f0 * 2 ** (2.0 * (t - DURATION_S / 2) / DURATION_S),
        "rise_then_fall": lambda t: f0 * 2 ** (0.7 * np.sin(2 * np.pi * t / DURATION_S)),
    }


def err_stats(err_cents: np.ndarray) -> dict:
    a = np.abs(err_cents)
    return {"mae": float(np.mean(a)), "median_ae": float(np.median(a)),
            "p95_ae": float(np.percentile(a, 95)), "max_ae": float(np.max(a))}


def run_one(name: str, spec, f0: float, contour_name: str, f_fn) -> dict:
    y = synth_tone(f_fn)
    native_times, mag, hz_of_bin = compute_frontend_native(y, spec)
    ridge_bin = mag.argmax(axis=0)
    ridge_hz = hz_of_bin[ridge_bin]
    true_hz = f_fn(native_times)
    err_cents = 1200.0 * np.log2(np.maximum(ridge_hz, 1e-6) / true_hz)

    out = {"overall": err_stats(err_cents)}

    if contour_name == "rise_then_fall":
        # curvature extrema at t = DURATION_S/4 (peak) and 3*DURATION_S/4 (trough)
        for label, t_ext in (("peak", DURATION_S / 4), ("trough", 3 * DURATION_S / 4)):
            m = np.abs(native_times - t_ext) <= 0.1
            out[f"turn_region_{label}"] = err_stats(err_cents[m]) if m.any() else None
        m_all_turn = ((np.abs(native_times - DURATION_S / 4) <= 0.1) | (np.abs(native_times - 3 * DURATION_S / 4) <= 0.1))
        out["turn_region"] = err_stats(err_cents[m_all_turn]) if m_all_turn.any() else None
    elif contour_name in ("ramp_slow", "ramp_fast"):
        m_start = native_times <= 0.15
        m_end = native_times >= DURATION_S - 0.15
        out["endpoint_region"] = err_stats(err_cents[m_start | m_end]) if (m_start | m_end).any() else None

    return out


def main() -> None:
    results = {}
    for name, spec in FRONTENDS.items():
        results[name] = {"primary_255hz": {}, "low_freq_stress": {}}
        contours = make_contours(PRIMARY_HZ)
        for cname, f_fn in contours.items():
            results[name]["primary_255hz"][cname] = run_one(name, spec, PRIMARY_HZ, cname, f_fn)
            print(f"{name} {cname} @255Hz: {results[name]['primary_255hz'][cname]['overall']}")

        for f0 in STRESS_HZ:
            contours_f0 = make_contours(f0)
            results[name]["low_freq_stress"][f"{f0:.0f}Hz"] = {}
            for cname in ("flat", "ramp_fast"):
                r = run_one(name, spec, f0, cname, contours_f0[cname])
                results[name]["low_freq_stress"][f"{f0:.0f}Hz"][cname] = r
                print(f"{name} {cname} @{f0:.0f}Hz: {r['overall']}")

    write_json(AUDIT_DIR / "frontend_synthetic.json", results)


if __name__ == "__main__":
    main()
