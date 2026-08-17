"""Step 19 section 18: synthetic, no-learning sanity test of the acoustic
representation's ACTUAL temporal resolution under clean, idealized
conditions. Generates pure-tone-plus-harmonics pitch contours (flat, linear
ramp at two speeds, rise->fall) at a representative frequency, and runs
them through the exact deterministic CQT extraction
(training/features.py::cqt_log_magnitude) -- the only feature-extraction
stage that can be meaningfully applied without training. The learned
salience model is real-audio-domain and would be meaningless on pure
synthetic tones (spec section 18's explicit permission to omit it), so this
test covers stage A only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.features import BINS_PER_OCTAVE, FMIN, N_BINS, SR, cqt_log_magnitude  # noqa: E402
from training.pitch_diagnostics.common import bin_from_hz, write_json  # noqa: E402
from training.pitch_diagnostics.pitch_audit.common import AUDIT_DIR  # noqa: E402

DURATION_S = 2.0
F0_CENTER = 250.0  # Hz, near the corpus median pitch (254.7 Hz, Step 19 CQT window-length calc)
N_HARMONICS = 4


def synth_tone(f_of_t, sr: int = SR, duration: float = DURATION_S) -> np.ndarray:
    t = np.arange(int(sr * duration)) / sr
    phase = 2 * np.pi * np.cumsum(f_of_t(t)) / sr
    y = np.zeros_like(t)
    for h in range(1, N_HARMONICS + 1):
        y += (1.0 / h) * np.sin(h * phase)
    y *= 0.3 / np.max(np.abs(y))
    return y.astype(np.float32)


CONTOURS = {
    "flat": lambda t: np.full_like(t, F0_CENTER),
    "ramp_slow": lambda t: F0_CENTER * 2 ** (0.5 * (t - DURATION_S / 2) / DURATION_S),   # ~0.5 octave over 2s
    "ramp_fast": lambda t: F0_CENTER * 2 ** (2.0 * (t - DURATION_S / 2) / DURATION_S),   # ~2 octaves over 2s (very fast)
    "rise_then_fall": lambda t: F0_CENTER * 2 ** (0.7 * np.sin(2 * np.pi * t / DURATION_S)),
}


def main() -> None:
    results = {}
    fig, axes = plt.subplots(len(CONTOURS), 1, figsize=(9, 10), sharex=True)
    for ax, (name, f_fn) in zip(axes, CONTOURS.items()):
        y = synth_tone(f_fn)
        native_times, log_cqt = cqt_log_magnitude(y, sr=SR)
        true_f = f_fn(native_times)
        true_bin = bin_from_hz(true_f)
        ridge_bin = log_cqt.argmax(axis=0)
        ridge_err_cents = (ridge_bin - true_bin) * (1200.0 / BINS_PER_OCTAVE)

        results[name] = {
            "mae_cents": float(np.mean(np.abs(ridge_err_cents))),
            "median_cents": float(np.median(np.abs(ridge_err_cents))),
            "max_abs_cents": float(np.max(np.abs(ridge_err_cents))),
        }

        ax.imshow(log_cqt, aspect="auto", origin="lower", extent=[native_times[0], native_times[-1], 0, N_BINS], cmap="magma")
        ax.plot(native_times, true_bin, color="cyan", lw=1.2, label="true f0 (bin)")
        ax.plot(native_times, ridge_bin, color="lime", lw=0.8, alpha=0.8, label="CQT argmax ridge")
        ax.set_ylabel(name, fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Step 19 synthetic CQT resolution test (f0~{F0_CENTER}Hz, {N_HARMONICS} harmonics)")
    fig.tight_layout()
    FIG_DIR = AUDIT_DIR / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "synthetic_resolution_test.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    write_json(AUDIT_DIR / "synthetic_resolution_test.json", results)
    print("=== synthetic CQT ridge-tracking error ===")
    for name, v in results.items():
        print(name, v)


if __name__ == "__main__":
    main()
