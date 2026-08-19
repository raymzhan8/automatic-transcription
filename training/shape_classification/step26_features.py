"""Step 26 section 4: ONE deterministic acoustic representation per oracle
primitive, reusing the CQT frontend already used throughout Steps 6-20
(`training/features.py`, SR=22050, CQT_HOP=220 -> 10ms hop, FMIN=75,
N_BINS=360, BINS_PER_OCTAVE=72, filter_scale=1 -- the config every trained
model in this repo has actually used; Step 20's fs=0.5 challenger is
audited only, never retrained on, per that step's own Phase B gate).

The legacy 1-second-clip spectrogram-PNG pipeline (`training/models.py`,
`training/spec_dataset.py`) is NOT reused: it assumes fixed-duration clips
and would require rebuilding its whole rendering step for variable-duration
primitives -- exactly the "spectrogram architecture project" section 3
warns against.

No waveform time-stretching: the native CQT is computed once per recording,
then each primitive's own log-magnitude patch is read off by interpolating
onto its own phase grid x=(t-start_s)/(end_s-start_s), t_grid = start_s +
x_grid*(end_s-start_s) -- byte-identical in spirit to `contours.crepe_contour`'s
own interpolation of CREPE onto the same N=64 grid, just per-frequency-bin
instead of scalar.

Caches to output/shape_classification/step26_audio_cache.pkl (rebuild with
--force): {primitive_id: np.ndarray[N_BINS, 64] float32 log10-magnitude}.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

import librosa
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.canonical.audio_refs import find_source_audio  # noqa: E402
from training.features import (  # noqa: E402
    BINS_PER_OCTAVE, CQT_HOP, FMIN, N_BINS, SR, cqt_log_magnitude, interpolate_cqt_to_target_grid,
)
from training.shape_classification.contours import X_GRID  # noqa: E402
from training.shape_classification.dataset import OUT_DIR  # noqa: E402

AUDIO_CACHE_PATH = OUT_DIR / "step26_audio_cache.pkl"


def frontend_metadata() -> dict[str, Any]:
    return {
        "representation": "CQT log10-magnitude (training/features.py, unmodified)",
        "sr": SR, "cqt_hop": CQT_HOP, "hop_s": CQT_HOP / SR, "fmin": FMIN,
        "n_bins": N_BINS, "bins_per_octave": BINS_PER_OCTAVE,
        "n_time_frames_per_primitive": len(X_GRID),
        "time_normalization": "interpolate native CQT time axis onto the primitive's own "
                               "N=64 phase grid (no waveform time-stretch)",
        "why_this_frontend": (
            "Already validated across Steps 6-20 (framewise trajectory/pitch models, the "
            "Step 20 frontend bake-off); the CQT->frequency-CNN pattern is production, not "
            "audit-only. filter_scale=0.5 (Step 20's challenger) is not used: Step 20's own "
            "Phase B (retrain on it) has not started, so it is not yet an 'already-tested' "
            "config for a trained classifier. The legacy fixed-1s-clip PNG pipeline was "
            "rejected because adapting it to variable-duration primitives means rebuilding "
            "its rendering step -- the spectrogram-architecture project section 3 rules out."
        ),
    }


def build(force: bool = False) -> dict[str, np.ndarray]:
    if AUDIO_CACHE_PATH.exists() and not force:
        print(f"Loading cached acoustic patches from {AUDIO_CACHE_PATH}")
        with open(AUDIO_CACHE_PATH, "rb") as fh:
            return pickle.load(fh)

    from training.shape_classification.contours import all_recording_ids, load_recording_and_primitives

    patches: dict[str, np.ndarray] = {}
    n_missing_audio = 0
    for rid in all_recording_ids(REPO_ROOT):
        _, primitives = load_recording_and_primitives(rid, REPO_ROOT)
        audio_path = find_source_audio(REPO_ROOT, rid)
        if audio_path is None:
            n_missing_audio += len(primitives)
            continue
        y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
        native_times, log_cqt = cqt_log_magnitude(y, sr=SR)

        for prim in primitives:
            start_s, end_s = prim["start_s"], prim["end_s"]
            t_grid = start_s + X_GRID * (end_s - start_s)
            patch = interpolate_cqt_to_target_grid(native_times, log_cqt, t_grid)  # [N_BINS, 64]
            patches[prim["primitive_id"]] = patch.astype(np.float32)
        print(f"{rid}: {len(primitives)} primitives")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(AUDIO_CACHE_PATH, "wb") as fh:
        pickle.dump(patches, fh)
    print(f"Built {len(patches)} acoustic patches ({n_missing_audio} skipped, no source audio)")
    print(f"Saved to {AUDIO_CACHE_PATH}")
    return patches


def main() -> None:
    import json
    print(json.dumps(frontend_metadata(), indent=2))
    build(force=True)


if __name__ == "__main__":
    main()
