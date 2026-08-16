"""Step 12: build the pooled per-frame dataset (HPS + learned) used by every
Phase-A diagnostic (spec sections 3-10, 13). One model-forward + one HPS
product per recording, across all 5 folds' test splits (= every recording
exactly once, since grouped 5-fold CV partitions the corpus).
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.folds import build_fold_split, load_kfold_manifest  # noqa: E402
from training.framewise_dataset import RecordingLaneIndex  # noqa: E402
from training.normalization import load_fold_cqt_stats  # noqa: E402
from training.pitch_diagnostics.register_resolution.common import (  # noqa: E402
    REG_DIR,
    candidate_hz,
    full_range_cand_cents,
    load_learned_model,
    native_range,
    per_recording_frame_data,
)

CACHE_PATH = REG_DIR / "frame_cache.pkl"  # full-range (0-360), §2 reconciliation only
NATIVE_CACHE_PATH = REG_DIR / "frame_cache_native.pkl"  # native Step-11 range (34-244), main Phase-A diagnostics


def _build(cache_path: Path, use_native_learned_range: bool, force: bool) -> list[dict]:
    if cache_path.exists() and not force:
        print(f"Loading cached frame data from {cache_path}")
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    index = RecordingLaneIndex.build(REPO_ROOT)
    manifest = load_kfold_manifest(REPO_ROOT)
    cand_cents_full = full_range_cand_cents()
    if use_native_learned_range:
        lrn_lo, lrn_hi = native_range()
        cand_cents_learned = 1200.0 * np.log2(candidate_hz(lrn_lo, lrn_hi))
    else:
        lrn_lo, lrn_hi = None, None
        cand_cents_learned = None

    records = []
    for fold in range(5):
        t0 = time.time()
        split = build_fold_split(manifest, fold, seed=42)
        if use_native_learned_range:
            model, _ckpt = load_learned_model("harmonic", fold, lrn_lo, lrn_hi)
        else:
            from training.pitch_diagnostics.register_resolution.common import load_learned_model_full_range
            model, _ckpt = load_learned_model_full_range("harmonic", fold)
        mu, sigma = load_fold_cqt_stats(fold, REPO_ROOT)
        for rec_id in split.test_recording_ids:
            out = per_recording_frame_data(rec_id, fold, index, model, mu, sigma, cand_cents_full, cand_cents_learned)
            if out is not None:
                records.append(out)
                print(f"  fold {fold} {rec_id}: n_valid={out['n_valid']}")
        print(f"fold {fold} done in {time.time()-t0:.1f}s")

    REG_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(records, fh)
    print(f"Saved {len(records)} recording records to {cache_path}")
    return records


def build(force: bool = False) -> list[dict]:
    """Native-range cache (learned model evaluated on its trained 34-244
    range) -- the default used by every Phase-A diagnostic script except the
    §2 reconciliation itself."""
    return _build(NATIVE_CACHE_PATH, use_native_learned_range=True, force=force)


def build_full_range(force: bool = False) -> list[dict]:
    """Full-range cache (0-360), used ONLY by candidate_range_fixed.py to
    quantify the out-of-distribution degradation the §2 fix causes."""
    return _build(CACHE_PATH, use_native_learned_range=False, force=force)


if __name__ == "__main__":
    recs = build(force=True)
    total = sum(r["n_valid"] for r in recs)
    print(f"Total valid frames across {len(recs)} recordings: {total}")
