"""Step 14 section 12: relative-pitch feature alignment tests, plus section
5's octave-unwrap safety check on real GT pitch (verifying unwrapping does
not commonly destroy legitimate local movement)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.framewise_dataset import RecordingLaneIndex  # noqa: E402
from training.normalization import log2_hz_to_cents  # noqa: E402
from training.relative_pitch_features import (  # noqa: E402
    PITCH_OFFSETS_FRAMES, compute_phi, octave_unwrap, phi_stats, standardize_phi,
)


def test_octave_unwrap_removes_exact_octave_jumps():
    delta = np.array([1200.0, -1200.0, 2400.0, 0.0])
    unwrapped = octave_unwrap(delta)
    np.testing.assert_allclose(unwrapped, [0.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_octave_unwrap_preserves_small_movement():
    delta = np.array([50.0, -75.0, 1250.0, -1180.0])
    unwrapped = octave_unwrap(delta)
    np.testing.assert_allclose(unwrapped, [50.0, -75.0, 50.0, 20.0], atol=1e-6)


def test_compute_phi_matches_manual_diff_at_offset_1():
    cents = np.array([0.0, 10.0, 25.0, 15.0, 40.0, 55.0], dtype=np.float64)
    valid = np.ones(len(cents), dtype=bool)
    phi = compute_phi(cents, valid, offsets=(1,))
    expected = np.zeros(len(cents))
    expected[1:] = np.diff(cents)
    np.testing.assert_allclose(phi[:, 0], expected, atol=1e-5)


def test_compute_phi_zeroes_leading_frames_before_offset():
    cents = np.arange(30, dtype=np.float64)
    valid = np.ones(len(cents), dtype=bool)
    phi = compute_phi(cents, valid, offsets=(5, 10, 20))
    for j, k in enumerate((5, 10, 20)):
        assert np.all(phi[:k, j] == 0.0)
        assert np.all(phi[k:, j] != 0.0)


def test_compute_phi_zeroes_across_invalid_gap():
    cents = np.array([0.0, 100.0, np.nan, np.nan, 300.0, 320.0], dtype=np.float64)
    valid = np.array([True, True, False, False, True, True])
    phi = compute_phi(cents, valid, offsets=(1, 2))
    # offset 1: pairs (0,1)=valid->100, (1,2)=invalid->0, (2,3)=invalid->0,
    # (3,4)=invalid->0, (4,5)=valid->20
    np.testing.assert_allclose(phi[:, 0], [0.0, 100.0, 0.0, 0.0, 0.0, 20.0], atol=1e-5)
    assert np.all(np.isfinite(phi))  # no NaN leaks through despite NaN cents input


def test_compute_phi_never_produces_nan_or_inf():
    rng = np.random.default_rng(0)
    cents = rng.normal(0, 500, size=200)
    cents[rng.integers(0, 200, size=40)] = np.nan
    valid = ~np.isnan(cents)
    # also scatter invalidity independent of NaN to exercise both_valid gating
    valid[rng.integers(0, 200, size=20)] = False
    phi = compute_phi(cents, valid)
    assert np.all(np.isfinite(phi))


def test_phi_stats_only_uses_both_valid_positions_and_train_data():
    cents = np.array([0.0, 100.0, 250.0, 400.0], dtype=np.float64)
    valid = np.array([True, True, True, True])
    phi = compute_phi(cents, valid, offsets=(1,))
    mu, sigma = phi_stats([phi], [valid])
    expected = np.diff(cents)  # [100, 150, 150]
    assert abs(mu[0] - expected.mean()) < 1e-4
    standardized = standardize_phi(phi, mu, sigma)
    assert standardized.shape == phi.shape


def test_octave_unwrap_rarely_changes_real_gt_deltas():
    """Section 5: verify on GT pitch that legitimate local movement is not
    commonly destroyed by unwrapping -- the fraction of valid consecutive
    pairs where |raw delta| > 600c (i.e. unwrapping actually does something)
    should be small at every offset."""
    index = RecordingLaneIndex.build(REPO_ROOT)
    fractions = {}
    for k in PITCH_OFFSETS_FRAMES:
        n_total = 0
        n_changed = 0
        for lane in index.lanes:
            frames = index._frames[(lane.recording_id, lane.lane_id)]
            valid = frames["valid_target"]
            pitch_log2 = frames["pitch_log2_hz"].astype(np.float64)
            cents = log2_hz_to_cents(pitch_log2, lane.fundamental_hz)
            if k >= len(cents):
                continue
            both_valid = valid[k:] & valid[:-k]
            raw = cents[k:] - cents[:-k]
            raw = raw[both_valid]
            n_total += len(raw)
            n_changed += int(np.sum(np.abs(raw) > 600.0))
        fractions[k] = n_changed / max(n_total, 1)

    for k, frac in fractions.items():
        print(f"offset {k} frames: {frac * 100:.3f}% of valid pairs have |raw delta| > 600c")
        assert frac < 0.02, f"offset {k}: unwrapping changes {frac:.4%} of pairs, too common to trust blindly"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
