"""Verify CQT features align to canonical 10 ms target frame centers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.canonical.frames import frame_centers  # noqa: E402
from dataset.canonical.schema import HOP_S  # noqa: E402
from training.features import (  # noqa: E402
    ALIGN_TOLERANCE_S,
    SR,
    cqt_log_magnitude,
    extract_features_at_target_centers,
)


def test_target_frame_centers_start_at_5ms():
    times = frame_centers(1.0, hop_s=HOP_S)
    assert len(times) > 0
    assert abs(times[0] - 0.005) < ALIGN_TOLERANCE_S
    assert abs(times[1] - 0.015) < ALIGN_TOLERANCE_S


def test_interpolated_features_share_target_timestamps():
    sr = SR
    duration = 2.0
    y = np.random.default_rng(0).standard_normal(int(sr * duration)).astype(np.float32)
    target_times, _ = extract_features_at_target_centers(y, duration)
    expected = frame_centers(duration, hop_s=HOP_S)
    np.testing.assert_allclose(target_times, expected, atol=ALIGN_TOLERANCE_S)


def test_native_cqt_times_are_not_used_as_supervision_grid():
    """Document why we interpolate: librosa native grid starts at 0 ms, not 5 ms."""
    sr = SR
    y = np.zeros(int(sr * 0.1))
    native_times, _ = cqt_log_magnitude(y, sr=sr)
    assert abs(native_times[0] - 0.0) < 1e-9
    target_times = frame_centers(0.1, hop_s=HOP_S)
    assert abs(native_times[0] - target_times[0]) > 0.001


def test_no_drift_over_one_minute_on_output_grid():
    duration = 60.0
    target_times = frame_centers(duration, hop_s=HOP_S)
    expected_last = (len(target_times) - 1 + 0.5) * HOP_S
    assert abs(target_times[-1] - expected_last) < ALIGN_TOLERANCE_S
    assert abs(target_times[-1] - (59.995)) < ALIGN_TOLERANCE_S
