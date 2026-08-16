"""Tests for precomputed CQT features."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.canonical.schema import features_dir, frames_dir, frames_npz_name  # noqa: E402
from training.features import ALIGN_TOLERANCE_S  # noqa: E402
from training.sampling import EXCERPT_FRAMES, choose_excerpt_start, slice_excerpt  # noqa: E402

PRIMARY_LANE = "0:0"


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="module")
def sample_recording_id(repo_root: Path) -> str:
    frames = list(frames_dir(repo_root).glob("*_0_0.npz"))
    if not frames:
        pytest.skip("no frames npz")
    return frames[0].name.split("_")[0]


def test_precomputed_frame_times_match_targets(repo_root: Path, sample_recording_id: str):
    feat_path = features_dir(repo_root) / f"{sample_recording_id}.npz"
    frame_path = frames_dir(repo_root) / frames_npz_name(sample_recording_id, PRIMARY_LANE)
    if not feat_path.exists():
        pytest.skip("features not built")
    feat = np.load(feat_path)
    frame = np.load(frame_path)
    np.testing.assert_allclose(
        feat["frame_time_s"], frame["frame_time_s"], atol=ALIGN_TOLERANCE_S
    )


def test_excerpt_has_400_frames(repo_root: Path, sample_recording_id: str):
    feat_path = features_dir(repo_root) / f"{sample_recording_id}.npz"
    frame_path = frames_dir(repo_root) / frames_npz_name(sample_recording_id, PRIMARY_LANE)
    if not feat_path.exists():
        pytest.skip("features not built")
    feat = np.load(feat_path)
    frame = np.load(frame_path)
    duration = float(frame["frame_time_s"][-1] + 0.005)
    rng = np.random.default_rng(0)
    valid = frame["valid_target"]
    if not valid.any():
        pytest.skip("no valid frames")
    anchor = int(np.flatnonzero(valid)[0])
    for start in (0, choose_excerpt_start(anchor, len(frame["frame_time_s"]), duration, rng=rng)):
        end = min(start + EXCERPT_FRAMES, feat["cqt_log"].shape[1])
        pad = EXCERPT_FRAMES - (end - start)
        assert end - start + pad == EXCERPT_FRAMES
        excerpt = slice_excerpt(
            len(frame["frame_time_s"]), duration, valid, start_idx=start
        )
        assert len(excerpt.padding_mask) == EXCERPT_FRAMES
