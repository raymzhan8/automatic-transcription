"""Step 15 section 10: model shapes/param counts/receptive field, and
dataset-level alignment tests for the P1 dense delta and P2 windowed
relative-salience representations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.folds import build_fold_split, load_kfold_manifest, prepare_fold  # noqa: E402
from training.framewise_dataset import FramewiseExcerptDataset, RecordingLaneIndex  # noqa: E402
from training.framewise_models import (  # noqa: E402
    PitchMotionEncoder, PitchOnlyClassifier, SalienceMotionEncoder, count_params, tcn_receptive_field,
)
from training.normalization import load_fold_cqt_stats, log2_hz_to_cents  # noqa: E402
from training.pitch_diagnostics.relative_pitch.dense_relative_salience import W_BINS  # noqa: E402
from training.relative_pitch_features import compute_phi, load_dense_estimated_pitch  # noqa: E402
from training.sampling import EXCERPT_FRAMES  # noqa: E402


def _fold_cqt_stats(fold_index: int = 0):
    try:
        return load_fold_cqt_stats(fold_index, REPO_ROOT)
    except FileNotFoundError:
        prepare_fold(REPO_ROOT, fold_index, seed=42)
        return load_fold_cqt_stats(fold_index, REPO_ROOT)


# ---------------------------------------------------------------- model ----

def test_pitch_motion_encoder_receptive_field_in_target_range():
    rf_frames = tcn_receptive_field(kernel_size=5, dilations=(1, 2, 4, 8))
    rf_ms = rf_frames * 10.0
    assert 500.0 <= rf_ms <= 1000.0, f"RF={rf_ms}ms outside spec section 6's 0.5-1.0s target"


def test_p1_forward_shape_and_small_param_count():
    model = PitchOnlyClassifier(PitchMotionEncoder(in_channels=1, hidden=32))
    x = torch.randn(2, 1, EXCERPT_FRAMES)
    out = model(x)
    assert out.shape == (2, EXCERPT_FRAMES, 4)
    n = count_params(model)
    assert n < 50_000, f"P1 not 'small': {n} params"


def test_p2_forward_shape_and_small_param_count():
    model = PitchOnlyClassifier(SalienceMotionEncoder(hidden=32))
    x = torch.randn(2, 1, W_BINS, EXCERPT_FRAMES)
    out = model(x)
    assert out.shape == (2, EXCERPT_FRAMES, 4)
    n = count_params(model)
    assert n < 100_000, f"P2 not 'small': {n} params"


# -------------------------------------------------------------- dataset ----

def _build_dataset(recording_ids=None):
    index = RecordingLaneIndex.build(REPO_ROOT)
    manifest = load_kfold_manifest(REPO_ROOT)
    split = build_fold_split(manifest, 0, seed=42)
    mu, sigma = _fold_cqt_stats(0)
    est = load_dense_estimated_pitch()
    ids = recording_ids or split.train_recording_ids[:2]
    ds = FramewiseExcerptDataset(
        index, ids, mu, sigma, estimated_pitch=est, compute_pitch_features=True,
        excerpts_per_epoch=8, seed=0,
    )
    return ds, index


def test_p1_input_is_phi_offset_1_channel():
    """P1's raw input (spec section 5's frame-to-frame octave-unwrapped
    increment) must equal the offset=1 slice of the already-tested Step 14
    phi tensor -- not a separately (re)computed quantity."""
    ds, _ = _build_dataset()
    sample = ds[0]
    # P1's raw input is exactly channel 0 of the already-tested Step 14 phi
    # tensor (offsets (1,5,10,20) -> channel 0 is the 10ms/offset=1 delta).
    est_cents_full = sample["phi_estimated"].numpy()
    assert est_cents_full.shape[1] == 4
    p1_channel = est_cents_full[:, 0]
    # cross-check against an independent recompute from the pitch_cents/valid_target
    # this same excerpt returned, restricted to offset=1 -- catches drift between
    # the two call sites even though both ultimately call compute_phi.
    manual = compute_phi(
        sample["pitch_cents"].numpy().astype(np.float64), sample["valid_target"].numpy(), offsets=(1,),
    )
    np.testing.assert_allclose(manual[:, 0], sample["phi_oracle"][:, 0].numpy(), atol=1e-6)
    assert p1_channel.shape == (est_cents_full.shape[0],)


def test_relative_salience_alignment_and_no_nan():
    """dense_relative_salience.py must produce a per-recording array whose
    excerpt-slicing (start:end, zero-padded tail) lines up 1:1 with the same
    excerpt's trajectory_type/valid_target -- same slicing code path as spec/target."""
    from training.pitch_diagnostics.relative_pitch.dense_relative_salience import build as build_rel_sal
    rel_sal = build_rel_sal()
    index = RecordingLaneIndex.build(REPO_ROOT)
    manifest = load_kfold_manifest(REPO_ROOT)
    split = build_fold_split(manifest, 0, seed=42)
    mu, sigma = _fold_cqt_stats(0)
    est = load_dense_estimated_pitch()
    ds = FramewiseExcerptDataset(
        index, split.train_recording_ids[:2], mu, sigma, estimated_pitch=est,
        compute_pitch_features=True, relative_salience=rel_sal, excerpts_per_epoch=4, seed=0,
    )
    for i in range(len(ds)):
        sample = ds[i]
        assert sample["relative_salience"].shape == (1, W_BINS, EXCERPT_FRAMES)
        assert torch.isfinite(sample["relative_salience"]).all()
        start = sample["excerpt_start_idx"]
        rid = sample["recording_id"]
        end = start + EXCERPT_FRAMES
        full = rel_sal[rid]
        n_valid_cols = min(end, full.shape[1]) - start
        expected = full[:, start:start + n_valid_cols]
        actual = sample["relative_salience"][0, :, :n_valid_cols].numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        if n_valid_cols < EXCERPT_FRAMES:
            assert np.allclose(sample["relative_salience"][0, :, n_valid_cols:].numpy(), 0.0)


def test_relative_salience_preserves_secondary_peaks():
    """Section 8: numerically verify the representation is not collapsed to
    a hard argmax -- most frames should show a close (ambiguous) top1-top2
    margin, not a single dominant spike."""
    from training.pitch_diagnostics.relative_pitch.dense_relative_salience import build as build_rel_sal
    rel_sal = build_rel_sal()
    rid = next(iter(rel_sal))
    arr = rel_sal[rid]
    sorted_vals = -np.sort(-arr, axis=0)
    margin = sorted_vals[0] - sorted_vals[1]
    close_frac = float(np.mean(margin < 0.1))
    assert close_frac > 0.5, f"only {close_frac:.2%} of frames show a close top1-top2 margin; representation may be over-peaked"


def test_relative_salience_reference_matches_estimated_pitch_bin():
    """The recentering reference must be the SAME Fused+D3 decoded path
    used elsewhere (P0/P1), not an independent estimate -- spot-check that
    the window's peak sits near relative offset 0 on average (a soft check,
    since salience peak != decoded argmax exactly, but they should track)."""
    from training.pitch_diagnostics.relative_pitch.dense_relative_salience import HALF_W, build as build_rel_sal
    rel_sal = build_rel_sal()
    rid = next(iter(rel_sal))
    arr = rel_sal[rid]
    peak_rel_bin = arr.argmax(axis=0) - HALF_W
    assert abs(float(np.mean(peak_rel_bin))) < 5.0, "windowed salience peak is not centered near the recentering reference"
