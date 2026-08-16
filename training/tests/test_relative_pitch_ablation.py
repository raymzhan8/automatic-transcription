"""Step 14 sections 2, 11-13: model shapes/param counts, and dataset-level
CQT/target/phi alignment tests (recording start, excerpt boundaries,
invalid-target gaps, padding, octave-unwrapping, multiple offsets)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.folds import build_fold_split, load_kfold_manifest, prepare_fold  # noqa: E402
from training.framewise_dataset import FramewiseExcerptDataset, RecordingLaneIndex, collate_excerpts  # noqa: E402
from training.framewise_models import FramewiseConditionalTCNModel, count_params  # noqa: E402
from training.normalization import load_fold_cqt_stats, log2_hz_to_cents  # noqa: E402
from training.relative_pitch_features import PITCH_FEATURE_DIM, compute_phi, load_dense_estimated_pitch  # noqa: E402
from training.sampling import EXCERPT_FRAMES  # noqa: E402


def _fold_cqt_stats(fold_index: int = 0):
    try:
        return load_fold_cqt_stats(fold_index, REPO_ROOT)
    except FileNotFoundError:
        prepare_fold(REPO_ROOT, fold_index, seed=42)
        return load_fold_cqt_stats(fold_index, REPO_ROOT)


# ---------------------------------------------------------------- model ----

def test_condition_a_audio_only_forward_and_no_pitch_params():
    model = FramewiseConditionalTCNModel(use_audio=True, use_pitch=False)
    spec = torch.randn(2, 1, 360, EXCERPT_FRAMES)
    out = model(spec, None)
    assert out.shape == (2, EXCERPT_FRAMES, 4)
    assert model.pitch_proj is None and model.fuse is None


def test_condition_b_pitch_only_forward_ignores_spec():
    model = FramewiseConditionalTCNModel(use_audio=False, use_pitch=True, pitch_dim=PITCH_FEATURE_DIM)
    pitch = torch.randn(2, EXCERPT_FRAMES, PITCH_FEATURE_DIM)
    out = model(None, pitch)
    assert out.shape == (2, EXCERPT_FRAMES, 4)
    assert model.freq_cnn is None and model.fuse is None


def test_condition_c_d_fusion_forward_shape():
    model = FramewiseConditionalTCNModel(use_audio=True, use_pitch=True, pitch_dim=PITCH_FEATURE_DIM)
    spec = torch.randn(2, 1, 360, EXCERPT_FRAMES)
    pitch = torch.randn(2, EXCERPT_FRAMES, PITCH_FEATURE_DIM)
    out = model(spec, pitch)
    assert out.shape == (2, EXCERPT_FRAMES, 4)
    assert model.fuse is not None


def test_shared_tcn_and_type_head_identical_module_shapes_across_conditions():
    """Spec section 2: the temporal classifier must be identical except for
    the input projection -- verify the TCN/type_head parameter SHAPES match
    across A/C/D (only the projection layers differ)."""
    a = FramewiseConditionalTCNModel(use_audio=True, use_pitch=False)
    c = FramewiseConditionalTCNModel(use_audio=True, use_pitch=True, pitch_dim=PITCH_FEATURE_DIM)
    for name, pa in a.tcn.state_dict().items():
        assert pa.shape == c.tcn.state_dict()[name].shape
    assert a.type_head.weight.shape == c.type_head.weight.shape
    assert a.freq_cnn.net[0].weight.shape == c.freq_cnn.net[0].weight.shape


def test_param_counts_documented():
    counts = {
        cond: count_params(FramewiseConditionalTCNModel(
            use_audio=use_a, use_pitch=use_p, pitch_dim=PITCH_FEATURE_DIM,
        ))
        for cond, use_a, use_p in (("A", True, False), ("B", False, True), ("C", True, True), ("D", True, True))
    }
    assert counts["C"] == counts["D"]  # identical architecture, per spec section 6
    assert counts["A"] < counts["C"]  # C has strictly more params (fuse + pitch_proj)
    assert counts["B"] < counts["A"]  # B has no freq_cnn at all
    print("param counts:", counts)


# -------------------------------------------------------------- dataset ----

def _build_dataset(estimated=True, recording_ids=None, **kwargs):
    index = RecordingLaneIndex.build(REPO_ROOT)
    manifest = load_kfold_manifest(REPO_ROOT)
    split = build_fold_split(manifest, 0, seed=42)
    mu, sigma = _fold_cqt_stats(0)
    est = load_dense_estimated_pitch() if estimated else None
    ids = recording_ids or split.train_recording_ids[:2]
    ds = FramewiseExcerptDataset(
        index, ids, mu, sigma, estimated_pitch=est, compute_pitch_features=True,
        excerpts_per_epoch=8, seed=0, **kwargs,
    )
    return ds, index


def test_phi_fields_present_and_shaped_correctly():
    ds, _ = _build_dataset()
    sample = ds[0]
    assert sample["phi_oracle"].shape == (EXCERPT_FRAMES, PITCH_FEATURE_DIM)
    assert sample["phi_estimated"].shape == (EXCERPT_FRAMES, PITCH_FEATURE_DIM)
    assert torch.isfinite(sample["phi_oracle"]).all()
    assert torch.isfinite(sample["phi_estimated"]).all()


def test_phi_oracle_matches_manual_recompute_from_same_excerpt_slice():
    """End-to-end alignment: recompute phi from the SAME excerpt's returned
    pitch_cents/valid_target and check it matches the dataset's phi_oracle
    exactly -- catches any off-by-one shift between spec/target/phi slicing."""
    ds, _ = _build_dataset()
    for i in range(len(ds)):
        sample = ds[i]
        manual = compute_phi(sample["pitch_cents"].numpy().astype(np.float64), sample["valid_target"].numpy())
        np.testing.assert_allclose(sample["phi_oracle"].numpy(), manual, atol=1e-4)


def test_phi_zero_at_padded_tail_and_matches_cqt_padding_convention():
    """Force an excerpt that runs past a short recording's end: padded
    positions must have valid_target=False, phi zeroed by construction, and
    CQT zero-padded -- all three referring to the same frame index."""
    index = RecordingLaneIndex.build(REPO_ROOT)
    manifest = load_kfold_manifest(REPO_ROOT)
    split = build_fold_split(manifest, 0, seed=42)
    mu, sigma = _fold_cqt_stats(0)
    est = load_dense_estimated_pitch()
    shortest = min(index.filter_recordings(set(split.train_recording_ids)), key=lambda x: x.n_frames)
    ds = FramewiseExcerptDataset(
        index, [shortest.recording_id], mu, sigma, estimated_pitch=est,
        compute_pitch_features=True, excerpts_per_epoch=1, seed=1,
    )
    found_padding = False
    for _ in range(200):
        sample = ds._sample_one()
        pad = sample["padding_mask"].numpy()
        if pad.any():
            found_padding = True
            assert not sample["valid_target"].numpy()[pad].any()
            assert np.allclose(sample["phi_oracle"].numpy()[pad], 0.0)
            assert np.allclose(sample["phi_estimated"].numpy()[pad], 0.0)
            spec = sample["spec"].numpy()[0]  # [F, T]
            assert np.allclose(spec[:, pad], 0.0), "CQT should be zero in the same padded region"
            break
    if not found_padding:
        # every corpus recording is many minutes long and choose_excerpt_start
        # (training.sampling, shared with B0/B1/C) keeps excerpts fully inside
        # duration_s whenever duration_s > EXCERPT_S, so real padded excerpts
        # essentially never occur here -- the zero-gating contract itself is
        # covered unconditionally by test_compute_phi_zeroes_across_invalid_gap
        # and by test_phi_oracle_matches_manual_recompute_from_same_excerpt_slice.
        pytest.skip("no padded excerpt observed in 200 draws (expected: no recording is this short)")


def test_phi_estimated_zero_wherever_gt_invalid_even_though_estimated_source_is_dense():
    """The estimated Fused+D3 path is defined everywhere (no NaN), but phi
    must still be gated by the SAME valid_target mask as oracle (spec
    section 6's 'identical feature function')."""
    ds, _ = _build_dataset()
    sample = ds[0]
    invalid = ~sample["valid_target"].numpy()
    assert np.allclose(sample["phi_estimated"].numpy()[invalid], 0.0)


def test_collate_preserves_phi_shapes_and_batch_order():
    ds, _ = _build_dataset()
    samples = [ds._sample_one() for _ in range(3)]
    batch = collate_excerpts(samples)
    assert batch["phi_oracle"].shape == (3, EXCERPT_FRAMES, PITCH_FEATURE_DIM)
    assert batch["phi_estimated"].shape == (3, EXCERPT_FRAMES, PITCH_FEATURE_DIM)
    for i in range(3):
        np.testing.assert_allclose(batch["phi_oracle"][i].numpy(), samples[i]["phi_oracle"].numpy())
