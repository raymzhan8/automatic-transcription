"""Normalization leakage tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.folds import build_fold_split, load_kfold_manifest, prepare_fold  # noqa: E402
from training.normalization import (  # noqa: E402
    compute_cqt_stats,
    compute_pitch_stats,
    denormalize_pitch,
    log2_hz_to_cents,
    standardize_pitch,
)


def test_cqt_stats_exclude_test_recordings():
    manifest = load_kfold_manifest(REPO_ROOT)
    split = build_fold_split(manifest, 0, seed=42)
    stats = compute_cqt_stats(split.train_recording_ids, REPO_ROOT)
    assert stats["count"] > 0
    for rid in split.test_recording_ids:
        assert rid not in split.train_recording_ids


def test_pitch_stats_use_train_only():
    manifest = load_kfold_manifest(REPO_ROOT)
    split = build_fold_split(manifest, 0, seed=42)
    stats = compute_pitch_stats(split.train_recording_ids, REPO_ROOT)
    assert stats["count"] > 0
    assert stats["unit"] == "cents"
    for rid in split.test_recording_ids + split.val_recording_ids:
        assert rid not in split.train_recording_ids


def test_pitch_denorm_recovers_cents():
    cents = np.array([-200.0, 0.0, 50.0, 700.0], dtype=np.float32)
    mu, sigma = 12.0, 180.0
    std = standardize_pitch(cents, mu, sigma)
    recovered = denormalize_pitch(std, mu, sigma)
    np.testing.assert_allclose(recovered, cents, atol=1e-4)


def test_log2_hz_to_cents_at_tonic_is_zero():
    fund = 146.83
    log2_hz = np.log2(fund)
    assert abs(float(log2_hz_to_cents(log2_hz, fund))) < 1e-6


def test_prepare_fold_writes_cents_pitch_stats():
    _, summary = prepare_fold(REPO_ROOT, 0, seed=42)
    assert summary["leakage_assertions"]["no_shared_group"]
    cqt_path = REPO_ROOT / summary["cqt_stats_path"]
    pitch_path = REPO_ROOT / summary["pitch_stats_path"]
    assert cqt_path.exists()
    data = np.load(cqt_path)
    assert data["mu"].shape == (360,)
    pitch = np.load(pitch_path)
    assert str(pitch["unit"]) == "cents" or pitch["unit"].item() == "cents"
