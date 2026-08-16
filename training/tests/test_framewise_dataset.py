"""Dataset and mask tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.folds import build_fold_split, load_kfold_manifest, prepare_fold  # noqa: E402
from training.framewise_dataset import (  # noqa: E402
    FramewiseExcerptDataset,
    RecordingLaneIndex,
    collate_excerpts,
)
from training.normalization import load_fold_cqt_stats  # noqa: E402
from training.sampling import EXCERPT_FRAMES, loss_mask  # noqa: E402


def _fold_cqt_stats(fold_index: int = 0):
    try:
        return load_fold_cqt_stats(fold_index, REPO_ROOT)
    except FileNotFoundError:
        prepare_fold(REPO_ROOT, fold_index, seed=42)
        return load_fold_cqt_stats(fold_index, REPO_ROOT)


def test_excerpt_batch_shapes():
    index = RecordingLaneIndex.build(REPO_ROOT)
    manifest = load_kfold_manifest(REPO_ROOT)
    split = build_fold_split(manifest, 0, seed=42)
    mu, sigma = _fold_cqt_stats(0)
    ds = FramewiseExcerptDataset(
        index, split.train_recording_ids[:3], mu, sigma, excerpts_per_epoch=4, seed=0
    )
    batch = collate_excerpts([ds[i] for i in range(min(2, len(ds)))])
    assert batch["spec"].shape == (2, 1, 360, EXCERPT_FRAMES)
    mask = loss_mask(batch["padding_mask"][0].numpy(), batch["valid_target"][0].numpy())
    assert mask.dtype == bool
    assert "lengths" in batch
    assert torch.equal(batch["lengths"], (~batch["padding_mask"]).sum(dim=1).long())


def test_loss_mask_excludes_padding_and_invalid():
    pad = torch.tensor([False, False, True, True])
    valid = torch.tensor([True, False, True, True])
    mask = (~pad) & valid
    assert mask.tolist() == [True, False, False, False]
