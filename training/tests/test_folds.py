"""Fold split and leakage tests."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.folds import (  # noqa: E402
    assert_no_split_leakage,
    build_fold_split,
    load_kfold_manifest,
)


def test_fold_splits_are_disjoint():
    manifest = load_kfold_manifest(REPO_ROOT)
    for fold in range(5):
        split = build_fold_split(manifest, fold, seed=42)
        train = set(split.train_recording_ids)
        val = set(split.val_recording_ids)
        test = set(split.test_recording_ids)
        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)
        leakage = assert_no_split_leakage(split, REPO_ROOT)
        assert leakage["no_shared_group"]


def test_val_not_in_train_groups():
    manifest = load_kfold_manifest(REPO_ROOT)
    split = build_fold_split(manifest, 0, seed=42)
    assert split.val_groups[0] not in split.train_groups
