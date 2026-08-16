"""Grouped k-fold split loading, validation holdout, and fold statistics."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dataset.canonical.schema import (
    HOP_S,
    frames_dir,
    frames_npz_name,
    splits_dir,
)
from dataset.canonical.splits import check_leakage, load_recordings
from training.normalization import (
    compute_cqt_stats,
    compute_pitch_stats,
    save_fold_stats,
)

PRIMARY_LANE = "0:0"
DEFAULT_MANIFEST = "grouped_kfold_k5_seed42.json"


@dataclass(frozen=True)
class FoldSplit:
    fold_index: int
    train_recording_ids: list[str]
    val_recording_ids: list[str]
    test_recording_ids: list[str]
    train_groups: list[str]
    val_groups: list[str]
    test_groups: list[str]


def load_kfold_manifest(repo_root: Path, manifest_name: str = DEFAULT_MANIFEST) -> dict[str, Any]:
    path = splits_dir(repo_root) / manifest_name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; run `python dataset/canonical/splits.py` first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _group_to_recordings(manifest: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for group_id, rids in manifest["group_members"].items():
        out[group_id] = sorted(rids)
    return out


def _recordings_in_fold(manifest: dict[str, Any], fold_name: str) -> list[str]:
    return sorted(
        rid for rid, fold in manifest["assignments"].items() if fold == fold_name
    )


def _groups_in_fold(manifest: dict[str, Any], fold_name: str) -> list[str]:
    return sorted(
        gid
        for gid, fold in manifest["group_assignments"].items()
        if fold == fold_name
    )


def pick_val_group(train_groups: list[str], seed: int) -> str:
    """Deterministic: lowest hash(group_id + seed) mod len."""
    if not train_groups:
        raise ValueError("no train groups for validation holdout")
    scored = sorted(
        train_groups,
        key=lambda g: int(hashlib.md5(f"{g}:{seed}".encode()).hexdigest(), 16),
    )
    return scored[0]


def build_fold_split(
    manifest: dict[str, Any],
    fold_index: int,
    *,
    seed: int = 42,
) -> FoldSplit:
    fold_name = f"fold_{fold_index}"
    test_groups = _groups_in_fold(manifest, fold_name)
    test_ids = _recordings_in_fold(manifest, fold_name)
    train_groups = sorted(
        gid for gid, fold in manifest["group_assignments"].items() if fold != fold_name
    )
    val_group = pick_val_group(train_groups, seed)
    val_groups = [val_group]
    group_map = _group_to_recordings(manifest)
    val_ids = list(group_map[val_group])
    train_ids = sorted(
        rid
        for gid in train_groups
        if gid != val_group
        for rid in group_map[gid]
    )
    return FoldSplit(
        fold_index=fold_index,
        train_recording_ids=train_ids,
        val_recording_ids=val_ids,
        test_recording_ids=test_ids,
        train_groups=[g for g in train_groups if g != val_group],
        val_groups=val_groups,
        test_groups=test_groups,
    )


def assert_no_split_leakage(
    split: FoldSplit,
    repo_root: Path,
) -> dict[str, Any]:
    """Verify train/val/test sets do not share performance groups or audio."""
    recordings = load_recordings(repo_root)
    by_id = {r["recording_id"]: r for r in recordings}

    def assignments(rids: list[str], name: str) -> dict[str, str]:
        return {rid: name for rid in rids}

    combined = {}
    combined.update(assignments(split.train_recording_ids, "train"))
    combined.update(assignments(split.val_recording_ids, "val"))
    combined.update(assignments(split.test_recording_ids, "test"))

    subset = [by_id[rid] for rid in combined if rid in by_id]
    return check_leakage(combined, subset)


def frame_class_histogram(recording_ids: list[str], repo_root: Path) -> Counter[int]:
    hist: Counter[int] = Counter()
    for recording_id in recording_ids:
        path = frames_dir(repo_root) / frames_npz_name(recording_id, PRIMARY_LANE)
        if not path.exists():
            continue
        data = np.load(path)
        valid = data["valid_target"]
        types = data["trajectory_type"][valid]
        for t in types:
            hist[int(t)] += 1
    return hist


def valid_frame_count(recording_ids: list[str], repo_root: Path) -> int:
    total = 0
    for recording_id in recording_ids:
        path = frames_dir(repo_root) / frames_npz_name(recording_id, PRIMARY_LANE)
        if not path.exists():
            continue
        data = np.load(path)
        total += int(data["valid_target"].sum())
    return total


def prepare_fold(
    repo_root: Path,
    fold_index: int,
    *,
    manifest_name: str = DEFAULT_MANIFEST,
    seed: int = 42,
) -> tuple[FoldSplit, dict[str, Any]]:
    manifest = load_kfold_manifest(repo_root, manifest_name)
    split = build_fold_split(manifest, fold_index, seed=seed)
    leakage = assert_no_split_leakage(split, repo_root)

    cqt_stats = compute_cqt_stats(split.train_recording_ids, repo_root)
    pitch_stats = compute_pitch_stats(split.train_recording_ids, repo_root)
    cqt_path, pitch_path = save_fold_stats(fold_index, cqt_stats, pitch_stats, repo_root)

    summary = {
        "fold_index": fold_index,
        "train_recordings": split.train_recording_ids,
        "val_recordings": split.val_recording_ids,
        "test_recordings": split.test_recording_ids,
        "train_groups": split.train_groups,
        "val_groups": split.val_groups,
        "test_groups": split.test_groups,
        "train_valid_frames": valid_frame_count(split.train_recording_ids, repo_root),
        "val_valid_frames": valid_frame_count(split.val_recording_ids, repo_root),
        "test_valid_frames": valid_frame_count(split.test_recording_ids, repo_root),
        "train_class_histogram": dict(sorted(frame_class_histogram(split.train_recording_ids, repo_root).items())),
        "val_class_histogram": dict(sorted(frame_class_histogram(split.val_recording_ids, repo_root).items())),
        "test_class_histogram": dict(sorted(frame_class_histogram(split.test_recording_ids, repo_root).items())),
        "cqt_stats_path": str(cqt_path.relative_to(repo_root)),
        "pitch_stats_path": str(pitch_path.relative_to(repo_root)),
        "leakage_assertions": leakage,
        "hop_s": HOP_S,
    }
    return split, summary
