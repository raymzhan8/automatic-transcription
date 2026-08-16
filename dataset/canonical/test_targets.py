"""Pytest sanity checks for Step 5 canonical framewise targets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dataset.canonical.validate_targets import duration_stats  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL = REPO_ROOT / "output" / "canonical" / "v1"


@pytest.fixture(scope="module")
def primitives_csv():
    path = CANONICAL / "index" / "primitives.csv"
    if not path.exists():
        pytest.skip("primitives not built")
    return path


@pytest.fixture(scope="module")
def sample_recording_id():
    recs = sorted((CANONICAL / "recordings").glob("*.json"))
    if not recs:
        pytest.skip("no recordings")
    return recs[0].stem


def test_primitives_exist(primitives_csv):
    assert primitives_csv.stat().st_size > 0


def test_primitive_types_valid(sample_recording_id):
    doc = json.loads((CANONICAL / "primitives" / f"{sample_recording_id}.json").read_text())
    for prim in doc["primitives"]:
        assert prim["canonical_type"] in {0, 1, 2, 3}
        assert prim["source_raw_trajectory_ids"]


def test_t4_decomposition():
    doc = json.loads((CANONICAL / "primitives" / "6417585554a0bfbd8de2d3ff.json").read_text())
    t4 = [p for p in doc["primitives"] if p["source_raw_type"] == 4]
    assert t4
    by_source = {}
    for p in t4:
        key = p["source_raw_indices"][0]
        by_source.setdefault(key, []).append(p)
    for segs in by_source.values():
        types = [s["canonical_type"] for s in sorted(segs, key=lambda x: x["source_subsegment_index"])]
        assert types == [2, 1]


def test_t6_only_type1():
    doc = json.loads((CANONICAL / "primitives" / "6417585554a0bfbd8de2d3ff.json").read_text())
    t6 = [p for p in doc["primitives"] if p["source_raw_type"] == 6]
    assert t6
    assert all(p["canonical_type"] == 1 for p in t6)


def test_frames_npz(sample_recording_id):
    npz_files = list((CANONICAL / "frames").glob(f"{sample_recording_id}_*.npz"))
    if not npz_files:
        pytest.skip("frames not built")
    data = np.load(npz_files[0], allow_pickle=True)
    valid = data["valid_target"]
    assert data["trajectory_type"].dtype == np.int8
    assert np.all(data["trajectory_type"][~valid] == -1)


def test_contour_verification_passed():
    path = CANONICAL / "step_5_contour_verification.json"
    if not path.exists():
        pytest.skip("contour verification not run")
    report = json.loads(path.read_text())
    assert report["n_failed"] == 0


def test_duration_stats_monotonic():
    prims_dir = CANONICAL / "primitives"
    if not prims_dir.exists():
        pytest.skip("primitives not built")
    durations = []
    for path in prims_dir.glob("*.json"):
        doc = json.loads(path.read_text())
        durations.extend(float(p["duration_s"]) for p in doc["primitives"])
    stats = duration_stats(durations)
    assert stats["count"] > 0
    assert stats["pct_lt_10_ms"] <= stats["pct_lt_20_ms"]
    assert stats["pct_lt_20_ms"] <= stats["pct_lt_50_ms"]
    assert stats["pct_lt_50_ms"] <= stats["pct_lt_100_ms"]
