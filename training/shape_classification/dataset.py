"""Step 22 section 1, 2, 12: build the full oracle-boundary shape corpus --
one record per canonical primitive (T0-T3), with both the oracle (O) and
CREPE (C) normalized-phase contours and their section 3/6 derived features
already attached, plus performance_group_id for grouped-fold filtering
downstream (section 12: no primitive is ever split across train/val/test
independently of its recording).

Caches to output/shape_classification/corpus.pkl (rebuild with --force).
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.canonical.contour import trajectories_by_index  # noqa: E402
from training.framewise_dataset import RecordingLaneIndex  # noqa: E402
from training.pitch_diagnostics.relative_pitch.dense_crepe_path import build as build_crepe  # noqa: E402
from training.shape_classification.contours import (  # noqa: E402
    X_GRID, all_recording_ids, crepe_contour, load_recording_and_primitives, oracle_contour,
)
from training.shape_classification.normalize import relative_cents, shape_normalize, slope_features, velocity  # noqa: E402

OUT_DIR = REPO_ROOT / "output" / "shape_classification"
CORPUS_PATH = OUT_DIR / "corpus.pkl"

# Step 22's semantic renaming of the existing canonical_type ids (unchanged
# from Step 5/schema.py's PRIMITIVE_TYPE_IDS={0,1,2,3}):
CLASS_NAMES = {0: "Fixed", 1: "Cosine", 2: "Sloped-start", 3: "Sloped-end"}


def derive_contour(log2_contour: np.ndarray | None) -> dict[str, Any] | None:
    if log2_contour is None or not np.all(np.isfinite(log2_contour)):
        return None
    r = relative_cents(log2_contour)
    q, span_cents, span_normalized = shape_normalize(r)
    v = velocity(q, X_GRID)
    feats = slope_features(q, v, X_GRID)
    return {
        "log2": log2_contour, "r": r, "q": q, "v": v,
        "span_cents": span_cents, "span_normalized": span_normalized,
        "features": feats,
    }


def load_recording_lookup(repo_root: Path = REPO_ROOT) -> dict[str, dict[str, Any]]:
    """Per-recording CREPE grid + duration, for on-demand re-extraction at
    perturbed boundaries (section 17) without rebuilding the whole corpus."""
    index = RecordingLaneIndex.build(repo_root)
    crepe_pitch = build_crepe()
    out = {}
    for lane in index.lanes:
        frames = index._frames[(lane.recording_id, lane.lane_id)]
        out[lane.recording_id] = {
            "frame_time_s": frames["frame_time_s"],
            "crepe_log2": crepe_pitch[lane.recording_id],
            "duration_s": lane.duration_s,
        }
    return out


def build(force: bool = False) -> list[dict[str, Any]]:
    if CORPUS_PATH.exists() and not force:
        print(f"Loading cached corpus from {CORPUS_PATH}")
        with open(CORPUS_PATH, "rb") as fh:
            return pickle.load(fh)

    index = RecordingLaneIndex.build(REPO_ROOT)
    lane_by_rid = {lane.recording_id: lane for lane in index.lanes}
    crepe_pitch = build_crepe()

    records: list[dict[str, Any]] = []
    n_oracle_fail = 0
    n_crepe_fail = 0

    for rid in all_recording_ids(REPO_ROOT):
        if rid not in lane_by_rid:
            continue
        lane = lane_by_rid[rid]
        rec_doc, primitives = load_recording_and_primitives(rid, REPO_ROOT)
        by_idx = trajectories_by_index(rec_doc)
        cache: dict[int, Any] = {}
        frames = index._frames[(rid, lane.lane_id)]
        frame_time_s = frames["frame_time_s"]
        crepe_log2 = crepe_pitch[rid]

        for prim in primitives:
            o_contour = oracle_contour(prim, rec_doc, by_idx, cache=cache)
            c_contour = crepe_contour(
                frame_time_s, crepe_log2, prim["start_s"], prim["end_s"],
                lane_duration_s=lane.duration_s,
            )
            o_derived = derive_contour(o_contour)
            c_derived = derive_contour(c_contour)
            if o_derived is None:
                n_oracle_fail += 1
            if c_derived is None:
                n_crepe_fail += 1

            records.append({
                "primitive_id": prim["primitive_id"],
                "recording_id": rid,
                "performance_group_id": lane.performance_group_id,
                "lane_id": lane.lane_id,
                "canonical_type": int(prim["canonical_type"]),
                "start_s": prim["start_s"], "end_s": prim["end_s"],
                "duration_s": prim["duration_s"],
                "oracle": o_derived,
                "crepe": c_derived,
            })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_PATH, "wb") as fh:
        pickle.dump(records, fh)
    print(f"Built corpus: {len(records)} primitives "
          f"({n_oracle_fail} oracle extraction failures, {n_crepe_fail} CREPE extraction failures)")
    print(f"Saved to {CORPUS_PATH}")
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    from collections import Counter
    counts = Counter(r["canonical_type"] for r in records)
    durs_by_type = {t: [] for t in range(4)}
    for r in records:
        durs_by_type[r["canonical_type"]].append(r["duration_s"])
    summary = {
        "total_primitives": len(records),
        "n_recordings": len(set(r["recording_id"] for r in records)),
        "class_counts": {CLASS_NAMES[t]: counts.get(t, 0) for t in range(4)},
        "class_duration_stats_s": {
            CLASS_NAMES[t]: {
                "median": float(np.median(durs_by_type[t])) if durs_by_type[t] else None,
                "mean": float(np.mean(durs_by_type[t])) if durs_by_type[t] else None,
                "min": float(np.min(durs_by_type[t])) if durs_by_type[t] else None,
                "max": float(np.max(durs_by_type[t])) if durs_by_type[t] else None,
            }
            for t in range(4)
        },
        "n_oracle_extraction_failures": sum(1 for r in records if r["oracle"] is None),
        "n_crepe_extraction_failures": sum(1 for r in records if r["crepe"] is None),
        "oracle_span_normalized_fraction": {
            CLASS_NAMES[t]: float(np.mean([r["oracle"]["span_normalized"] for r in records
                                            if r["canonical_type"] == t and r["oracle"] is not None]))
            for t in range(4)
        },
    }
    return summary


def main() -> None:
    records = build(force=True)
    summary = summarize(records)
    (OUT_DIR / "corpus_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
