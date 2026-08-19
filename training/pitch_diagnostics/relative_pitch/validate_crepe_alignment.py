"""Step 21 section 2: pre-training alignment/coverage check for the CREPE
dense pitch path (dense_crepe_path.py) against the canonical framewise
dataset, before it is used to train anything. No model, no decoding --
pure bookkeeping over the already-built CREPE cache and RecordingLaneIndex.

Checks, per spec: recording IDs, lane IDs, frame count, 10ms frame
timestamps, valid_target mask, finite pitch values, missing pitch values,
recording coverage. Fails loudly (non-zero exit) on any systematic
mismatch instead of silently dropping recordings/frames.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.framewise_dataset import RecordingLaneIndex  # noqa: E402
from training.pitch_diagnostics.relative_pitch.dense_crepe_path import build as build_crepe  # noqa: E402

OUT_PATH = REPO_ROOT / "output" / "pitch_diagnostics" / "relative_pitch" / "crepe_alignment_validation.json"
NATIVE_HOP_S = 0.01
HOP_TOL_S = 1e-6


def main() -> None:
    index = RecordingLaneIndex.build(REPO_ROOT)
    crepe = build_crepe()  # loads the cached pkl if present, else builds it

    index_ids = {lane.recording_id for lane in index.lanes}
    crepe_ids = set(crepe.keys())

    per_recording = []
    total_frames = 0
    total_valid = 0
    total_covered_valid = 0  # valid_target frames with a finite CREPE value
    total_nan_or_inf = 0
    problems: list[str] = []

    for lane in index.lanes:
        rid = lane.recording_id
        frames = index._frames[(rid, lane.lane_id)]
        times = frames["frame_time_s"]
        valid = frames["valid_target"]
        n_expected = len(times)

        row = {
            "recording_id": rid, "lane_id": lane.lane_id,
            "n_frames_canonical": int(n_expected),
            "in_crepe_cache": rid in crepe,
        }

        if rid not in crepe:
            problems.append(f"{rid}: missing from CREPE cache entirely")
            per_recording.append(row)
            continue

        est = np.asarray(crepe[rid])
        row["n_frames_crepe"] = int(len(est))
        if len(est) != n_expected:
            problems.append(f"{rid}: frame count mismatch (canonical={n_expected}, crepe={len(est)})")

        # 10ms native grid: uniform hop, matches canonical frame_time_s exactly.
        if len(times) > 1:
            diffs = np.diff(times)
            max_hop_err = float(np.max(np.abs(diffs - NATIVE_HOP_S)))
            row["max_hop_error_s"] = max_hop_err
            if max_hop_err > HOP_TOL_S:
                problems.append(f"{rid}: frame_time_s not a uniform 10ms grid (max err {max_hop_err:.2e}s)")

        n = min(len(est), n_expected)
        est_n = est[:n]
        valid_n = valid[:n]
        finite = np.isfinite(est_n)
        n_nan_or_inf = int((~finite).sum())
        n_valid = int(valid_n.sum())
        n_covered_valid = int((finite & valid_n).sum())

        row.update({
            "n_valid_target": n_valid,
            "n_finite": int(finite.sum()),
            "n_nan_or_inf": n_nan_or_inf,
            "n_covered_valid_target": n_covered_valid,
            "coverage_pct_of_valid": 100.0 * n_covered_valid / n_valid if n_valid else None,
        })

        total_frames += n
        total_valid += n_valid
        total_covered_valid += n_covered_valid
        total_nan_or_inf += n_nan_or_inf
        if n_nan_or_inf > 0:
            problems.append(f"{rid}: {n_nan_or_inf} NaN/Inf CREPE values in native grid")
        if n_covered_valid != n_valid:
            problems.append(f"{rid}: {n_valid - n_covered_valid} valid_target frames lack a finite CREPE value")

        per_recording.append(row)

    extra_in_crepe = crepe_ids - index_ids
    missing_from_crepe = index_ids - crepe_ids
    if extra_in_crepe:
        problems.append(f"CREPE cache has recordings not in canonical index: {sorted(extra_in_crepe)}")
    if missing_from_crepe:
        problems.append(f"canonical index has recordings missing from CREPE cache: {sorted(missing_from_crepe)}")

    summary = {
        "total_recordings_canonical": len(index_ids),
        "total_recordings_crepe": len(crepe_ids),
        "recording_ids_match": index_ids == crepe_ids,
        "total_frames": total_frames,
        "total_valid_target_frames": total_valid,
        "total_crepe_covered_valid_frames": total_covered_valid,
        "total_missing_or_invalid_valid_frames": total_valid - total_covered_valid,
        "coverage_pct": 100.0 * total_covered_valid / total_valid if total_valid else None,
        "total_nan_or_inf_frames_native_grid": total_nan_or_inf,
        "problems": problems,
        "status": "OK" if not problems else "PROBLEMS_FOUND",
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"summary": summary, "per_recording": per_recording}, indent=2) + "\n")

    print("=== Step 21 CREPE alignment validation ===")
    print(f"recordings: canonical={summary['total_recordings_canonical']} "
          f"crepe={summary['total_recordings_crepe']} match={summary['recording_ids_match']}")
    print(f"total valid_target frames: {total_valid}")
    print(f"CREPE-covered valid frames: {total_covered_valid} ({summary['coverage_pct']:.4f}%)")
    print(f"missing/invalid valid frames: {summary['total_missing_or_invalid_valid_frames']}")
    print(f"NaN/Inf on native grid: {total_nan_or_inf}")
    print(f"status: {summary['status']}")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(" -", p)
        sys.exit(1)


if __name__ == "__main__":
    main()
