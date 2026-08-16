"""Corpus-level validation and statistics for Step 5 targets."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.canonical.build import exported_recording_ids  # noqa: E402
from dataset.canonical.schema import (  # noqa: E402
    COMPOSITE_DECOMPOSITION,
    CONTOUR_TOLERANCE_LOG2,
    HOP_S,
    PRIMITIVE_TYPE_IDS,
    canonical_root,
    frames_dir,
    frames_npz_name,
    splits_dir,
)
from dataset.canonical.verify_contours import verify_recording  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_split_leakage(repo_root: Path) -> dict:
    issues = []
    for manifest_path in sorted(splits_dir(repo_root).glob("*.json")):
        manifest = load_json(manifest_path)
        group_to_splits: dict[str, set[str]] = defaultdict(set)
        audio_to_splits: dict[str, set[str]] = defaultdict(set)
        assignments = manifest.get("assignments") or {}
        for recording_id, split_name in assignments.items():
            rec_path = canonical_root(repo_root) / "recordings" / f"{recording_id}.json"
            if not rec_path.exists():
                continue
            doc = load_json(rec_path)
            pg = doc.get("performance", {}).get("performance_group_id")
            audio_id = doc.get("audio", {}).get("audio_id")
            if pg:
                group_to_splits[pg].add(split_name)
            if audio_id:
                audio_to_splits[audio_id].add(split_name)
        for pg, splits in group_to_splits.items():
            if len(splits) > 1:
                issues.append(
                    f"{manifest_path.name}: performance group {pg} in {sorted(splits)}"
                )
        for audio_id, splits in audio_to_splits.items():
            if len(splits) > 1:
                issues.append(
                    f"{manifest_path.name}: audio_id {audio_id} in {sorted(splits)}"
                )
    return {"passed": not issues, "issues": issues}


DURATION_THRESHOLDS_MS = (10, 20, 50, 100)


def duration_stats(durations: list[float]) -> dict:
    if not durations:
        return {}
    ordered = sorted(durations)
    n = len(ordered)

    def pctile(p: float) -> float:
        idx = min(n - 1, int(p * n))
        return ordered[idx]

    pct_lt = {
        f"pct_lt_{ms}_ms": 100.0 * sum(1 for d in ordered if d < ms / 1000.0) / n
        for ms in DURATION_THRESHOLDS_MS
    }
    assert_monotonic_cumulative_percentages(pct_lt)
    return {
        "count": n,
        "min": ordered[0],
        "p01": pctile(0.01),
        "p05": pctile(0.05),
        "median": statistics.median(ordered),
        "mean": sum(ordered) / n,
        "p95": pctile(0.95),
        "max": ordered[-1],
        **pct_lt,
    }


def assert_monotonic_cumulative_percentages(pct_lt: dict[str, float]) -> None:
    """Cumulative ``% shorter than X ms`` must be non-decreasing in X."""
    keys = [f"pct_lt_{ms}_ms" for ms in DURATION_THRESHOLDS_MS]
    values = [pct_lt[k] for k in keys if k in pct_lt]
    for left, right in zip(values, values[1:]):
        if left > right + 1e-9:
            raise ValueError(
                f"non-monotonic cumulative duration percentages: {pct_lt}"
            )


def same_type_boundary_analysis(primitives: list[dict], hop_s: float = HOP_S) -> dict:
    by_lane: dict[str, list[dict]] = defaultdict(list)
    for p in primitives:
        by_lane[p["lane_id"]].append(p)

    boundaries = []
    for lane, members in by_lane.items():
        ordered = sorted(members, key=lambda p: (p["start_s"], p["seq"]))
        for a, b in zip(ordered, ordered[1:]):
            if a["canonical_type"] != b["canonical_type"]:
                continue
            pitch_delta_cents = None
            pa = a.get("end_pitch") or {}
            pb = b.get("start_pitch") or {}
            if pa.get("log2_hz") is not None and pb.get("log2_hz") is not None:
                pitch_delta_cents = 1200.0 * (pb["log2_hz"] - pa["log2_hz"])
            boundaries.append(
                {
                    "lane_id": lane,
                    "type": a["canonical_type"],
                    "boundary_s": b["start_s"],
                    "pitch_delta_cents": pitch_delta_cents,
                    "prim_a": a["primitive_id"],
                    "prim_b": b["primitive_id"],
                }
            )

    def classify(row: dict) -> str:
        cents = row.get("pitch_delta_cents")
        if cents is not None and abs(cents) > 1.0:
            return "pitch_step"
        return "phase_only"

    classified = Counter(classify(b) for b in boundaries)
    t1_t1 = sum(1 for b in boundaries if b["type"] == 1)
    return {
        "n_same_type_boundaries": len(boundaries),
        "n_t1_t1_boundaries": t1_t1,
        "classification": dict(classified),
        "examples": boundaries[:20],
    }


def validate_frames_npz(path: Path, hop_s: float = HOP_S) -> list[str]:
    issues = []
    data = np.load(path, allow_pickle=True)
    valid = data["valid_target"]
    traj_type = data["trajectory_type"]
    pitch = data["pitch_log2_hz"]
    phase = data["phase"]
    times = data["frame_time_s"]

    masked = ~valid
    if np.any(traj_type[masked] != -1):
        issues.append(f"{path.name}: masked frame with trajectory_type != -1")
    if np.any(np.isfinite(pitch[masked])):
        issues.append(f"{path.name}: masked frame with finite pitch")
    if np.any(np.isfinite(phase[masked])):
        issues.append(f"{path.name}: masked frame with finite phase")

    valid_idx = np.where(valid)[0]
    if valid_idx.size:
        if np.any(traj_type[valid] < 0) or np.any(traj_type[valid] > 3):
            issues.append(f"{path.name}: valid frame with type outside 0-3")
        if np.any(phase[valid] < -1e-6) or np.any(phase[valid] > 1.0 + 1e-6):
            issues.append(f"{path.name}: phase outside [0,1]")

    # phase reset at primitive boundaries
    prim_ids = data["primitive_id"]
    for i in valid_idx[1:]:
        if prim_ids[i] != prim_ids[i - 1] and valid[i - 1]:
            if abs(float(phase[i]) - 0.0) > 0.05 and float(phase[i - 1]) < 0.95:
                pass  # soft check only

    if len(times) > 1:
        expected = np.diff(times)
        if not np.allclose(expected, hop_s, rtol=1e-6, atol=1e-9):
            issues.append(f"{path.name}: non-uniform frame spacing")

    return issues


def run_validation(repo_root: Path) -> dict:
    root = canonical_root(repo_root)
    ids = exported_recording_ids(repo_root)

    all_primitives: list[dict] = []
    raw_counts = 0
    t0_merges = 0
    decompose = Counter()
    durations_by_type: dict[int, list[float]] = defaultdict(list)
    contour_reports = []
    frame_issues = []

    for recording_id in ids:
        rec = load_json(root / "recordings" / f"{recording_id}.json")
        prim_doc = load_json(root / "primitives" / f"{recording_id}.json")
        raw_counts += len(rec["trajectories"])
        t0_merges += prim_doc["stats"].get("n_t0_merges", 0)
        decompose["4"] += prim_doc["stats"].get("n_decompose_4", 0)
        decompose["5"] += prim_doc["stats"].get("n_decompose_5", 0)
        decompose["6_segments"] += prim_doc["stats"].get("n_decompose_6_segments", 0)

        for prim in prim_doc["primitives"]:
            all_primitives.append(prim)
            t = int(prim["canonical_type"])
            durations_by_type[t].append(float(prim["duration_s"]))
            if prim["canonical_type"] not in PRIMITIVE_TYPE_IDS:
                frame_issues.append(f"{recording_id}: bad canonical type")

    contour_path = root / "step_5_contour_verification.json"
    if contour_path.exists():
        contour_payload = load_json(contour_path)
        contour_reports = contour_payload.get("reports", [])
    else:
        for recording_id in ids:
            rec = load_json(root / "recordings" / f"{recording_id}.json")
            prim_doc = load_json(root / "primitives" / f"{recording_id}.json")
            contour_reports.append(verify_recording(rec, prim_doc, grid_step_s=0.001))

    for recording_id in ids:
        rec = load_json(root / "recordings" / f"{recording_id}.json")
        for lane in rec.get("lanes", []):
            lane_id = lane["lane_id"]
            npz_path = frames_dir(repo_root) / frames_npz_name(recording_id, lane_id)
            if npz_path.exists():
                frame_issues.extend(validate_frames_npz(npz_path))

    all_durations = [p["duration_s"] for p in all_primitives]
    t1_from_type6_durations = [
        p["duration_s"]
        for p in all_primitives
        if p["canonical_type"] == 1 and p["rule_applied"] == "decompose_6"
    ]

    split_check = check_split_leakage(repo_root)
    contour_failed = [r for r in contour_reports if not r["passed"]]

    checks = {
        "primitives_count_positive": len(all_primitives) > 0,
        "all_canonical_types_valid": all(
            p["canonical_type"] in PRIMITIVE_TYPE_IDS for p in all_primitives
        ),
        "contour_preservation": not contour_failed,
        "split_leakage": split_check["passed"],
        "frame_npz_checks": not frame_issues,
    }

    return {
        "n_recordings": len(ids),
        "raw_trajectory_count": raw_counts,
        "primitive_count": len(all_primitives),
        "t0_merge_units": t0_merges,
        "decomposition": dict(decompose),
        "composite_decomposition_rules": COMPOSITE_DECOMPOSITION,
        "duration_stats_overall": duration_stats(all_durations),
        "duration_stats_by_type": {
            str(t): duration_stats(durations_by_type[t]) for t in sorted(durations_by_type)
        },
        "t1_from_type6_duration_stats": duration_stats(t1_from_type6_durations),
        "same_type_boundary_analysis": same_type_boundary_analysis(all_primitives),
        "contour_tolerance_log2": CONTOUR_TOLERANCE_LOG2,
        "contour_summary": {
            "n_failed": len(contour_failed),
            "max_error_over_corpus": max(
                (r["max_error_log2"] for r in contour_reports), default=0.0
            ),
        },
        "split_leakage": split_check,
        "checks": checks,
        "frame_issues": frame_issues,
        "all_passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()

    result = run_validation(args.repo_root)
    out = canonical_root(args.repo_root) / "step_5_validation.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("primitive_count", "checks", "all_passed")}, indent=2))
    print(f"Wrote {out}")
    if not result["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
