"""Verify raw vs canonical primitive pitch contours."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.canonical.build import exported_recording_ids  # noqa: E402
from dataset.canonical.contour import compare_lane_contours  # noqa: E402
from dataset.canonical.primitives import primitives_by_lane  # noqa: E402
from dataset.canonical.schema import CONTOUR_TOLERANCE_LOG2, canonical_root  # noqa: E402


def audio_duration_s(recording_doc: dict) -> float:
    files = recording_doc.get("audio", {}).get("files") or []
    for item in files:
        if item.get("role") == "source":
            return float(item["duration_s"])
    return float(recording_doc.get("coverage", {}).get("audio_duration_s") or 0.0)


def verify_recording(recording_doc: dict, primitives_doc: dict, *, grid_step_s: float) -> dict:
    duration = audio_duration_s(recording_doc)
    lanes: dict[str, list] = {}
    for traj in recording_doc["trajectories"]:
        lanes.setdefault(traj["lane_id"], []).append(traj)

    prim_lanes = primitives_by_lane(primitives_doc)
    lane_reports = []
    max_error = 0.0
    failures = []

    for lane, trajs in sorted(lanes.items()):
        report = compare_lane_contours(
            recording_doc,
            lane,
            trajs,
            prim_lanes.get(lane, []),
            duration_s=duration,
            grid_step_s=grid_step_s,
        )
        lane_reports.append(report)
        lane_max = report["stats"]["max"]
        if lane_max is not None:
            max_error = max(max_error, lane_max)
            if lane_max > CONTOUR_TOLERANCE_LOG2:
                failures.append(
                    {
                        "lane_id": lane,
                        "max": lane_max,
                        "worst": report["worst"],
                    }
                )

    return {
        "recording_id": recording_doc["recording_id"],
        "duration_s": duration,
        "grid_step_s": grid_step_s,
        "tolerance_log2": CONTOUR_TOLERANCE_LOG2,
        "max_error_log2": max_error,
        "passed": not failures,
        "failures": failures,
        "lanes": lane_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-ids", nargs="*")
    parser.add_argument("--grid-step-s", type=float, default=0.001)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    root = canonical_root(args.repo_root)
    ids = args.recording_ids or exported_recording_ids(args.repo_root)
    reports = []
    any_failed = False

    for recording_id in ids:
        rec_path = root / "recordings" / f"{recording_id}.json"
        prim_path = root / "primitives" / f"{recording_id}.json"
        if not rec_path.exists() or not prim_path.exists():
            print(f"skip missing {recording_id}", file=sys.stderr)
            continue
        recording_doc = json.loads(rec_path.read_text(encoding="utf-8"))
        primitives_doc = json.loads(prim_path.read_text(encoding="utf-8"))
        report = verify_recording(
            recording_doc, primitives_doc, grid_step_s=args.grid_step_s
        )
        reports.append(report)
        status = "PASS" if report["passed"] else "FAIL"
        print(
            f"{recording_id}: {status} max_error={report['max_error_log2']:.2e}"
        )
        if not report["passed"]:
            any_failed = True
            if args.fail_fast:
                break

    out = {
        "tolerance_log2": CONTOUR_TOLERANCE_LOG2,
        "grid_step_s": args.grid_step_s,
        "n_recordings": len(reports),
        "n_failed": sum(1 for r in reports if not r["passed"]),
        "reports": reports,
    }
    out_path = root / "step_5_contour_verification.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")

    if any_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
