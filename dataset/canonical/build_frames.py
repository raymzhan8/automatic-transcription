"""CLI: build framewise .npz targets from primitives."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.canonical.build import exported_recording_ids  # noqa: E402
from dataset.canonical.frames import build_framewise_targets  # noqa: E402
from dataset.canonical.schema import (  # noqa: E402
    FRAMES_SUMMARY_COLUMNS,
    HOP_S,
    frames_dir,
    frames_npz_name,
    index_dir,
    primitives_dir,
    recordings_dir,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-ids", nargs="*")
    parser.add_argument("--hop-s", type=float, default=HOP_S)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()

    ids = args.recording_ids or exported_recording_ids(args.repo_root)
    out_dir = frames_dir(args.repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    for recording_id in ids:
        rec_path = recordings_dir(args.repo_root) / f"{recording_id}.json"
        prim_path = primitives_dir(args.repo_root) / f"{recording_id}.json"
        if not rec_path.exists() or not prim_path.exists():
            print(f"skip missing {recording_id}", file=sys.stderr)
            continue
        recording_doc = json.loads(rec_path.read_text(encoding="utf-8"))
        primitives_doc = json.loads(prim_path.read_text(encoding="utf-8"))
        lane_frames = build_framewise_targets(
            recording_doc, primitives_doc, hop_s=args.hop_s
        )
        pg_id = recording_doc.get("performance", {}).get("performance_group_id")
        for lane, arrays in lane_frames.items():
            fname = frames_npz_name(recording_id, lane)
            relpath = str(Path("frames") / fname)
            out_path = out_dir / fname
            np.savez_compressed(out_path, recording_id=recording_id, lane_id=lane, **arrays)
            n_frames = int(arrays["n_frames"])
            n_valid = int(arrays["n_valid_frames"])
            duration = float(arrays["frame_time_s"][-1] + args.hop_s / 2) if n_frames else 0.0
            summary_rows.append(
                {
                    "recording_id": recording_id,
                    "lane_id": lane,
                    "performance_group_id": pg_id,
                    "n_frames": n_frames,
                    "n_valid_frames": n_valid,
                    "valid_fraction": n_valid / n_frames if n_frames else 0.0,
                    "duration_s": duration,
                    "hop_s": args.hop_s,
                    "npz_relpath": relpath,
                }
            )
            print(f"{recording_id} {lane}: {n_valid}/{n_frames} valid frames -> {fname}")

    csv_path = index_dir(args.repo_root) / "frames_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FRAMES_SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {csv_path} ({len(summary_rows)} rows)")


if __name__ == "__main__":
    main()
