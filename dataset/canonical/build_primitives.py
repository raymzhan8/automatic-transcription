"""CLI: build primitive trajectory tables from canonical recordings."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.canonical.build import exported_recording_ids  # noqa: E402
from dataset.canonical.primitives import build_primitive_trajectories  # noqa: E402
from dataset.canonical.schema import (  # noqa: E402
    PRIMITIVES_COLUMNS,
    index_dir,
    primitives_dir,
    recordings_dir,
)


def write_primitives_csv(rows: list[dict], repo_root: Path) -> Path:
    out = index_dir(repo_root) / "primitives.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PRIMITIVES_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out


def primitives_to_csv_rows(doc: dict) -> list[dict]:
    rows = []
    for prim in doc["primitives"]:
        start_pitch = prim.get("start_pitch") or {}
        end_pitch = prim.get("end_pitch") or {}
        rows.append(
            {
                "recording_id": doc["recording_id"],
                "primitive_id": prim["primitive_id"],
                "lane_id": prim["lane_id"],
                "seq": prim["seq"],
                "canonical_type": prim["canonical_type"],
                "source_raw_type": prim["source_raw_type"],
                "source_subsegment_index": prim.get("source_subsegment_index"),
                "source_raw_trajectory_ids": "|".join(
                    prim["source_raw_trajectory_ids"]
                ),
                "rule_applied": prim["rule_applied"],
                "start_s": prim["start_s"],
                "end_s": prim["end_s"],
                "duration_s": prim["duration_s"],
                "start_pitch_key": start_pitch.get("pitch_key"),
                "end_pitch_key": end_pitch.get("pitch_key"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recording-ids",
        nargs="*",
        help="Limit to these recording IDs (default: all exported)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
    )
    args = parser.parse_args()

    ids = args.recording_ids or exported_recording_ids(args.repo_root)
    out_dir = primitives_dir(args.repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for recording_id in ids:
        path = recordings_dir(args.repo_root) / f"{recording_id}.json"
        if not path.exists():
            print(f"skip missing recording {recording_id}", file=sys.stderr)
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        prim_doc = build_primitive_trajectories(doc)
        out_path = out_dir / f"{recording_id}.json"
        out_path.write_text(
            json.dumps(prim_doc, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        all_rows.extend(primitives_to_csv_rows(prim_doc))
        print(
            f"{recording_id}: {len(prim_doc['primitives'])} primitives "
            f"(raw {prim_doc['stats']['n_raw_trajectories']})"
        )

    csv_path = write_primitives_csv(all_rows, args.repo_root)
    print(f"Wrote {len(all_rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
