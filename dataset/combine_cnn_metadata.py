"""Combine per-piece CNN dataset metadata into a single CSV."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

METADATA_COLUMNS = [
    "piece_id",
    "piece_title",
    "traj_index",
    "segment_index",
    "unique_id",
    "idtap_id",
    "idtap_name",
    "label",
    "abs_start",
    "abs_end",
    "duration",
    "image_path",
    "clip_path",
]
SKIPPED_COLUMNS = [
    "piece_id",
    "traj_index",
    "unique_id",
    "idtap_id",
    "idtap_name",
    "reason",
]


def reconstruct_from_images(piece_dir: Path, project_root: Path) -> list[dict[str, object]]:
    images_dir = piece_dir / "images"
    clips_dir = piece_dir / "clips"
    piece_id = piece_dir.name
    pattern = re.compile(r"^(\d{4})_(.+)\.png$")
    rows: list[dict[str, object]] = []

    for image_path in sorted(images_dir.glob("*.png")):
        match = pattern.match(image_path.name)
        if not match:
            continue
        traj_index = int(match.group(1))
        label = match.group(2)
        clip_path = clips_dir / f"{image_path.stem}.wav"
        rows.append(
            {
                "piece_id": piece_id,
                "piece_title": piece_id,
                "traj_index": traj_index,
                "segment_index": 0,
                "unique_id": f"{piece_id}_{traj_index:04d}",
                "idtap_id": "",
                "idtap_name": "",
                "label": label,
                "abs_start": "",
                "abs_end": "",
                "duration": 1.0,
                "image_path": str(image_path.relative_to(project_root)),
                "clip_path": str(clip_path.relative_to(project_root))
                if clip_path.exists()
                else "",
            }
        )
    return rows


def combine_cnn_metadata(cnn_dir: Path, project_root: Path) -> tuple[Path, Path, int, int]:
    all_dir = cnn_dir / "all"
    all_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    all_skipped: list[dict[str, object]] = []
    piece_counts: Counter[str] = Counter()

    for piece_dir in sorted(cnn_dir.iterdir()):
        if not piece_dir.is_dir() or piece_dir.name == "all":
            continue

        metadata_path = piece_dir / "metadata.csv"
        if metadata_path.exists():
            with metadata_path.open(newline="") as f:
                rows = list(csv.DictReader(f))
            all_rows.extend(rows)
            piece_counts[piece_dir.name] = len(rows)
        else:
            rows = reconstruct_from_images(piece_dir, project_root)
            if rows:
                all_rows.extend(rows)
                piece_counts[piece_dir.name] = len(rows)

        skipped_path = piece_dir / "skipped.csv"
        if skipped_path.exists():
            with skipped_path.open(newline="") as f:
                all_skipped.extend(list(csv.DictReader(f)))

    combined_metadata = all_dir / "metadata.csv"
    with combined_metadata.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    combined_skipped = all_dir / "skipped.csv"
    with combined_skipped.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SKIPPED_COLUMNS)
        writer.writeheader()
        writer.writerows(all_skipped)

    return combined_metadata, combined_skipped, len(all_rows), len(piece_counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cnn-dir",
        type=Path,
        default=REPO_ROOT / "output" / "cnn_dataset",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=REPO_ROOT,
    )
    args = parser.parse_args()

    metadata_path, skipped_path, row_count, piece_count = combine_cnn_metadata(
        args.cnn_dir,
        args.project_root,
    )
    print(f"Combined metadata: {metadata_path}")
    print(f"  rows: {row_count}")
    print(f"  pieces: {piece_count}")
    print(f"  skipped: {skipped_path}")


if __name__ == "__main__":
    main()
