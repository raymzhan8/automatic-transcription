"""Survey trajectory type counts across viewable transcriptions (no audio download)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from idtap import Piece, SwaraClient
from idtap.classes.trajectory import Trajectory

from dataset_utils import (
    LABEL_MAP,
    TARGET_IDTAP_IDS,
    iter_trajectory_candidates,
)

ROOT = Path(__file__).parent


def main() -> None:
    client = SwaraClient()
    transcriptions = client.get_viewable_transcriptions()
    print(f"Found {len(transcriptions)} viewable transcriptions\n")

    target_counts: Counter[str] = Counter()
    all_type_counts: Counter[int] = Counter()
    usable_counts: Counter[str] = Counter()
    loaded = 0
    failed = 0
    with_audio = 0

    for entry in transcriptions:
        recording_id = entry["_id"]
        has_audio = bool(entry.get("audioID"))
        if has_audio:
            with_audio += 1

        try:
            piece = Piece.from_json(client.get_piece(recording_id))
            loaded += 1
        except Exception as exc:
            failed += 1
            print(f"  skip {recording_id}: {exc}")
            continue

        for phrase in piece.phrase_grid[0]:
            if 0 < len(phrase.trajectory_grid):
                for traj in phrase.trajectory_grid[0]:
                    all_type_counts[traj.id] += 1

        for candidate in iter_trajectory_candidates(
            piece, recording_id, has_audio=has_audio
        ):
            target_counts[candidate.label] += 1
            if has_audio:
                usable_counts[candidate.label] += 1

    print("=== Target classes (IDs 0-3, duration-filtered) ===")
    for label, cfg in LABEL_MAP.items():
        total = target_counts[label]
        usable = usable_counts[label]
        print(
            f"  {label} (ID {cfg['idtap_id']}, {cfg['name']}): "
            f"{total} total, {usable} with audio"
        )

    print("\n=== All trajectory type counts (track 0, string 0) ===")
    traj_names = Trajectory().names
    for traj_id in sorted(all_type_counts):
        name = traj_names[traj_id] if traj_id < len(traj_names) else "Unknown"
        marker = " *" if traj_id in TARGET_IDTAP_IDS else ""
        print(f"  ID {traj_id:2d} ({name}): {all_type_counts[traj_id]}{marker}")

    print("\n=== Summary ===")
    print(f"  Loaded: {loaded}")
    print(f"  Failed: {failed}")
    print(f"  With audio: {with_audio}")
    print(f"  Target candidates (all): {sum(target_counts.values())}")
    print(f"  Target candidates (with audio): {sum(usable_counts.values())}")


if __name__ == "__main__":
    main()
