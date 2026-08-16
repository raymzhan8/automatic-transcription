"""Build Phase 1 trajectory classification dataset from Swara Studio transcriptions."""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from idtap import Piece, SwaraClient  # noqa: E402
from scipy.io import wavfile  # noqa: E402

from inventory.dataset_utils import (  # noqa: E402
    CLIP_DURATION,
    LABEL_MAP,
    TrajectoryCandidate,
    audio_dir,
    cache_dir,
    data_dir,
    iter_trajectory_candidates,
)
AUDIO_FORMAT = "wav"
MAX_PER_CLASS = 100
MIN_PER_CLASS = 50
RANDOM_SEED = 42


def collect_candidates(client: SwaraClient) -> list[TrajectoryCandidate]:
    transcriptions = client.get_viewable_transcriptions()
    candidates: list[TrajectoryCandidate] = []
    seen: set[tuple[str, str]] = set()

    for entry in transcriptions:
        recording_id = entry["_id"]
        if not entry.get("audioID"):
            continue

        try:
            piece = Piece.from_json(client.get_piece(recording_id))
        except Exception as exc:
            print(f"  skip {recording_id}: {exc}")
            continue

        for candidate in iter_trajectory_candidates(
            piece, recording_id, has_audio=True
        ):
            key = (candidate.recording_id, candidate.traj.unique_id)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)

    random.seed(RANDOM_SEED)
    random.shuffle(candidates)
    return candidates


def select_and_build(
    client: SwaraClient,
    candidates: list[TrajectoryCandidate],
    clips_dir: Path,
    recordings_cache: Path,
) -> tuple[list[dict[str, str | float]], Counter[str]]:
    counts: Counter[str] = Counter()
    rows: list[dict[str, str | float]] = []
    piece_cache: dict[str, Piece] = {}
    example_idx = 0

    for candidate in candidates:
        if counts[candidate.label] >= MAX_PER_CLASS:
            continue
        if all(counts[label] >= MAX_PER_CLASS for label in LABEL_MAP):
            break

        try:
            source_path = ensure_recording_cached(
                client,
                piece_cache,
                candidate.recording_id,
                recordings_cache,
            )
            example_idx += 1
            filename = f"example_{example_idx:04d}.wav"
            dest_path = clips_dir / filename
            slice_and_write_clip(
                source_path,
                candidate.abs_start,
                CLIP_DURATION,
                dest_path,
            )
        except Exception as exc:
            print(
                f"  skip {candidate.recording_id} "
                f"({candidate.label}, traj {candidate.traj.unique_id[:8]}): {exc}"
            )
            continue

        counts[candidate.label] += 1
        rows.append(
            {
                "file": filename,
                "label": candidate.label,
                "start_time": 0.0,
                "end_time": CLIP_DURATION,
                "tonic_hz": candidate.tonic_hz if candidate.tonic_hz else "",
                "performer_id": candidate.performer_id,
                "recording_id": candidate.recording_id,
            }
        )

        if example_idx % 25 == 0:
            print(f"  Wrote {example_idx} clips")

    return rows, counts


def ensure_recording_cached(
    client: SwaraClient,
    piece_cache: dict[str, Piece],
    recording_id: str,
    cache_root: Path,
) -> Path:
    cached_path = cache_root / f"{recording_id}.{AUDIO_FORMAT}"
    if cached_path.exists():
        return cached_path

    if recording_id not in piece_cache:
        piece_cache[recording_id] = Piece.from_json(client.get_piece(recording_id))

    piece = piece_cache[recording_id]
    saved = client.download_and_save_transcription_audio(
        piece,
        format=AUDIO_FORMAT,
        filepath=str(cache_root),
        filename=f"{recording_id}.{AUDIO_FORMAT}",
    )
    if saved is None:
        raise RuntimeError(f"Failed to download audio for {recording_id}")
    return Path(saved)


def slice_and_write_clip(
    source_path: Path,
    abs_start: float,
    clip_duration: float,
    dest_path: Path,
) -> None:
    sample_rate, audio = wavfile.read(source_path)
    start_sample = int(abs_start * sample_rate)
    end_sample = int((abs_start + clip_duration) * sample_rate)
    start_sample = max(0, start_sample)
    end_sample = min(len(audio), end_sample)
    if end_sample - start_sample < int(clip_duration * sample_rate * 0.99):
        raise ValueError(
            f"Insufficient audio for {clip_duration}s clip at {abs_start:.3f}s "
            f"in {source_path.name}"
        )
    clip = audio[start_sample:end_sample]
    wavfile.write(dest_path, sample_rate, clip)


def write_label_map(out_dir: Path) -> None:
    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(LABEL_MAP, f, indent=2)
        f.write("\n")


def write_metadata(out_dir: Path, rows: list[dict[str, str | float]]) -> None:
    fieldnames = [
        "file",
        "label",
        "start_time",
        "end_time",
        "tonic_hz",
        "performer_id",
        "recording_id",
    ]
    with open(out_dir / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    client = SwaraClient()
    out_dir = data_dir(ROOT)
    clips_dir = audio_dir(ROOT)
    recordings_cache = cache_dir(ROOT)

    out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    recordings_cache.mkdir(parents=True, exist_ok=True)

    write_label_map(out_dir)

    print("Collecting candidates...")
    candidates = collect_candidates(client)
    print(f"  Found {len(candidates)} unique candidates with audio")

    rows, counts = select_and_build(
        client, candidates, clips_dir, recordings_cache
    )
    print(f"  Built {len(rows)} examples:")
    for label in sorted(LABEL_MAP):
        print(f"    {label}: {counts[label]}")

    write_metadata(out_dir, rows)

    print("\n=== Build complete ===")
    print(f"  Output: {out_dir}")
    print(f"  Clips: {len(rows)}")
    for label in sorted(LABEL_MAP):
        count = counts[label]
        status = "ok" if count >= MIN_PER_CLASS else "below target"
        print(f"    {label}: {count} ({status})")


if __name__ == "__main__":
    main()
