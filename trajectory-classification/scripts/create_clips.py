"""Create annotated audio clips from full recordings using metadata.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"
RECORDINGS_DIR = PROJECT_ROOT / "data" / "recordings"
CLIPS_DIR = PROJECT_ROOT / "data" / "clips"


def load_metadata(path: Path) -> pd.DataFrame:
    """Load metadata CSV from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    return pd.read_csv(path)


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to mono by averaging channels."""
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1)


def compute_clip_bounds(
    start_time: float,
    end_time: float,
    context: float,
    duration: float,
) -> tuple[float, float]:
    """Compute clip start/end times with optional context, clamped to recording bounds."""
    clip_start = max(0.0, start_time - context)
    clip_end = min(duration, end_time + context)
    return clip_start, clip_end


def create_clip(
    recording_path: Path,
    clip_path: Path,
    start_time: float,
    end_time: float,
    context: float,
) -> None:
    """Load a recording segment and write a mono clip without normalization."""
    audio, sample_rate = sf.read(recording_path, always_2d=False)
    audio = to_mono(np.asarray(audio, dtype=np.float64))

    duration = len(audio) / sample_rate
    clip_start, clip_end = compute_clip_bounds(start_time, end_time, context, duration)

    start_sample = int(round(clip_start * sample_rate))
    end_sample = int(round(clip_end * sample_rate))
    start_sample = max(0, min(start_sample, len(audio)))
    end_sample = max(start_sample, min(end_sample, len(audio)))

    clip = audio[start_sample:end_sample]
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(clip_path, clip, sample_rate)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create annotated audio clips from metadata.csv"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing clip files",
    )
    parser.add_argument(
        "--context",
        type=float,
        default=0.0,
        help="Seconds of audio to include before and after each annotation (default: 0)",
    )
    return parser.parse_args()


def main() -> None:
    """Read metadata and create clip files for each example."""
    args = parse_args()

    if args.context < 0:
        print("ERROR: --context must be nonnegative")
        sys.exit(1)

    try:
        df = load_metadata(METADATA_PATH)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    created = 0
    skipped = 0
    failed = 0

    for _, row in df.iterrows():
        example_id = str(row["example_id"])
        recording_file = str(row["recording_file"])
        clip_path = CLIPS_DIR / f"{example_id}.wav"

        if clip_path.exists() and not args.overwrite:
            print(f"SKIP: {example_id} (clip already exists; use --overwrite to replace)")
            skipped += 1
            continue

        recording_path = RECORDINGS_DIR / recording_file
        if not recording_path.exists():
            print(f"FAIL: {example_id} (recording not found: {recording_path})")
            failed += 1
            continue

        start_time = pd.to_numeric(row["start_time"], errors="coerce")
        end_time = pd.to_numeric(row["end_time"], errors="coerce")
        if pd.isna(start_time) or pd.isna(end_time):
            print(f"FAIL: {example_id} (non-numeric start_time or end_time)")
            failed += 1
            continue

        try:
            create_clip(
                recording_path=recording_path,
                clip_path=clip_path,
                start_time=float(start_time),
                end_time=float(end_time),
                context=args.context,
            )
            print(f"CREATE: {example_id} -> {clip_path.relative_to(PROJECT_ROOT)}")
            created += 1
        except Exception as exc:
            print(f"FAIL: {example_id} ({exc})")
            failed += 1

    print("\n=== Clip Creation Summary ===")
    print(f"Created: {created}")
    print(f"Skipped: {skipped}")
    print(f"Failed:  {failed}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
