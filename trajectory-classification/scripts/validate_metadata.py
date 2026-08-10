"""Validate trajectory-classification dataset metadata and print summary statistics."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"
RECORDINGS_DIR = PROJECT_ROOT / "data" / "recordings"

REQUIRED_COLUMNS = [
    "example_id",
    "recording_file",
    "start_time",
    "end_time",
    "label",
    "performer_id",
    "recording_id",
    "tonic_hz",
    "confidence",
    "notes",
]

VALID_LABELS = {
    "trajectory_1",
    "trajectory_2",
    "trajectory_3",
    "trajectory_4",
}

VALID_CONFIDENCE = {"high", "medium", "low"}


def load_metadata(path: Path) -> pd.DataFrame:
    """Load metadata CSV from disk.

    Raises:
        FileNotFoundError: If the metadata file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    return pd.read_csv(path)


def check_required_columns(df: pd.DataFrame) -> list[str]:
    """Return errors for any missing required columns."""
    errors: list[str] = []
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
    return errors


def check_unique_example_ids(df: pd.DataFrame) -> list[str]:
    """Return errors for duplicate example_id values."""
    errors: list[str] = []
    if "example_id" not in df.columns:
        return errors

    duplicated = df["example_id"][df["example_id"].duplicated(keep=False)]
    if duplicated.empty:
        return errors

    duplicate_ids = sorted(duplicated.unique())
    errors.append(
        "Duplicate example_id values found: " + ", ".join(str(v) for v in duplicate_ids)
    )
    return errors


def get_recording_duration(recording_path: Path) -> float | None:
    """Return recording duration in seconds, or None if unreadable."""
    try:
        return float(sf.info(recording_path).duration)
    except Exception as exc:
        raise RuntimeError(f"Could not read audio duration: {exc}") from exc


def validate_rows(df: pd.DataFrame, recordings_dir: Path) -> list[str]:
    """Validate each metadata row and collect all errors."""
    errors: list[str] = []

    if not all(col in df.columns for col in REQUIRED_COLUMNS):
        return errors

    for idx, row in df.iterrows():
        row_label = f"Row {idx + 2} (example_id={row['example_id']!r})"

        recording_file = row["recording_file"]
        if pd.isna(recording_file) or str(recording_file).strip() == "":
            errors.append(f"{row_label}: recording_file is missing or empty")
            continue

        recording_path = recordings_dir / str(recording_file)
        recording_exists = recording_path.exists()
        if not recording_exists:
            errors.append(
                f"{row_label}: recording file not found: {recording_path.relative_to(PROJECT_ROOT)}"
            )

        start_time = pd.to_numeric(row["start_time"], errors="coerce")
        end_time = pd.to_numeric(row["end_time"], errors="coerce")

        if pd.isna(start_time):
            errors.append(f"{row_label}: start_time is not numeric ({row['start_time']!r})")
        if pd.isna(end_time):
            errors.append(f"{row_label}: end_time is not numeric ({row['end_time']!r})")

        if not pd.isna(start_time) and start_time < 0:
            errors.append(f"{row_label}: start_time must be nonnegative (got {start_time})")

        if not pd.isna(start_time) and not pd.isna(end_time) and end_time <= start_time:
            errors.append(
                f"{row_label}: end_time must be greater than start_time "
                f"(got start_time={start_time}, end_time={end_time})"
            )

        label = row["label"]
        if pd.isna(label) or str(label) not in VALID_LABELS:
            errors.append(
                f"{row_label}: invalid label {label!r}; "
                f"must be one of {sorted(VALID_LABELS)}"
            )

        confidence = row["confidence"]
        if pd.isna(confidence) or str(confidence) not in VALID_CONFIDENCE:
            errors.append(
                f"{row_label}: invalid confidence {confidence!r}; "
                f"must be one of {sorted(VALID_CONFIDENCE)}"
            )

        if (
            recording_exists
            and not pd.isna(start_time)
            and not pd.isna(end_time)
            and end_time > start_time
        ):
            try:
                duration = get_recording_duration(recording_path)
            except RuntimeError as exc:
                errors.append(f"{row_label}: {exc}")
                continue

            if end_time > duration:
                errors.append(
                    f"{row_label}: end_time ({end_time}) exceeds recording duration "
                    f"({duration:.3f}s) for {recording_file}"
                )

    return errors


def print_summary(df: pd.DataFrame) -> None:
    """Print readable dataset summary statistics."""
    print("\n=== Dataset Summary ===")
    print(f"Number of examples: {len(df)}")

    if "label" in df.columns and not df.empty:
        print("\nExamples per class:")
        for label, count in df["label"].value_counts().sort_index().items():
            print(f"  {label}: {count}")

    if "confidence" in df.columns and not df.empty:
        print("\nExamples per confidence level:")
        for level, count in df["confidence"].value_counts().sort_index().items():
            print(f"  {level}: {count}")

    if "recording_file" in df.columns:
        print(f"\nUnique recordings: {df['recording_file'].nunique()}")

    if "performer_id" in df.columns:
        print(f"Unique performers: {df['performer_id'].nunique()}")

    if {"start_time", "end_time"}.issubset(df.columns):
        start_times = pd.to_numeric(df["start_time"], errors="coerce")
        end_times = pd.to_numeric(df["end_time"], errors="coerce")
        durations = end_times - start_times
        valid_durations = durations[durations > 0]

        print("\nTrajectory duration (seconds):")
        if valid_durations.empty:
            print("  No valid durations to summarize")
        else:
            print(f"  Average: {valid_durations.mean():.3f}")
            print(f"  Minimum: {valid_durations.min():.3f}")
            print(f"  Maximum: {valid_durations.max():.3f}")


def main() -> None:
    """Load metadata, run all validation checks, and exit with appropriate status."""
    errors: list[str] = []

    try:
        df = load_metadata(METADATA_PATH)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    errors.extend(check_required_columns(df))
    errors.extend(check_unique_example_ids(df))
    errors.extend(validate_rows(df, RECORDINGS_DIR))

    if errors:
        print("=== Validation Errors ===")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Validation passed: no errors found.")

    print_summary(df)

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
