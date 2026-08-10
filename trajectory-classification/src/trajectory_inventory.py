"""Extract, validate, and export IDTAP trajectory inventories."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from idtap.classes.piece import Piece
from idtap.classes.pitch import Pitch
from idtap.classes.trajectory import Trajectory

INVENTORY_COLUMNS = [
    "piece_id",
    "piece_title",
    "trajectory_uid",
    "idtap_trajectory_id",
    "idtap_trajectory_name",
    "trajectory_duration",
    "trajectory_start_time",
    "trajectory_end_time",
    "track_index",
    "string_index",
    "phrase_index",
    "instrument",
    "fundamental",
    "audio_id",
    "audio_filename",
    "performer",
    "raga",
    "pitches",
    "log_frequencies",
    "tags",
]

COUNTS_COLUMNS = [
    "idtap_trajectory_id",
    "idtap_trajectory_name",
    "count",
    "number_of_pieces",
    "mean_duration",
    "median_duration",
    "min_duration",
    "max_duration",
]

FAILED_COLUMNS = [
    "piece_id",
    "piece_title",
    "error_type",
    "error_message",
]

VALID_TRAJECTORY_IDS = set(range(14))


@dataclass
class InventoryWarning:
    """A non-fatal validation warning for one trajectory or piece."""

    piece_id: str
    message: str
    trajectory_uid: str | None = None


@dataclass
class FailedTranscription:
    """A transcription that could not be processed."""

    piece_id: str
    piece_title: str
    error_type: str
    error_message: str


@dataclass
class InventoryResult:
    """Collected inventory rows, warnings, and failures."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[InventoryWarning] = field(default_factory=list)
    failures: list[FailedTranscription] = field(default_factory=list)


def performer_from_piece(piece: Piece) -> str:
    """Return a human-readable performer label from piece metadata."""
    if piece.soloist:
        return str(piece.soloist)
    parts = [part for part in (piece.given_name, piece.family_name) if part]
    if parts:
        return " ".join(parts)
    if piece.name:
        return str(piece.name)
    return ""


def raga_name_from_piece(piece: Piece) -> str:
    """Return the raga name when available."""
    if piece.raga and piece.raga.name:
        return str(piece.raga.name)
    return ""


def fundamental_from_piece(piece: Piece) -> float | None:
    """Return the piece tonic in Hz when available."""
    if piece.raga and piece.raga.fundamental is not None:
        return float(piece.raga.fundamental)
    return None


def instrument_name(piece: Piece, track_index: int) -> str:
    """Return the instrument name for a track index."""
    if track_index >= len(piece.instrumentation):
        return ""
    instrument = piece.instrumentation[track_index]
    return getattr(instrument, "name", str(instrument))


def serialize_pitch(pitch: Pitch) -> dict[str, Any]:
    """Convert a Pitch object to a JSON-serializable dictionary."""
    try:
        return pitch.to_json()
    except Exception:
        return {
            "swara": getattr(pitch, "swara", None),
            "oct": getattr(pitch, "oct", None),
            "raised": getattr(pitch, "raised", None),
            "frequency": getattr(pitch, "frequency", None),
        }


def serialize_json_field(value: Any) -> str:
    """Serialize complex trajectory fields for CSV export."""
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=True)


def max_string_count(piece: Piece, track_index: int) -> int:
    """Return the maximum number of string lanes used on a track."""
    phrases = piece.phrase_grid[track_index]
    if not phrases:
        return 0
    return max(len(phrase.trajectory_grid) for phrase in phrases)


def iter_trajectory_records(
    piece: Piece,
    *,
    piece_id: str,
    piece_title: str,
    audio_filename: str = "",
) -> list[dict[str, Any]]:
    """Extract inventory rows by walking all tracks, strings, and phrases."""
    records: list[dict[str, Any]] = []
    fundamental = fundamental_from_piece(piece)
    performer = performer_from_piece(piece)
    raga = raga_name_from_piece(piece)
    audio_id = piece.audio_id or ""

    for track_index in range(len(piece.phrase_grid)):
        string_count = max_string_count(piece, track_index)
        for string_index in range(string_count):
            trajectories = piece.all_trajectories(
                inst=track_index,
                string_idx=string_index,
            )
            start_times = piece.traj_start_times(
                inst=track_index,
                string_idx=string_index,
            )

            phrase_indices = _phrase_indices_for_trajectories(
                piece,
                track_index=track_index,
                string_index=string_index,
            )

            pair_count = min(
                len(trajectories),
                len(start_times),
                len(phrase_indices),
            )
            for idx in range(pair_count):
                trajectory = trajectories[idx]
                start_time = float(start_times[idx])
                end_time = start_time + float(trajectory.dur_tot)
                records.append(
                    _trajectory_record(
                        piece=piece,
                        piece_id=piece_id,
                        piece_title=piece_title,
                        trajectory=trajectory,
                        track_index=track_index,
                        string_index=string_index,
                        phrase_index=phrase_indices[idx],
                        start_time=start_time,
                        end_time=end_time,
                        audio_id=audio_id,
                        audio_filename=audio_filename,
                        performer=performer,
                        raga=raga,
                        fundamental=fundamental,
                    )
                )
    return records


def _phrase_indices_for_trajectories(
    piece: Piece,
    *,
    track_index: int,
    string_index: int,
) -> list[int]:
    """Mirror ``all_trajectories`` phrase order to assign phrase indices."""
    phrase_indices: list[int] = []
    for phrase_index, phrase in enumerate(piece.phrase_grid[track_index]):
        if string_index < len(phrase.trajectory_grid):
            phrase_indices.extend([phrase_index] * len(phrase.trajectory_grid[string_index]))
    return phrase_indices


def _trajectory_record(
    *,
    piece: Piece,
    piece_id: str,
    piece_title: str,
    trajectory: Trajectory,
    track_index: int,
    string_index: int,
    phrase_index: int,
    start_time: float,
    end_time: float,
    audio_id: str,
    audio_filename: str,
    performer: str,
    raga: str,
    fundamental: float | None,
) -> dict[str, Any]:
    """Build one inventory row for a trajectory."""
    pitches = [serialize_pitch(p) for p in trajectory.pitches]
    log_frequencies = list(trajectory.log_freqs)
    tags = list(trajectory.tags) if trajectory.tags is not None else []

    return {
        "piece_id": piece_id,
        "piece_title": piece_title,
        "trajectory_uid": trajectory.unique_id,
        "idtap_trajectory_id": trajectory.id,
        "idtap_trajectory_name": trajectory.name_,
        "trajectory_duration": float(trajectory.dur_tot),
        "trajectory_start_time": start_time,
        "trajectory_end_time": end_time,
        "track_index": track_index,
        "string_index": string_index,
        "phrase_index": phrase_index,
        "instrument": instrument_name(piece, track_index),
        "fundamental": fundamental if fundamental is not None else "",
        "audio_id": audio_id,
        "audio_filename": audio_filename,
        "performer": performer,
        "raga": raga,
        "pitches": serialize_json_field(pitches),
        "log_frequencies": serialize_json_field(log_frequencies),
        "tags": serialize_json_field(tags),
    }


def validate_trajectory_lengths(
    piece: Piece,
    *,
    piece_id: str,
    warnings: list[InventoryWarning],
) -> None:
    """Warn when trajectory and start-time arrays differ in length."""
    for track_index in range(len(piece.phrase_grid)):
        string_count = max_string_count(piece, track_index)
        for string_index in range(string_count):
            trajectories = piece.all_trajectories(
                inst=track_index,
                string_idx=string_index,
            )
            start_times = piece.traj_start_times(
                inst=track_index,
                string_idx=string_index,
            )
            if len(trajectories) != len(start_times):
                warnings.append(
                    InventoryWarning(
                        piece_id=piece_id,
                        message=(
                            f"track {track_index}, string {string_index}: "
                            f"{len(trajectories)} trajectories but "
                            f"{len(start_times)} start times"
                        ),
                    )
                )


def validate_inventory_rows(
    rows: list[dict[str, Any]],
    *,
    warnings: list[InventoryWarning],
) -> None:
    """Validate exported rows and append warnings for inconsistent data."""
    id_to_names: dict[int, set[str]] = {}

    for row in rows:
        piece_id = str(row["piece_id"])
        trajectory_uid = str(row["trajectory_uid"])
        traj_id = int(row["idtap_trajectory_id"])
        traj_name = str(row["idtap_trajectory_name"])
        duration = float(row["trajectory_duration"])
        start_time = float(row["trajectory_start_time"])
        end_time = float(row["trajectory_end_time"])

        if traj_id not in VALID_TRAJECTORY_IDS:
            warnings.append(
                InventoryWarning(
                    piece_id=piece_id,
                    trajectory_uid=trajectory_uid,
                    message=f"trajectory ID {traj_id} is outside the expected range 0-13",
                )
            )

        id_to_names.setdefault(traj_id, set()).add(traj_name)

        if duration <= 0:
            warnings.append(
                InventoryWarning(
                    piece_id=piece_id,
                    trajectory_uid=trajectory_uid,
                    message=(
                        f"trajectory ID {traj_id} ({traj_name}) has non-positive "
                        f"duration {duration}"
                    ),
                )
            )

        if not math.isfinite(start_time) or not math.isfinite(end_time):
            warnings.append(
                InventoryWarning(
                    piece_id=piece_id,
                    trajectory_uid=trajectory_uid,
                    message=(
                        f"non-finite timing: start={start_time}, end={end_time}"
                    ),
                )
            )

    for traj_id, names in sorted(id_to_names.items()):
        if len(names) > 1:
            warnings.append(
                InventoryWarning(
                    piece_id="*",
                    message=(
                        f"inconsistent names for trajectory ID {traj_id}: "
                        f"{sorted(names)}"
                    ),
                )
            )


def extract_piece_inventory(
    piece: Piece,
    *,
    piece_id: str,
    piece_title: str,
    audio_filename: str = "",
) -> tuple[list[dict[str, Any]], list[InventoryWarning]]:
    """Extract inventory rows and validation warnings for one piece."""
    warnings: list[InventoryWarning] = []
    validate_trajectory_lengths(piece, piece_id=piece_id, warnings=warnings)
    rows = iter_trajectory_records(
        piece,
        piece_id=piece_id,
        piece_title=piece_title,
        audio_filename=audio_filename,
    )
    validate_inventory_rows(rows, warnings=warnings)
    return rows, warnings


def build_counts_summary(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate trajectory counts and duration statistics by IDTAP type."""
    if not rows:
        return pd.DataFrame(columns=COUNTS_COLUMNS)

    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(["idtap_trajectory_id", "idtap_trajectory_name"], dropna=False)
        .agg(
            count=("trajectory_uid", "size"),
            number_of_pieces=("piece_id", pd.Series.nunique),
            mean_duration=("trajectory_duration", "mean"),
            median_duration=("trajectory_duration", "median"),
            min_duration=("trajectory_duration", "min"),
            max_duration=("trajectory_duration", "max"),
        )
        .reset_index()
        .sort_values(["idtap_trajectory_id", "idtap_trajectory_name"])
    )
    return grouped[COUNTS_COLUMNS]


def write_inventory_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write the detailed trajectory inventory CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=INVENTORY_COLUMNS)
    df.to_csv(path, index=False)


def write_counts_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write the trajectory summary counts CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    build_counts_summary(rows).to_csv(path, index=False)


def write_failures_csv(failures: list[FailedTranscription], path: Path) -> None:
    """Write failed transcription records to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "piece_id": failure.piece_id,
                "piece_title": failure.piece_title,
                "error_type": failure.error_type,
                "error_message": failure.error_message,
            }
            for failure in failures
        ],
        columns=FAILED_COLUMNS,
    )
    df.to_csv(path, index=False)


def print_inventory_summary(rows: list[dict[str, Any]]) -> None:
    """Print a readable summary sorted by trajectory ID."""
    counts = build_counts_summary(rows)
    print("\n=== Trajectory Inventory Summary ===")
    print(f"Total trajectories: {len(rows)}")
    if counts.empty:
        print("No trajectories found.")
        return

    unique_pieces = len({row["piece_id"] for row in rows})
    print(f"Unique pieces: {unique_pieces}")
    print("\nCounts by IDTAP trajectory type:")
    for _, row in counts.iterrows():
        print(
            f"  ID {int(row['idtap_trajectory_id']):2d} "
            f"({row['idtap_trajectory_name']}): "
            f"{int(row['count'])} trajectories across "
            f"{int(row['number_of_pieces'])} pieces, "
            f"mean duration {row['mean_duration']:.3f}s"
        )


def print_warnings(warnings: list[InventoryWarning]) -> None:
    """Print collected validation warnings."""
    if not warnings:
        return
    print("\n=== Validation Warnings ===")
    for warning in warnings:
        prefix = f"[{warning.piece_id}]"
        if warning.trajectory_uid:
            prefix = f"[{warning.piece_id} | {warning.trajectory_uid}]"
        print(f"  - {prefix} {warning.message}")
