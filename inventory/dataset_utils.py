"""Shared helpers for Phase 1 trajectory dataset building."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from idtap.classes.piece import Piece
from idtap.classes.trajectory import Trajectory

TARGET_IDTAP_IDS = {0, 1, 2, 3}

LABEL_MAP = {
    "trajectory_1": {"idtap_id": 0, "name": "Fixed"},
    "trajectory_2": {"idtap_id": 1, "name": "Bend: Simple"},
    "trajectory_3": {"idtap_id": 2, "name": "Bend: Sloped Start"},
    "trajectory_4": {"idtap_id": 3, "name": "Bend: Sloped End"},
}

IDTAP_ID_TO_LABEL = {
    cfg["idtap_id"]: label for label, cfg in LABEL_MAP.items()
}

MIN_DURATION = 0.15
MAX_DURATION = 5.0
CLIP_DURATION = 1.0
INST_TRACK = 0
STRING_IDX = 0


@dataclass(frozen=True)
class TrajectoryCandidate:
    recording_id: str
    traj: Trajectory
    abs_start: float
    abs_end: float
    label: str
    performer_id: str
    tonic_hz: Optional[float]
    has_audio: bool


def slugify_performer(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    slug = slug.strip("_")
    return slug or "unknown"


def performer_id_from_piece(piece: Piece) -> str:
    if piece.soloist:
        return slugify_performer(piece.soloist)
    if piece.given_name or piece.family_name:
        name = " ".join(
            part for part in (piece.given_name, piece.family_name) if part
        )
        return slugify_performer(name)
    if piece.name:
        return slugify_performer(piece.name)
    return "unknown"


def tonic_hz_from_piece(piece: Piece) -> Optional[float]:
    if piece.raga and piece.raga.fundamental:
        return float(piece.raga.fundamental)
    return None


def iter_trajectory_candidates(
    piece: Piece,
    recording_id: str,
    has_audio: bool,
    *,
    inst_track: int = INST_TRACK,
    string_idx: int = STRING_IDX,
) -> Iterator[TrajectoryCandidate]:
    performer_id = performer_id_from_piece(piece)
    tonic_hz = tonic_hz_from_piece(piece)

    for phrase in piece.phrase_grid[inst_track]:
        phrase_start = phrase.start_time or 0.0
        if string_idx >= len(phrase.trajectory_grid):
            continue
        for traj in phrase.trajectory_grid[string_idx]:
            if traj.id not in TARGET_IDTAP_IDS:
                continue
            if not (MIN_DURATION <= traj.dur_tot <= MAX_DURATION):
                continue
            abs_start = phrase_start + (traj.start_time or 0.0)
            abs_end = abs_start + traj.dur_tot
            if abs_start + CLIP_DURATION > (piece.dur_tot or 0.0):
                continue
            yield TrajectoryCandidate(
                recording_id=recording_id,
                traj=traj,
                abs_start=abs_start,
                abs_end=abs_end,
                label=IDTAP_ID_TO_LABEL[traj.id],
                performer_id=performer_id,
                tonic_hz=tonic_hz,
                has_audio=has_audio,
            )


def data_dir(root: Path) -> Path:
    return root / "data"


def audio_dir(root: Path) -> Path:
    return data_dir(root) / "audio"


def cache_dir(root: Path) -> Path:
    return root / ".cache" / "recordings"
