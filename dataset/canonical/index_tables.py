"""Flat CSV projections of the canonical documents.

These are pure projections — deletable and regenerable from
``recordings/*.json``. Written with stdlib ``csv`` because ``pandas`` is
currently broken in this environment.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .schema import (
    RECORDINGS_COLUMNS,
    TRAJECTORIES_COLUMNS,
    TRANSITIONS_COLUMNS,
    index_dir,
    recordings_dir,
)


def _recording_row(doc: dict[str, Any]) -> dict[str, Any]:
    performance = doc.get("performance", {})
    coverage = doc.get("coverage", {})
    audio = doc.get("audio", {})
    source = next((f for f in audio.get("files", []) if f["role"] == "source"), None)
    annotation_source = doc.get("annotation_source", {})
    return {
        "recording_id": doc["recording_id"],
        "title": doc.get("title"),
        "soloist": performance.get("soloist"),
        "solo_instrument": performance.get("solo_instrument"),
        "raga_name": performance.get("raga_name"),
        "performance_group_id": performance.get("performance_group_id"),
        "audio_id": audio.get("audio_id"),
        "audio_relpath": source["relpath"] if source else None,
        "audio_duration_s": coverage.get("audio_duration_s"),
        "annotation_start_s": coverage.get("annotation_start_s"),
        "annotation_end_s": coverage.get("annotation_end_s"),
        "audio_vs_annotation_delta_s": coverage.get("audio_vs_annotation_delta_s"),
        "n_lanes": len(doc.get("lanes", [])),
        "n_phrases": len(doc.get("structure", {}).get("phrases", [])),
        "n_trajectories": coverage.get("n_trajectories"),
        "n_canonical_units": doc.get("canonicalization", {}).get("n_canonical_units"),
        "n_silent_annotations": coverage.get("n_silent_annotations"),
        "silent_annotation_duration_s": coverage.get("silent_annotation_duration_s"),
        "silent_annotation_fraction": coverage.get("silent_annotation_fraction"),
        "max_single_silent_annotation_s": coverage.get("max_single_silent_annotation_s"),
        "annotated_non_silent_fraction": coverage.get("annotated_non_silent_fraction"),
        "date_created": annotation_source.get("date_created"),
        "date_modified": annotation_source.get("date_modified"),
        "transcriber_name": annotation_source.get("transcriber_name"),
    }


def _trajectory_rows(doc: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for traj in doc.get("trajectories", []):
        raw = traj["raw"]
        derived = traj["derived"]
        start_pitch = derived.get("start_pitch")
        end_pitch = derived.get("end_pitch")
        rows.append(
            {
                "recording_id": doc["recording_id"],
                "traj_id": traj["traj_id"],
                "index": traj["index"],
                "lane_id": traj["lane_id"],
                "phrase_index": raw.get("phrase_index"),
                "num": raw.get("num"),
                "type_id": derived["type_id"],
                "type_name": derived["type_name"],
                "start_s": derived["start_s"],
                "end_s": derived["end_s"],
                "duration_s": derived["duration_s"],
                "is_silent_annotation": derived["is_silent_annotation"],
                "is_composite_type": derived["is_composite_type"],
                "n_control_points": derived["n_control_points"],
                "pitch_endpoints_available": derived["pitch_endpoints_available"],
                "start_pitch_key": start_pitch["pitch_key"] if start_pitch else None,
                "end_pitch_key": end_pitch["pitch_key"] if end_pitch else None,
                "start_frequency_hz": start_pitch["frequency_hz"] if start_pitch else None,
                "end_frequency_hz": end_pitch["frequency_hz"] if end_pitch else None,
                "pitch_range_cents": derived.get("pitch_range_cents"),
                "group_id": raw.get("group_id"),
                "canonical_unit_id": derived.get("canonical_unit_id"),
                "canonical_unit_role": derived.get("canonical_unit_role"),
            }
        )
    return rows


def _transition_rows(doc: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for transition in doc.get("transitions", []):
        row = {"recording_id": doc["recording_id"]}
        row.update({k: transition.get(k) for k in TRANSITIONS_COLUMNS if k != "recording_id"})
        rows.append(row)
    return rows


def _write(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_index_tables(repo_root: Path) -> dict[str, Path]:
    """Read every canonical recording document and emit the three CSVs."""
    docs_dir = recordings_dir(repo_root)
    out_dir = index_dir(repo_root)

    recording_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []

    for path in sorted(docs_dir.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            doc = json.load(handle)
        recording_rows.append(_recording_row(doc))
        trajectory_rows.extend(_trajectory_rows(doc))
        transition_rows.extend(_transition_rows(doc))

    paths = {
        "recordings": out_dir / "recordings.csv",
        "trajectories": out_dir / "trajectories.csv",
        "transitions": out_dir / "transitions.csv",
    }
    _write(paths["recordings"], RECORDINGS_COLUMNS, recording_rows)
    _write(paths["trajectories"], TRAJECTORIES_COLUMNS, trajectory_rows)
    _write(paths["transitions"], TRANSITIONS_COLUMNS, transition_rows)
    return paths
