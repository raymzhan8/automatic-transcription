"""Generate 10 ms framewise targets from canonical primitives."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .contour import (
    compute_log2_at_time,
    primitive_trajectory_at_time,
    trajectories_by_index,
)
from .schema import FRAME_MEMBERSHIP, HOP_S, MASKED_TRAJECTORY_TYPE, PHASE_DEFINITION


def frame_centers(duration_s: float, hop_s: float = HOP_S) -> np.ndarray:
    """Half-open grid: centers while t < duration_s."""
    if duration_s <= 0:
        return np.array([], dtype=np.float64)
    n = int(math.floor((duration_s - hop_s / 2.0) / hop_s)) + 1
    if n <= 0:
        return np.array([], dtype=np.float64)
    k = np.arange(n, dtype=np.float64)
    return (k + 0.5) * hop_s


def build_lane_frames(
    recording_doc: dict[str, Any],
    lane_primitives: list[dict[str, Any]],
    *,
    duration_s: float,
    hop_s: float = HOP_S,
) -> dict[str, Any]:
    """Build framewise arrays for one lane."""
    times = frame_centers(duration_s, hop_s=hop_s)
    n = len(times)
    by_idx = trajectories_by_index(recording_doc)

    valid = np.zeros(n, dtype=bool)
    traj_type = np.full(n, MASKED_TRAJECTORY_TYPE, dtype=np.int8)
    pitch = np.full(n, np.nan, dtype=np.float32)
    phase = np.full(n, np.nan, dtype=np.float32)
    prim_ids = np.array([""] * n, dtype=object)
    source_ids = np.array([""] * n, dtype=object)

    ordered = sorted(lane_primitives, key=lambda p: (p["start_s"], p["seq"]))
    obj_cache: dict[int, Any] = {}
    for prim in ordered:
        mask = (times >= prim["start_s"]) & (times < prim["end_s"])
        if not np.any(mask):
            continue
        valid[mask] = True
        traj_type[mask] = int(prim["canonical_type"])
        prim_ids[mask] = prim["primitive_id"]
        source_ids[mask] = "|".join(prim["source_raw_trajectory_ids"])
        duration = prim["duration_s"]
        if duration > 0:
            phase[mask] = np.clip((times[mask] - prim["start_s"]) / duration, 0.0, 1.0)
        for i in np.where(mask)[0]:
            t = float(times[i])
            hit = primitive_trajectory_at_time(prim, by_idx, t)
            if hit is None:
                valid[i] = False
                traj_type[i] = MASKED_TRAJECTORY_TYPE
                pitch[i] = np.nan
                phase[i] = np.nan
                continue
            entry, x = hit
            val = compute_log2_at_time(recording_doc, entry, x, cache=obj_cache)
            if val is None:
                valid[i] = False
                traj_type[i] = MASKED_TRAJECTORY_TYPE
                pitch[i] = np.nan
                phase[i] = np.nan
            else:
                pitch[i] = val

    dp_dt = np.full(n, np.nan, dtype=np.float32)
    if n > 1:
        dp_dt[1:] = (pitch[1:] - pitch[:-1]) / hop_s

    boundary_starts: list[int] = []
    boundary_ends: list[int] = []
    for prim in ordered:
        start_idx = int(np.searchsorted(times, prim["start_s"], side="left"))
        end_idx = int(np.searchsorted(times, prim["end_s"], side="left")) - 1
        if start_idx < n:
            boundary_starts.append(start_idx)
        if end_idx >= 0:
            boundary_ends.append(end_idx)

    return {
        "frame_time_s": times,
        "valid_target": valid,
        "trajectory_type": traj_type,
        "pitch_log2_hz": pitch,
        "phase": phase,
        "primitive_id": prim_ids,
        "source_raw_trajectory_ids": source_ids,
        "dp_dt_log2_hz_per_s": dp_dt,
        "boundary_start_frame_indices": np.array(boundary_starts, dtype=np.int32),
        "boundary_end_frame_indices": np.array(boundary_ends, dtype=np.int32),
        "hop_s": hop_s,
        "frame_membership": FRAME_MEMBERSHIP,
        "phase_definition": PHASE_DEFINITION,
        "n_frames": n,
        "n_valid_frames": int(valid.sum()),
    }


def build_framewise_targets(
    recording_doc: dict[str, Any],
    primitives_doc: dict[str, Any],
    *,
    hop_s: float = HOP_S,
) -> dict[str, dict[str, Any]]:
    duration = _audio_duration(recording_doc)
    lanes: dict[str, list[dict[str, Any]]] = {}
    for prim in primitives_doc["primitives"]:
        lanes.setdefault(prim["lane_id"], []).append(prim)

    out: dict[str, dict[str, Any]] = {}
    for lane, prims in sorted(lanes.items()):
        out[lane] = build_lane_frames(
            recording_doc, prims, duration_s=duration, hop_s=hop_s
        )
    return out


def _audio_duration(recording_doc: dict[str, Any]) -> float:
    files = recording_doc.get("audio", {}).get("files") or []
    for item in files:
        if item.get("role") == "source":
            return float(item["duration_s"])
    return float(recording_doc.get("coverage", {}).get("audio_duration_s") or 0.0)
