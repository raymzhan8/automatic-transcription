"""Additive Type-0 canonicalization overlay.

The trajectory array is never mutated or collapsed. This builds a sibling block
of canonical units that reference raw indices, and back-references
``canonical_unit_id`` / ``canonical_unit_role`` onto each trajectory's derived
block. Every trajectory belongs to exactly one unit, so the canonical view is a
complete tiling rather than a sparse patch.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from .schema import (
    CANONICALIZATION_RULE,
    CANONICALIZATION_RULE_APPLIED,
    CANONICALIZATION_RULE_VERSION,
    canonical_unit_id,
)


class CanonicalUnitError(AssertionError):
    """Raised when a merged unit's span disagrees with its members' durations."""


def build_canonicalization(
    recording_id: str,
    trajectories: Sequence[dict[str, Any]],
    transitions: Sequence[dict[str, Any]],
    *,
    merge_across_phrase_boundary: bool,
    time_tolerance_s: float,
) -> dict[str, Any]:
    """Build the overlay and stamp unit back-references onto each trajectory."""
    by_index = {t["index"]: t for t in trajectories}
    mergeable: dict[int, dict[str, Any]] = {}
    for transition in transitions:
        if not transition["type0_same_pitch_adjacent"]:
            continue
        if transition["crosses_phrase_boundary"] and not merge_across_phrase_boundary:
            continue
        mergeable[transition["from_index"]] = transition

    lanes: dict[str, list[dict[str, Any]]] = {}
    for traj in trajectories:
        lanes.setdefault(traj["lane_id"], []).append(traj)

    units: list[dict[str, Any]] = []
    run_lengths: Counter[int] = Counter()

    for lane, members in sorted(lanes.items()):
        ordered = sorted(members, key=lambda t: t["index"])
        run: list[dict[str, Any]] = []
        for position, traj in enumerate(ordered):
            run.append(traj)
            following = ordered[position + 1] if position + 1 < len(ordered) else None
            transition = mergeable.get(traj["index"])
            continues = (
                following is not None
                and transition is not None
                and transition["to_index"] == following["index"]
            )
            if continues:
                continue
            units.append(_unit(recording_id, lane, run, mergeable, time_tolerance_s))
            run_lengths[len(run)] += 1
            run = []

    _stamp_back_references(units, by_index)

    n_merged = sum(1 for u in units if u["merged"])
    return {
        "rule_version": CANONICALIZATION_RULE_VERSION,
        "rule": CANONICALIZATION_RULE,
        "merge_across_phrase_boundary": merge_across_phrase_boundary,
        "n_raw_trajectories": len(trajectories),
        "n_canonical_units": len(units),
        "n_merged_units": n_merged,
        "run_length_histogram": {str(k): v for k, v in sorted(run_lengths.items())},
        "units": units,
    }


def _unit(
    recording_id: str,
    lane: str,
    run: Sequence[dict[str, Any]],
    mergeable: dict[int, dict[str, Any]],
    time_tolerance_s: float,
) -> dict[str, Any]:
    first = run[0]
    last = run[-1]
    start_s = first["derived"]["start_s"]
    end_s = last["derived"]["end_s"]
    merged = len(run) > 1

    start_pitch = first["derived"].get("start_pitch")
    gaps = [
        mergeable[member["index"]]["gap_s"]
        for member in run[:-1]
        if member["index"] in mergeable
    ]

    if merged:
        member_total = sum(member["derived"]["duration_s"] for member in run)
        if abs(member_total - (end_s - start_s)) > time_tolerance_s:
            raise CanonicalUnitError(
                f"{recording_id} unit at index {first['index']}: member durations sum to "
                f"{member_total} but span is {end_s - start_s}"
            )

    return {
        "canonical_unit_id": canonical_unit_id(recording_id, first["index"]),
        "lane_id": lane,
        "member_indices": [member["index"] for member in run],
        "member_traj_ids": [member["traj_id"] for member in run],
        "merged": merged,
        "rule_applied": CANONICALIZATION_RULE_APPLIED if merged else "singleton",
        "type_id": first["derived"]["type_id"],
        "pitch_key": start_pitch["pitch_key"] if start_pitch else None,
        "start_s": start_s,
        "end_s": end_s,
        "duration_s": end_s - start_s,
        "max_gap_within_unit_s": max((abs(g) for g in gaps), default=0.0),
        "crosses_phrase_boundary": len(
            {member["raw"]["phrase_index"] for member in run}
        )
        > 1,
    }


def _stamp_back_references(
    units: Sequence[dict[str, Any]],
    by_index: dict[int, dict[str, Any]],
) -> None:
    for unit in units:
        indices = unit["member_indices"]
        for position, index in enumerate(indices):
            if len(indices) == 1:
                role = "single"
            elif position == 0:
                role = "run_start"
            elif position == len(indices) - 1:
                role = "run_end"
            else:
                role = "run_member"
            derived = by_index[index]["derived"]
            derived["canonical_unit_id"] = unit["canonical_unit_id"]
            derived["canonical_unit_role"] = role
