"""Per-lane transitions between consecutive trajectories.

One entry per consecutive pair *within a lane*, never across lanes: a second
string or a second track is a simultaneous voice, not a continuation.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from .pitch import cents_between, log2_between
from .schema import FIXED_TRAJECTORY_ID


def interval_relation(gap_s: float, tolerance_s: float) -> str:
    if gap_s > tolerance_s:
        return "gap"
    if gap_s < -tolerance_s:
        return "overlap"
    return "meets"


def build_transitions(
    trajectories: Sequence[dict[str, Any]],
    *,
    section_of_phrase: Callable[[int, int], int],
    time_tolerance_s: float,
) -> list[dict[str, Any]]:
    """Consecutive-pair records, computed independently inside each lane."""
    lanes: dict[str, list[dict[str, Any]]] = {}
    for traj in trajectories:
        lanes.setdefault(traj["lane_id"], []).append(traj)

    out: list[dict[str, Any]] = []
    for lane, members in lanes.items():
        ordered = sorted(members, key=lambda t: t["index"])
        for previous, current in zip(ordered, ordered[1:]):
            out.append(
                _transition(
                    previous,
                    current,
                    lane=lane,
                    section_of_phrase=section_of_phrase,
                    time_tolerance_s=time_tolerance_s,
                )
            )
    out.sort(key=lambda t: (t["lane_id"], t["from_index"]))
    return out


def _transition(
    previous: dict[str, Any],
    current: dict[str, Any],
    *,
    lane: str,
    section_of_phrase: Callable[[int, int], int],
    time_tolerance_s: float,
) -> dict[str, Any]:
    from_derived = previous["derived"]
    to_derived = current["derived"]
    from_raw = previous["raw"]
    to_raw = current["raw"]

    from_end_s = from_derived["end_s"]
    to_start_s = to_derived["start_s"]
    gap_s = to_start_s - from_end_s

    from_pitch = from_derived.get("end_pitch")
    to_pitch = to_derived.get("start_pitch")
    endpoints_available = from_pitch is not None and to_pitch is not None

    from_key = from_pitch["pitch_key"] if from_pitch else None
    to_key = to_pitch["pitch_key"] if to_pitch else None
    same_pitch = bool(endpoints_available and from_key == to_key)

    both_type_0 = (
        from_derived["type_id"] == FIXED_TRAJECTORY_ID
        and to_derived["type_id"] == FIXED_TRAJECTORY_ID
    )
    relation = interval_relation(gap_s, time_tolerance_s)

    from_group = from_raw.get("group_id")
    to_group = to_raw.get("group_id")

    from_section = section_of_phrase(from_raw["track_index"], from_raw["phrase_index"])
    to_section = section_of_phrase(to_raw["track_index"], to_raw["phrase_index"])

    return {
        "from_index": previous["index"],
        "to_index": current["index"],
        "lane_id": lane,
        "order_delta": current["index"] - previous["index"],
        "gap_s": gap_s,
        "interval_relation": relation,
        "overlap_s": max(0.0, -gap_s),
        "from_end_s": from_end_s,
        "to_start_s": to_start_s,
        "crosses_phrase_boundary": from_raw["phrase_index"] != to_raw["phrase_index"],
        "crosses_section_boundary": from_section != to_section,
        "from_type_id": from_derived["type_id"],
        "to_type_id": to_derived["type_id"],
        "pitch_endpoints_available": endpoints_available,
        "from_end_pitch_key": from_key,
        "to_start_pitch_key": to_key,
        "pitch_delta_cents": cents_between(from_pitch, to_pitch),
        "pitch_delta_log2": log2_between(from_pitch, to_pitch),
        "same_pitch": same_pitch,
        "both_type_0": both_type_0,
        "type0_same_pitch_adjacent": bool(
            both_type_0 and same_pitch and relation == "meets"
        ),
        "same_group_id": bool(
            from_group is not None and to_group is not None and from_group == to_group
        ),
    }


def non_meets_transitions(transitions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [t for t in transitions if t["interval_relation"] != "meets"]
