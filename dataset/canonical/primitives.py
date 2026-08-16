"""Build four-primitive canonical trajectories from raw recording documents.

Raw trajectories in ``recordings/*.json`` are never modified. Output lives in
``primitives/<recording_id>.json``.
"""

from __future__ import annotations

from typing import Any, Sequence

from .decompose import iter_primitive_segments, rule_for_source_type
from .schema import (
    FIXED_TRAJECTORY_ID,
    PRIMITIVES_RULE_VERSION,
    SKIP_PRIMITIVE_SOURCE_TYPES,
    primitive_id,
)


class PrimitiveBuildError(AssertionError):
    """Raised when generated primitives violate invariants."""


def build_primitive_trajectories(recording_doc: dict[str, Any]) -> dict[str, Any]:
    """Return a primitives document for one recording."""
    recording_id = recording_doc["recording_id"]
    trajectories = recording_doc["trajectories"]
    transitions = recording_doc["transitions"]
    mergeable = {
        t["from_index"]: t
        for t in transitions
        if t.get("type0_same_pitch_adjacent")
    }

    lanes: dict[str, list[dict[str, Any]]] = {}
    for traj in trajectories:
        lanes.setdefault(traj["lane_id"], []).append(traj)

    all_primitives: list[dict[str, Any]] = []
    stats = {
        "n_raw_trajectories": len(trajectories),
        "n_masked_raw": 0,
        "n_t0_merges": 0,
        "n_decompose_4": 0,
        "n_decompose_5": 0,
        "n_decompose_6_segments": 0,
        "by_canonical_type": {0: 0, 1: 0, 2: 0, 3: 0},
    }

    for lane, members in sorted(lanes.items()):
        ordered = sorted(members, key=lambda t: t["index"])
        lane_seq = 0
        position = 0
        while position < len(ordered):
            traj = ordered[position]
            derived = traj["derived"]
            raw = traj["raw"]
            type_id = int(derived["type_id"])

            if type_id in SKIP_PRIMITIVE_SOURCE_TYPES:
                stats["n_masked_raw"] += 1
                position += 1
                continue

            if type_id == FIXED_TRAJECTORY_ID:
                run = [traj]
                while position + len(run) < len(ordered):
                    current = run[-1]
                    following = ordered[position + len(run)]
                    transition = mergeable.get(current["index"])
                    if (
                        transition is None
                        or transition["to_index"] != following["index"]
                        or int(following["derived"]["type_id"]) != FIXED_TRAJECTORY_ID
                    ):
                        break
                    run.append(following)
                    position += 1
                position += 1
                merged = len(run) > 1
                if merged:
                    stats["n_t0_merges"] += 1
                first = run[0]
                last = run[-1]
                start_s = first["derived"]["start_s"]
                end_s = last["derived"]["end_s"]
                start_pitch = first["derived"].get("start_pitch")
                end_pitch = last["derived"].get("end_pitch")
                prim = _primitive_record(
                    recording_id=recording_id,
                    lane=lane,
                    seq=lane_seq,
                    canonical_type=0,
                    source_raw_type=0,
                    source_subsegment_index=None,
                    source_raw_trajectory_ids=[t["traj_id"] for t in run],
                    source_raw_indices=[t["index"] for t in run],
                    rule_applied=rule_for_source_type(0, merged_t0=merged),
                    start_s=start_s,
                    end_s=end_s,
                    start_pitch=start_pitch,
                    end_pitch=end_pitch,
                    source_x_start=0.0,
                    source_x_end=1.0,
                    source_entry=first,
                )
                all_primitives.append(prim)
                stats["by_canonical_type"][0] += 1
                lane_seq += 1
                continue

            segments = iter_primitive_segments(raw, derived)
            if not segments:
                stats["n_masked_raw"] += 1
                position += 1
                continue

            start_s = derived["start_s"]
            duration_s = derived["duration_s"]
            for spec in segments:
                seg_start = start_s + spec.start_frac * duration_s
                seg_end = start_s + spec.end_frac * duration_s
                start_pitch = _pitch_at_fraction(traj, spec.start_frac)
                end_pitch = _pitch_at_fraction(traj, spec.end_frac)
                rule = rule_for_source_type(type_id)
                if rule == "decompose_4":
                    stats["n_decompose_4"] += 1
                elif rule == "decompose_5":
                    stats["n_decompose_5"] += 1
                elif rule == "decompose_6":
                    stats["n_decompose_6_segments"] += 1
                prim = _primitive_record(
                    recording_id=recording_id,
                    lane=lane,
                    seq=lane_seq,
                    canonical_type=spec.canonical_type,
                    source_raw_type=type_id,
                    source_subsegment_index=spec.subsegment_index,
                    source_raw_trajectory_ids=[traj["traj_id"]],
                    source_raw_indices=[traj["index"]],
                    rule_applied=rule,
                    start_s=seg_start,
                    end_s=seg_end,
                    start_pitch=start_pitch,
                    end_pitch=end_pitch,
                    source_x_start=spec.start_frac,
                    source_x_end=spec.end_frac,
                    source_entry=traj,
                )
                all_primitives.append(prim)
                stats["by_canonical_type"][spec.canonical_type] += 1
                lane_seq += 1
            position += 1

    _assert_invariants(all_primitives, mergeable)
    return {
        "schema_version": recording_doc.get("schema_version"),
        "recording_id": recording_id,
        "performance_group_id": recording_doc.get("performance", {}).get(
            "performance_group_id"
        ),
        "rule_version": PRIMITIVES_RULE_VERSION,
        "pitch_match_rule": recording_doc.get("derivation_params", {}).get(
            "pitch_match_rule", "exact_swara_oct_raised_log_offset"
        ),
        "primitives": all_primitives,
        "stats": stats,
    }


def _pitch_at_fraction(
    traj: dict[str, Any],
    frac: float,
) -> dict[str, Any] | None:
    derived = traj["derived"]
    cps = derived.get("control_points") or []
    if not cps:
        if frac <= 0.0:
            return derived.get("start_pitch")
        return derived.get("end_pitch")
    if frac <= 0.0:
        return cps[0].get("pitch")
    if frac >= 1.0:
        return cps[-1].get("pitch")
    for cp in cps:
        if cp.get("x_frac") is not None and float(cp["x_frac"]) >= frac:
            return cp.get("pitch")
    return cps[-1].get("pitch")


def _primitive_record(
    *,
    recording_id: str,
    lane: str,
    seq: int,
    canonical_type: int,
    source_raw_type: int,
    source_subsegment_index: int | None,
    source_raw_trajectory_ids: Sequence[str],
    source_raw_indices: Sequence[int],
    rule_applied: str,
    start_s: float,
    end_s: float,
    start_pitch: dict[str, Any] | None,
    end_pitch: dict[str, Any] | None,
    source_x_start: float,
    source_x_end: float,
    source_entry: dict[str, Any],
) -> dict[str, Any]:
    return {
        "primitive_id": primitive_id(recording_id, lane, seq),
        "lane_id": lane,
        "seq": seq,
        "canonical_type": canonical_type,
        "source_raw_trajectory_ids": list(source_raw_trajectory_ids),
        "source_raw_indices": list(source_raw_indices),
        "source_raw_type": source_raw_type,
        "source_subsegment_index": source_subsegment_index,
        "source_x_start": source_x_start,
        "source_x_end": source_x_end,
        "rule_applied": rule_applied,
        "start_s": start_s,
        "end_s": end_s,
        "duration_s": end_s - start_s,
        "start_pitch": start_pitch,
        "end_pitch": end_pitch,
        "primary_source_index": source_entry["index"],
    }


def _assert_invariants(
    primitives: Sequence[dict[str, Any]],
    mergeable: dict[int, dict[str, Any]],
) -> None:
    by_lane: dict[str, list[dict[str, Any]]] = {}
    for prim in primitives:
        if prim["canonical_type"] not in {0, 1, 2, 3}:
            raise PrimitiveBuildError(
                f"invalid canonical_type {prim['canonical_type']} on {prim['primitive_id']}"
            )
        if not prim["source_raw_trajectory_ids"]:
            raise PrimitiveBuildError(f"missing provenance on {prim['primitive_id']}")
        by_lane.setdefault(prim["lane_id"], []).append(prim)

    for lane, members in by_lane.items():
        ordered = sorted(members, key=lambda p: (p["start_s"], p["seq"]))
        for prev, curr in zip(ordered, ordered[1:]):
            if curr["start_s"] < prev["start_s"]:
                raise PrimitiveBuildError(
                    f"lane {lane}: primitive {curr['primitive_id']} starts before "
                    f"{prev['primitive_id']}"
                )
            if curr["start_s"] < prev["end_s"] - 1e-9:
                raise PrimitiveBuildError(
                    f"lane {lane}: overlap between {prev['primitive_id']} and "
                    f"{curr['primitive_id']}"
                )

    for transition in mergeable.values():
        from_idx = transition["from_index"]
        to_idx = transition["to_index"]
        from_key = transition.get("from_end_pitch_key")
        to_key = transition.get("to_start_pitch_key")
        if from_key != to_key:
            raise PrimitiveBuildError(
                f"mergeable transition {from_idx}->{to_idx} has different pitch keys"
            )


def primitives_by_lane(
    primitives_doc: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    lanes: dict[str, list[dict[str, Any]]] = {}
    for prim in primitives_doc["primitives"]:
        lanes.setdefault(prim["lane_id"], []).append(prim)
    for lane in lanes:
        lanes[lane].sort(key=lambda p: (p["start_s"], p["seq"]))
    return lanes
