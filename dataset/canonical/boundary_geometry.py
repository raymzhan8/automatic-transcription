"""Pitch geometry at canonical primitive boundaries.

Uses the ground-truth parametric pitch contour (``Trajectory.compute``), not
measured F0. Derivative estimates:

* **Grid:** 1 ms samples over a local window (default ±250 ms), clipped to lane span.
* **Pitch unit:** cents = ``1200 * log2_hz`` (log-frequency displacement).
* **Velocity:** central difference with ±10 ms offset → cents/s.
* **Acceleration:** second central difference with ±10 ms → cents/s².
* **Boundary v/a:** mean over [−50, −20] ms before and [+20, +50] ms after.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .contour import compute_log2_at_time, raw_trajectory_at_time
from .pitch import cents_between
from .primitives import primitives_by_lane
from .verify_roundtrip import build_from_raw

VELOCITY_OFFSET_S = 0.01
ACCEL_OFFSET_S = 0.01
V_BEFORE_WINDOW = (-0.05, -0.02)
V_AFTER_WINDOW = (0.02, 0.05)
DEFAULT_WINDOW_S = 0.25
DECOMPOSE_RULES = frozenset({"decompose_4", "decompose_5", "decompose_6"})


@dataclass
class BoundaryRecord:
    recording_id: str
    lane_id: str
    boundary_s: float
    prim_a_id: str
    prim_b_id: str
    prev_type: int
    next_type: int
    same_type: bool
    prim_a_rule: str
    prim_b_rule: str
    source_raw_type_a: int
    source_raw_type_b: int
    source_raw_indices_a: list[int]
    source_raw_indices_b: list[int]
    same_raw_trajectory: bool
    introduced_by_decomposition: bool
    raw_preserved: bool
    pitch_step_cents: float | None = None
    v_before: float | None = None
    v_after: float | None = None
    delta_v: float | None = None
    a_before: float | None = None
    a_after: float | None = None
    delta_a: float | None = None
    at_control_point_transition: bool | None = None
    internal_kink_log2: float | None = None
    prim_a_duration_s: float = 0.0
    prim_b_duration_s: float = 0.0
    is_t6_internal: bool = False
    is_raw_t1_t1: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _same_raw_indices(a: Sequence[int], b: Sequence[int]) -> bool:
    return list(a) == list(b)


def _introduced_by_decomposition(
    prim_a: dict[str, Any], prim_b: dict[str, Any], same_raw: bool
) -> bool:
    if not same_raw:
        return False
    if prim_a["rule_applied"] in DECOMPOSE_RULES or prim_b["rule_applied"] in DECOMPOSE_RULES:
        return True
    return (
        prim_a.get("source_subsegment_index") is not None
        and prim_b.get("source_subsegment_index") is not None
    )


def iter_canonical_boundaries(
    recording_doc: dict[str, Any],
    primitives_doc: dict[str, Any],
) -> list[BoundaryRecord]:
    """Enumerate consecutive primitive pairs per lane (metadata only)."""
    recording_id = recording_doc["recording_id"]
    out: list[BoundaryRecord] = []
    for lane_id, members in primitives_by_lane(primitives_doc).items():
        for prim_a, prim_b in zip(members, members[1:]):
            same_raw = _same_raw_indices(
                prim_a["source_raw_indices"], prim_b["source_raw_indices"]
            )
            decomp = _introduced_by_decomposition(prim_a, prim_b, same_raw)
            prev_type = int(prim_a["canonical_type"])
            next_type = int(prim_b["canonical_type"])
            is_t6_internal = bool(
                decomp
                and prim_a["source_raw_type"] == 6
                and prim_b["source_raw_type"] == 6
                and prev_type == 1
                and next_type == 1
            )
            is_raw_t1_t1 = bool(
                not decomp
                and prim_a["rule_applied"] == "keep"
                and prim_b["rule_applied"] == "keep"
                and prim_a["source_raw_type"] == 1
                and prim_b["source_raw_type"] == 1
                and not same_raw
                and prev_type == 1
                and next_type == 1
            )
            out.append(
                BoundaryRecord(
                    recording_id=recording_id,
                    lane_id=lane_id,
                    boundary_s=float(prim_b["start_s"]),
                    prim_a_id=prim_a["primitive_id"],
                    prim_b_id=prim_b["primitive_id"],
                    prev_type=prev_type,
                    next_type=next_type,
                    same_type=prev_type == next_type,
                    prim_a_rule=str(prim_a["rule_applied"]),
                    prim_b_rule=str(prim_b["rule_applied"]),
                    source_raw_type_a=int(prim_a["source_raw_type"]),
                    source_raw_type_b=int(prim_b["source_raw_type"]),
                    source_raw_indices_a=list(prim_a["source_raw_indices"]),
                    source_raw_indices_b=list(prim_b["source_raw_indices"]),
                    same_raw_trajectory=same_raw,
                    introduced_by_decomposition=decomp,
                    raw_preserved=not decomp,
                    prim_a_duration_s=float(prim_a["duration_s"]),
                    prim_b_duration_s=float(prim_b["duration_s"]),
                    is_t6_internal=is_t6_internal,
                    is_raw_t1_t1=is_raw_t1_t1,
                )
            )
    return out


def sample_dense_cents(
    recording_doc: dict[str, Any],
    lane_trajectories: Sequence[dict[str, Any]],
    t0: float,
    t1: float,
    *,
    step_s: float = 0.001,
) -> tuple[list[float], list[float]]:
    """Sample parametric pitch in cents on a uniform time grid."""
    ordered = sorted(lane_trajectories, key=lambda t: t["index"])
    obj_cache: dict[int, Any] = {}
    times: list[float] = []
    cents: list[float] = []
    t = t0 + step_s / 2.0
    while t < t1:
        hit = raw_trajectory_at_time(ordered, t)
        if hit is not None:
            entry, x = hit
            log2_hz = compute_log2_at_time(
                recording_doc, entry, x, cache=obj_cache
            )
            if log2_hz is not None:
                times.append(t)
                cents.append(1200.0 * log2_hz)
        t += step_s
    return times, cents


def _central_velocity(times: list[float], cents: list[float]) -> list[float]:
    if len(times) < 3:
        return []
    dt = 2.0 * VELOCITY_OFFSET_S
    out: list[float] = []
    for i, t in enumerate(times):
        t_lo = t - VELOCITY_OFFSET_S
        t_hi = t + VELOCITY_OFFSET_S
        c_lo = _interp(times, cents, t_lo)
        c_hi = _interp(times, cents, t_hi)
        if c_lo is None or c_hi is None:
            out.append(float("nan"))
        else:
            out.append((c_hi - c_lo) / dt)
    return out


def _central_acceleration(times: list[float], cents: list[float]) -> list[float]:
    if len(times) < 3:
        return []
    dt = ACCEL_OFFSET_S
    out: list[float] = []
    for t in times:
        c_lo = _interp(times, cents, t - dt)
        c_mid = _interp(times, cents, t)
        c_hi = _interp(times, cents, t + dt)
        if c_lo is None or c_mid is None or c_hi is None:
            out.append(float("nan"))
        else:
            out.append((c_hi - 2.0 * c_mid + c_lo) / (dt * dt))
    return out


def _interp(times: list[float], values: list[float], t: float) -> float | None:
    if not times:
        return None
    if t <= times[0]:
        return values[0]
    if t >= times[-1]:
        return values[-1]
    for i in range(len(times) - 1):
        if times[i] <= t <= times[i + 1]:
            span = times[i + 1] - times[i]
            if span <= 0:
                return values[i]
            frac = (t - times[i]) / span
            return values[i] + frac * (values[i + 1] - values[i])
    return None


def _mean_in_window(
    times: list[float],
    values: list[float],
    center: float,
    window: tuple[float, float],
) -> float | None:
    lo, hi = window
    selected = [
        v
        for t, v in zip(times, values)
        if center + lo <= t <= center + hi and v == v
    ]
    if not selected:
        return None
    return sum(selected) / len(selected)


def _control_point_info(
    recording_doc: dict[str, Any],
    prim_a: dict[str, Any],
    prim_b: dict[str, Any],
) -> tuple[bool | None, float | None]:
    if not _same_raw_indices(prim_a["source_raw_indices"], prim_b["source_raw_indices"]):
        return None, None
    if len(prim_a["source_raw_indices"]) != 1:
        return None, None
    idx = prim_a["source_raw_indices"][0]
    entry = next(
        (t for t in recording_doc["trajectories"] if t["index"] == idx),
        None,
    )
    if entry is None:
        return None, None
    raw = entry["raw"]
    dur_array = raw.get("dur_array") or []
    if len(dur_array) < 2:
        return False, 0.0
    boundary_x = float(prim_a.get("source_x_end") or prim_b.get("source_x_start") or 0.0)
    fracs = [0.0]
    for frac in dur_array:
        fracs.append(fracs[-1] + float(frac))
    fracs[-1] = 1.0
    internal = fracs[1:-1]
    at_cp = any(abs(boundary_x - b) < 1e-6 for b in internal)
    ratios = recording_doc["raga"]["stratified_ratios"]
    fundamental = float(recording_doc["raga"]["fundamental_hz"])
    eps = 1e-5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = build_from_raw(raw, ratios=ratios, fundamental=fundamental)
    kink = 0.0
    for b in internal:
        if abs(boundary_x - b) < 1e-6:
            left = obj.compute(max(0.0, b - eps), log_scale=True)
            right = obj.compute(min(1.0, b + eps), log_scale=True)
            kink = abs(left - right)
            break
    return at_cp, kink


def enrich_boundary_geometry(
    record: BoundaryRecord,
    recording_doc: dict[str, Any],
    prim_a: dict[str, Any],
    prim_b: dict[str, Any],
    *,
    window_s: float = DEFAULT_WINDOW_S,
) -> BoundaryRecord:
    """Fill pitch/velocity/acceleration fields for one boundary."""
    record.pitch_step_cents = cents_between(
        prim_a.get("end_pitch"), prim_b.get("start_pitch")
    )
    lane = record.lane_id
    lane_trajs = [t for t in recording_doc["trajectories"] if t["lane_id"] == lane]
    if lane_trajs:
        span_lo = min(t["derived"]["start_s"] for t in lane_trajs)
        span_hi = max(t["derived"]["end_s"] for t in lane_trajs)
    else:
        span_lo, span_hi = 0.0, record.boundary_s
    t0 = max(span_lo, record.boundary_s - window_s)
    t1 = min(span_hi, record.boundary_s + window_s)
    times, cents = sample_dense_cents(recording_doc, lane_trajs, t0, t1)
    if len(times) >= 3:
        velocities = _central_velocity(times, cents)
        accelerations = _central_acceleration(times, cents)
        record.v_before = _mean_in_window(
            times, velocities, record.boundary_s, V_BEFORE_WINDOW
        )
        record.v_after = _mean_in_window(
            times, velocities, record.boundary_s, V_AFTER_WINDOW
        )
        record.a_before = _mean_in_window(
            times, accelerations, record.boundary_s, V_BEFORE_WINDOW
        )
        record.a_after = _mean_in_window(
            times, accelerations, record.boundary_s, V_AFTER_WINDOW
        )
        if record.v_before is not None and record.v_after is not None:
            record.delta_v = record.v_after - record.v_before
        if record.a_before is not None and record.a_after is not None:
            record.delta_a = record.a_after - record.a_before
    at_cp, kink = _control_point_info(recording_doc, prim_a, prim_b)
    record.at_control_point_transition = at_cp
    record.internal_kink_log2 = kink
    return record


def build_boundary_records(
    recording_doc: dict[str, Any],
    primitives_doc: dict[str, Any],
    *,
    compute_geometry: bool = True,
) -> list[BoundaryRecord]:
    prim_by_id = {p["primitive_id"]: p for p in primitives_doc["primitives"]}
    records = iter_canonical_boundaries(recording_doc, primitives_doc)
    if not compute_geometry:
        return records
    enriched: list[BoundaryRecord] = []
    for rec in records:
        prim_a = prim_by_id[rec.prim_a_id]
        prim_b = prim_by_id[rec.prim_b_id]
        enriched.append(enrich_boundary_geometry(rec, recording_doc, prim_a, prim_b))
    return enriched
