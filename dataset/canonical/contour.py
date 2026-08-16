"""Shared pitch contour evaluation for primitives and framewise targets."""

from __future__ import annotations

import warnings
from typing import Any, Sequence

from .verify_roundtrip import build_from_raw


def trajectories_by_index(recording_doc: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {t["index"]: t for t in recording_doc["trajectories"]}


def raw_trajectory_at_time(
    lane_trajectories: Sequence[dict[str, Any]],
    t: float,
) -> tuple[dict[str, Any], float] | None:
    """Return (trajectory entry, normalized x) for wall time t."""
    for traj in lane_trajectories:
        d = traj["derived"]
        if d["start_s"] <= t < d["end_s"]:
            duration = d["duration_s"]
            if duration <= 0:
                return traj, 0.0
            x = (t - d["start_s"]) / duration
            return traj, min(max(x, 0.0), 1.0)
    return None


def primitive_trajectory_at_time(
    primitive: dict[str, Any],
    trajectories_by_idx: dict[int, dict[str, Any]],
    t: float,
) -> tuple[dict[str, Any], float] | None:
    """Map wall time inside a primitive to source raw traj + normalized x."""
    if not (primitive["start_s"] <= t < primitive["end_s"]):
        return None

    indices = primitive["source_raw_indices"]
    if len(indices) == 1:
        entry = trajectories_by_idx[indices[0]]
        x0 = float(primitive["source_x_start"])
        x1 = float(primitive["source_x_end"])
        local = (t - primitive["start_s"]) / primitive["duration_s"]
        x = x0 + local * (x1 - x0)
        return entry, min(max(x, 0.0), 1.0)

    # Merged T0 run spans multiple raw trajectories.
    for index in indices:
        entry = trajectories_by_idx[index]
        d = entry["derived"]
        if d["start_s"] <= t < d["end_s"]:
            duration = d["duration_s"]
            if duration <= 0:
                return entry, 0.0
            x = (t - d["start_s"]) / duration
            return entry, min(max(x, 0.0), 1.0)
    return None


def compute_log2_at_time(
    recording_doc: dict[str, Any],
    entry: dict[str, Any],
    x: float,
    *,
    cache: dict[int, Any] | None = None,
) -> float | None:
    ratios = recording_doc["raga"]["stratified_ratios"]
    fundamental = float(recording_doc["raga"]["fundamental_hz"])
    if entry["derived"].get("is_silent_annotation"):
        return None
    index = entry["index"]
    if cache is not None and index in cache:
        obj = cache[index]
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj = build_from_raw(entry["raw"], ratios=ratios, fundamental=fundamental)
        if cache is not None:
            cache[index] = obj
    try:
        return float(obj.compute(x, log_scale=True))
    except Exception:
        return None


def _sample_times_in_spans(
    spans: Sequence[tuple[float, float]],
    grid_step_s: float,
) -> list[float]:
    times: list[float] = []
    for start_s, end_s in spans:
        if end_s <= start_s:
            continue
        t = start_s + grid_step_s / 2.0
        while t < end_s:
            times.append(t)
            t += grid_step_s
    return times


def compare_lane_contours(
    recording_doc: dict[str, Any],
    lane: str,
    lane_trajectories: Sequence[dict[str, Any]],
    lane_primitives: Sequence[dict[str, Any]],
    *,
    duration_s: float,
    grid_step_s: float = 0.001,
) -> dict[str, Any]:
    """Compare raw vs primitive-derived pitch contours on a uniform time grid."""
    by_idx = trajectories_by_index(recording_doc)
    ordered_trajs = sorted(lane_trajectories, key=lambda t: t["index"])
    ordered_prims = sorted(lane_primitives, key=lambda p: (p["start_s"], p["seq"]))

    spans = [(t["derived"]["start_s"], t["derived"]["end_s"]) for t in ordered_trajs]
    sample_times = _sample_times_in_spans(spans, grid_step_s)

    errors: list[float] = []
    worst: list[dict[str, Any]] = []
    n_compared = 0
    n_skipped = 0
    obj_cache: dict[int, Any] = {}

    for t in sample_times:
        raw_hit = raw_trajectory_at_time(ordered_trajs, t)
        prim = next(
            (p for p in ordered_prims if p["start_s"] <= t < p["end_s"]),
            None,
        )

        if raw_hit is None and prim is None:
            continue

        if raw_hit is None or prim is None:
            n_skipped += 1
            continue

        raw_entry, raw_x = raw_hit
        prim_hit = primitive_trajectory_at_time(prim, by_idx, t)
        if prim_hit is None:
            n_skipped += 1
            continue

        prim_entry, prim_x = prim_hit
        raw_pitch = compute_log2_at_time(
            recording_doc, raw_entry, raw_x, cache=obj_cache
        )
        prim_pitch = compute_log2_at_time(
            recording_doc, prim_entry, prim_x, cache=obj_cache
        )

        if raw_pitch is None and prim_pitch is None:
            continue
        if raw_pitch is None or prim_pitch is None:
            n_skipped += 1
            continue

        delta = abs(raw_pitch - prim_pitch)
        errors.append(delta)
        n_compared += 1
        if len(worst) < 10 or delta > worst[0]["delta"]:
            item = {
                "t": t,
                "delta": delta,
                "raw_index": raw_entry["index"],
                "prim_id": prim["primitive_id"],
            }
            worst.append(item)
            worst.sort(key=lambda w: w["delta"])
            if len(worst) > 10:
                worst.pop(0)

    if errors:
        sorted_err = sorted(errors)
        p99_idx = min(len(sorted_err) - 1, int(0.99 * len(sorted_err)))
        stats = {
            "max": max(errors),
            "mean": sum(errors) / len(errors),
            "p99": sorted_err[p99_idx],
            "n_compared": n_compared,
            "n_skipped": n_skipped,
            "n_sample_times": len(sample_times),
        }
    else:
        stats = {
            "max": None,
            "mean": None,
            "p99": None,
            "n_compared": n_compared,
            "n_skipped": n_skipped,
            "n_sample_times": len(sample_times),
        }

    return {"lane_id": lane, "stats": stats, "worst": worst}
