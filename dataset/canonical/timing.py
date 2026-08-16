"""Absolute time derivation from stored wire offsets, plus its assertions.

``Piece.traj_start_times(inst, string_idx=0)`` ignores the stored ``startTime``
and re-derives it by cumulatively summing ``dur_tot``, which would silently mask
any real non-contiguity. So the canonical builder derives times from
``phrase.start_time + traj.start_time`` and only *checks* the library helper,
recording any disagreement instead of preferring one silently.
"""

from __future__ import annotations

from typing import Any, Iterator, Sequence

from .schema import lane_id


class TimingDisagreement(AssertionError):
    """Raised when derived times and the idtap helper disagree beyond tolerance."""


def phrase_tracks(piece_json: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Return the phrase grid, tolerating documents that carry only ``phrases``."""
    grid = piece_json.get("phraseGrid")
    if grid is None:
        grid = [piece_json.get("phrases") or []]
    return grid


def phrase_start_times(phrases: Sequence[dict[str, Any]]) -> list[tuple[float, str]]:
    """Absolute start of each phrase, and whether it came from the wire."""
    out: list[tuple[float, str]] = []
    cumulative = 0.0
    for phrase in phrases:
        stored = phrase.get("startTime")
        if stored is None:
            out.append((cumulative, "derived_cumulative"))
        else:
            out.append((float(stored), "stored"))
        cumulative += float(phrase.get("durTot") or 0.0)
    return out


def iter_placements(piece_json: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield one record per wire trajectory with absolute times attached.

    Yielded in wire order (track, phrase, string, position). ``num`` is taken
    verbatim from the wire and falls back to the position within the row.
    """
    for track_index, phrases in enumerate(phrase_tracks(piece_json)):
        starts = phrase_start_times(phrases)
        for phrase_index, phrase in enumerate(phrases):
            phrase_start_s, phrase_source = starts[phrase_index]
            for string_index, row in enumerate(phrase.get("trajectoryGrid") or []):
                cumulative = 0.0
                for position, wire in enumerate(row):
                    dur_tot = float(wire.get("durTot") or 0.0)
                    stored = wire.get("startTime")
                    if stored is None:
                        offset = cumulative
                        source = "derived_cumulative"
                    else:
                        offset = float(stored)
                        source = "stored"
                    if phrase_source != "stored":
                        source = "derived_cumulative"
                    start_s = phrase_start_s + offset
                    num = wire.get("num")
                    yield {
                        "track_index": track_index,
                        "string_index": string_index,
                        "lane_id": lane_id(track_index, string_index),
                        "phrase_index": phrase_index,
                        "position": position,
                        "num": position if num is None else int(num),
                        "wire": wire,
                        "phrase_start_s": phrase_start_s,
                        "start_time_offset_s": offset,
                        "start_s": start_s,
                        "end_s": start_s + dur_tot,
                        "duration_s": dur_tot,
                        "start_time_source": source,
                    }
                    cumulative += dur_tot


def control_point_fracs(n_pitches: int, dur_array: Sequence[float] | None) -> list[float]:
    """Normalised positions of each control point inside a trajectory."""
    if n_pitches <= 0:
        return []
    if n_pitches == 1:
        return [0.0]
    segments = [float(d) for d in dur_array] if dur_array else []
    if len(segments) == n_pitches - 1:
        fracs = [0.0]
        for seg in segments:
            fracs.append(fracs[-1] + seg)
        fracs[-1] = 1.0
        return fracs
    step = 1.0 / (n_pitches - 1)
    return [min(1.0, i * step) for i in range(n_pitches)]


def segment_boundaries_s(
    start_s: float,
    duration_s: float,
    dur_array: Sequence[float] | None,
) -> list[float]:
    """Absolute times of the ``dur_array`` segment boundaries, endpoints included."""
    if not dur_array:
        return [start_s, start_s + duration_s]
    boundaries = [start_s]
    running = 0.0
    for seg in dur_array:
        running += float(seg)
        boundaries.append(start_s + running * duration_s)
    boundaries[-1] = start_s + duration_s
    return boundaries


def check_library_agreement(
    piece: Any,
    placements: Sequence[dict[str, Any]],
    *,
    tolerance_s: float,
) -> list[dict[str, Any]]:
    """Compare derived starts against ``Piece.traj_start_times`` per lane."""
    reports: list[dict[str, Any]] = []
    lanes: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for placement in placements:
        key = (placement["track_index"], placement["string_index"])
        lanes.setdefault(key, []).append(placement)

    for (track_index, string_index), lane_placements in sorted(lanes.items()):
        try:
            library_times = [
                float(t)
                for t in piece.traj_start_times(inst=track_index, string_idx=string_index)
            ]
            error: str | None = None
        except Exception as exc:  # pragma: no cover - depends on wire shape
            library_times = []
            error = f"{type(exc).__name__}: {exc}"

        derived = [p["start_s"] for p in lane_placements]
        comparable = min(len(derived), len(library_times))
        deltas = [abs(derived[i] - library_times[i]) for i in range(comparable)]
        max_delta = max(deltas) if deltas else None
        n_disagreements = sum(1 for d in deltas if d > tolerance_s)

        reports.append(
            {
                "lane_id": lane_id(track_index, string_index),
                "n_derived": len(derived),
                "n_library": len(library_times),
                "count_matches": len(derived) == len(library_times),
                "max_abs_delta_s": max_delta,
                "n_disagreements": n_disagreements,
                "within_tolerance": n_disagreements == 0,
                "library_error": error,
            }
        )
    return reports


def loud_timing_problems(reports: Sequence[dict[str, Any]]) -> list[str]:
    """Human-readable descriptions of every timing check that did not pass."""
    problems: list[str] = []
    for report in reports:
        if report["library_error"]:
            problems.append(
                f"lane {report['lane_id']}: traj_start_times raised {report['library_error']}"
            )
        if not report["count_matches"]:
            problems.append(
                f"lane {report['lane_id']}: derived {report['n_derived']} trajectories "
                f"but Piece.traj_start_times returned {report['n_library']}"
            )
        if not report["within_tolerance"]:
            problems.append(
                f"lane {report['lane_id']}: {report['n_disagreements']} start times differ "
                f"from Piece.traj_start_times (max {report['max_abs_delta_s']} s)"
            )
    return problems
