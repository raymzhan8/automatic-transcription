"""The coverage block and its assertions.

``coverage`` deliberately surfaces the finding that some recordings are almost
entirely "Silent" because they were never transcribed. It states facts and draws
no conclusion: consumers filter on ``silent_annotation_fraction`` and
``max_single_silent_annotation_s``.
"""

from __future__ import annotations

import sys
from typing import Any, Sequence

from .audio_refs import source_entry
from .schema import SILENT_TRAJECTORY_ID
from .transitions import non_meets_transitions


class CoverageAssertionError(AssertionError):
    """Raised when annotated span and audio duration disagree beyond tolerance."""


def build_coverage(
    trajectories: Sequence[dict[str, Any]],
    audio_block: dict[str, Any],
    transitions: Sequence[dict[str, Any]],
    *,
    primary_lane_id: str,
    audio_duration_tolerance_s: float,
) -> dict[str, Any]:
    starts = [t["derived"]["start_s"] for t in trajectories]
    ends = [t["derived"]["end_s"] for t in trajectories]
    annotation_start_s = min(starts) if starts else 0.0
    annotation_end_s = max(ends) if ends else 0.0
    span_s = annotation_end_s - annotation_start_s

    source = source_entry(audio_block)
    audio_duration_s = source["duration_s"] if source else None
    delta_s = (
        audio_duration_s - annotation_end_s if audio_duration_s is not None else None
    )

    # Silence statistics are scoped to a single lane; concurrent lanes overlap in
    # time, so pooling them would make the fractions meaningless.
    lane = [t for t in trajectories if t["lane_id"] == primary_lane_id]
    silent = [t for t in lane if t["derived"]["type_id"] == SILENT_TRAJECTORY_ID]
    silent_duration_s = sum(t["derived"]["duration_s"] for t in silent)
    silent_fraction = silent_duration_s / span_s if span_s > 0 else None

    offenders = non_meets_transitions(transitions)

    return {
        "annotation_start_s": annotation_start_s,
        "annotation_end_s": annotation_end_s,
        "audio_duration_s": audio_duration_s,
        "audio_vs_annotation_delta_s": delta_s,
        "n_trajectories": len(trajectories),
        "silence_scope_lane_id": primary_lane_id,
        "n_silent_annotations": len(silent),
        "silent_annotation_duration_s": silent_duration_s,
        "silent_annotation_fraction": silent_fraction,
        "max_single_silent_annotation_s": max(
            (t["derived"]["duration_s"] for t in silent), default=0.0
        ),
        "annotated_non_silent_fraction": (
            None if silent_fraction is None else 1.0 - silent_fraction
        ),
        "assertions": {
            "audio_present": source is not None,
            "audio_duration_tolerance_s": audio_duration_tolerance_s,
            "audio_matches_annotation": (
                None if delta_s is None else abs(delta_s) <= audio_duration_tolerance_s
            ),
            "all_transitions_meet": not offenders,
            "n_non_meets_transitions": len(offenders),
        },
    }


def assert_coverage(recording_id: str, coverage: dict[str, Any]) -> None:
    """Fail loudly on a coverage violation; non-meets transitions warn loudly."""
    assertions = coverage["assertions"]
    if assertions["audio_matches_annotation"] is False:
        raise CoverageAssertionError(
            f"{recording_id}: audio duration {coverage['audio_duration_s']} s and "
            f"annotation end {coverage['annotation_end_s']} s differ by "
            f"{coverage['audio_vs_annotation_delta_s']} s, above tolerance "
            f"{assertions['audio_duration_tolerance_s']} s"
        )
    if not assertions["all_transitions_meet"]:
        print(
            f"  WARNING {recording_id}: {assertions['n_non_meets_transitions']} "
            "transitions are not 'meets'",
            file=sys.stderr,
        )
    if not assertions["audio_present"]:
        print(
            f"  WARNING {recording_id}: no source audio found under output/",
            file=sys.stderr,
        )
