"""Approved four-primitive decomposition rules.

Mirrors ``iter_labeled_segments`` in ``dataset/export_denoised_cnn_dataset.py``
without importing librosa or the export pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .schema import (
    COMPOSITE_DECOMPOSITION,
    FIXED_TRAJECTORY_ID,
    SKIP_PRIMITIVE_SOURCE_TYPES,
)


@dataclass(frozen=True)
class PrimitiveSegmentSpec:
    """One canonical primitive span inside a single raw trajectory."""

    canonical_type: int
    start_frac: float
    end_frac: float
    subsegment_index: int | None


def _dur_fractions(n_segments: int, dur_array: Sequence[float] | None) -> list[float]:
    if dur_array and len(dur_array) >= n_segments:
        return [float(x) for x in dur_array[:n_segments]]
    if n_segments == 2:
        return [1.0 / 3.0, 2.0 / 3.0]
    return [1.0 / n_segments] * n_segments


def _segments_from_fractions(
    labels: Sequence[int],
    fracs: Sequence[float],
) -> list[PrimitiveSegmentSpec]:
    starts = [0.0]
    for frac in fracs:
        starts.append(starts[-1] + float(frac))
    starts[-1] = 1.0
    return [
        PrimitiveSegmentSpec(
            canonical_type=int(label),
            start_frac=starts[i],
            end_frac=starts[i + 1],
            subsegment_index=i,
        )
        for i, label in enumerate(labels)
    ]


def iter_primitive_segments(
    raw: dict[str, Any],
    derived: dict[str, Any],
) -> list[PrimitiveSegmentSpec]:
    """Return ordered primitive segment specs for one raw trajectory."""
    type_id = int(derived["type_id"])
    if type_id in SKIP_PRIMITIVE_SOURCE_TYPES:
        return []

    if type_id in {0, 1, 2, 3}:
        return [
            PrimitiveSegmentSpec(
                canonical_type=type_id,
                start_frac=0.0,
                end_frac=1.0,
                subsegment_index=None,
            )
        ]

    dur_array = raw.get("dur_array") or []

    if type_id in COMPOSITE_DECOMPOSITION:
        labels = COMPOSITE_DECOMPOSITION[type_id]
        fracs = _dur_fractions(len(labels), dur_array)
        return _segments_from_fractions(labels, fracs)

    if type_id == 6:
        pitches = raw.get("pitches") or []
        n_segments = len(dur_array) if dur_array else max(len(pitches) - 1, 1)
        fracs = _dur_fractions(n_segments, dur_array)
        labels = [1] * n_segments
        return _segments_from_fractions(labels, fracs)

    return []


def is_mergeable_type0_transition(transition: dict[str, Any]) -> bool:
    """True when consecutive Fixed trajectories may merge."""
    return bool(transition.get("type0_same_pitch_adjacent"))


def type0_run_rule() -> str:
    return "merge_t0_same_pitch_contiguous"


def rule_for_source_type(type_id: int, *, merged_t0: bool = False) -> str:
    if type_id == FIXED_TRAJECTORY_ID and merged_t0:
        return "merge_t0"
    if type_id == FIXED_TRAJECTORY_ID:
        return "keep"
    if type_id in {1, 2, 3}:
        return "keep"
    if type_id == 4:
        return "decompose_4"
    if type_id == 5:
        return "decompose_5"
    if type_id == 6:
        return "decompose_6"
    return "masked"
