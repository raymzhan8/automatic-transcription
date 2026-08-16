"""Single source of truth for the canonical dataset schema (v1).

Every other module under ``dataset.canonical`` imports its constants, enums and
identifier formats from here so that a schema change is a one-file edit.
"""

from __future__ import annotations

from pathlib import Path

SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# storage layout
# ---------------------------------------------------------------------------

CANONICAL_SUBDIR = Path("output") / "canonical" / f"v{SCHEMA_VERSION.split('.')[0]}"


def canonical_root(repo_root: Path) -> Path:
    return repo_root / CANONICAL_SUBDIR


def recordings_dir(repo_root: Path) -> Path:
    return canonical_root(repo_root) / "recordings"


def raw_api_dir(repo_root: Path) -> Path:
    return canonical_root(repo_root) / "raw_api"


def index_dir(repo_root: Path) -> Path:
    return canonical_root(repo_root) / "index"


def splits_dir(repo_root: Path) -> Path:
    return canonical_root(repo_root) / "splits"


def failed_csv_path(repo_root: Path) -> Path:
    return canonical_root(repo_root) / "failed.csv"


FAILED_COLUMNS = ["recording_id", "error_type", "error_message"]

# ---------------------------------------------------------------------------
# trajectory types
# ---------------------------------------------------------------------------

# Mirrors ``idtap.classes.trajectory.Trajectory.names`` (index == wire ``id``).
TRAJECTORY_ID_TO_NAME: dict[int, str] = {
    0: "Fixed",
    1: "Bend: Simple",
    2: "Bend: Sloped Start",
    3: "Bend: Sloped End",
    4: "Bend: Ladle",
    5: "Bend: Reverse Ladle",
    6: "Bend: Simple Multiple",
    7: "Krintin",
    8: "Krintin Slide",
    9: "Krintin Slide Hammer",
    10: "Dense Krintin Slide Hammer",
    11: "Slide",
    12: "Silent",
    13: "Vibrato",
}

SILENT_TRAJECTORY_ID = 12
FIXED_TRAJECTORY_ID = 0

# Types whose ``dur_array`` describes more than one internal segment.
COMPOSITE_TYPE_IDS = frozenset({4, 5, 6, 7, 8, 9, 10})

DEFAULT_SLOPE = 2.0

# ---------------------------------------------------------------------------
# enums
# ---------------------------------------------------------------------------

INTERVAL_RELATIONS = ("meets", "gap", "overlap")
CANONICAL_UNIT_ROLES = ("single", "run_start", "run_member", "run_end")
AUDIO_ROLES = ("source", "denoised", "vocals", "accompaniment", "noise")

# ``role`` -> filename suffix under ``output/denoised/<recording_id>/``.
DERIVED_STEM_SUFFIXES = {
    "denoised": ".denoised.wav",
    "vocals": ".vocals.wav",
    "accompaniment": ".accompaniment.wav",
    "noise": ".noise.wav",
}

# ---------------------------------------------------------------------------
# derivation parameters
# ---------------------------------------------------------------------------

DERIVATION_PARAMS: dict[str, object] = {
    "time_tolerance_s": 1e-6,
    "pitch_match_rule": "exact_swara_oct_raised_log_offset",
    "start_time_source": "stored_offsets",
    "traj_id_scheme": "recording:lane:phrase:num",
    # Added beyond the plan text: coverage.py needs an explicit threshold for
    # the audio-vs-annotation assertion, and the Type-0 overlay needs the
    # phrase-boundary decision recorded rather than hardcoded.
    "audio_duration_tolerance_s": 0.05,
    "type0_merge_across_phrase_boundary": True,
    "pitch_ratios_source": "raga_stratified_ratios",
}

CANONICALIZATION_RULE_VERSION = "type0_v1"
CANONICALIZATION_RULE = (
    "Consecutive same-lane trajectories with type_id == 0, identical pitch_key, "
    "and interval_relation == 'meets' form one canonical unit."
)
CANONICALIZATION_RULE_APPLIED = "type0_same_pitch_contiguous"

# ---------------------------------------------------------------------------
# Step 5 — four-primitive targets
# ---------------------------------------------------------------------------

PRIMITIVE_TYPE_IDS = frozenset({0, 1, 2, 3})
SKIP_PRIMITIVE_SOURCE_TYPES = frozenset({7, 8, 9, 10, 11, 12, 13})
COMPOSITE_DECOMPOSITION: dict[int, list[int]] = {4: [2, 1], 5: [1, 3]}

PRIMITIVES_RULE_VERSION = "primitives_v1"
HOP_S = 0.01
PITCH_TARGET = "log2_hz"
PHASE_DEFINITION = "linear_elapsed_over_duration"
FRAME_MEMBERSHIP = "half_open_center"
CONTOUR_TOLERANCE_LOG2 = 1e-9

MASKED_TRAJECTORY_TYPE = -1


def primitives_dir(repo_root: Path) -> Path:
    return canonical_root(repo_root) / "primitives"


def frames_dir(repo_root: Path) -> Path:
    return canonical_root(repo_root) / "frames"


def features_dir(repo_root: Path) -> Path:
    return canonical_root(repo_root) / "features"


def normalization_dir(repo_root: Path) -> Path:
    return canonical_root(repo_root) / "normalization"


def figures_step5_dir(repo_root: Path) -> Path:
    return canonical_root(repo_root) / "figures" / "step5"


def figures_step5_5_dir(repo_root: Path) -> Path:
    return canonical_root(repo_root) / "figures" / "step5_5"


def primitive_id(recording_id: str, lane: str, seq: int) -> str:
    lane_token = lane.replace(":", "_")
    return f"{recording_id}:{lane_token}:p{int(seq):05d}"


def frames_npz_name(recording_id: str, lane: str) -> str:
    lane_token = lane.replace(":", "_")
    return f"{recording_id}_{lane_token}.npz"


PRIMITIVES_COLUMNS = [
    "recording_id",
    "primitive_id",
    "lane_id",
    "seq",
    "canonical_type",
    "source_raw_type",
    "source_subsegment_index",
    "source_raw_trajectory_ids",
    "rule_applied",
    "start_s",
    "end_s",
    "duration_s",
    "start_pitch_key",
    "end_pitch_key",
]

FRAMES_SUMMARY_COLUMNS = [
    "recording_id",
    "lane_id",
    "performance_group_id",
    "n_frames",
    "n_valid_frames",
    "valid_fraction",
    "duration_s",
    "hop_s",
    "npz_relpath",
]


def lane_id(track_index: int, string_index: int) -> str:
    return f"{track_index}:{string_index}"


def traj_id(
    recording_id: str,
    track_index: int,
    string_index: int,
    phrase_index: int,
    num: int | None,
) -> str:
    num_token = "nXXX" if num is None else f"n{int(num):03d}"
    return (
        f"{recording_id}:{track_index}:{string_index}"
        f":p{int(phrase_index):03d}:{num_token}"
    )


def canonical_unit_id(recording_id: str, first_member_index: int) -> str:
    return f"{recording_id}:u{int(first_member_index):04d}"


# ---------------------------------------------------------------------------
# index table projections
# ---------------------------------------------------------------------------

RECORDINGS_COLUMNS = [
    "recording_id",
    "title",
    "soloist",
    "solo_instrument",
    "raga_name",
    "performance_group_id",
    "audio_id",
    "audio_relpath",
    "audio_duration_s",
    "annotation_start_s",
    "annotation_end_s",
    "audio_vs_annotation_delta_s",
    "n_lanes",
    "n_phrases",
    "n_trajectories",
    "n_canonical_units",
    "n_silent_annotations",
    "silent_annotation_duration_s",
    "silent_annotation_fraction",
    "max_single_silent_annotation_s",
    "annotated_non_silent_fraction",
    "date_created",
    "date_modified",
    "transcriber_name",
]

TRAJECTORIES_COLUMNS = [
    "recording_id",
    "traj_id",
    "index",
    "lane_id",
    "phrase_index",
    "num",
    "type_id",
    "type_name",
    "start_s",
    "end_s",
    "duration_s",
    "is_silent_annotation",
    "is_composite_type",
    "n_control_points",
    "pitch_endpoints_available",
    "start_pitch_key",
    "end_pitch_key",
    "start_frequency_hz",
    "end_frequency_hz",
    "pitch_range_cents",
    "group_id",
    "canonical_unit_id",
    "canonical_unit_role",
]

TRANSITIONS_COLUMNS = [
    "recording_id",
    "lane_id",
    "from_index",
    "to_index",
    "gap_s",
    "interval_relation",
    "overlap_s",
    "from_end_s",
    "to_start_s",
    "crosses_phrase_boundary",
    "crosses_section_boundary",
    "from_type_id",
    "to_type_id",
    "pitch_endpoints_available",
    "from_end_pitch_key",
    "to_start_pitch_key",
    "pitch_delta_cents",
    "pitch_delta_log2",
    "same_pitch",
    "both_type_0",
    "type0_same_pitch_adjacent",
    "same_group_id",
]
