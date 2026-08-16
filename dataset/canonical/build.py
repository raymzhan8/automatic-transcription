"""Build the canonical recording-level transcription dataset (schema v1).

Fetches each transcription via ``SwaraClient``, caches the verbatim response
under ``output/canonical/v1/raw_api/``, and emits one JSON document per
recording that keeps IDTAP's annotation values in a ``raw`` block and quarantines
everything computed in ``derived`` blocks.

Run from the repository root:

    python dataset/canonical/build.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import humps  # noqa: E402
from idtap import Piece, SwaraClient  # noqa: E402
from idtap.classes.raga import Raga  # noqa: E402

from dataset.canonical.audio_refs import build_audio_block  # noqa: E402
from dataset.canonical.canonicalize import build_canonicalization  # noqa: E402
from dataset.canonical.coverage import assert_coverage, build_coverage  # noqa: E402
from dataset.canonical.index_tables import write_index_tables  # noqa: E402
from dataset.canonical.pitch import (  # noqa: E402
    derive_pitch,
    pitch_range_cents,
    pitch_wire_keys,
    raw_pitch,
)
from dataset.canonical.schema import (  # noqa: E402
    COMPOSITE_TYPE_IDS,
    DEFAULT_SLOPE,
    DERIVATION_PARAMS,
    FAILED_COLUMNS,
    SCHEMA_VERSION,
    SILENT_TRAJECTORY_ID,
    TRAJECTORY_ID_TO_NAME,
    failed_csv_path,
    lane_id,
    raw_api_dir,
    recordings_dir,
    traj_id,
)
from dataset.canonical.timing import (  # noqa: E402
    check_library_agreement,
    control_point_fracs,
    iter_placements,
    loud_timing_problems,
    phrase_start_times,
    phrase_tracks,
    segment_boundaries_s,
)
from dataset.canonical.transitions import build_transitions  # noqa: E402

# raw field name -> wire key. Every lookup goes through ``.get()`` because these
# keys vary between older and newer transcriptions; a missing key becomes null
# and is recorded in ``wire_keys_present`` rather than being synthesized.
TRAJ_WIRE_FIELDS: dict[str, str] = {
    "id": "id",
    "num": "num",
    "unique_id": "uniqueId",
    "name": "name",
    "instrumentation": "instrumentation",
    "dur_tot": "durTot",
    "dur_array": "durArray",
    "slope": "slope",
    "start_time": "startTime",
    "fund_id12": "fundID12",
    "vowel": "vowel",
    "vowel_ipa": "vowelIpa",
    "vowel_hindi": "vowelHindi",
    "vowel_eng_trans": "vowelEngTrans",
    "start_consonant": "startConsonant",
    "start_consonant_ipa": "startConsonantIpa",
    "start_consonant_hindi": "startConsonantHindi",
    "start_consonant_eng_trans": "startConsonantEngTrans",
    "end_consonant": "endConsonant",
    "end_consonant_ipa": "endConsonantIpa",
    "end_consonant_hindi": "endConsonantHindi",
    "end_consonant_eng_trans": "endConsonantEngTrans",
    "group_id": "groupId",
    "tags": "tags",
}

TRAJ_NESTED_FIELDS: dict[str, str] = {
    "articulations": "articulations",
    "vib_obj": "vibObj",
    "automation": "automation",
}

RATIOS_CONFLICT_MARKER = "transcription ratios"


# ---------------------------------------------------------------------------
# raw fetch + cache
# ---------------------------------------------------------------------------


def raw_cache_path(repo_root: Path, recording_id: str) -> Path:
    return raw_api_dir(repo_root) / f"{recording_id}.json.gz"


def load_raw(
    recording_id: str,
    *,
    client: Any,
    repo_root: Path,
    use_cached_raw: bool = True,
) -> tuple[dict[str, Any], str]:
    """Return the verbatim wire document plus an ISO ``fetched_at`` stamp."""
    path = raw_cache_path(repo_root, recording_id)
    if use_cached_raw and path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle), _iso(path.stat().st_mtime)

    if client is None:
        raise FileNotFoundError(
            f"no cached raw response for {recording_id} and no client was supplied"
        )
    piece_json = client.get_piece(recording_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(piece_json, handle)
    return piece_json, _iso(path.stat().st_mtime)


def _iso(timestamp: float) -> str:
    return (
        datetime.fromtimestamp(timestamp, tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


# ---------------------------------------------------------------------------
# structure helpers
# ---------------------------------------------------------------------------


def _flatten_categories(categorization: dict[str, Any] | None) -> list[str]:
    if not categorization:
        return []
    out: list[str] = []
    for group, entries in categorization.items():
        if isinstance(entries, dict):
            for name, enabled in entries.items():
                if enabled:
                    out.append(f"{group}/{name}")
        elif entries:
            out.append(str(group))
    return out


def _section_starts_grid(piece_json: dict[str, Any], n_tracks: int) -> list[list[int]]:
    grid = piece_json.get("sectionStartsGrid")
    if grid is None:
        grid = [piece_json.get("sectionStarts") or [0]]
    grid = [sorted(int(s) for s in row) or [0] for row in grid]
    while len(grid) < n_tracks:
        grid.append([0])
    return grid


def _make_section_lookup(section_starts_grid: list[list[int]]):
    def section_of_phrase(track_index: int, phrase_index: int) -> int:
        starts = (
            section_starts_grid[track_index]
            if track_index < len(section_starts_grid)
            else [0]
        )
        section = 0
        for position, start in enumerate(starts):
            if phrase_index >= start:
                section = position
            else:
                break
        return section

    return section_of_phrase


def build_structure(
    piece_json: dict[str, Any],
    section_starts_grid: list[list[int]],
) -> dict[str, Any]:
    tracks = phrase_tracks(piece_json)
    phrases_out: list[dict[str, Any]] = []
    for track_index, phrases in enumerate(tracks):
        starts = phrase_start_times(phrases)
        for phrase_index, phrase in enumerate(phrases):
            categorization_grid = phrase.get("categorizationGrid") or []
            phrases_out.append(
                {
                    "track_index": track_index,
                    "phrase_index": phrase_index,
                    "start_s": starts[phrase_index][0],
                    "duration_s": float(phrase.get("durTot") or 0.0),
                    "is_section_start": phrase.get("isSectionStart"),
                    "categories": _flatten_categories(
                        categorization_grid[0] if categorization_grid else None
                    ),
                    "ad_hoc_categories": list(phrase.get("adHocCategorizationGrid") or []),
                    "unique_id": phrase.get("uniqueId"),
                    "wire_keys_present": sorted(phrase.keys()),
                }
            )

    section_cat_grid = piece_json.get("sectionCatGrid") or []
    ad_hoc_section_grid = piece_json.get("adHocSectionCatGrid") or []
    sections_out: list[dict[str, Any]] = []
    for track_index, starts in enumerate(section_starts_grid):
        categories_row = (
            section_cat_grid[track_index] if track_index < len(section_cat_grid) else []
        )
        ad_hoc_row = (
            ad_hoc_section_grid[track_index]
            if track_index < len(ad_hoc_section_grid)
            else []
        )
        for section_index, start_phrase in enumerate(starts):
            sections_out.append(
                {
                    "track_index": track_index,
                    "section_index": section_index,
                    "start_phrase_index": start_phrase,
                    "categories": _flatten_categories(
                        categories_row[section_index]
                        if section_index < len(categories_row)
                        else None
                    ),
                    "ad_hoc_categories": list(
                        ad_hoc_row[section_index]
                        if section_index < len(ad_hoc_row)
                        else []
                    ),
                }
            )

    return {
        "phrases": phrases_out,
        "sections": sections_out,
        "meters": list(piece_json.get("meters") or []),
    }


# ---------------------------------------------------------------------------
# trajectory blocks
# ---------------------------------------------------------------------------


def build_raw_block(placement: dict[str, Any]) -> dict[str, Any]:
    wire = placement["wire"]
    raw: dict[str, Any] = {
        field: wire.get(key) for field, key in TRAJ_WIRE_FIELDS.items()
    }
    for field, key in TRAJ_NESTED_FIELDS.items():
        value = wire.get(key)
        raw[field] = humps.decamelize(value) if isinstance(value, (dict, list)) else value

    wire_pitches = wire.get("pitches") or []
    raw["pitches"] = [raw_pitch(p) for p in wire_pitches]
    raw["track_index"] = placement["track_index"]
    raw["string_index"] = placement["string_index"]
    raw["phrase_index"] = placement["phrase_index"]
    raw["wire_keys_present"] = sorted(wire.keys())
    raw["pitch_wire_keys_present"] = pitch_wire_keys(wire_pitches)
    return raw


def build_derived_block(
    placement: dict[str, Any],
    raw: dict[str, Any],
    *,
    ratios: Any,
    fundamental: float,
) -> dict[str, Any]:
    type_id = int(raw["id"]) if raw["id"] is not None else -1
    is_silent = type_id == SILENT_TRAJECTORY_ID
    raw_pitches = raw["pitches"]
    start_s = placement["start_s"]
    duration_s = placement["duration_s"]

    endpoints_available = (not is_silent) and len(raw_pitches) > 0

    control_points: list[dict[str, Any]] = []
    if endpoints_available:
        fracs = control_point_fracs(len(raw_pitches), raw.get("dur_array"))
        for point_index, (frac, raw_p) in enumerate(zip(fracs, raw_pitches)):
            control_points.append(
                {
                    "point_index": point_index,
                    "frac": frac,
                    "time_s": start_s + frac * duration_s,
                    "pitch": derive_pitch(raw_p, ratios=ratios, fundamental=fundamental),
                }
            )

    pitches = [cp["pitch"] for cp in control_points]
    start_pitch = pitches[0] if pitches else None
    end_pitch = pitches[-1] if pitches else None

    return {
        "type_id": type_id,
        "type_name": TRAJECTORY_ID_TO_NAME.get(type_id, raw.get("name")),
        "start_s": start_s,
        "end_s": placement["end_s"],
        "duration_s": duration_s,
        "start_time_source": placement["start_time_source"],
        "phrase_start_s": placement["phrase_start_s"],
        "is_silent_annotation": is_silent,
        "is_composite_type": type_id in COMPOSITE_TYPE_IDS,
        "n_control_points": len(raw_pitches),
        "start_pitch": start_pitch,
        "end_pitch": end_pitch,
        "pitch_endpoints_available": endpoints_available,
        "control_points": control_points,
        "segment_boundaries_s": segment_boundaries_s(
            start_s, duration_s, raw.get("dur_array")
        ),
        "pitch_range_cents": pitch_range_cents(pitches),
        "slope_is_default": float(raw.get("slope") or DEFAULT_SLOPE) == DEFAULT_SLOPE,
        "has_vowel": raw.get("vowel") is not None,
        "has_consonant": raw.get("start_consonant") is not None
        or raw.get("end_consonant") is not None,
        "canonical_unit_id": None,
        "canonical_unit_role": None,
    }


# ---------------------------------------------------------------------------
# the builder
# ---------------------------------------------------------------------------


def build_recording(
    recording_id: str,
    *,
    client,
    repo_root: Path,
    performance_group_id: str | None = None,
    use_cached_raw: bool = True,
) -> dict:
    """Return the complete canonical recording document.

    ``performance_group_id`` is dependency-injected: grouping is a corpus-level
    property computed by ``performance_groups.assign_performance_groups`` over
    every transcription, so a single-recording build can never derive it.
    """
    piece_json, fetched_at = load_raw(
        recording_id,
        client=client,
        repo_root=repo_root,
        use_cached_raw=use_cached_raw,
    )

    raga_json = piece_json.get("raga") or {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        raga = Raga.from_json(json.loads(json.dumps(raga_json)))
        ratios = raga.stratified_ratios
        try:
            piece = Piece.from_json(json.loads(json.dumps(piece_json)))
            piece_load_error = None
        except Exception as exc:
            piece = None
            piece_load_error = f"{type(exc).__name__}: {exc}"
    warning_texts = [str(w.message) for w in caught]
    ratios_conflict = any(RATIOS_CONFLICT_MARKER in text for text in warning_texts)

    fundamental = float(raga.fundamental)
    tolerance_s = float(DERIVATION_PARAMS["time_tolerance_s"])

    placements = list(iter_placements(piece_json))
    trajectories: list[dict[str, Any]] = []
    for placement in placements:
        raw = build_raw_block(placement)
        derived = build_derived_block(
            placement, raw, ratios=ratios, fundamental=fundamental
        )
        trajectories.append(
            {
                "index": -1,
                "lane_id": placement["lane_id"],
                "traj_id": traj_id(
                    recording_id,
                    placement["track_index"],
                    placement["string_index"],
                    placement["phrase_index"],
                    placement["num"],
                ),
                "raw": raw,
                "derived": derived,
            }
        )

    trajectories.sort(
        key=lambda t: (
            t["raw"]["track_index"],
            t["raw"]["string_index"],
            t["derived"]["start_s"],
            t["raw"]["phrase_index"],
            t["raw"]["num"] if t["raw"]["num"] is not None else 0,
        )
    )
    for index, traj in enumerate(trajectories):
        traj["index"] = index

    tracks = phrase_tracks(piece_json)
    section_starts_grid = _section_starts_grid(piece_json, len(tracks))
    section_of_phrase = _make_section_lookup(section_starts_grid)

    transitions = build_transitions(
        trajectories,
        section_of_phrase=section_of_phrase,
        time_tolerance_s=tolerance_s,
    )
    canonicalization = build_canonicalization(
        recording_id,
        trajectories,
        transitions,
        merge_across_phrase_boundary=bool(
            DERIVATION_PARAMS["type0_merge_across_phrase_boundary"]
        ),
        time_tolerance_s=tolerance_s,
    )

    timing_checks: list[dict[str, Any]] = []
    if piece is not None:
        timing_checks = check_library_agreement(
            piece, placements, tolerance_s=tolerance_s
        )
        for problem in loud_timing_problems(timing_checks):
            print(f"  TIMING {recording_id}: {problem}", file=sys.stderr)

    instrumentation = list(piece_json.get("instrumentation") or [])
    lanes: list[dict[str, Any]] = []
    lane_counts: dict[str, int] = {}
    for traj in trajectories:
        lane_counts[traj["lane_id"]] = lane_counts.get(traj["lane_id"], 0) + 1
    for track_index, phrases in enumerate(tracks):
        n_strings = max(
            (len(p.get("trajectoryGrid") or []) for p in phrases), default=0
        )
        for string_index in range(n_strings):
            lane = lane_id(track_index, string_index)
            lanes.append(
                {
                    "lane_id": lane,
                    "track_index": track_index,
                    "string_index": string_index,
                    "instrument": (
                        instrumentation[track_index]
                        if track_index < len(instrumentation)
                        else None
                    ),
                    "n_trajectories": lane_counts.get(lane, 0),
                }
            )

    audio = build_audio_block(recording_id, piece_json.get("audioID"), repo_root)
    primary_lane = lanes[0]["lane_id"] if lanes else lane_id(0, 0)
    coverage = build_coverage(
        trajectories,
        audio,
        transitions,
        primary_lane_id=primary_lane,
        audio_duration_tolerance_s=float(
            DERIVATION_PARAMS["audio_duration_tolerance_s"]
        ),
    )
    assert_coverage(recording_id, coverage)

    return {
        "schema_version": SCHEMA_VERSION,
        "recording_id": recording_id,
        "title": piece_json.get("title"),
        "performance": {
            "soloist": piece_json.get("soloist"),
            "solo_instrument": piece_json.get("soloInstrument"),
            "instrumentation": instrumentation,
            "raga_name": raga_json.get("name"),
            "performance_group_id": performance_group_id,
            "composition_title": None,
        },
        "raga": {
            "name": raga_json.get("name"),
            "fundamental_hz": fundamental,
            "ratios": raga_json.get("ratios"),
            "tuning": raga_json.get("tuning"),
            "rule_set_present": bool(raga_json.get("ruleSet")),
            "ratios_rule_set_conflict": ratios_conflict,
            "stratified_ratios": ratios,
        },
        "audio": audio,
        "annotation_source": {
            "provider": "idtap",
            "piece_id": recording_id,
            "date_created": piece_json.get("dateCreated"),
            "date_modified": piece_json.get("dateModified"),
            "transcriber_name": piece_json.get("name"),
            "transcriber_user_id": piece_json.get("userID"),
            "permissions": piece_json.get("permissions"),
            "collections": piece_json.get("collections"),
            "idtap_package_version": _idtap_version(),
            "fetched_at": fetched_at,
            "wire_keys_present": sorted(piece_json.keys()),
            "library_load_error": piece_load_error,
            "library_warnings": warning_texts,
        },
        "lanes": lanes,
        "structure": build_structure(piece_json, section_starts_grid),
        "coverage": coverage,
        "timing_checks": timing_checks,
        "trajectories": trajectories,
        "transitions": transitions,
        "canonicalization": canonicalization,
        "derivation_params": dict(DERIVATION_PARAMS),
    }


def _idtap_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("idtap")
    except Exception:
        return None


def write_recording(doc: dict, repo_root: Path) -> Path:
    """Write the canonical document to ``recordings/<recording_id>.json``."""
    path = recordings_dir(repo_root) / f"{doc['recording_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(doc, handle, ensure_ascii=False, indent=1)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def exported_recording_ids(repo_root: Path) -> list[str]:
    """The already-exported recordings, ignoring ``all`` and dotfiles."""
    cnn_dir = repo_root / "output" / "cnn_dataset"
    return sorted(
        entry.name
        for entry in cnn_dir.iterdir()
        if entry.is_dir() and entry.name != "all" and not entry.name.startswith(".")
    )


def _write_failed(repo_root: Path, rows: list[dict[str, str]]) -> Path:
    path = failed_csv_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAILED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _load_performance_groups(client) -> dict[str, str]:
    """Corpus-level grouping, injected into each per-recording build.

    ``performance_groups`` is authored separately; until it lands, recordings are
    built with a null group rather than a guessed one.
    """
    try:
        from dataset.canonical.performance_groups import assign_performance_groups
    except ImportError:
        print(
            "  NOTE dataset.canonical.performance_groups not available; "
            "performance_group_id will be null",
            file=sys.stderr,
        )
        return {}
    transcriptions = client.get_viewable_transcriptions()
    return assign_performance_groups(transcriptions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recording-ids",
        nargs="*",
        default=None,
        help="Limit to specific recording ids (default: the exported cnn_dataset ids)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
    )
    parser.add_argument(
        "--refresh-raw",
        action="store_true",
        help="Re-fetch from IDTAP even when a cached raw_api response exists",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Skip regenerating the index CSVs",
    )
    args = parser.parse_args()

    repo_root: Path = args.repo_root
    recording_ids = args.recording_ids or exported_recording_ids(repo_root)

    client = SwaraClient()
    groups = _load_performance_groups(client)

    failures: list[dict[str, str]] = []
    for position, recording_id in enumerate(recording_ids, start=1):
        print(f"[{position}/{len(recording_ids)}] {recording_id}")
        try:
            doc = build_recording(
                recording_id,
                client=client,
                repo_root=repo_root,
                performance_group_id=groups.get(recording_id),
                use_cached_raw=not args.refresh_raw,
            )
            path = write_recording(doc, repo_root)
            coverage = doc["coverage"]
            print(
                f"  {coverage['n_trajectories']} trajectories, "
                f"{doc['canonicalization']['n_canonical_units']} canonical units "
                f"-> {path.relative_to(repo_root)}"
            )
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            failures.append(
                {
                    "recording_id": recording_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    failed_path = _write_failed(repo_root, failures)
    if failures:
        print(f"\n{len(failures)} recordings failed -> {failed_path.relative_to(repo_root)}")

    if not args.no_index:
        paths = write_index_tables(repo_root)
        for name, path in paths.items():
            print(f"index/{name}.csv -> {path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
