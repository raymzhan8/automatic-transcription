"""Read-only analysis for Step 4.5 — Canonical Trajectory Target Report.

Computes transition statistics, Type-0 merge safety, composite internal kinks,
and curated before/after example structs from the canonical dataset on disk.

Run from the repository root::

    python dataset/canonical/analyze_step_4_5.py
    python dataset/canonical/analyze_step_4_5.py --output output/canonical/v1/step_4_5_analysis.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.canonical.schema import (  # noqa: E402
    COMPOSITE_TYPE_IDS,
    FIXED_TRAJECTORY_ID,
    SILENT_TRAJECTORY_ID,
    TRAJECTORY_ID_TO_NAME,
    canonical_root,
)
from dataset.canonical.verify_roundtrip import build_from_raw  # noqa: E402

COMPOSITE_LABELS: dict[int, list[int]] = {4: [2, 1], 5: [1, 3]}
SKIP_IDTAP_IDS = {7, 8, 9, 10, 11, 13}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def transition_category(row: dict[str, str]) -> str:
    same_type = row["from_type_id"] == row["to_type_id"]
    same_pitch = row["same_pitch"] == "True"
    if same_type and same_pitch:
        return "same_type + same_pitch"
    if same_type and not same_pitch:
        return "same_type + different_pitch"
    if not same_type and same_pitch:
        return "different_type + same_pitch"
    return "different_type + different_pitch"


def load_recording(canonical: Path, recording_id: str) -> dict[str, Any]:
    return json.loads((canonical / "recordings" / f"{recording_id}.json").read_text())


def traj_by_index(doc: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {entry["index"]: entry for entry in doc["trajectories"]}


def pitch_label(pitch: dict[str, Any] | None) -> str:
    if not pitch:
        return "?"
    sargam = pitch.get("sargam") or "?"
    return str(sargam)


def type_label(type_id: int) -> str:
    name = TRAJECTORY_ID_TO_NAME.get(type_id, "?")
    return f"T{type_id}({name})"


def summarize_trajectory(entry: dict[str, Any]) -> dict[str, Any]:
    raw = entry["raw"]
    derived = entry["derived"]
    start_pitch = derived.get("start_pitch") or {}
    end_pitch = derived.get("end_pitch") or {}
    return {
        "index": entry["index"],
        "type_id": derived["type_id"],
        "type_name": derived.get("type_name"),
        "start_s": derived["start_s"],
        "end_s": derived["end_s"],
        "duration_s": derived["duration_s"],
        "start_pitch_key": (start_pitch.get("pitch_key") if start_pitch else None),
        "end_pitch_key": (end_pitch.get("pitch_key") if end_pitch else None),
        "start_sargam": start_pitch.get("sargam"),
        "end_sargam": end_pitch.get("sargam"),
        "vowel": raw.get("vowel"),
        "consonant": raw.get("consonant"),
        "articulation": raw.get("articulation"),
        "slope": raw.get("slope"),
        "dur_array_len": len(raw.get("dur_array") or []),
    }


def internal_kinks(
    entry: dict[str, Any],
    *,
    ratios: list[float],
    fundamental: float,
    eps: float = 1e-5,
) -> list[float]:
    raw = entry["raw"]
    dur_array = raw.get("dur_array") or []
    if len(dur_array) < 2:
        return []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = build_from_raw(raw, ratios=ratios, fundamental=fundamental)
    fracs = [0.0]
    for frac in dur_array:
        fracs.append(fracs[-1] + float(frac))
    fracs[-1] = 1.0
    kinks: list[float] = []
    for boundary in fracs[1:-1]:
        left = obj.compute(max(0.0, boundary - eps), log_scale=True)
        right = obj.compute(min(1.0, boundary + eps), log_scale=True)
        kinks.append(abs(left - right))
    return kinks


def analyze_transitions(transitions: list[dict[str, str]]) -> dict[str, Any]:
    categories = Counter(transition_category(row) for row in transitions)
    t0_same = [
        row
        for row in transitions
        if row["both_type_0"] == "True" and row["same_pitch"] == "True"
    ]
    t0_diff = [
        row
        for row in transitions
        if row["both_type_0"] == "True" and row["same_pitch"] == "False"
    ]
    non_t0_same = [
        row
        for row in transitions
        if row["from_type_id"] == row["to_type_id"]
        and row["same_pitch"] == "True"
        and row["both_type_0"] == "False"
    ]
    diff_type_same = [
        row
        for row in transitions
        if row["from_type_id"] != row["to_type_id"] and row["same_pitch"] == "True"
    ]
    t0_diff_cents = [
        abs(float(row["pitch_delta_cents"]))
        for row in t0_diff
        if row["pitch_delta_cents"]
    ]
    return {
        "n_transitions": len(transitions),
        "categories": dict(categories),
        "t0_same_pitch": len(t0_same),
        "t0_diff_pitch": len(t0_diff),
        "t0_diff_pitch_cents": {
            "min": min(t0_diff_cents) if t0_diff_cents else None,
            "p50": statistics.median(t0_diff_cents) if t0_diff_cents else None,
            "max": max(t0_diff_cents) if t0_diff_cents else None,
        },
        "non_t0_same_type_same_pitch": len(non_t0_same),
        "non_t0_same_type_same_pitch_by_type": dict(
            Counter(row["from_type_id"] for row in non_t0_same)
        ),
        "top_diff_type_same_pitch_pairs": [
            {"from_type_id": a, "to_type_id": b, "count": count}
            for (a, b), count in Counter(
                (row["from_type_id"], row["to_type_id"]) for row in diff_type_same
            ).most_common(12)
        ],
        "t6_same_pitch": sum(
            1
            for row in non_t0_same
            if row["from_type_id"] == "6" and row["to_type_id"] == "6"
        ),
    }


def analyze_t0_merges(
    transitions: list[dict[str, str]],
    canonical: Path,
) -> dict[str, Any]:
    merge_rows = [
        row
        for row in transitions
        if row["type0_same_pitch_adjacent"] == "True"
    ]
    cache: dict[str, dict[str, Any]] = {}
    meta = Counter()
    combined_durations: list[float] = []
    examples: list[dict[str, Any]] = []

    for row in merge_rows:
        rid = row["recording_id"]
        if rid not in cache:
            cache[rid] = load_recording(canonical, rid)
        doc = cache[rid]
        by_idx = traj_by_index(doc)
        left = by_idx[int(row["from_index"])]
        right = by_idx[int(row["to_index"])]
        left_raw = left["raw"]
        right_raw = right["raw"]
        if left_raw.get("vowel") != right_raw.get("vowel"):
            meta["vowel_diff"] += 1
        if left_raw.get("consonant") != right_raw.get("consonant"):
            meta["consonant_diff"] += 1
        if left_raw.get("articulation") != right_raw.get("articulation"):
            meta["articulation_diff"] += 1
        if left_raw.get("slope") != right_raw.get("slope"):
            meta["slope_diff"] += 1
        if left_raw.get("automation") != right_raw.get("automation"):
            meta["automation_diff"] += 1
        if row["crosses_phrase_boundary"] == "True":
            meta["crosses_phrase_boundary"] += 1
        combined_durations.append(
            left["derived"]["duration_s"] + right["derived"]["duration_s"]
        )
        examples.append(
            {
                "recording_id": rid,
                "from_index": int(row["from_index"]),
                "to_index": int(row["to_index"]),
                "pitch_key": row["from_end_pitch_key"],
                "from_sargam": pitch_label(left["derived"].get("end_pitch")),
                "dur_left_s": left["derived"]["duration_s"],
                "dur_right_s": right["derived"]["duration_s"],
                "vowel_left": left_raw.get("vowel"),
                "vowel_right": right_raw.get("vowel"),
                "consonant_left": left_raw.get("consonant"),
                "consonant_right": right_raw.get("consonant"),
                "articulation_left": left_raw.get("articulation"),
                "articulation_right": right_raw.get("articulation"),
            }
        )

    vowel_diff_examples = [ex for ex in examples if ex["vowel_left"] != ex["vowel_right"]]

    return {
        "n_merge_candidates": len(merge_rows),
        "metadata_diffs": dict(meta),
        "combined_duration_s": {
            "p50": statistics.median(combined_durations) if combined_durations else None,
            "max": max(combined_durations) if combined_durations else None,
        },
        "examples": examples[:20],
        "vowel_diff_examples": vowel_diff_examples[:5],
    }


def analyze_composites(
    trajectories: list[dict[str, str]],
    canonical: Path,
) -> dict[str, Any]:
    by_type: Counter[int] = Counter()
    seg_counts: dict[int, Counter[int]] = defaultdict(Counter)
    kink_stats: dict[int, list[float]] = defaultdict(list)
    cache: dict[str, dict[str, Any]] = {}

    for row in trajectories:
        if row["is_silent_annotation"] == "True":
            continue
        type_id = int(row["type_id"])
        if type_id not in COMPOSITE_TYPE_IDS:
            continue
        by_type[type_id] += 1
        rid = row["recording_id"]
        if rid not in cache:
            cache[rid] = load_recording(canonical, rid)
        doc = cache[rid]
        entry = next(
            item for item in doc["trajectories"] if item["index"] == int(row["index"])
        )
        dur_array = entry["raw"].get("dur_array") or []
        seg_counts[type_id][len(dur_array)] += 1
        kinks = internal_kinks(
            entry,
            ratios=doc["raga"]["stratified_ratios"],
            fundamental=float(doc["raga"]["fundamental_hz"]),
        )
        if kinks:
            kink_stats[type_id].append(max(kinks))

    composite_summary: dict[str, Any] = {}
    for type_id in sorted(by_type):
        kinks = kink_stats.get(type_id, [])
        composite_summary[str(type_id)] = {
            "name": TRAJECTORY_ID_TO_NAME.get(type_id, "?"),
            "count": by_type[type_id],
            "dur_array_segment_counts": {
                str(k): v for k, v in sorted(seg_counts[type_id].items())
            },
            "internal_kink_log2": {
                "p50": statistics.median(kinks) if kinks else None,
                "max": max(kinks) if kinks else None,
                "n_with_internal_boundaries": len(kinks),
            },
        }
    return composite_summary


def find_merged_unit(doc: dict[str, Any], min_members: int = 1) -> dict[str, Any] | None:
    for unit in doc.get("canonicalization", {}).get("units", []):
        if unit.get("merged") and len(unit["member_indices"]) >= min_members:
            return unit
    return None


def build_examples(
    transitions: list[dict[str, str]],
    trajectories: list[dict[str, str]],
    canonical: Path,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    cache: dict[str, dict[str, Any]] = {}

    def doc_for(rid: str) -> dict[str, Any]:
        if rid not in cache:
            cache[rid] = load_recording(canonical, rid)
        return cache[rid]

    def add_example(
        example_id: str,
        category: str,
        recording_id: str,
        *,
        raw_segments: list[dict[str, Any]],
        canonical_segments: list[dict[str, Any]],
        rule: str,
        ambiguous: bool = False,
        notes: str = "",
    ) -> None:
        examples.append(
            {
                "example_id": example_id,
                "category": category,
                "recording_id": recording_id,
                "raw": raw_segments,
                "canonical": canonical_segments,
                "rule": rule,
                "ambiguous": ambiguous,
                "notes": notes,
            }
        )

    # 1. T0 same-pitch 2-member merge
    rid = "645ff354deeaf2d1e33b3c44"
    doc = doc_for(rid)
    unit = next(
        u
        for u in doc["canonicalization"]["units"]
        if u["canonical_unit_id"] == "645ff354deeaf2d1e33b3c44:u0129"
    )
    by_idx = traj_by_index(doc)
    raw_segs = [summarize_trajectory(by_idx[i]) for i in unit["member_indices"]]
    add_example(
        "t0_same_pitch_merge_2",
        "T0 same-pitch merge",
        rid,
        raw_segments=raw_segs,
        canonical_segments=[
            {
                "member_indices": unit["member_indices"],
                "type_id": unit["type_id"],
                "pitch_key": unit["pitch_key"],
                "start_s": unit["start_s"],
                "end_s": unit["end_s"],
                "duration_s": unit["duration_s"],
            }
        ],
        rule="MERGE contiguous T0 with identical pitch_key",
    )

    # 2. T0 same-pitch 4-member merge
    rid = "6491d48d608d1718e0311003"
    doc = doc_for(rid)
    unit = next(
        u
        for u in doc["canonicalization"]["units"]
        if u["canonical_unit_id"] == "6491d48d608d1718e0311003:u0033"
    )
    by_idx = traj_by_index(doc)
    raw_segs = [summarize_trajectory(by_idx[i]) for i in unit["member_indices"]]
    add_example(
        "t0_same_pitch_merge_4",
        "T0 same-pitch merge (run of 4)",
        rid,
        raw_segments=raw_segs,
        canonical_segments=[
            {
                "member_indices": unit["member_indices"],
                "type_id": unit["type_id"],
                "pitch_key": unit["pitch_key"],
                "duration_s": unit["duration_s"],
            }
        ],
        rule="MERGE contiguous T0 with identical pitch_key",
    )

    # 3. T0 diff-pitch kept distinct
    row = next(
        t
        for t in transitions
        if t["both_type_0"] == "True"
        and t["same_pitch"] == "False"
        and t["recording_id"] == "6417585554a0bfbd8de2d3ff"
        and t["from_index"] == "396"
    )
    doc = doc_for(row["recording_id"])
    by_idx = traj_by_index(doc)
    left = summarize_trajectory(by_idx[int(row["from_index"])])
    right = summarize_trajectory(by_idx[int(row["to_index"])])
    add_example(
        "t0_diff_pitch_keep",
        "T0 different pitch — keep separate",
        row["recording_id"],
        raw_segments=[left, right],
        canonical_segments=[left, right],
        rule="KEEP — geometric pitch step at boundary",
        notes=f"pitch_delta_cents={row['pitch_delta_cents']}",
    )

    # 4. T0 diff-pitch large jump
    large = max(
        (
            t
            for t in transitions
            if t["both_type_0"] == "True" and t["same_pitch"] == "False"
        ),
        key=lambda t: abs(float(t["pitch_delta_cents"] or 0)),
    )
    doc = doc_for(large["recording_id"])
    by_idx = traj_by_index(doc)
    left = summarize_trajectory(by_idx[int(large["from_index"])])
    right = summarize_trajectory(by_idx[int(large["to_index"])])
    add_example(
        "t0_diff_pitch_large",
        "T0 different pitch — large jump",
        large["recording_id"],
        raw_segments=[left, right],
        canonical_segments=[left, right],
        rule="KEEP — geometric pitch step at boundary",
        notes=f"pitch_delta_cents={large['pitch_delta_cents']}",
    )

    # 5. T3 -> T0 same pitch
    row = next(
        t
        for t in transitions
        if t["from_type_id"] == "3"
        and t["to_type_id"] == "0"
        and t["same_pitch"] == "True"
    )
    doc = doc_for(row["recording_id"])
    by_idx = traj_by_index(doc)
    left = summarize_trajectory(by_idx[int(row["from_index"])])
    right = summarize_trajectory(by_idx[int(row["to_index"])])
    add_example(
        "t3_to_t0_same_pitch",
        "Different type, same pitch",
        row["recording_id"],
        raw_segments=[left, right],
        canonical_segments=[
            {**left, "target_type": 3},
            {**right, "target_type": 0},
        ],
        rule="KEEP both — type change is a canonical ML boundary even without pitch step",
    )

    # 6. T6 -> T6 same pitch (ambiguous)
    row = next(
        t
        for t in transitions
        if t["from_type_id"] == "6"
        and t["to_type_id"] == "6"
        and t["same_pitch"] == "True"
    )
    doc = doc_for(row["recording_id"])
    by_idx = traj_by_index(doc)
    left = summarize_trajectory(by_idx[int(row["from_index"])])
    right = summarize_trajectory(by_idx[int(row["to_index"])])
    add_example(
        "t6_to_t6_same_pitch",
        "T6 same-pitch consecutive split",
        row["recording_id"],
        raw_segments=[left, right],
        canonical_segments=[left, right],
        rule="KEEP (ambiguous) — no deterministic merge without over-merging Simple Multiple runs",
        ambiguous=True,
    )

    # 7. Type 4 Ladle
    row = next(t for t in trajectories if t["type_id"] == "4")
    doc = doc_for(row["recording_id"])
    entry = next(
        item for item in doc["trajectories"] if item["index"] == int(row["index"])
    )
    raw = summarize_trajectory(entry)
    dur_array = entry["raw"].get("dur_array") or []
    labels = COMPOSITE_LABELS[4]
    fracs = [0.0]
    for frac in dur_array[: len(labels)]:
        fracs.append(fracs[-1] + float(frac))
    fracs[-1] = 1.0
    canonical_segs = []
    for i, label in enumerate(labels):
        canonical_segs.append(
            {
                "target_type": label,
                "start_frac": fracs[i],
                "end_frac": fracs[i + 1],
                "within_index": entry["index"],
            }
        )
    add_example(
        "type4_ladle_map",
        "Type 4 Ladle decomposition",
        row["recording_id"],
        raw_segments=[raw],
        canonical_segments=canonical_segs,
        rule="MAP_TYPE 4 -> [2, 1] using dur_array fractions; contour unchanged",
    )

    # 8. Type 5 Reverse Ladle
    row = next(t for t in trajectories if t["type_id"] == "5")
    doc = doc_for(row["recording_id"])
    entry = next(
        item for item in doc["trajectories"] if item["index"] == int(row["index"])
    )
    raw = summarize_trajectory(entry)
    dur_array = entry["raw"].get("dur_array") or []
    labels = COMPOSITE_LABELS[5]
    fracs = [0.0]
    for frac in dur_array[: len(labels)]:
        fracs.append(fracs[-1] + float(frac))
    fracs[-1] = 1.0
    canonical_segs = [
        {
            "target_type": label,
            "start_frac": fracs[i],
            "end_frac": fracs[i + 1],
            "within_index": entry["index"],
        }
        for i, label in enumerate(labels)
    ]
    add_example(
        "type5_reverse_ladle_map",
        "Type 5 Reverse Ladle decomposition",
        row["recording_id"],
        raw_segments=[raw],
        canonical_segments=canonical_segs,
        rule="MAP_TYPE 5 -> [1, 3] using dur_array fractions; contour unchanged",
    )

    # 9. Type 6 Simple Multiple — keep composite
    row = next(
        t
        for t in trajectories
        if t["type_id"] == "6" and float(t["duration_s"]) > 0.5
    )
    doc = doc_for(row["recording_id"])
    entry = next(
        item for item in doc["trajectories"] if item["index"] == int(row["index"])
    )
    raw = summarize_trajectory(entry)
    add_example(
        "type6_keep_composite",
        "Type 6 Simple Multiple — keep as one unit",
        row["recording_id"],
        raw_segments=[raw],
        canonical_segments=[
            {
                "target_type": 1,
                "member_index": entry["index"],
                "n_internal_segments": raw["dur_array_len"],
                "note": "internal dur_array boundaries are parametric, not geometric",
            }
        ],
        rule="KEEP_COMPOSITE labeled as target_type=1",
        ambiguous=True,
        notes="Alternative MAP_TYPE decomposition exists but is not unique",
    )

    # 10. Type 7 Krintin — mask
    row = next(t for t in trajectories if t["type_id"] == "7")
    doc = doc_for(row["recording_id"])
    entry = next(
        item for item in doc["trajectories"] if item["index"] == int(row["index"])
    )
    raw = summarize_trajectory(entry)
    kinks = internal_kinks(
        entry,
        ratios=doc["raga"]["stratified_ratios"],
        fundamental=float(doc["raga"]["fundamental_hz"]),
    )
    add_example(
        "type7_mask",
        "Type 7 Krintin — excluded from 4-class target",
        row["recording_id"],
        raw_segments=[raw],
        canonical_segments=[{"masked": True, "reason": "SKIP_IDTAP_IDS"}],
        rule="MASK/EXCLUDE — outside four-class vocabulary",
        notes=f"max_internal_kink_log2={max(kinks) if kinks else 0:.4f}",
    )

    # 11. Type 13 Vibrato — mask
    row = next(t for t in trajectories if t["type_id"] == "13")
    doc = doc_for(row["recording_id"])
    entry = next(
        item for item in doc["trajectories"] if item["index"] == int(row["index"])
    )
    raw = summarize_trajectory(entry)
    add_example(
        "type13_mask",
        "Type 13 Vibrato — excluded from 4-class target",
        row["recording_id"],
        raw_segments=[raw],
        canonical_segments=[{"masked": True, "reason": "SKIP_IDTAP_IDS"}],
        rule="MASK/EXCLUDE — outside four-class vocabulary",
    )

    # 12. Silent
    row = next(t for t in trajectories if t["type_id"] == "12")
    doc = doc_for(row["recording_id"])
    entry = next(
        item for item in doc["trajectories"] if item["index"] == int(row["index"])
    )
    raw = summarize_trajectory(entry)
    add_example(
        "type12_silent_mask",
        "Type 12 Silent — mask for type/pitch loss",
        row["recording_id"],
        raw_segments=[raw],
        canonical_segments=[{"masked": True, "reason": "silent annotation"}],
        rule="MASK — do not label as inactive; exclude from shape/pitch supervision",
    )

    # 13. Vowel-diff T0 merge
    merge_info = analyze_t0_merges(transitions, canonical)
    if merge_info["vowel_diff_examples"]:
        ex = merge_info["vowel_diff_examples"][0]
        doc = doc_for(ex["recording_id"])
        by_idx = traj_by_index(doc)
        left = summarize_trajectory(by_idx[ex["from_index"]])
        right = summarize_trajectory(by_idx[ex["to_index"]])
        add_example(
            "t0_merge_vowel_diff",
            "T0 same-pitch merge with vowel metadata change",
            ex["recording_id"],
            raw_segments=[left, right],
            canonical_segments=[
                {
                    "member_indices": [ex["from_index"], ex["to_index"]],
                    "type_id": 0,
                    "pitch_key": ex["pitch_key"],
                    "duration_s": ex["dur_left_s"] + ex["dur_right_s"],
                }
            ],
            rule="MERGE for pitch/type targets; symbolic vowel boundary lost",
            ambiguous=True,
            notes=f"vowel {ex['vowel_left']!r} -> {ex['vowel_right']!r}",
        )

    return examples


def count_canonical_units(canonical: Path) -> dict[str, Any]:
    n_raw = 0
    n_units = 0
    n_merged_units = 0
    n_fixed_raw = 0
    fixed_units: set[str] = set()
    for path in sorted((canonical / "recordings").glob("*.json")):
        doc = json.loads(path.read_text())
        n_raw += len(doc["trajectories"])
        canon = doc.get("canonicalization", {})
        n_units += canon.get("n_canonical_units", 0)
        n_merged_units += canon.get("n_merged_units", 0)
        for entry in doc["trajectories"]:
            if entry["derived"]["type_id"] == FIXED_TRAJECTORY_ID:
                n_fixed_raw += 1
                unit_id = entry["derived"].get("canonical_unit_id")
                if unit_id:
                    fixed_units.add(unit_id)
    return {
        "n_raw_trajectories": n_raw,
        "n_canonical_units_current_overlay": n_units,
        "n_merged_units": n_merged_units,
        "n_fixed_raw": n_fixed_raw,
        "n_fixed_canonical_units": len(fixed_units),
        "boundaries_removed_by_t0_merge": n_fixed_raw - len(fixed_units),
    }


def run_analysis(canonical: Path) -> dict[str, Any]:
    transitions = load_csv(canonical / "index" / "transitions.csv")
    trajectories = load_csv(canonical / "index" / "trajectories.csv")
    non_silent_types = Counter(
        int(row["type_id"])
        for row in trajectories
        if row["is_silent_annotation"] == "False"
    )
    return {
        "pitch_tolerance": {
            "rule": "exact pitch_key match (swara|oct|raised|log_offset)",
            "source": "dataset/canonical/schema.py derivation_params.pitch_match_rule",
        },
        "transitions": analyze_transitions(transitions),
        "t0_merge_analysis": analyze_t0_merges(transitions, canonical),
        "composite_analysis": analyze_composites(trajectories, canonical),
        "canonical_unit_counts": count_canonical_units(canonical),
        "non_silent_type_counts": {
            str(k): v for k, v in sorted(non_silent_types.items())
        },
        "examples": build_examples(transitions, trajectories, canonical),
        "mapping_reference": {
            "legacy_four_class": "inventory/dataset_utils.py TARGET_IDTAP_IDS = {0,1,2,3}",
            "cnn_five_class": "training/spec_dataset.py FULL_LABEL_TO_IDX",
            "export_rules": "dataset/export_denoised_cnn_dataset.py iter_labeled_segments",
            "skip_ids": sorted(SKIP_IDTAP_IDS),
            "composite_labels": COMPOSITE_LABELS,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=canonical_root(REPO_ROOT),
        help="Path to output/canonical/v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path (default: print summary to stdout)",
    )
    args = parser.parse_args()
    result = run_analysis(args.canonical_root)
    payload = json.dumps(result, indent=2, sort_keys=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
