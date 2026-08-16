"""Grouped split manifests, written next to the data but never inside it.

Manifests live in ``output/canonical/v1/splits/<split_name>.json`` so re-splitting
never rewrites a recording document.

The atomic unit is the **performance**, never the recording. Twelve audioIDs in
the corpus carry two or three separate transcriptions, and the D.V. Paluskar
Bhopali upload was segmented across three audioIDs; splitting on ``recording_id``
would put byte-identical audio on both sides of a train/test boundary.
``performance_groups`` already settled that question and the builder stamped its
answer onto every recording document, so this module groups on
``performance.performance_group_id`` and refuses to run if any recording lacks
one.

Everything the manifest reports is read back from the index tables, never
recomputed from the IDTAP API:

* ``type_counts`` come from ``index/trajectories.csv`` and count every lane,
  because a count of annotations is not a duration and concurrent lanes cannot
  double-count it.
* ``non_silent_duration_s`` comes from the ``coverage`` block via
  ``index/recordings.csv``. Coverage scopes its silence statistics to a single
  lane (``coverage.silence_scope_lane_id``) precisely because concurrent lanes
  overlap in time, so this module uses coverage's own span and silent duration
  rather than pooling lanes itself. The same applies to the
  ``min_non_silent_fraction`` eligibility filter, which reads
  ``annotated_non_silent_fraction`` straight from coverage.

Imbalance is reported, never silently corrected: no split is reshuffled to
equalize class counts.

Run from the repository root::

    python dataset/canonical/splits.py
    python dataset/canonical/splits.py --min-non-silent-fraction 0.3
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.canonical.schema import index_dir, splits_dir  # noqa: E402

STRATEGY_RATIO = "grouped_by_performance"
STRATEGY_KFOLD = "grouped_kfold_by_performance"
GROUP_KEY = "performance_group_id"

DEFAULT_SEED = 42
DEFAULT_RATIOS: dict[str, float] = {"train": 0.6, "val": 0.2, "test": 0.2}
DEFAULT_K = 5

# Stated in the manifest so a consumer never has to guess what a number means.
STATS_DEFINITIONS = {
    "type_counts": (
        "Trajectories per raw type_id, counted across every lane of the "
        "recording, from index/trajectories.csv."
    ),
    "non_silent_duration_s": (
        "(coverage.annotation_end_s - coverage.annotation_start_s) minus "
        "coverage.silent_annotation_duration_s, i.e. scoped to "
        "coverage.silence_scope_lane_id rather than pooled across lanes."
    ),
    "n_performances": "Distinct performance_group_id values in the split.",
}


class LeakageError(AssertionError):
    """Raised when one audio_id or one performance group spans two splits."""


class GroupingError(AssertionError):
    """Raised when a recording carries no performance_group_id to split on."""


# ---------------------------------------------------------------------------
# reading the index tables
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run `python dataset/canonical/build.py` first"
        )
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def load_recordings(repo_root: Path) -> list[dict[str, Any]]:
    """One record per recording, carrying only what a split manifest needs."""
    rows = _read_csv(index_dir(repo_root) / "recordings.csv")
    out: list[dict[str, Any]] = []
    for row in rows:
        group_id = row.get(GROUP_KEY) or None
        if group_id is None:
            raise GroupingError(
                f"{row['recording_id']} has no {GROUP_KEY}; splitting on recording "
                "id would leak shared audio across splits"
            )
        start_s = _float(row.get("annotation_start_s")) or 0.0
        end_s = _float(row.get("annotation_end_s")) or 0.0
        silent_s = _float(row.get("silent_annotation_duration_s")) or 0.0
        out.append(
            {
                "recording_id": row["recording_id"],
                "performance_group_id": group_id,
                "audio_id": row.get("audio_id") or None,
                "non_silent_fraction": _float(row.get("annotated_non_silent_fraction")),
                "non_silent_duration_s": (end_s - start_s) - silent_s,
            }
        )
    out.sort(key=lambda r: r["recording_id"])
    return out


def load_type_counts(repo_root: Path) -> dict[str, Counter[int]]:
    """Per-recording trajectory-type histogram, pooled over every lane."""
    rows = _read_csv(index_dir(repo_root) / "trajectories.csv")
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    for row in rows:
        counts[row["recording_id"]][int(row["type_id"])] += 1
    return counts


# ---------------------------------------------------------------------------
# eligibility and grouping
# ---------------------------------------------------------------------------


def apply_eligibility(
    recordings: Sequence[dict[str, Any]],
    *,
    min_non_silent_fraction: float | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split the corpus into eligible recordings and excluded ones with reasons."""
    if min_non_silent_fraction is None:
        return list(recordings), []

    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for recording in recordings:
        fraction = recording["non_silent_fraction"]
        if fraction is None or fraction < min_non_silent_fraction:
            excluded.append(
                {
                    "recording_id": recording["recording_id"],
                    "reason": "non_silent_fraction_below_threshold",
                    "non_silent_fraction": fraction,
                }
            )
        else:
            eligible.append(recording)
    return eligible, excluded


def group_recordings(recordings: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    """``performance_group_id`` -> sorted recording ids, with sorted keys.

    Sorting on both levels is what makes the manifest independent of the order
    the index table happened to be written in.
    """
    members: dict[str, list[str]] = defaultdict(list)
    for recording in recordings:
        members[recording["performance_group_id"]].append(recording["recording_id"])
    return {group: sorted(members[group]) for group in sorted(members)}


def shuffled_groups(group_ids: Sequence[str], seed: int) -> list[str]:
    """Seeded permutation of the *sorted* group ids, so input order cannot leak in."""
    ordered = sorted(group_ids)
    random.Random(seed).shuffle(ordered)
    return ordered


# ---------------------------------------------------------------------------
# allocation
# ---------------------------------------------------------------------------


def allocate_by_ratio(
    group_ids: Sequence[str],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """Assign whole groups to named splits by largest remainder.

    Groups are the unit, matching the plan's "with 14 groups a 60/20/20 is
    roughly 8/3/3". No group is ever divided, and no split is rebalanced to
    equalize class counts.
    """
    names = list(ratios)
    ordered = shuffled_groups(group_ids, seed)
    total = len(ordered)

    exact = {name: total * ratios[name] for name in names}
    quota = {name: int(exact[name]) for name in names}
    leftover = total - sum(quota.values())
    remainders = sorted(
        names,
        key=lambda name: (-(exact[name] - quota[name]), names.index(name)),
    )
    for name in remainders[:leftover]:
        quota[name] += 1

    assignment: dict[str, str] = {}
    position = 0
    for name in names:
        for group_id in ordered[position : position + quota[name]]:
            assignment[group_id] = name
        position += quota[name]
    return assignment


def allocate_by_fold(group_ids: Sequence[str], k: int, seed: int) -> dict[str, str]:
    """Deal whole groups round-robin into ``k`` folds."""
    ordered = shuffled_groups(group_ids, seed)
    return {
        group_id: fold_name(position % k, k)
        for position, group_id in enumerate(ordered)
    }


def fold_name(index: int, k: int) -> str:
    return f"fold_{index}"


# ---------------------------------------------------------------------------
# statistics and leakage
# ---------------------------------------------------------------------------


def build_stats(
    split_names: Sequence[str],
    assignments: dict[str, str],
    recordings: Sequence[dict[str, Any]],
    type_counts: dict[str, Counter[int]],
) -> dict[str, dict[str, Any]]:
    """Per-split counts, histograms and non-silent duration. Nothing is balanced."""
    by_id = {r["recording_id"]: r for r in recordings}
    observed_types = sorted(
        {type_id for rid in assignments for type_id in type_counts.get(rid, {})}
    )

    stats: dict[str, dict[str, Any]] = {}
    for name in split_names:
        members = sorted(rid for rid, split in assignments.items() if split == name)
        histogram: Counter[int] = Counter()
        for rid in members:
            histogram.update(type_counts.get(rid, Counter()))
        stats[name] = {
            "n_recordings": len(members),
            "n_performances": len(
                {by_id[rid]["performance_group_id"] for rid in members}
            ),
            "type_counts": {str(t): histogram.get(t, 0) for t in observed_types},
            "non_silent_duration_s": sum(
                by_id[rid]["non_silent_duration_s"] for rid in members
            ),
        }
    return stats


def check_leakage(
    assignments: dict[str, str],
    recordings: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Compute — never assume — that no audio and no performance spans two splits."""
    by_id = {r["recording_id"]: r for r in recordings}

    splits_by_audio: dict[str, set[str]] = defaultdict(set)
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    for rid, split in assignments.items():
        recording = by_id[rid]
        if recording["audio_id"]:
            splits_by_audio[recording["audio_id"]].add(split)
        splits_by_group[recording["performance_group_id"]].add(split)

    straddling_audio = {
        audio_id: sorted(names)
        for audio_id, names in sorted(splits_by_audio.items())
        if len(names) > 1
    }
    straddling_groups = {
        group_id: sorted(names)
        for group_id, names in sorted(splits_by_group.items())
        if len(names) > 1
    }

    assertions = {
        "no_shared_audio_id": not straddling_audio,
        "no_shared_group": not straddling_groups,
        "n_audio_ids_checked": len(splits_by_audio),
        "n_groups_checked": len(splits_by_group),
        "audio_ids_in_multiple_splits": straddling_audio,
        "groups_in_multiple_splits": straddling_groups,
    }
    if straddling_audio or straddling_groups:
        raise LeakageError(
            f"audio ids spanning splits: {straddling_audio}; "
            f"performance groups spanning splits: {straddling_groups}"
        )
    return assertions


# ---------------------------------------------------------------------------
# manifests
# ---------------------------------------------------------------------------


def _manifest(
    *,
    split_name: str,
    strategy: str,
    seed: int,
    ratios: dict[str, float],
    min_non_silent_fraction: float | None,
    excluded: Sequence[dict[str, Any]],
    split_names: Sequence[str],
    group_assignment: dict[str, str],
    members: dict[str, list[str]],
    recordings: Sequence[dict[str, Any]],
    type_counts: dict[str, Counter[int]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    assignments = {
        rid: group_assignment[group_id]
        for group_id, rids in members.items()
        for rid in rids
    }
    assignments = {rid: assignments[rid] for rid in sorted(assignments)}

    manifest: dict[str, Any] = {
        "split_name": split_name,
        "strategy": strategy,
        "group_key": GROUP_KEY,
        "seed": seed,
        "ratios": ratios,
        "eligibility_filter": {"min_non_silent_fraction": min_non_silent_fraction},
        "excluded_recordings": list(excluded),
        "assignments": assignments,
        "group_members": members,
        "group_assignments": {
            group_id: group_assignment[group_id] for group_id in sorted(members)
        },
        "stats_definitions": STATS_DEFINITIONS,
        "stats_per_split": build_stats(
            split_names, assignments, recordings, type_counts
        ),
        "leakage_assertions": check_leakage(assignments, recordings),
    }
    if extra:
        manifest.update(extra)
    return manifest


def build_ratio_manifest(
    recordings: Sequence[dict[str, Any]],
    type_counts: dict[str, Counter[int]],
    *,
    split_name: str,
    ratios: dict[str, float] | None = None,
    seed: int = DEFAULT_SEED,
    min_non_silent_fraction: float | None = None,
) -> dict[str, Any]:
    ratios = dict(ratios or DEFAULT_RATIOS)
    eligible, excluded = apply_eligibility(
        recordings, min_non_silent_fraction=min_non_silent_fraction
    )
    members = group_recordings(eligible)
    return _manifest(
        split_name=split_name,
        strategy=STRATEGY_RATIO,
        seed=seed,
        ratios=ratios,
        min_non_silent_fraction=min_non_silent_fraction,
        excluded=excluded,
        split_names=list(ratios),
        group_assignment=allocate_by_ratio(list(members), ratios, seed),
        members=members,
        recordings=eligible,
        type_counts=type_counts,
    )


def build_kfold_manifest(
    recordings: Sequence[dict[str, Any]],
    type_counts: dict[str, Counter[int]],
    *,
    split_name: str,
    k: int = DEFAULT_K,
    seed: int = DEFAULT_SEED,
    min_non_silent_fraction: float | None = None,
) -> dict[str, Any]:
    """Every fold is a held-out test set; train on the other ``k - 1``.

    A single 20% test split of ~14 performances is 3 performances, so any
    headline metric is dominated by which ones landed there. This manifest is
    the antidote, not a replacement.
    """
    eligible, excluded = apply_eligibility(
        recordings, min_non_silent_fraction=min_non_silent_fraction
    )
    members = group_recordings(eligible)
    names = [fold_name(i, k) for i in range(k)]
    return _manifest(
        split_name=split_name,
        strategy=STRATEGY_KFOLD,
        seed=seed,
        ratios={name: 1.0 / k for name in names},
        min_non_silent_fraction=min_non_silent_fraction,
        excluded=excluded,
        split_names=names,
        group_assignment=allocate_by_fold(list(members), k, seed),
        members=members,
        recordings=eligible,
        type_counts=type_counts,
        extra={
            "k": k,
            "fold_usage": (
                "Each fold is the held-out set for one CV round; the training "
                "set is every recording not assigned to that fold."
            ),
        },
    )


def write_manifest(manifest: dict[str, Any], repo_root: Path) -> Path:
    """Write with sorted keys and a trailing newline, so two runs are byte-identical."""
    path = splits_dir(repo_root) / f"{manifest['split_name']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_summary(manifest: dict[str, Any]) -> None:
    print(f"{manifest['split_name']}  ({manifest['strategy']}, seed {manifest['seed']})")
    for name, stats in manifest["stats_per_split"].items():
        histogram = " ".join(
            f"{type_id}:{count}" for type_id, count in stats["type_counts"].items()
        )
        print(
            f"  {name:<8} {stats['n_recordings']:>3} recordings, "
            f"{stats['n_performances']:>2} performances, "
            f"{stats['non_silent_duration_s']:>9.1f} s non-silent"
        )
        print(f"           types {histogram}")
    assertions = manifest["leakage_assertions"]
    print(
        f"  leakage: no_shared_audio_id={assertions['no_shared_audio_id']} "
        f"({assertions['n_audio_ids_checked']} audio ids), "
        f"no_shared_group={assertions['no_shared_group']} "
        f"({assertions['n_groups_checked']} groups)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument(
        "--min-non-silent-fraction",
        type=float,
        default=None,
        help=(
            "Restrict to recordings whose coverage.annotated_non_silent_fraction "
            "is at least this (default: no filter)"
        ),
    )
    parser.add_argument(
        "--name-suffix",
        default="",
        help="Appended to both manifest names, for filtered split variants",
    )
    args = parser.parse_args()

    repo_root: Path = args.repo_root
    recordings = load_recordings(repo_root)
    type_counts = load_type_counts(repo_root)
    print(
        f"{len(recordings)} recordings -> "
        f"{len(group_recordings(recordings))} performance groups"
    )

    ratio_manifest = build_ratio_manifest(
        recordings,
        type_counts,
        split_name=f"grouped_60_20_20_seed{args.seed}{args.name_suffix}",
        seed=args.seed,
        min_non_silent_fraction=args.min_non_silent_fraction,
    )
    kfold_manifest = build_kfold_manifest(
        recordings,
        type_counts,
        split_name=f"grouped_kfold_k{args.k}_seed{args.seed}{args.name_suffix}",
        k=args.k,
        seed=args.seed,
        min_non_silent_fraction=args.min_non_silent_fraction,
    )

    for manifest in (ratio_manifest, kfold_manifest):
        path = write_manifest(manifest, repo_root)
        _print_summary(manifest)
        print(f"  -> {path.relative_to(repo_root)}\n")


if __name__ == "__main__":
    main()
