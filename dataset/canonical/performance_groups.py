"""Group IDTAP transcriptions into performances, so splits never leak the same audio.

Two transcriptions of one performance must never land on opposite sides of a
train/test split. Recording ids do not express that: 12 audioIDs in the corpus
carry 2-3 separate transcriptions, and a few performances were even re-uploaded
under fresh audioIDs (`(clone)`, `(denoised)`). This module assigns every
transcription a `performance_group_id`; the group, never the recording, is the
atomic unit a splitter is allowed to divide.

Grouping is a union-find over three merge rules, applied in this order:

1. **Shared `audioID`.** Byte-identical audio, so this is not a heuristic.
2. **Same normalized (soloist, raga) plus a title match.** Both soloist and raga
   must be real names -- blank and placeholder values like ``other`` never match.
   The title match is deliberately narrow, see `title_key` below.
3. **Explicit groups from `overrides.json`**, hand-maintained for performances
   that neither rule can see (segmented uploads under different audioIDs).

`title_key` normalization, stated in full so a merge can be audited:

* NFKD-fold, lowercase, replace every non-alphanumeric character with a space.
* Drop tokens that come from fields the group is already keyed on or that
  describe the upload rather than the performance: tokens of the soloist name,
  tokens of the raga name, tokens of `soloInstrument` / `instrumentation`,
  provenance markers (`clone`, `denoised`, `test`, `transcription`, ...), and
  English filler (`the`, `of`, `for`, ...).
* Drop single-character alphabetic leftovers, keep digits.
* The key is the *set* of surviving tokens: order and repeats do not matter.
* **A title match requires a non-empty key on both sides.** A title that says
  nothing beyond the artist and the raga cannot merge anything -- otherwise the
  rule would degenerate into "same artist, same raga", which merges genuinely
  distinct performances.

No spelling normalization is applied, so `Chaap` / `Chhaap` and
`Raisili` / `Rasili` stay distinct. The rule under-merges on purpose: rule 1 is
the safety net for real duplicates and rule 3 is the escape hatch.

`performance_group_id` is derived from the group's contents, never minted:
``audio:<smallest audioID in the group>``, or ``rec:<smallest recording id>``
for groups whose members have no audio at all. Rebuilding the same corpus
therefore reproduces byte-identical ids regardless of input order or dict
iteration order. A group spanning several audioIDs is named after the smallest
of them, and admitting a new member with a smaller audioID does rename the
group -- ids identify group *content*, not a group lifetime.

Run as a script for an audit report (this is the only code path that hits the
network)::

    python dataset/canonical/performance_groups.py
    python dataset/canonical/performance_groups.py --json output/canonical/v1/performance_groups.json
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OVERRIDES_PATH = Path(__file__).resolve().parent / "overrides.json"
DEFAULT_EXPORTED_DIR = PROJECT_ROOT / "output" / "cnn_dataset"

RULE_SHARED_AUDIO = "shared_audio_id"
RULE_TITLE_MATCH = "soloist_raga_title_match"
RULE_OVERRIDE = "override"

# Soloist / raga values that name nobody. Rule 2 refuses to match on these.
PLACEHOLDER_NAMES = frozenset(
    {"", "other", "others", "unknown", "unspecified", "none", "n a", "na", "misc", "test"}
)

# Tokens that describe the upload, not the performance.
PROVENANCE_TOKENS = frozenset(
    {
        "clone",
        "cloned",
        "clones",
        "copy",
        "dup",
        "duplicate",
        "denoise",
        "denoised",
        "cleaned",
        "backup",
        "test",
        "tests",
        "testing",
        "demo",
        "draft",
        "wip",
        "final",
        "new",
        "old",
        "rev",
        "revised",
        "version",
        "v1",
        "v2",
        "transcription",
        "transcriptions",
        "transcribed",
        "transcript",
        "dn",
        "sip",
    }
)

FILLER_TOKENS = frozenset(
    {"a", "an", "the", "and", "of", "for", "in", "on", "by", "with", "raga", "rag", "raag"}
)


def _tokenize(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", str(text or "")).lower()
    cleaned = "".join(ch if ch.isalnum() else " " for ch in folded)
    return cleaned.split()


def normalize_name(value: str | None) -> str:
    """Whitespace/punctuation-insensitive form of a soloist or raga name."""
    return " ".join(_tokenize(value or ""))


def is_placeholder_name(value: str | None) -> bool:
    return normalize_name(value) in PLACEHOLDER_NAMES


def entry_recording_id(entry: dict) -> str:
    return str(entry["_id"])


def entry_audio_id(entry: dict) -> str | None:
    audio_id = entry.get("audioID")
    return str(audio_id) if audio_id else None


def entry_raga_name(entry: dict) -> str:
    raga = entry.get("raga") or {}
    if isinstance(raga, dict):
        return str(raga.get("name") or "")
    return str(getattr(raga, "name", "") or "")


def entry_instrument_tokens(entry: dict) -> set[str]:
    tokens: set[str] = set()
    tokens.update(_tokenize(entry.get("soloInstrument") or ""))
    instrumentation = entry.get("instrumentation") or []
    if isinstance(instrumentation, (list, tuple)):
        for instrument in instrumentation:
            tokens.update(_tokenize(instrument if isinstance(instrument, str) else ""))
    else:
        tokens.update(_tokenize(str(instrumentation)))
    return tokens


def title_key(entry: dict) -> tuple[str, ...]:
    """Sorted tuple of the title tokens that carry performance identity.

    Empty when the title says nothing beyond soloist, raga, instrument and
    provenance markers; an empty key never matches anything.
    """
    dropped = set(PROVENANCE_TOKENS) | set(FILLER_TOKENS)
    dropped.update(_tokenize(entry.get("soloist") or ""))
    dropped.update(_tokenize(entry_raga_name(entry)))
    dropped.update(entry_instrument_tokens(entry))

    kept = {
        token
        for token in _tokenize(entry.get("title") or "")
        if token not in dropped and (len(token) > 1 or token.isdigit())
    }
    return tuple(sorted(kept))


def load_overrides(path: Path | None = None) -> list[list[str]]:
    """Explicit groups of recording ids that must be unioned.

    Reads `overrides.json` (see that file for the documented structure).
    Returns `[]` when the file is absent.
    """
    overrides_path = Path(path) if path is not None else DEFAULT_OVERRIDES_PATH
    if not overrides_path.exists():
        return []
    with overrides_path.open() as f:
        payload = json.load(f)
    groups: list[list[str]] = []
    for group in payload.get("groups", []):
        ids = [str(rid) for rid in group.get("recording_ids", [])]
        if len(ids) > 1:
            groups.append(ids)
    return groups


def merge_edges(
    entries: list[dict],
    *,
    overrides: list[list[str]] | None = None,
    rules: frozenset[str] | None = None,
) -> list[dict]:
    """Every merge decision, as auditable records of `{rule, key, members}`.

    `rules` restricts which merge rules fire, which is how the report shows what
    the title heuristic contributes on its own.
    """
    active = rules if rules is not None else frozenset({RULE_SHARED_AUDIO, RULE_TITLE_MATCH, RULE_OVERRIDE})
    known_ids = {entry_recording_id(entry) for entry in entries}
    edges: list[dict] = []

    if RULE_SHARED_AUDIO in active:
        by_audio: dict[str, list[str]] = defaultdict(list)
        for entry in entries:
            audio_id = entry_audio_id(entry)
            if audio_id:
                by_audio[audio_id].append(entry_recording_id(entry))
        for audio_id, members in sorted(by_audio.items()):
            if len(members) > 1:
                edges.append(
                    {"rule": RULE_SHARED_AUDIO, "key": audio_id, "members": sorted(members)}
                )

    if RULE_TITLE_MATCH in active:
        by_title: dict[tuple[str, str, tuple[str, ...]], list[str]] = defaultdict(list)
        for entry in entries:
            soloist = normalize_name(entry.get("soloist"))
            raga = normalize_name(entry_raga_name(entry))
            key = title_key(entry)
            if not key or is_placeholder_name(soloist) or is_placeholder_name(raga):
                continue
            by_title[(soloist, raga, key)].append(entry_recording_id(entry))
        for (soloist, raga, key), members in sorted(by_title.items()):
            if len(members) > 1:
                edges.append(
                    {
                        "rule": RULE_TITLE_MATCH,
                        "key": f"{soloist} | {raga} | {' '.join(key)}",
                        "members": sorted(members),
                    }
                )

    if RULE_OVERRIDE in active:
        override_groups = load_overrides() if overrides is None else overrides
        for group in override_groups:
            members = sorted({rid for rid in group if rid in known_ids})
            if len(members) > 1:
                edges.append({"rule": RULE_OVERRIDE, "key": ",".join(members), "members": members})

    return edges


def _union_find(ids: list[str], edges: list[dict]) -> dict[str, list[str]]:
    """Map each id to its component's sorted member list."""
    parent = {rid: rid for rid in ids}

    def find(rid: str) -> str:
        root = rid
        while parent[root] != root:
            root = parent[root]
        while parent[rid] != root:
            parent[rid], rid = root, parent[rid]
        return root

    for edge in edges:
        members = [rid for rid in edge["members"] if rid in parent]
        for other in members[1:]:
            root_a, root_b = find(members[0]), find(other)
            if root_a != root_b:
                # Deterministic root choice, so components do not depend on edge order.
                low, high = sorted((root_a, root_b))
                parent[high] = low

    components: dict[str, list[str]] = defaultdict(list)
    for rid in sorted(parent):
        components[find(rid)].append(rid)
    return {root: sorted(members) for root, members in components.items()}


def performance_group_id(members: list[str], audio_ids_by_recording: dict[str, str | None]) -> str:
    """Content-derived id: smallest audioID in the group, else smallest recording id."""
    audio_ids = sorted(
        audio_ids_by_recording[rid] for rid in members if audio_ids_by_recording.get(rid)
    )
    if audio_ids:
        return f"audio:{audio_ids[0]}"
    return f"rec:{sorted(members)[0]}"


def assign_performance_groups(
    entries: list[dict],
    *,
    overrides: list[list[str]] | None = None,
    rules: frozenset[str] | None = None,
) -> dict[str, str]:
    """Map every recording id in `entries` to its `performance_group_id`.

    `entries` is the list returned by `SwaraClient.get_viewable_transcriptions()`.
    Pure and offline: identical input gives identical output, and input order
    does not matter.
    """
    recording_ids = sorted({entry_recording_id(entry) for entry in entries})
    audio_ids_by_recording = {
        entry_recording_id(entry): entry_audio_id(entry) for entry in entries
    }
    components = _union_find(recording_ids, merge_edges(entries, overrides=overrides, rules=rules))

    assignments: dict[str, str] = {}
    for members in components.values():
        group_id = performance_group_id(members, audio_ids_by_recording)
        for rid in members:
            assignments[rid] = group_id
    return {rid: assignments[rid] for rid in recording_ids}


def group_members(
    entries: list[dict],
    *,
    overrides: list[list[str]] | None = None,
    rules: frozenset[str] | None = None,
) -> dict[str, list[str]]:
    """Inverse of `assign_performance_groups`, with sorted keys and members."""
    assignments = assign_performance_groups(entries, overrides=overrides, rules=rules)
    members: dict[str, list[str]] = defaultdict(list)
    for rid, group_id in assignments.items():
        members[group_id].append(rid)
    return {group_id: sorted(members[group_id]) for group_id in sorted(members)}


def unjustified_title_merges(entries: list[dict], *, overrides: list[list[str]] | None = None) -> list[dict]:
    """Title-rule merges that shared audio and the overrides do not already imply.

    These are exactly the merges a human should sign off on, so the report lists
    them individually rather than only counting them.
    """
    safe_rules = frozenset({RULE_SHARED_AUDIO, RULE_OVERRIDE})
    safe = assign_performance_groups(entries, overrides=overrides, rules=safe_rules)
    full = group_members(entries, overrides=overrides)
    by_id = {entry_recording_id(entry): entry for entry in entries}

    findings: list[dict] = []
    for group_id, members in full.items():
        safe_groups = sorted({safe[rid] for rid in members})
        if len(safe_groups) < 2:
            continue
        findings.append(
            {
                "performance_group_id": group_id,
                "merged_subgroups": safe_groups,
                "members": [
                    {
                        "recording_id": rid,
                        "audio_id": entry_audio_id(by_id[rid]),
                        "title": by_id[rid].get("title"),
                        "soloist": by_id[rid].get("soloist"),
                        "raga": entry_raga_name(by_id[rid]),
                        "dur_tot_s": round(float(by_id[rid].get("durTot") or 0.0), 1),
                        "title_key": " ".join(title_key(by_id[rid])),
                    }
                    for rid in members
                ],
            }
        )
    return findings


def exported_recording_ids(exported_dir: Path = DEFAULT_EXPORTED_DIR) -> list[str]:
    """Recording ids already exported under `output/cnn_dataset/`."""
    if not exported_dir.exists():
        return []
    return sorted(
        d.name
        for d in exported_dir.iterdir()
        if d.is_dir() and d.name != "all" and not d.name.startswith(".")
    )


def build_report(entries: list[dict], *, exported_dir: Path = DEFAULT_EXPORTED_DIR) -> dict:
    overrides = load_overrides()
    assignments = assign_performance_groups(entries, overrides=overrides)
    members = group_members(entries, overrides=overrides)
    edges = merge_edges(entries, overrides=overrides)
    exported = exported_recording_ids(exported_dir)
    exported_groups = sorted({assignments[rid] for rid in exported if rid in assignments})

    return {
        "n_entries": len(entries),
        "n_recordings": len(assignments),
        "n_with_audio_id": sum(1 for entry in entries if entry_audio_id(entry)),
        "n_performance_groups": len(members),
        "n_multi_recording_groups": sum(1 for m in members.values() if len(m) > 1),
        "n_shared_audio_groups": sum(1 for e in edges if e["rule"] == RULE_SHARED_AUDIO),
        "n_title_match_edges": sum(1 for e in edges if e["rule"] == RULE_TITLE_MATCH),
        "n_override_edges": sum(1 for e in edges if e["rule"] == RULE_OVERRIDE),
        "exported": {
            "dir": str(exported_dir),
            "n_recordings": len(exported),
            "n_performance_groups": len(exported_groups),
            "missing_from_api": [rid for rid in exported if rid not in assignments],
            "groups": {
                group_id: [rid for rid in members[group_id] if rid in exported]
                for group_id in exported_groups
            },
        },
        "merge_edges": edges,
        "unjustified_title_merges": unjustified_title_merges(entries, overrides=overrides),
        "assignments": assignments,
        "group_members": members,
    }


def print_report(report: dict, entries: list[dict]) -> None:
    by_id = {entry_recording_id(entry): entry for entry in entries}

    def describe(rid: str) -> str:
        entry = by_id.get(rid)
        if entry is None:
            return f"{rid}  (not in transcription list)"
        return (
            f"{rid}  audio={entry_audio_id(entry) or '-':<24} "
            f"{round(float(entry.get('durTot') or 0.0), 1):>8}s  {entry.get('title')!r}"
        )

    print(
        f"{report['n_entries']} transcriptions, "
        f"{report['n_with_audio_id']} with an audioID, "
        f"{report['n_performance_groups']} performance groups "
        f"({report['n_multi_recording_groups']} hold more than one transcription)"
    )
    print(
        f"merge edges: {report['n_shared_audio_groups']} shared-audio, "
        f"{report['n_title_match_edges']} title-match, "
        f"{report['n_override_edges']} override"
    )

    print("\nShared audioID groups")
    for edge in report["merge_edges"]:
        if edge["rule"] != RULE_SHARED_AUDIO:
            continue
        print(f"  {edge['key']}")
        for rid in edge["members"]:
            print(f"    {describe(rid)}")

    print("\nTitle-match edges")
    for edge in report["merge_edges"]:
        if edge["rule"] != RULE_TITLE_MATCH:
            continue
        print(f"  {edge['key']}")
        for rid in edge["members"]:
            print(f"    {describe(rid)}")

    print("\nOverride edges")
    for edge in report["merge_edges"]:
        if edge["rule"] != RULE_OVERRIDE:
            continue
        group_id = report["assignments"].get(edge["members"][0], "?")
        print(f"  {group_id}")
        for rid in edge["members"]:
            print(f"    {describe(rid)}")

    findings = report["unjustified_title_merges"]
    print(f"\nGroups the title heuristic created beyond audio/override evidence: {len(findings)}")
    for finding in findings:
        print(f"  {finding['performance_group_id']}  joins {finding['merged_subgroups']}")
        for member in finding["members"]:
            print(
                f"    {member['recording_id']}  audio={member['audio_id'] or '-':<24}"
                f"{member['dur_tot_s']:>8}s  {member['title']!r}  key={member['title_key']!r}"
            )

    exported = report["exported"]
    print(
        f"\nExported recordings under {exported['dir']}: "
        f"{exported['n_recordings']} recordings -> {exported['n_performance_groups']} performances"
    )
    for group_id, group in exported["groups"].items():
        marker = "  <-- multi-recording group" if len(group) > 1 else ""
        print(f"  {group_id}: {group}{marker}")
    if exported["missing_from_api"]:
        print(f"  not in transcription list: {exported['missing_from_api']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--exported-dir",
        type=Path,
        default=DEFAULT_EXPORTED_DIR,
        help="directory whose subdirectory names are already-exported recording ids",
    )
    parser.add_argument(
        "--entries-json",
        type=Path,
        default=None,
        help="read the transcription list from a JSON file instead of the IDTAP API",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="also write the full report (assignments, group members, merge edges) here",
    )
    args = parser.parse_args()

    if args.entries_json is not None:
        with args.entries_json.open() as f:
            entries = json.load(f)
    else:
        from idtap import SwaraClient

        entries = SwaraClient().get_viewable_transcriptions()

    report = build_report(entries, exported_dir=args.exported_dir)
    print_report(report, entries)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
