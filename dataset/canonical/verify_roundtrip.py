"""Prove the ``raw`` block lost no shape parameter.

The whole point of storing IDTAP's annotation verbatim is that a consumer can
rebuild the annotation itself — in particular that framewise f0 targets can be
regenerated with ``Trajectory.compute(x, log_scale=True)`` without re-fetching
anything. This script checks that claim directly:

1. Rebuild an ``idtap`` ``Trajectory`` from a canonical document's ``raw`` block
   alone, using only the fields that ``raw.wire_keys_present`` says existed on
   the wire (plus the recording's own ``raga`` block, which is where the raw
   layer deliberately keeps ratios and fundamental instead of duplicating them
   on every pitch).
2. Build the reference ``Trajectory`` **straight from the cached wire JSON** in
   ``raw_api/`` and compare shape parameters and contours over a dense grid.

The reference is built from the wire and not by walking a ``Piece`` on purpose:
``Piece.from_json`` calls ``Phrase.consolidate_silent_trajs``, which merges runs
of silent trajectories, so a ``Piece``'s trajectory list can be shorter than the
wire's and the indices would not line up.

Run from the repository root::

    python dataset/canonical/verify_roundtrip.py
    python dataset/canonical/verify_roundtrip.py --sample-per-type 25
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from idtap.classes.trajectory import Trajectory  # noqa: E402

from dataset.canonical.build import (  # noqa: E402
    TRAJ_NESTED_FIELDS,
    TRAJ_WIRE_FIELDS,
    raw_cache_path,
)
from dataset.canonical.schema import (  # noqa: E402
    SILENT_TRAJECTORY_ID,
    TRAJECTORY_ID_TO_NAME,
    recordings_dir,
    traj_id,
)
from dataset.canonical.timing import iter_placements  # noqa: E402

DEFAULT_GRID_POINTS = 201
CONTOUR_TOLERANCE = 1e-9

# ``raw`` also carries bookkeeping that is ours, not IDTAP's; the constructor
# rejects unknown keys, and passing them would weaken the test anyway.
NON_WIRE_RAW_FIELDS = frozenset(
    {
        "track_index",
        "string_index",
        "phrase_index",
        "wire_keys_present",
        "pitch_wire_keys_present",
    }
)

# Everything that determines the contour, plus the annotation payload that rides
# along with it. ``unique_id`` is excluded: the schema stores null when the wire
# had none, and the library mints a fresh UUID each load.
SHAPE_ATTRS = (
    "id",
    "dur_tot",
    "dur_array",
    "slope",
    "vib_obj",
    "fund_id12",
    "num",
    "start_time",
    "group_id",
    "vowel",
    "start_consonant",
    "end_consonant",
    "tags",
)


class RoundTripError(AssertionError):
    """Raised when a rebuilt trajectory disagrees with the wire it came from."""


# ---------------------------------------------------------------------------
# reconstruction
# ---------------------------------------------------------------------------


def raw_to_wire(raw: dict[str, Any]) -> dict[str, Any]:
    """Re-emit the wire trajectory dict from a canonical ``raw`` block.

    A field is emitted only when ``raw.wire_keys_present`` says its key existed,
    so a value that was absent on the wire stays absent rather than becoming an
    explicit null — which matters because the ``Trajectory`` constructor
    distinguishes "missing" from "None" for ``tags``, ``dur_array`` and friends.
    """
    present = set(raw.get("wire_keys_present") or [])
    wire: dict[str, Any] = {}
    for field, key in {**TRAJ_WIRE_FIELDS, **TRAJ_NESTED_FIELDS}.items():
        if key in present:
            wire[key] = raw.get(field)
    wire["pitches"] = [dict(p) for p in raw.get("pitches") or []]
    return wire


def build_from_raw(
    raw: dict[str, Any],
    *,
    ratios: list[Any],
    fundamental: float,
) -> Trajectory:
    """The reconstruction under test: raw block plus the recording's raga."""
    return Trajectory.from_json(
        raw_to_wire(raw), ratios=ratios, fundamental=fundamental
    )


def build_from_wire(
    wire: dict[str, Any],
    *,
    ratios: list[Any],
    fundamental: float,
) -> Trajectory:
    """The reference, built from the cached verbatim response."""
    return Trajectory.from_json(
        json.loads(json.dumps(wire)), ratios=ratios, fundamental=fundamental
    )


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------


def compare_shape(rebuilt: Trajectory, reference: Trajectory) -> list[str]:
    """Every shape parameter that differs, described in full."""
    problems = []
    for attr in SHAPE_ATTRS:
        left, right = getattr(rebuilt, attr), getattr(reference, attr)
        if left != right:
            problems.append(f"{attr}: rebuilt {left!r} != wire {right!r}")

    if len(rebuilt.pitches) != len(reference.pitches):
        problems.append(
            f"pitches: rebuilt {len(rebuilt.pitches)} != wire {len(reference.pitches)}"
        )
    else:
        for position, (a, b) in enumerate(zip(rebuilt.pitches, reference.pitches)):
            if (a.swara, a.oct, a.raised, a.log_offset) != (
                b.swara,
                b.oct,
                b.raised,
                b.log_offset,
            ):
                problems.append(f"pitches[{position}]: rebuilt {a.to_json()} != wire {b.to_json()}")
            elif a.frequency != b.frequency:
                problems.append(
                    f"pitches[{position}].frequency: rebuilt {a.frequency} != wire {b.frequency}"
                )

    left_arts = {k: v.to_json() for k, v in rebuilt.articulations.items()}
    right_arts = {k: v.to_json() for k, v in reference.articulations.items()}
    if left_arts != right_arts:
        problems.append(f"articulations: rebuilt {left_arts} != wire {right_arts}")

    left_auto = rebuilt.automation.to_json() if rebuilt.automation else None
    right_auto = reference.automation.to_json() if reference.automation else None
    if left_auto != right_auto:
        problems.append(f"automation: rebuilt {left_auto} != wire {right_auto}")
    return problems


def _compute(trajectory: Trajectory, x: float) -> tuple[float | None, str | None]:
    try:
        return trajectory.compute(x, log_scale=True), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def compare_contour(
    rebuilt: Trajectory,
    reference: Trajectory,
    *,
    grid_points: int,
) -> dict[str, Any]:
    """Max absolute log2-Hz difference over a dense grid, plus any mismatch."""
    problems: list[str] = []
    max_delta: float | None = None
    values: list[float] = []
    n_errors = 0
    for step in range(grid_points):
        x = step / (grid_points - 1)
        left, left_error = _compute(rebuilt, x)
        right, right_error = _compute(reference, x)
        if left_error or right_error:
            # Both sides must fail the same way; a one-sided failure is a lost
            # parameter (e.g. a silent trajectory whose fundamental went missing).
            n_errors += 1
            if left_error != right_error:
                problems.append(
                    f"x={x:.4f}: rebuilt raised {left_error}, wire raised {right_error}"
                )
            continue
        delta = abs(left - right)
        max_delta = delta if max_delta is None else max(max_delta, delta)
        values.append(left)
        if delta > CONTOUR_TOLERANCE or not math.isfinite(delta):
            problems.append(
                f"x={x:.4f}: rebuilt log2 {left!r} != wire log2 {right!r} (delta {delta})"
            )
    return {
        "max_log2_delta": max_delta,
        "n_grid_errors": n_errors,
        "contour_is_constant": bool(values) and max(values) == min(values),
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# walking the corpus
# ---------------------------------------------------------------------------


def wire_by_traj_id(recording_id: str, piece_json: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map each canonical ``traj_id`` back to the wire dict it was built from.

    Uses the builder's own placement walk, so alignment cannot drift from the
    document, and asserts the ids are unique.
    """
    out: dict[str, dict[str, Any]] = {}
    for placement in iter_placements(piece_json):
        key = traj_id(
            recording_id,
            placement["track_index"],
            placement["string_index"],
            placement["phrase_index"],
            placement["num"],
        )
        if key in out:
            raise RoundTripError(f"{recording_id}: duplicate traj_id {key} in the wire")
        out[key] = placement["wire"]
    return out


def select(
    trajectories: Iterable[dict[str, Any]],
    *,
    sample_per_type: int,
    seed: int,
) -> list[dict[str, Any]]:
    """All trajectories, or a seeded per-type sample spread across the corpus."""
    items = list(trajectories)
    if sample_per_type <= 0:
        return items
    by_type: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_type[item["derived"]["type_id"]].append(item)
    chosen: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for type_id in sorted(by_type):
        members = by_type[type_id]
        chosen.extend(
            members
            if len(members) <= sample_per_type
            else rng.sample(members, sample_per_type)
        )
    return chosen


def verify_recording(
    doc: dict[str, Any],
    piece_json: dict[str, Any],
    *,
    grid_points: int,
    sample_per_type: int,
    seed: int,
) -> list[dict[str, Any]]:
    recording_id = doc["recording_id"]
    ratios = doc["raga"]["stratified_ratios"]
    fundamental = float(doc["raga"]["fundamental_hz"])
    wires = wire_by_traj_id(recording_id, piece_json)

    results: list[dict[str, Any]] = []
    for trajectory in select(
        doc["trajectories"], sample_per_type=sample_per_type, seed=seed
    ):
        key = trajectory["traj_id"]
        wire = wires.get(key)
        if wire is None:
            raise RoundTripError(f"{recording_id}: no wire trajectory for {key}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rebuilt = build_from_raw(
                trajectory["raw"], ratios=ratios, fundamental=fundamental
            )
            reference = build_from_wire(wire, ratios=ratios, fundamental=fundamental)

        shape_problems = compare_shape(rebuilt, reference)
        contour = compare_contour(rebuilt, reference, grid_points=grid_points)
        results.append(
            {
                "recording_id": recording_id,
                "traj_id": key,
                "type_id": trajectory["derived"]["type_id"],
                "n_pitches": len(trajectory["raw"]["pitches"]),
                "is_silent_with_pitches": (
                    trajectory["derived"]["is_silent_annotation"]
                    and bool(trajectory["raw"]["pitches"])
                ),
                "max_log2_delta": contour["max_log2_delta"],
                "n_grid_errors": contour["n_grid_errors"],
                "contour_is_constant": contour["contour_is_constant"],
                "problems": shape_problems + contour["problems"],
            }
        )
    return results


# ---------------------------------------------------------------------------
# dispatch-table facts worth asserting once
# ---------------------------------------------------------------------------


def check_dispatch_table() -> dict[str, Any]:
    """``id == 11`` has no method of its own; the table aliases ``id7``."""
    probe = Trajectory()
    return {
        "has_id11_method": hasattr(Trajectory, "id11"),
        "id11_aliases_id7": probe.ids[11].__func__ is Trajectory.id7,
        "n_dispatch_entries": len(probe.ids),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _report(results: list[dict[str, Any]]) -> int:
    by_type: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_type[result["type_id"]].append(result)

    print(f"{'type':>5}  {'name':<22} {'n':>5}  {'max log2 delta':>15}  status")
    failures = 0
    for type_id in sorted(by_type):
        members = by_type[type_id]
        bad = [m for m in members if m["problems"]]
        failures += len(bad)
        deltas = [m["max_log2_delta"] for m in members if m["max_log2_delta"] is not None]
        max_delta = max(deltas) if deltas else None
        status = "OK" if not bad else f"{len(bad)} FAILED"
        n_constant = sum(1 for m in members if m["contour_is_constant"])
        if type_id == SILENT_TRAJECTORY_ID:
            # id12 returns the constant fund_id12, so its contour check is
            # degenerate by construction; the ones that compute nothing at all
            # are the annotations that carry no fundamental to return.
            uncomputable = sum(1 for m in members if m["n_grid_errors"])
            status += f" ({n_constant} constant, {uncomputable} with no fund_id12)"
        print(
            f"{type_id:>5}  {TRAJECTORY_ID_TO_NAME.get(type_id, '?'):<22} "
            f"{len(members):>5}  "
            f"{'n/a' if max_delta is None else format(max_delta, '.3e'):>15}  {status}"
        )

    silent_with_pitches = [r for r in results if r["is_silent_with_pitches"]]
    print(
        f"\nsilent annotations carrying a pitches array: {len(silent_with_pitches)} "
        f"({sum(1 for r in silent_with_pitches if r['problems'])} failed)"
    )

    missing = sorted(set(TRAJECTORY_ID_TO_NAME) - set(by_type))
    if missing:
        print(
            "types absent from the corpus, so not exercised: "
            + ", ".join(f"{t} ({TRAJECTORY_ID_TO_NAME[t]})" for t in missing)
        )

    for result in results:
        for problem in result["problems"][:5]:
            print(f"  FAIL {result['traj_id']}: {problem}", file=sys.stderr)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--recording-ids", nargs="*", default=None)
    parser.add_argument("--grid-points", type=int, default=DEFAULT_GRID_POINTS)
    parser.add_argument(
        "--sample-per-type",
        type=int,
        default=0,
        help="Trajectories per type to check (default: 0, meaning every one)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo_root: Path = args.repo_root
    paths = sorted(recordings_dir(repo_root).glob("*.json"))
    if args.recording_ids:
        wanted = set(args.recording_ids)
        paths = [p for p in paths if p.stem in wanted]
    if not paths:
        raise SystemExit("no canonical recording documents found")

    dispatch = check_dispatch_table()
    print(
        f"dispatch table: {dispatch['n_dispatch_entries']} entries, "
        f"Trajectory.id11 defined={dispatch['has_id11_method']}, "
        f"ids[11] is id7={dispatch['id11_aliases_id7']}\n"
    )
    if dispatch["has_id11_method"] or not dispatch["id11_aliases_id7"]:
        raise SystemExit("dispatch table no longer aliases id 11 onto id7")

    results: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            doc = json.load(handle)
        cache = raw_cache_path(repo_root, doc["recording_id"])
        with gzip.open(cache, "rt", encoding="utf-8") as handle:
            piece_json = json.load(handle)
        results.extend(
            verify_recording(
                doc,
                piece_json,
                grid_points=args.grid_points,
                sample_per_type=args.sample_per_type,
                seed=args.seed,
            )
        )

    failures = _report(results)
    print(
        f"\n{len(results)} trajectories checked across {len(paths)} recordings "
        f"on a {args.grid_points}-point grid: {len(results) - failures} passed, "
        f"{failures} failed"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
