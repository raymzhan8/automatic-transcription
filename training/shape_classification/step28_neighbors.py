"""Step 28 sections 3, 7, 16: build the previous/next neighbor map for every
canonical primitive, and the TRAIN-only descriptive transition matrix.

Adjacency rule (calibrated empirically, not guessed): two primitives in the
SAME recording_id + lane_id, consecutive in start_s order, are neighbors
only if `next.start_s - cur.end_s < GAP_TOLERANCE_S`. Measured across all
7,159 consecutive same-lane pairs in the corpus: the gap distribution is
sharply bimodal -- 82.8% are truly contiguous (< 1ms, matching decompose.py's
own segment construction), then a long tail starting well past 20ms (p90 =
141ms, up to 1505s for pairs separated by an untranscribed/masked raw
trajectory, e.g. a skipped type-12 Silent or type-7/13 krintin/slide). 20ms
sits cleanly in the gap between these two populations (5,938/7,159 pairs
below it vs. only 14 more between 1ms and 20ms), so it is used as the
adjacency threshold -- primitives separated by more than this are treated
as NOT neighbors (an "invalid-target gap", per spec section 3), even though
they are numerically consecutive in the primitives list.

primitive_id encodes (recording_id, lane_id) but NOT time order on its own
(SKIP_PRIMITIVE_SOURCE_TYPES trajectories are silently omitted from the
primitives list, so consecutive primitive_ids can still straddle a gap) --
ordering is therefore always by start_s within (recording_id, lane_id), not
by primitive_id string or `seq`.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GAP_TOLERANCE_S = 0.02
CLASS_NAMES = ("Fixed", "Cosine", "Sloped-start", "Sloped-end")


def build_neighbor_map(records: list[dict]) -> dict[str, dict[str, str | None]]:
    by_lane: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        by_lane[(r["recording_id"], r["lane_id"])].append(r)

    neighbor_map: dict[str, dict[str, str | None]] = {}
    for _, members in by_lane.items():
        ordered = sorted(members, key=lambda r: r["start_s"])
        for i, r in enumerate(ordered):
            prev_id = None
            next_id = None
            if i > 0:
                prev = ordered[i - 1]
                if r["start_s"] - prev["end_s"] < GAP_TOLERANCE_S:
                    prev_id = prev["primitive_id"]
            if i < len(ordered) - 1:
                nxt = ordered[i + 1]
                if nxt["start_s"] - r["end_s"] < GAP_TOLERANCE_S:
                    next_id = nxt["primitive_id"]
            neighbor_map[r["primitive_id"]] = {"prev": prev_id, "next": next_id}
    return neighbor_map


def neighbor_coverage_report(neighbor_map: dict[str, dict[str, str | None]]) -> dict[str, int]:
    both = prev_only = next_only = neither = 0
    for v in neighbor_map.values():
        has_p, has_n = v["prev"] is not None, v["next"] is not None
        if has_p and has_n:
            both += 1
        elif has_p:
            prev_only += 1
        elif has_n:
            next_only += 1
        else:
            neither += 1
    return {"both": both, "prev_only": prev_only, "next_only": next_only, "neither": neither, "total": len(neighbor_map)}


def train_only_transition_matrix(
    records: list[dict], neighbor_map: dict[str, dict[str, str | None]], train_recording_ids: set[str],
) -> dict[str, Any]:
    """Section 16: descriptive only, TRAIN labels only, never used as a
    model input. P(type_i | type_{i-1})."""
    by_pid = {r["primitive_id"]: r for r in records}
    counts = Counter()
    row_totals = Counter()
    for r in records:
        if r["recording_id"] not in train_recording_ids:
            continue
        prev_id = neighbor_map[r["primitive_id"]]["prev"]
        if prev_id is None or prev_id not in by_pid:
            continue
        prev_type = by_pid[prev_id]["canonical_type"]
        cur_type = r["canonical_type"]
        counts[(prev_type, cur_type)] += 1
        row_totals[prev_type] += 1

    matrix = {}
    for prev_t in range(4):
        total = row_totals[prev_t]
        matrix[CLASS_NAMES[prev_t]] = {
            CLASS_NAMES[cur_t]: (counts[(prev_t, cur_t)] / total if total else 0.0) for cur_t in range(4)
        }
    counts_out = {f"{CLASS_NAMES[a]}->{CLASS_NAMES[b]}": n for (a, b), n in counts.items()}
    return {"transition_probs": matrix, "transition_counts": counts_out, "row_totals": {CLASS_NAMES[k]: v for k, v in row_totals.items()}}
