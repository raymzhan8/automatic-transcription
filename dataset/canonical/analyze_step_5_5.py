"""Step 5.5 — boundary learnability and class-balance audit (diagnostic only)."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.canonical.boundary_geometry import (  # noqa: E402
    BoundaryRecord,
    build_boundary_records,
)
from dataset.canonical.build import exported_recording_ids  # noqa: E402
from dataset.canonical.frames import frame_centers  # noqa: E402
from dataset.canonical.schema import HOP_S, canonical_root, frames_dir  # noqa: E402
from dataset.canonical.validate_targets import duration_stats  # noqa: E402

PITCH_THRESHOLDS_CENTS = (1, 5, 10, 20)
VELOCITY_THRESHOLDS = (50, 100, 200, 500)
ACCEL_THRESHOLDS = (500, 1000, 2000, 5000)

T1_ORIGIN_RULES = {
    "raw_t1": lambda p: p["rule_applied"] == "keep" and p["source_raw_type"] == 1,
    "t4_decomposition": lambda p: p["rule_applied"] == "decompose_4",
    "t5_decomposition": lambda p: p["rule_applied"] == "decompose_5",
    "t6_decomposition": lambda p: p["rule_applied"] == "decompose_6",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def t1_origin(prim: dict[str, Any]) -> str | None:
    if prim["canonical_type"] != 1:
        return None
    for name, pred in T1_ORIGIN_RULES.items():
        if pred(prim):
            return name
    return "other"


def distribution_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "median": None, "p25": None, "p75": None, "mean": None}
    ordered = sorted(values)
    n = len(ordered)

    def q(p: float) -> float:
        return ordered[min(n - 1, int(p * n))]

    return {
        "count": n,
        "median": statistics.median(ordered),
        "p25": q(0.25),
        "p75": q(0.75),
        "mean": sum(ordered) / n,
    }


def pct_above(values: list[float | None], threshold: float) -> float | None:
    present = [abs(v) for v in values if v is not None]
    if not present:
        return None
    return 100.0 * sum(1 for v in present if v > threshold) / len(present)


def count_frames_for_primitive(
    prim: dict[str, Any],
    times: np.ndarray,
) -> int:
    if prim["duration_s"] <= 0:
        return 0
    mask = (times >= prim["start_s"]) & (times < prim["end_s"])
    return int(np.sum(mask))


def analyze_duration_stats(all_primitives: list[dict]) -> dict:
    by_type: dict[int, list[float]] = defaultdict(list)
    t1_t6: list[float] = []
    for p in all_primitives:
        by_type[int(p["canonical_type"])].append(float(p["duration_s"]))
        if p["canonical_type"] == 1 and p["rule_applied"] == "decompose_6":
            t1_t6.append(float(p["duration_s"]))
    return {
        "overall": duration_stats([p["duration_s"] for p in all_primitives]),
        "by_canonical_type": {
            str(t): duration_stats(durs) for t, durs in sorted(by_type.items())
        },
        "t1_from_type6": duration_stats(t1_t6),
        "step5_error_diagnosis": "reporting_only",
    }


def analyze_class_balance(
    all_primitives: list[dict],
    repo_root: Path,
    recording_ids: list[str],
) -> dict:
    prim_counts = Counter(int(p["canonical_type"]) for p in all_primitives)
    prim_total = sum(prim_counts.values())
    prim_pct = {
        str(t): 100.0 * prim_counts[t] / prim_total for t in sorted(prim_counts)
    }

    t1_prim_origin = Counter(t1_origin(p) for p in all_primitives if p["canonical_type"] == 1)

    frame_counts = Counter()
    t1_frame_origin = Counter()
    prim_by_id: dict[str, dict] = {p["primitive_id"]: p for p in all_primitives}

    for recording_id in recording_ids:
        rec = load_json(canonical_root(repo_root) / "recordings" / f"{recording_id}.json")
        for lane in rec.get("lanes", []):
            lane_id = lane["lane_id"]
            npz_path = frames_dir(repo_root) / f"{recording_id}_{lane_id.replace(':', '_')}.npz"
            if not npz_path.exists():
                continue
            data = np.load(npz_path, allow_pickle=True)
            valid = data["valid_target"]
            types = data["trajectory_type"]
            prim_ids = data["primitive_id"]
            for i in np.where(valid)[0]:
                t = int(types[i])
                frame_counts[t] += 1
                pid = str(prim_ids[i])
                prim = prim_by_id.get(pid)
                if prim and prim["canonical_type"] == 1:
                    origin = t1_origin(prim)
                    if origin:
                        t1_frame_origin[origin] += 1

    frame_total = sum(frame_counts.values())
    frame_pct = {
        str(t): 100.0 * frame_counts[t] / frame_total for t in sorted(frame_counts)
    }
    t1_prim_total = sum(t1_prim_origin.values())
    t1_frame_total = sum(t1_frame_origin.values())

    return {
        "primitive_level": {
            "counts": {str(k): v for k, v in sorted(prim_counts.items())},
            "percentages": prim_pct,
            "total": prim_total,
        },
        "frame_level_valid_target": {
            "counts": {str(k): v for k, v in sorted(frame_counts.items())},
            "percentages": frame_pct,
            "total": frame_total,
        },
        "t1_by_origin_primitives": {
            k: {"count": v, "pct": 100.0 * v / t1_prim_total if t1_prim_total else 0}
            for k, v in sorted(t1_prim_origin.items())
        },
        "t1_by_origin_frames": {
            k: {"count": v, "pct": 100.0 * v / t1_frame_total if t1_frame_total else 0}
            for k, v in sorted(t1_frame_origin.items())
        },
    }


def boundary_inventory(boundaries: list[BoundaryRecord]) -> dict:
    type_pairs = Counter((b.prev_type, b.next_type) for b in boundaries)
    return {
        "n_boundaries": len(boundaries),
        "n_same_type": sum(1 for b in boundaries if b.same_type),
        "n_different_type": sum(1 for b in boundaries if not b.same_type),
        "n_introduced_by_decomposition": sum(
            1 for b in boundaries if b.introduced_by_decomposition
        ),
        "n_raw_preserved": sum(1 for b in boundaries if b.raw_preserved),
        "type_pair_counts": {
            f"T{a}|T{b}": c for (a, b), c in sorted(type_pairs.items())
        },
    }


def sensitivity_matrix(
    boundaries: list[BoundaryRecord],
    *,
    pitch_thresholds: tuple[int, ...] = PITCH_THRESHOLDS_CENTS,
    velocity_thresholds: tuple[int, ...] = VELOCITY_THRESHOLDS,
    accel_thresholds: tuple[int, ...] = ACCEL_THRESHOLDS,
) -> dict:
    pitch_steps = [abs(b.pitch_step_cents or 0.0) for b in boundaries]
    n = len(boundaries)

    return {
        "pitch_step_pct": {
            str(t): 100.0 * sum(1 for v in pitch_steps if v > t) / n if n else 0
            for t in pitch_thresholds
        },
        "delta_v_pct": {
            str(t): pct_above([b.delta_v for b in boundaries], t)
            for t in velocity_thresholds
        },
        "delta_a_pct": {
            str(t): pct_above([b.delta_a for b in boundaries], t)
            for t in accel_thresholds
        },
    }


def oracle_analysis(boundaries: list[BoundaryRecord]) -> dict:
    def run_subset(subset: list[BoundaryRecord]) -> dict:
        n = len(subset)
        if n == 0:
            return {}
        type_rec = sum(1 for b in subset if not b.same_type)
        results: dict[str, Any] = {
            "n": n,
            "pct_type_change": 100.0 * type_rec / n,
        }
        remaining = [b for b in subset if b.same_type]
        for pt in PITCH_THRESHOLDS_CENTS:
            pitch_rec = sum(
                1
                for b in remaining
                if b.pitch_step_cents is not None and abs(b.pitch_step_cents) > pt
            )
            results[f"pct_plus_pitch_gt_{pt}cents"] = (
                100.0 * (type_rec + pitch_rec) / n
            )
        for vt in VELOCITY_THRESHOLDS:
            dyn_rec = sum(
                1
                for b in remaining
                if b.delta_v is not None and abs(b.delta_v) > vt
            )
            results[f"pct_plus_delta_v_gt_{vt}"] = (
                100.0 * (type_rec + dyn_rec) / n
            )
        for at in ACCEL_THRESHOLDS:
            acc_rec = sum(
                1
                for b in remaining
                if b.delta_a is not None and abs(b.delta_a) > at
            )
            results[f"pct_plus_delta_a_gt_{at}"] = (
                100.0 * (type_rec + acc_rec) / n
            )
        no_cue = sum(
            1
            for b in remaining
            if (b.pitch_step_cents is None or abs(b.pitch_step_cents) <= 1)
            and (b.delta_v is None or abs(b.delta_v) <= 50)
            and (b.delta_a is None or abs(b.delta_a) <= 500)
        )
        results["pct_no_obvious_cue_strict"] = 100.0 * no_cue / n
        return results

    return {
        "all_boundaries": run_subset(boundaries),
        "same_type_only": run_subset([b for b in boundaries if b.same_type]),
        "t6_internal_only": run_subset([b for b in boundaries if b.is_t6_internal]),
    }


def compare_populations(
    a: list[BoundaryRecord], b: list[BoundaryRecord]
) -> dict:
    def vals(records: list[BoundaryRecord], attr: str) -> list[float]:
        return [
            float(getattr(r, attr))
            for r in records
            if getattr(r, attr) is not None
        ]

    out: dict[str, Any] = {
        "population_a_n": len(a),
        "population_b_n": len(b),
    }
    for attr in ("pitch_step_cents", "delta_v", "delta_a"):
        va = [abs(v) for v in vals(a, attr)]
        vb = [abs(v) for v in vals(b, attr)]
        out[f"{attr}_a"] = distribution_summary(va)
        out[f"{attr}_b"] = distribution_summary(vb)
        if va and vb:
            out[f"{attr}_ks_pvalue"] = float(stats.ks_2samp(va, vb).pvalue)
    da = [r.prim_a_duration_s for r in a] + [r.prim_b_duration_s for r in a]
    db = [r.prim_a_duration_s for r in b] + [r.prim_b_duration_s for r in b]
    out["duration_a"] = distribution_summary(da)
    out["duration_b"] = distribution_summary(db)
    if da and db:
        out["duration_ks_pvalue"] = float(stats.ks_2samp(da, db).pvalue)
    return out


def analyze_t6_internal(boundaries: list[BoundaryRecord]) -> dict:
    t6 = [b for b in boundaries if b.is_t6_internal]
    n = len(t6)
    at_cp = sum(1 for b in t6 if b.at_control_point_transition)
    return {
        "n_internal_t6_boundaries": n,
        "pct_at_control_point_transition": 100.0 * at_cp / n if n else 0,
        "sensitivity": sensitivity_matrix(t6),
        "internal_kink_log2": distribution_summary(
            [b.internal_kink_log2 for b in t6 if b.internal_kink_log2 is not None]
        ),
    }


def analyze_phase_correlations(
    repo_root: Path,
    recording_ids: list[str],
    all_primitives: list[dict],
) -> dict:
    prim_by_id = {p["primitive_id"]: p for p in all_primitives}
    by_type: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for recording_id in recording_ids:
        rec = load_json(canonical_root(repo_root) / "recordings" / f"{recording_id}.json")
        for lane in rec.get("lanes", []):
            lane_id = lane["lane_id"]
            npz_path = frames_dir(repo_root) / f"{recording_id}_{lane_id.replace(':', '_')}.npz"
            if not npz_path.exists():
                continue
            data = np.load(npz_path, allow_pickle=True)
            valid = data["valid_target"]
            phase = data["phase"]
            pitch = data["pitch_log2_hz"]
            types = data["trajectory_type"]
            dp_dt = data["dp_dt_log2_hz_per_s"]
            prim_ids = data["primitive_id"]
            for i in np.where(valid)[0]:
                t = int(types[i])
                pid = str(prim_ids[i])
                prim = prim_by_id.get(pid)
                if prim is None:
                    continue
                ph = float(phase[i])
                p = float(pitch[i])
                start_log2 = (prim.get("start_pitch") or {}).get("log2_hz")
                end_log2 = (prim.get("end_pitch") or {}).get("log2_hz")
                if start_log2 is not None:
                    by_type[t]["disp_from_start_cents"].append(
                        1200.0 * (p - start_log2)
                    )
                if end_log2 is not None:
                    by_type[t]["disp_to_end_cents"].append(1200.0 * (end_log2 - p))
                by_type[t]["phase"].append(ph)
                if np.isfinite(dp_dt[i]):
                    by_type[t]["dp_dt_cents_per_s"].append(
                        1200.0 * float(dp_dt[i])
                    )

    out: dict[str, Any] = {}
    for t, series in sorted(by_type.items()):
        ph = np.array(series["phase"])
        entry: dict[str, Any] = {"n_frames": len(ph)}
        for key in ("disp_from_start_cents", "disp_to_end_cents", "dp_dt_cents_per_s"):
            if key not in series or len(series[key]) != len(ph):
                continue
            y = np.array(series[key])
            if len(ph) > 10:
                rho, _ = stats.spearmanr(ph, y)
                entry[f"spearman_phase_vs_{key}"] = float(rho)
                if np.std(ph) > 0 and np.std(y) > 0:
                    slope, _, r, _, _ = stats.linregress(ph, y)
                    entry[f"linregress_phase_vs_{key}"] = {
                        "slope": float(slope),
                        "r2": float(r * r),
                    }
        out[str(t)] = entry
    return out


def analyze_frame_resolution(
    all_primitives: list[dict],
    repo_root: Path,
    recording_ids: list[str],
) -> dict:
    by_type: dict[int, Counter] = defaultdict(Counter)
    t1_origin_frames: Counter = Counter()
    duration_by_bucket: dict[str, list[float]] = defaultdict(list)

    rec_durations: dict[str, float] = {}
    for recording_id in recording_ids:
        rec = load_json(canonical_root(repo_root) / "recordings" / f"{recording_id}.json")
        cov = rec.get("coverage", {})
        rec_durations[recording_id] = float(cov.get("audio_duration_s") or 0.0)

    for prim in all_primitives:
        recording_id = prim["primitive_id"].split(":")[0]
        dur_s = rec_durations.get(recording_id, 0.0)
        times = frame_centers(dur_s, hop_s=HOP_S)
        n_frames = count_frames_for_primitive(prim, times)
        if n_frames == 0:
            bucket = "0"
        elif n_frames == 1:
            bucket = "1"
        elif n_frames == 2:
            bucket = "2"
        else:
            bucket = "3+"
        ct = int(prim["canonical_type"])
        by_type[ct][bucket] += 1
        duration_by_bucket[bucket].append(float(prim["duration_s"]))
        origin = t1_origin(prim)
        if origin == "t6_decomposition":
            t1_origin_frames[bucket] += 1

    def pct_hist(counter: Counter) -> dict[str, float]:
        total = sum(counter.values())
        return {k: 100.0 * counter[k] / total for k in sorted(counter)} if total else {}

    t6_total = sum(t1_origin_frames.values())
    return {
        "frame_count_histogram_by_type": {
            str(t): dict(sorted(c.items())) for t, c in sorted(by_type.items())
        },
        "frame_count_pct_by_type": {
            str(t): pct_hist(c) for t, c in sorted(by_type.items())
        },
        "t6_origin_t1_frame_histogram": dict(sorted(t1_origin_frames.items())),
        "t6_origin_t1_pct_1_or_2_frames": (
            100.0
            * (t1_origin_frames.get("1", 0) + t1_origin_frames.get("2", 0))
            / t6_total
            if t6_total
            else 0
        ),
        "median_duration_by_frame_bucket": {
            k: distribution_summary(v)["median"] for k, v in duration_by_bucket.items()
        },
    }


def select_visualization_examples(
    boundaries: list[BoundaryRecord],
) -> dict[str, list[dict]]:
    """Pick representative boundaries for each required category."""

    def pick(
        pred,
        n: int = 2,
        sort_key=None,
        reverse: bool = True,
    ) -> list[dict]:
        items = [b for b in boundaries if pred(b)]
        if sort_key:
            items.sort(key=sort_key, reverse=reverse)
        return [b.to_dict() for b in items[:n]]

    def no_cue(b: BoundaryRecord) -> bool:
        return (
            b.same_type
            and (b.pitch_step_cents is None or abs(b.pitch_step_cents) <= 1)
            and (b.delta_v is None or abs(b.delta_v) <= 50)
            and (b.delta_a is None or abs(b.delta_a) <= 500)
        )

    return {
        "t0_t0_diff_pitch": pick(
            lambda b: b.prev_type == 0
            and b.next_type == 0
            and b.pitch_step_cents is not None
            and abs(b.pitch_step_cents) > 5,
            sort_key=lambda b: abs(b.pitch_step_cents or 0),
        ),
        "raw_t1_t1": pick(lambda b: b.is_raw_t1_t1),
        "t6_t1_t1_obvious": pick(
            lambda b: b.is_t6_internal
            and (
                (b.pitch_step_cents is not None and abs(b.pitch_step_cents) > 10)
                or (b.delta_v is not None and abs(b.delta_v) > 200)
            ),
            sort_key=lambda b: max(
                abs(b.pitch_step_cents or 0), abs(b.delta_v or 0)
            ),
        ),
        "t6_t1_t1_subtle": pick(
            lambda b: b.is_t6_internal
            and not no_cue(b)
            and (b.pitch_step_cents is None or abs(b.pitch_step_cents) <= 10)
            and (b.delta_v is not None and 50 < abs(b.delta_v) <= 200),
            sort_key=lambda b: abs(b.delta_v or 0),
        ),
        "t6_t1_t1_no_cue": pick(
            lambda b: b.is_t6_internal and no_cue(b),
        ),
        "t1_t2": pick(lambda b: b.prev_type == 1 and b.next_type == 2),
        "t2_t1_from_t4": pick(
            lambda b: b.prev_type == 2
            and b.next_type == 1
            and b.introduced_by_decomposition
            and b.prim_a_rule == "decompose_4",
        ),
        "t1_t3_from_t5": pick(
            lambda b: b.prev_type == 1
            and b.next_type == 3
            and b.introduced_by_decomposition
            and b.prim_b_rule == "decompose_5",
        ),
    }


def run_analysis(repo_root: Path) -> dict:
    root = canonical_root(repo_root)
    recording_ids = exported_recording_ids(repo_root)
    all_primitives: list[dict] = []
    all_boundaries: list[BoundaryRecord] = []

    for recording_id in recording_ids:
        rec = load_json(root / "recordings" / f"{recording_id}.json")
        prim_doc = load_json(root / "primitives" / f"{recording_id}.json")
        all_primitives.extend(prim_doc["primitives"])
        all_boundaries.extend(build_boundary_records(rec, prim_doc))

    same_type = [b for b in all_boundaries if b.same_type]
    t6_internal = [b for b in all_boundaries if b.is_t6_internal]
    raw_t1_t1 = [b for b in all_boundaries if b.is_raw_t1_t1]

    return {
        "n_recordings": len(recording_ids),
        "duration_statistics": analyze_duration_stats(all_primitives),
        "class_balance": analyze_class_balance(all_primitives, repo_root, recording_ids),
        "boundary_inventory": boundary_inventory(all_boundaries),
        "same_type_boundary_analysis": {
            "n_same_type": len(same_type),
            "n_t1_t1": sum(1 for b in same_type if b.prev_type == 1),
            "sensitivity": sensitivity_matrix(same_type),
            "pitch_step_distribution": distribution_summary(
                [abs(b.pitch_step_cents or 0) for b in same_type]
            ),
        },
        "t6_internal_boundary_analysis": analyze_t6_internal(all_boundaries),
        "raw_t1_t1_vs_t6_t1_t1": compare_populations(raw_t1_t1, t6_internal),
        "oracle_boundary_recovery": oracle_analysis(all_boundaries),
        "phase_correlation": analyze_phase_correlations(
            repo_root, recording_ids, all_primitives
        ),
        "frame_resolution_10ms": analyze_frame_resolution(
            all_primitives, repo_root, recording_ids
        ),
        "visualization_examples": select_visualization_examples(all_boundaries),
        "derivative_estimation": {
            "grid_step_s": 0.001,
            "window_s": 0.25,
            "velocity_offset_s": 0.01,
            "accel_offset_s": 0.01,
            "v_before_window_s": [-0.05, -0.02],
            "v_after_window_s": [0.02, 0.05],
            "pitch_unit": "cents = 1200 * log2_hz",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()

    result = run_analysis(args.repo_root)
    out = canonical_root(args.repo_root) / "step_5_5_analysis.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "n_boundaries": result["boundary_inventory"]["n_boundaries"],
                "n_same_type": result["boundary_inventory"]["n_same_type"],
                "t6_internal": result["t6_internal_boundary_analysis"][
                    "n_internal_t6_boundaries"
                ],
            },
            indent=2,
        )
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
