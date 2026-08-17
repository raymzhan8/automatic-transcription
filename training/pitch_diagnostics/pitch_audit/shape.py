"""Step 16 sections 7-10: turning-point recall/timing, local-shape
(flat/rising/falling/rise->fall/fall->rise) confusion, quantization/
staircase run-length analysis, and jitter in GT-flat (T0) regions.
"""

from __future__ import annotations

import numpy as np

from training.metrics import TYPE_NAMES
from training.pitch_diagnostics.common import write_json
from training.pitch_diagnostics.pitch_audit.common import AUDIT_DIR, NATIVE_HOP_S, build_bundles, delta_at_offset

TURN_VELOCITY_THRESHOLD_CENTS_S = 100.0  # reused from Steps 13-15's direction deadband
TURN_TOLERANCES_MS = (20, 50, 100)
EST_TURN_OFFSET = 5  # 50ms-smoothed velocity proxy for the estimated side (see turning_points docstring)

SHAPE_HALF_WINDOW_FRAMES = 5  # 50ms each side
SHAPE_MOVE_THRESHOLD_CENTS = 20.0  # ~1.2 CQT bins
SHAPE_CLASSES = ("flat", "rising", "falling", "rise_fall", "fall_rise")

STAIRCASE_TOLERANCE_CENTS = 8.0  # half a native CQT bin (16.67c/bin)


def _sign(x: np.ndarray, thresh: float) -> np.ndarray:
    s = np.zeros(len(x), dtype=np.int8)
    s[x > thresh] = 1
    s[x < -thresh] = -1
    return s


def turning_points(bundles: list[dict]) -> dict:
    """GT turns: sign change in the analytic dp/dt with both sides
    exceeding the reused 100c/s deadband (spec section 7's "conservative
    rule... avoid counting tiny fluctuations"). EST turns: sign change in
    the SAME-style velocity computed from the 50ms (k=5) delta / 0.05s --
    matching GT's own smoothness scale rather than the raw noisy 10ms
    finite difference, for a fair comparison."""
    by_type: dict[str, dict] = {TYPE_NAMES[t]: {"n_gt_turns": 0, "matched": {tol: 0 for tol in TURN_TOLERANCES_MS},
                                                 "timing_errors_ms": [], "n_est_turns": 0,
                                                 "est_false": {tol: 0 for tol in TURN_TOLERANCES_MS}}
                                 for t in range(4)}
    per_recording: dict[str, dict] = {}

    for b in bundles:
        n = b["n"]
        gt_vel = b["dp_dt_cents_s"]
        gt_vel_valid = b["valid"] & np.isfinite(gt_vel)
        gt_sign = _sign(gt_vel, TURN_VELOCITY_THRESHOLD_CENTS_S)
        gt_turn = np.zeros(n, dtype=bool)
        gt_turn[1:] = (gt_sign[1:] != 0) & (gt_sign[:-1] != 0) & (gt_sign[1:] != gt_sign[:-1]) & gt_vel_valid[1:] & gt_vel_valid[:-1]

        est_vel = delta_at_offset(b["est_cents"], EST_TURN_OFFSET) / (EST_TURN_OFFSET * NATIVE_HOP_S)
        est_vel = np.nan_to_num(est_vel, nan=0.0)
        est_sign = _sign(est_vel, TURN_VELOCITY_THRESHOLD_CENTS_S)
        est_turn = np.zeros(n, dtype=bool)
        est_turn[1:] = (est_sign[1:] != 0) & (est_sign[:-1] != 0) & (est_sign[1:] != est_sign[:-1])

        gt_turn_idx = np.flatnonzero(gt_turn)
        est_turn_idx = np.flatnonzero(est_turn)
        rec_entry = {"n_gt_turns": len(gt_turn_idx), "n_est_turns": len(est_turn_idx)}

        matched_est = np.zeros(len(est_turn_idx), dtype=bool)
        for gi in gt_turn_idx:
            tt = int(b["trajectory_type"][gi])
            if tt not in range(4):
                continue
            by_type[TYPE_NAMES[tt]]["n_gt_turns"] += 1
            if len(est_turn_idx) == 0:
                continue
            dists_frames = est_turn_idx - gi
            nearest_i = int(np.argmin(np.abs(dists_frames)))
            nearest_ms = abs(int(dists_frames[nearest_i])) * 10
            for tol in TURN_TOLERANCES_MS:
                if nearest_ms <= tol:
                    by_type[TYPE_NAMES[tt]]["matched"][tol] += 1
            if nearest_ms <= max(TURN_TOLERANCES_MS):
                by_type[TYPE_NAMES[tt]]["timing_errors_ms"].append(nearest_ms)
                matched_est[nearest_i] = True

        for ei in est_turn_idx:
            tt = int(b["trajectory_type"][ei])
            if tt not in range(4) or len(gt_turn_idx) == 0:
                continue
            by_type[TYPE_NAMES[tt]]["n_est_turns"] += 1
            nearest_ms = int(np.min(np.abs(gt_turn_idx - ei))) * 10
            for tol in TURN_TOLERANCES_MS:
                if nearest_ms > tol:
                    by_type[TYPE_NAMES[tt]]["est_false"][tol] += 1

        per_recording[b["recording_id"]] = rec_entry

    summary = {}
    for name, d in by_type.items():
        n_gt = max(d["n_gt_turns"], 1)
        n_est = max(d["n_est_turns"], 1)
        summary[name] = {
            "n_gt_turns": d["n_gt_turns"], "n_est_turns": d["n_est_turns"],
            "recall_by_tolerance": {f"{tol}ms": d["matched"][tol] / n_gt for tol in TURN_TOLERANCES_MS},
            "false_turn_rate_by_tolerance": {f"{tol}ms": d["est_false"][tol] / n_est for tol in TURN_TOLERANCES_MS},
            "median_timing_error_ms": float(np.median(d["timing_errors_ms"])) if d["timing_errors_ms"] else None,
        }
    return {"by_type": summary, "per_recording": per_recording,
            "velocity_threshold_cents_s": TURN_VELOCITY_THRESHOLD_CENTS_S, "est_offset_ms": EST_TURN_OFFSET * 10}


def _classify_shape(cents: np.ndarray, t: int, n: int) -> int | None:
    lo = t - SHAPE_HALF_WINDOW_FRAMES
    hi = t + SHAPE_HALF_WINDOW_FRAMES
    if lo < 0 or hi >= n:
        return None
    first_half = cents[lo:t]
    second_half = cents[t:hi]
    if len(first_half) == 0 or len(second_half) == 0 or not np.all(np.isfinite(first_half)) or not np.all(np.isfinite(second_half)):
        return None
    d1 = first_half[-1] - first_half[0]
    d2 = second_half[-1] - second_half[0]
    s1 = 0 if abs(d1) < SHAPE_MOVE_THRESHOLD_CENTS else (1 if d1 > 0 else -1)
    s2 = 0 if abs(d2) < SHAPE_MOVE_THRESHOLD_CENTS else (1 if d2 > 0 else -1)
    if s1 == 0 and s2 == 0:
        return 0  # flat
    if s1 == 1 and s2 == -1:
        return 3  # rise_fall
    if s1 == -1 and s2 == 1:
        return 4  # fall_rise
    if s2 == 1 or (s2 == 0 and s1 == 1):
        return 1  # rising
    if s2 == -1 or (s2 == 0 and s1 == -1):
        return 2  # falling
    return 0


def shape_confusion(bundles: list[dict]) -> dict:
    n_classes = len(SHAPE_CLASSES)
    overall = np.zeros((n_classes, n_classes), dtype=np.int64)
    by_type = {TYPE_NAMES[t]: np.zeros((n_classes, n_classes), dtype=np.int64) for t in range(4)}

    for b in bundles:
        n = b["n"]
        valid = b["valid"]
        idxs = np.flatnonzero(valid)
        idxs = idxs[(idxs >= SHAPE_HALF_WINDOW_FRAMES) & (idxs < n - SHAPE_HALF_WINDOW_FRAMES)]
        for t in idxs:
            gt_c = _classify_shape(b["gt_cents"], int(t), n)
            est_c = _classify_shape(b["est_cents"], int(t), n)
            if gt_c is None or est_c is None:
                continue
            overall[gt_c, est_c] += 1
            tt = int(b["trajectory_type"][t])
            if tt in range(4):
                by_type[TYPE_NAMES[tt]][gt_c, est_c] += 1

    def to_dict(mat):
        return {"labels": list(SHAPE_CLASSES), "matrix": mat.tolist(),
                "row_normalized": (mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)).round(4).tolist()}

    return {"overall": to_dict(overall), "by_type": {k: to_dict(v) for k, v in by_type.items()},
            "half_window_ms": SHAPE_HALF_WINDOW_FRAMES * 10, "move_threshold_cents": SHAPE_MOVE_THRESHOLD_CENTS}


def staircase_runs(bundles: list[dict]) -> dict:
    """Run-length distributions (native frames staying within
    STAIRCASE_TOLERANCE_CENTS of the run's start value), GT vs estimated,
    restricted to valid frames, pooled and by type."""
    def run_lengths(cents: np.ndarray, valid: np.ndarray, tt: np.ndarray | None = None, want_type: int | None = None):
        lengths = []
        i = 0
        n = len(cents)
        while i < n:
            if not valid[i] or not np.isfinite(cents[i]) or (want_type is not None and tt[i] != want_type):
                i += 1
                continue
            j = i + 1
            while j < n and valid[j] and np.isfinite(cents[j]) and abs(cents[j] - cents[i]) <= STAIRCASE_TOLERANCE_CENTS \
                    and (want_type is None or tt[j] == want_type):
                j += 1
            lengths.append(j - i)
            i = j
        return lengths

    gt_all, est_all = [], []
    gt_by_type = {t: [] for t in range(4)}
    est_by_type = {t: [] for t in range(4)}
    for b in bundles:
        gt_all.extend(run_lengths(b["gt_cents"], b["valid"]))
        est_all.extend(run_lengths(b["est_cents"], b["valid"]))
        for t in range(4):
            gt_by_type[t].extend(run_lengths(b["gt_cents"], b["valid"], b["trajectory_type"], t))
            est_by_type[t].extend(run_lengths(b["est_cents"], b["valid"], b["trajectory_type"], t))

    def summ(lst):
        if not lst:
            return {"n": 0}
        a = np.array(lst)
        return {"n": len(a), "median_frames": float(np.median(a)), "median_ms": float(np.median(a) * 10),
                "p90_frames": float(np.percentile(a, 90)), "mean_frames": float(np.mean(a))}

    return {
        "gt_overall": summ(gt_all), "est_overall": summ(est_all),
        "gt_by_type": {TYPE_NAMES[t]: summ(v) for t, v in gt_by_type.items()},
        "est_by_type": {TYPE_NAMES[t]: summ(v) for t, v in est_by_type.items()},
        "tolerance_cents": STAIRCASE_TOLERANCE_CENTS,
    }


def jitter_audit(bundles: list[dict]) -> dict:
    """In T0 (GT-flat) regions specifically: estimated frame-to-frame
    variance, nonzero-delta fraction, and direction-reversal (flip-flop)
    rate among nonzero deltas -- the failure mode opposite to smoothing."""
    out_by_type = {}
    for t in range(4):
        est_deltas, gt_deltas = [], []
        for b in bundles:
            m = b["valid"] & (b["trajectory_type"] == t)
            d_est = delta_at_offset(b["est_cents"], 1)
            d_gt = delta_at_offset(b["gt_cents"], 1)
            mm = m & np.isfinite(d_est) & np.isfinite(d_gt)
            est_deltas.append(d_est[mm]); gt_deltas.append(d_gt[mm])
        e = np.concatenate(est_deltas) if est_deltas else np.array([])
        g = np.concatenate(gt_deltas) if gt_deltas else np.array([])
        if len(e) == 0:
            out_by_type[TYPE_NAMES[t]] = {"n": 0}
            continue
        nonzero = np.abs(e) > 1e-6
        sign = np.sign(e)
        reversal = 0
        prev_sign = 0
        n_compare = 0
        for s, nz in zip(sign, nonzero):
            if nz:
                if prev_sign != 0:
                    n_compare += 1
                    if s != prev_sign:
                        reversal += 1
                prev_sign = s
        out_by_type[TYPE_NAMES[t]] = {
            "n": len(e), "est_std_cents": float(np.std(e)), "gt_std_cents": float(np.std(g)),
            "frac_est_nonzero": float(nonzero.mean()),
            "direction_reversal_rate_among_nonzero": (reversal / n_compare) if n_compare > 0 else None,
        }
    return out_by_type


def main() -> None:
    bundles = build_bundles()
    out = {
        "turning_points": turning_points(bundles),
        "shape_confusion": shape_confusion(bundles),
        "staircase_runs": staircase_runs(bundles),
        "jitter": jitter_audit(bundles),
    }
    write_json(AUDIT_DIR / "shape_audit.json", out)

    print("=== turning points by type ===")
    for k, v in out["turning_points"]["by_type"].items():
        print(k, v)
    print("\n=== staircase run lengths (median frames): GT vs EST by type ===")
    for t in TYPE_NAMES.values():
        g = out["staircase_runs"]["gt_by_type"][t]
        e = out["staircase_runs"]["est_by_type"][t]
        print(t, "GT median", g.get("median_frames"), "EST median", e.get("median_frames"))
    print("\n=== jitter by type ===")
    for k, v in out["jitter"].items():
        print(k, v)
    print("\n=== shape confusion overall (row=GT, col=EST) ===")
    print(out["shape_confusion"]["overall"]["labels"])
    for row in out["shape_confusion"]["overall"]["row_normalized"]:
        print(row)


if __name__ == "__main__":
    main()
