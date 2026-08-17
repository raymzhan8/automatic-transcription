"""Step 19: the core predecoder audit -- acoustic ridge visibility (A),
motion-vs-flat/counterfactual contrast (A), salience fidelity (S), A-to-S
degradation, candidate continuity, the D0 staircase mechanism, the zero-
delta causal decomposition (the central table), frequency-resolution vs.
GT motion, turning-point temporal response, and harmonic/drone association
for zero-delta failures. One pass per recording (predecoder_common.py),
GT used for evaluation only, never for feature construction.
"""

from __future__ import annotations

import numpy as np

from training.metrics import TYPE_NAMES
from training.pitch_diagnostics.common import write_json
from training.pitch_diagnostics.pitch_audit.common import (
    AUDIT_DIR, NATIVE_HOP_S, both_valid_mask, delta_at_offset,
)
from training.pitch_diagnostics.pitch_audit.predecoder_common import CENTS_PER_BIN, iter_recording_records

DPDT_BUCKET_EDGES = (0.0, 100.0, 400.0, 1000.0, np.inf)
DPDT_BUCKET_NAMES = ("0-100c/s", "100-400c/s", "400-1000c/s", ">1000c/s")
TURN_THRESHOLD_CENTS_S = 100.0
A_RANK_WEAK_THRESHOLD = 20     # rank in candidate-range acoustic energy beyond which "acoustic weak"
S_RANK_LOST_THRESHOLD = 10     # rank in salience beyond which "salience lost the candidate"
CAND_LO, CAND_HI = 34, 244     # LRN_LO, LRN_HI, duplicated here to avoid import cycle
N_CAND = CAND_HI - CAND_LO

HARMONIC_TARGETS = {"octave_+1": 1200.0, "octave_-1": -1200.0}
HARMONIC_TOL_CENTS = 50.0


def _vectorized_run_lengths(mask: np.ndarray) -> list[int]:
    """Lengths of contiguous True runs in a boolean array, vectorized."""
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diffs == 1)
    ends = np.flatnonzero(diffs == -1)
    return (ends - starts).tolist()


def _rank_of_bin(col: np.ndarray, bin_idx: int) -> int:
    """1-based rank of `bin_idx` within `col` (descending)."""
    if bin_idx < 0 or bin_idx >= len(col):
        return len(col) + 1
    return int((col > col[bin_idx]).sum()) + 1


def _bin_rel(gt_bin_abs_t: float) -> int:
    return int(round(gt_bin_abs_t)) - CAND_LO


def main() -> None:
    # accumulators
    a_rank_rows, s_rank_rows = [], []  # each: dict(type, dpdt_bucket, boundary, value)
    ma_rows, ms_rows = [], []          # motion contrast at 100ms window
    counterfactual_rows = []           # stationary vs half-speed preference
    continuity_runs_a, continuity_runs_s = {t: [] for t in range(4)}, {t: [] for t in range(4)}
    zero_delta_causes = {t: {} for t in range(4)}
    freq_res_rows = {t: [] for t in range(4)}  # |delta GT 10ms| in bins, by type
    turn_response_rows = []
    harmonic_assoc = {t: {"n": 0, "octave_+1": 0, "octave_-1": 0, "other": 0} for t in range(4)}

    n_recordings = 0
    for rec in iter_recording_records():
        n_recordings += 1
        n = rec["n"]
        valid = rec["valid"]
        mag = rec["mag"]  # [360, n]
        fused = rec["fused_probs"]  # [210, n]
        gt_bin_abs = rec["gt_bin_abs"]
        gt_cents = rec["gt_cents_rel"]
        d0_cents = rec["d0_cents_rel"]
        ttype = rec["trajectory_type"]
        dpdt = rec["dp_dt_cents_s"]
        times = rec["times"]

        idx = np.flatnonzero(valid)
        # ---- boundary proximity (evaluation only) ----
        near_boundary = np.zeros(n, dtype=bool)
        for prim in rec["primitives"]:
            near_boundary |= (np.abs(times - prim["start_s"]) <= 0.05) | (np.abs(times - prim["end_s"]) <= 0.05)

        for t in idx:
            tt = int(ttype[t])
            if tt not in range(4):
                continue
            gb = int(round(gt_bin_abs[t]))
            gb_rel_full = gb  # index into mag's full 360 range
            gb_rel_cand = gb - CAND_LO  # index into fused's 210 range

            # ---- sec 4: acoustic ridge visibility (rank in full 360-bin magnitude) ----
            col_mag = mag[:, t]
            a_rank = _rank_of_bin(col_mag, gb_rel_full)
            dpdt_bucket = None
            adpdt = abs(dpdt[t]) if np.isfinite(dpdt[t]) else None
            if adpdt is not None:
                for name, lo, hi in zip(DPDT_BUCKET_NAMES, DPDT_BUCKET_EDGES[:-1], DPDT_BUCKET_EDGES[1:]):
                    if lo <= adpdt < hi:
                        dpdt_bucket = name
                        break
            a_rank_rows.append((tt, dpdt_bucket, bool(near_boundary[t]), a_rank))

            # ---- sec 8: salience fidelity (rank in the 210-bin candidate/fused array) ----
            if 0 <= gb_rel_cand < N_CAND:
                col_s = fused[:, t]
                s_rank = _rank_of_bin(col_s, gb_rel_cand)
            else:
                s_rank = N_CAND + 1
            s_rank_rows.append((tt, dpdt_bucket, bool(near_boundary[t]), s_rank))

            # ---- sec 13: frequency resolution vs GT 10ms motion ----
            if t >= 1 and valid[t - 1]:
                dgt10 = abs(gt_cents[t] - gt_cents[t - 1])
                if np.isfinite(dgt10):
                    freq_res_rows[tt].append(dgt10 / CENTS_PER_BIN)

        # ---- sec 6: motion contrast M_A / M_S at 100ms (k=10) window, GT-moving frames ----
        k = 10
        both = both_valid_mask(valid, k)
        for t in np.flatnonzero(both):
            tt = int(ttype[t])
            if tt not in range(4):
                continue
            d_gt = gt_cents[t] - gt_cents[t - k]
            if not np.isfinite(d_gt) or abs(d_gt) < 1e-6:
                continue
            # energy along true GT path vs stationary path anchored at t-k
            gt_path_bins = np.round(gt_bin_abs[t - k: t + 1]).astype(np.int64)
            gt_path_bins = np.clip(gt_path_bins, 0, mag.shape[0] - 1)
            e_gt_path_a = float(np.sum(mag[gt_path_bins, np.arange(t - k, t + 1)]))
            stat_bin = gt_path_bins[0]
            e_stat_path_a = float(np.sum(mag[stat_bin, t - k: t + 1]))
            denom = max(e_gt_path_a + e_stat_path_a, 1e-12)
            m_a = (e_gt_path_a - e_stat_path_a) / denom
            ma_rows.append((tt, m_a))

            gt_path_bins_cand = np.clip(gt_path_bins - CAND_LO, 0, N_CAND - 1)
            e_gt_path_s = float(np.sum(fused[gt_path_bins_cand, np.arange(t - k, t + 1)]))
            stat_bin_cand = gt_path_bins_cand[0]
            e_stat_path_s = float(np.sum(fused[stat_bin_cand, t - k: t + 1]))
            denom_s = max(e_gt_path_s + e_stat_path_s, 1e-12)
            m_s = (e_gt_path_s - e_stat_path_s) / denom_s
            ms_rows.append((tt, m_s))

            # sec 7: counterfactual -- half-speed path
            half_bins = np.round(np.linspace(gt_bin_abs[t - k], gt_bin_abs[t - k] + (gt_bin_abs[t] - gt_bin_abs[t - k]) * 0.5, k + 1)).astype(np.int64)
            half_bins = np.clip(half_bins, 0, mag.shape[0] - 1)
            e_half_a = float(np.sum(mag[half_bins, np.arange(t - k, t + 1)]))
            counterfactual_rows.append((tt, e_gt_path_a > e_stat_path_a, e_gt_path_a > e_half_a))

        # ---- sec 10: candidate continuity -- run length of "GT-near candidate in top-5" ----
        s_rank_dense = np.full(n, N_CAND + 2, dtype=np.int64)
        a_rank_dense = np.full(n, 362, dtype=np.int64)
        for t in idx:
            gb = int(round(gt_bin_abs[t]))
            gb_cand = gb - CAND_LO
            if 0 <= gb_cand < N_CAND:
                s_rank_dense[t] = _rank_of_bin(fused[:, t], gb_cand)
            a_rank_dense[t] = _rank_of_bin(mag[:, t], gb)
        for tt in range(4):
            type_valid = valid & (ttype == tt)
            continuity_runs_s[tt].extend(_vectorized_run_lengths(type_valid & (s_rank_dense <= 5)))
            continuity_runs_a[tt].extend(_vectorized_run_lengths(type_valid & (a_rank_dense <= 5)))

        # ---- sec 11-12: D0 staircase mechanism / zero-delta causal decomposition ----
        d0_delta1 = delta_at_offset(d0_cents, 1)
        gt_vel = np.zeros(n)
        gt_vel[1:] = (gt_cents[1:] - gt_cents[:-1]) / NATIVE_HOP_S
        bv1 = both_valid_mask(valid, 1)
        zero_delta_mask = bv1 & (np.abs(gt_vel) > 100.0) & (np.abs(d0_delta1) < 1e-6)
        for t in np.flatnonzero(zero_delta_mask):
            tt = int(ttype[t])
            if tt not in range(4):
                continue
            a_ok_t = a_rank_dense[t] <= A_RANK_WEAK_THRESHOLD
            a_ok_tm1 = a_rank_dense[t - 1] <= A_RANK_WEAK_THRESHOLD
            if not (a_ok_t and a_ok_tm1):
                cause = "acoustic_weak_or_absent"
            elif s_rank_dense[t] > S_RANK_LOST_THRESHOLD:
                cause = "salience_lost_candidate"
            else:
                dgt10 = abs(gt_cents[t] - gt_cents[t - 1])
                if np.isfinite(dgt10) and dgt10 < CENTS_PER_BIN:
                    cause = "sub_bin_quantization"
                elif np.isfinite(dgt10):
                    cause = "present_but_not_selected"
                else:
                    cause = "ambiguous_unresolved"
            zero_delta_causes[tt][cause] = zero_delta_causes[tt].get(cause, 0) + 1

            # sec 17: harmonic/drone association for THIS zero-delta failure
            d0_abs_cents_here = d0_cents[t] - gt_cents[t]  # error relative to GT at this frame
            h = harmonic_assoc[tt]
            h["n"] += 1
            matched = False
            for name, target in HARMONIC_TARGETS.items():
                if abs(d0_abs_cents_here - target) < HARMONIC_TOL_CENTS:
                    h[name] += 1
                    matched = True
                    break
            if not matched:
                h["other"] += 1

        # ---- sec 14: turning-point temporal response (A/S sharpness around GT turns) ----
        gt_vel_full = np.zeros(n)
        gt_vel_full[1:] = (gt_cents[1:] - gt_cents[:-1]) / NATIVE_HOP_S
        sign = np.zeros(n, dtype=np.int8)
        sign[gt_vel_full > TURN_THRESHOLD_CENTS_S] = 1
        sign[gt_vel_full < -TURN_THRESHOLD_CENTS_S] = -1
        turn_idx = np.flatnonzero((sign[1:] != 0) & (sign[:-1] != 0) & (sign[1:] != sign[:-1]) & valid[1:] & valid[:-1]) + 1
        for ti in turn_idx:
            tt = int(ttype[ti])
            if tt not in range(4) or ti < 5 or ti + 5 >= n:
                continue
            before = a_rank_dense[ti - 5:ti]
            after = a_rank_dense[ti:ti + 5]
            turn_response_rows.append((tt, float(np.mean(before)), float(np.mean(after))))

    out = {
        "n_recordings": n_recordings,
        "acoustic_rank": _summ_rows(a_rank_rows),
        "salience_rank": _summ_rows(s_rank_rows),
        "motion_contrast_A": _summ_scalar_by_type(ma_rows),
        "motion_contrast_S": _summ_scalar_by_type(ms_rows),
        "counterfactual_preference": _summ_counterfactual(counterfactual_rows),
        "candidate_continuity_A": {TYPE_NAMES[t]: _run_summ(v) for t, v in continuity_runs_a.items()},
        "candidate_continuity_S": {TYPE_NAMES[t]: _run_summ(v) for t, v in continuity_runs_s.items()},
        "zero_delta_decomposition": {TYPE_NAMES[t]: v for t, v in zero_delta_causes.items()},
        "frequency_resolution": {
            TYPE_NAMES[t]: {
                "n": len(v),
                "frac_lt_0.5_bin": float(np.mean(np.array(v) < 0.5)) if v else None,
                "frac_lt_1_bin": float(np.mean(np.array(v) < 1.0)) if v else None,
                "frac_lt_2_bin": float(np.mean(np.array(v) < 2.0)) if v else None,
            } for t, v in freq_res_rows.items()
        },
        "turning_point_response": _summ_turn(turn_response_rows),
        "harmonic_association_zero_delta": {TYPE_NAMES[t]: v for t, v in harmonic_assoc.items()},
    }
    write_json(AUDIT_DIR / "predecoder_audit.json", out)
    _print_summary(out)


def _summ_rows(rows):
    by_type = {t: [] for t in range(4)}
    by_dpdt = {n: [] for n in DPDT_BUCKET_NAMES}
    by_boundary = {"near": [], "away": []}
    for tt, dpdt_bucket, near_b, val in rows:
        by_type[tt].append(val)
        if dpdt_bucket:
            by_dpdt[dpdt_bucket].append(val)
        by_boundary["near" if near_b else "away"].append(val)
    return {
        "by_type": {TYPE_NAMES[t]: {"n": len(v), "median_rank": float(np.median(v)) if v else None,
                                     "mean_rank": float(np.mean(v)) if v else None} for t, v in by_type.items()},
        "by_dpdt": {k: {"n": len(v), "median_rank": float(np.median(v)) if v else None} for k, v in by_dpdt.items()},
        "by_boundary": {k: {"n": len(v), "median_rank": float(np.median(v)) if v else None} for k, v in by_boundary.items()},
    }


def _summ_scalar_by_type(rows):
    by_type = {t: [] for t in range(4)}
    for tt, val in rows:
        by_type[tt].append(val)
    return {TYPE_NAMES[t]: {"n": len(v), "mean": float(np.mean(v)) if v else None,
                             "median": float(np.median(v)) if v else None,
                             "frac_positive": float(np.mean(np.array(v) > 0)) if v else None} for t, v in by_type.items()}


def _summ_counterfactual(rows):
    by_type = {t: {"n": 0, "gt_beats_stationary": 0, "gt_beats_half_speed": 0} for t in range(4)}
    for tt, beats_stat, beats_half in rows:
        d = by_type[tt]
        d["n"] += 1
        d["gt_beats_stationary"] += int(beats_stat)
        d["gt_beats_half_speed"] += int(beats_half)
    return {TYPE_NAMES[t]: {"n": d["n"],
                             "frac_gt_beats_stationary": d["gt_beats_stationary"] / max(d["n"], 1),
                             "frac_gt_beats_half_speed": d["gt_beats_half_speed"] / max(d["n"], 1)}
            for t, d in by_type.items()}


def _run_summ(runs):
    if not runs:
        return {"n_runs": 0}
    a = np.array(runs)
    return {"n_runs": len(a), "median_frames": float(np.median(a)), "median_ms": float(np.median(a) * 10),
            "p90_frames": float(np.percentile(a, 90))}


def _summ_turn(rows):
    by_type = {t: {"before": [], "after": []} for t in range(4)}
    for tt, before, after in rows:
        by_type[tt]["before"].append(before)
        by_type[tt]["after"].append(after)
    return {TYPE_NAMES[t]: {"n": len(d["before"]),
                             "mean_A_rank_before_turn": float(np.mean(d["before"])) if d["before"] else None,
                             "mean_A_rank_after_turn": float(np.mean(d["after"])) if d["after"] else None}
            for t, d in by_type.items()}


def _print_summary(out):
    print("=== acoustic (A) rank by type ===")
    for k, v in out["acoustic_rank"]["by_type"].items():
        print(k, v)
    print("\n=== salience (S) rank by type ===")
    for k, v in out["salience_rank"]["by_type"].items():
        print(k, v)
    print("\n=== motion contrast M_A / M_S by type ===")
    for t in TYPE_NAMES.values():
        print(t, "M_A", out["motion_contrast_A"][t], "M_S", out["motion_contrast_S"][t])
    print("\n=== zero-delta causal decomposition ===")
    for t, causes in out["zero_delta_decomposition"].items():
        total = sum(causes.values())
        print(t, "n=", total, {k: round(v / max(total, 1), 3) for k, v in causes.items()})
    print("\n=== frequency resolution vs GT 10ms motion ===")
    for k, v in out["frequency_resolution"].items():
        print(k, v)
    print("\n=== candidate continuity (S) by type ===")
    for k, v in out["candidate_continuity_S"].items():
        print(k, v)


if __name__ == "__main__":
    main()
