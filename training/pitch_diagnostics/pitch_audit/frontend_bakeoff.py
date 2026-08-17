"""Step 20 Phase A: real-data acoustic-only bake-off across every candidate
frontend (frontends.py). No learned model, no salience, no decoder -- every
metric here is a property of a frontend's own raw magnitude spectrum,
computed over its FULL native bin grid (not restricted to the 104-778Hz
candidate band), mirroring Step 19's exact acoustic-rank (A-rank) definition
so A0's numbers are a direct reproduction check against Step 19's frozen
results (spec section 1). Because different frontends have very different
native bin counts (CQT: 360; STFT n_fft=4096: 2049), every rank is also
reported as a normalized percentile (rank / n_bins) for cross-frontend
comparability (spec section 5/9). GT is used for evaluation only, never for
feature construction. Fully vectorized per recording -- no per-frame Python
loop over up to 220k native frames.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.canonical.visualize_targets import source_audio_path  # noqa: E402
from training.features import SR  # noqa: E402
from training.framewise_dataset import PRIMARY_LANE, RecordingLaneIndex  # noqa: E402
from training.metrics import TYPE_NAMES  # noqa: E402
from training.normalization import log2_hz_to_cents  # noqa: E402
from training.pitch_diagnostics.common import linear_mag, write_json  # noqa: E402
from training.pitch_diagnostics.pitch_audit.common import AUDIT_DIR, NATIVE_HOP_S, both_valid_mask  # noqa: E402
from training.pitch_diagnostics.pitch_audit.frontends import (  # noqa: E402
    FRONTENDS, align_to_canonical_grid, compute_frontend_native, cqt_bin_hz, hz_to_bin,
)

TURN_THRESHOLD_CENTS_S = 100.0
A_RANK_STRONG, A_RANK_WEAK = 5, 20
K_WINDOW = 10  # 100ms, matches Step 19
DPDT_EDGES = (0.0, 100.0, 400.0, 1000.0, np.inf)
DPDT_NAMES = ("0-100c/s", "100-400c/s", "400-1000c/s", ">1000c/s")


def _vectorized_run_lengths(mask: np.ndarray) -> list[int]:
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diffs == 1)
    ends = np.flatnonzero(diffs == -1)
    return (ends - starts).tolist()


def _frontend_local_bin_width_cents(spec, hz: np.ndarray) -> np.ndarray:
    """Local bin width in cents at frequency `hz` on this frontend's own
    native grid (constant 16.667c for CQT; frequency-dependent, linear-Hz
    for STFT/multires)."""
    if spec.kind == "cqt":
        return np.full_like(hz, 1200.0 / 72.0)
    n_fft = spec.params.get("win_long")
    if n_fft is None:
        win = spec.params["win_length"]
        n_fft = max(1 << (win - 1).bit_length(), win)
    d_hz = SR / n_fft
    return 1200.0 * np.log2((hz + d_hz) / hz)


def process_recording_frontend(spec, mag: np.ndarray, hz_of_bin: np.ndarray,
                                gt_hz: np.ndarray, gt_cents: np.ndarray, valid: np.ndarray,
                                ttype: np.ndarray, dpdt: np.ndarray, times: np.ndarray,
                                primitives: list[dict]) -> dict:
    """Fully vectorized per-recording metrics for one frontend. Returns a
    dict of numpy arrays (masked to the relevant subset already) ready to be
    concatenated across recordings."""
    n_bins, n = mag.shape
    bin_idx_f = hz_to_bin(spec, np.clip(gt_hz, 1.0, None))
    in_grid = valid & np.isfinite(bin_idx_f) & (bin_idx_f >= 0) & (bin_idx_f < n_bins)

    idx_clip = np.clip(np.round(np.nan_to_num(bin_idx_f, nan=0.0)), 0, n_bins - 1).astype(np.int64)
    gt_val = mag[idx_clip, np.arange(n)]
    rank = (mag > gt_val[None, :]).sum(axis=0) + 1
    rank = np.where(in_grid, rank, n_bins + 1).astype(np.float64)
    norm_rank = rank / n_bins

    near_boundary = np.zeros(n, dtype=bool)
    near_boundary_100 = np.zeros(n, dtype=bool)
    for prim in primitives:
        near_boundary |= (np.abs(times - prim["start_s"]) <= 0.05) | (np.abs(times - prim["end_s"]) <= 0.05)
        near_boundary_100 |= (np.abs(times - prim["start_s"]) <= 0.10) | (np.abs(times - prim["end_s"]) <= 0.10)

    dpdt_bucket = np.full(n, -1, dtype=np.int8)
    adpdt = np.abs(dpdt)
    for bi, (lo, hi) in enumerate(zip(DPDT_EDGES[:-1], DPDT_EDGES[1:])):
        dpdt_bucket[(adpdt >= lo) & (adpdt < hi)] = bi

    m = in_grid
    rank_out = {
        "tt": ttype[m].astype(np.int8), "dpdt": dpdt_bucket[m],
        "near50": near_boundary[m], "near100": near_boundary_100[m],
        "rank": rank[m], "norm_rank": norm_rank[m],
    }

    # ---- motion contrast + counterfactual, k=10 (100ms) window ----
    k = K_WINDOW
    both = both_valid_mask(valid & in_grid, k)
    idxs = np.flatnonzero(both)
    if len(idxs) > 0:
        cols = idxs[:, None] - k + np.arange(k + 1)[None, :]
        rows_path = idx_clip[cols]
        e_gt_path = mag[rows_path, cols].sum(axis=1)
        stat_bin = idx_clip[idxs - k]
        stat_rows = np.repeat(stat_bin[:, None], k + 1, axis=1)
        e_stat_path = mag[stat_rows, cols].sum(axis=1)
        denom = np.maximum(e_gt_path + e_stat_path, 1e-12)
        m_a = (e_gt_path - e_stat_path) / denom

        start_idx = idx_clip[idxs - k].astype(np.float64)
        end_idx = idx_clip[idxs].astype(np.float64)
        frac = np.arange(k + 1) / k
        half_target = start_idx[:, None] + (end_idx - start_idx)[:, None] * 0.5 * frac[None, :]
        half_rows = np.clip(np.round(half_target), 0, n_bins - 1).astype(np.int64)
        e_half_path = mag[half_rows, cols].sum(axis=1)

        d_gt = gt_cents[idxs] - gt_cents[idxs - k]
        moving = np.isfinite(d_gt) & (np.abs(d_gt) >= 1e-6)
        beats_stat = e_gt_path > e_stat_path
        beats_half = e_gt_path > e_half_path
        mot_out = {
            "tt": ttype[idxs][moving].astype(np.int8), "m_a": m_a[moving],
            "beats_stat": beats_stat[moving], "beats_half": beats_half[moving],
        }
    else:
        mot_out = {"tt": np.array([], dtype=np.int8), "m_a": np.array([]),
                    "beats_stat": np.array([], dtype=bool), "beats_half": np.array([], dtype=bool)}

    # ---- continuity: run length of rank<=5 ----
    continuity = {}
    for tt in range(4):
        type_valid = in_grid & (ttype == tt)
        continuity[tt] = _vectorized_run_lengths(type_valid & (rank <= A_RANK_STRONG))

    # ---- turning-point response ----
    gt_vel = np.zeros(n)
    gt_vel[1:] = (gt_cents[1:] - gt_cents[:-1]) / NATIVE_HOP_S
    sign = np.zeros(n, dtype=np.int8)
    sign[gt_vel > TURN_THRESHOLD_CENTS_S] = 1
    sign[gt_vel < -TURN_THRESHOLD_CENTS_S] = -1
    turn_idx = np.flatnonzero((sign[1:] != 0) & (sign[:-1] != 0) & (sign[1:] != sign[:-1]) & valid[1:] & valid[:-1]) + 1
    turn_idx = turn_idx[(turn_idx >= 5) & (turn_idx < n - 5)]
    turn_out = {"tt": [], "before": [], "after": []}
    for ti in turn_idx:
        tt = int(ttype[ti])
        if tt not in range(4):
            continue
        turn_out["tt"].append(tt)
        turn_out["before"].append(float(np.mean(rank[ti - 5:ti])))
        turn_out["after"].append(float(np.mean(rank[ti:ti + 5])))

    # ---- frontend-only causal proxy (§14): classify GT-moving frames ----
    local_bin_width = _frontend_local_bin_width_cents(spec, np.clip(gt_hz, 1.0, None))
    both1 = both_valid_mask(valid & in_grid, 1)
    moving_mask = both1 & (np.abs(gt_vel) > 100.0)
    dgt10 = np.full(n, np.nan)
    dgt10[1:] = np.abs(gt_cents[1:] - gt_cents[:-1])
    cat = np.full(n, -1, dtype=np.int8)  # 0 strong 1 weak 2 ambiguous 3 sub_resolution
    sub_res = moving_mask & (dgt10 < local_bin_width)
    strong = moving_mask & ~sub_res & (rank <= A_RANK_STRONG)
    weak = moving_mask & ~sub_res & ~strong & (rank <= A_RANK_WEAK)
    ambig = moving_mask & ~sub_res & ~strong & ~weak
    cat[sub_res] = 3
    cat[strong] = 0
    cat[weak] = 1
    cat[ambig] = 2
    causal_out = {"tt": ttype[moving_mask].astype(np.int8), "cat": cat[moving_mask]}

    return {"rank": rank_out, "motion": mot_out, "continuity": continuity, "turn": turn_out, "causal": causal_out}


def main() -> None:
    index = RecordingLaneIndex.build(REPO_ROOT)
    acc = {name: {
        "rank": {"tt": [], "dpdt": [], "near50": [], "near100": [], "rank": [], "norm_rank": []},
        "motion": {"tt": [], "m_a": [], "beats_stat": [], "beats_half": []},
        "continuity": {t: [] for t in range(4)},
        "turn": {"tt": [], "before": [], "after": []},
        "causal": {"tt": [], "cat": []},
    } for name in FRONTENDS}

    t_start = time.time()
    for i, lane in enumerate(index.lanes):
        rid = lane.recording_id
        frames = index._frames[(rid, PRIMARY_LANE)]
        n_cache = min(index._features[rid]["cqt_log"].shape[1], lane.n_frames, len(frames["frame_time_s"]))
        times = frames["frame_time_s"][:n_cache]
        valid = frames["valid_target"][:n_cache] & (times < lane.duration_s)
        tonic_hz = lane.fundamental_hz
        gt_log2 = frames["pitch_log2_hz"][:n_cache].astype(np.float64)
        gt_hz = np.exp2(gt_log2)
        gt_cents = log2_hz_to_cents(gt_log2, tonic_hz)
        ttype = frames["trajectory_type"][:n_cache]
        dpdt = frames["dp_dt_log2_hz_per_s"][:n_cache].astype(np.float64) * 1200.0
        primitives = index.primitives_for_recording(rid)

        rec_path = REPO_ROOT / "output" / "canonical" / "v1" / "recordings" / f"{rid}.json"
        recording_doc = json.loads(rec_path.read_text(encoding="utf-8"))
        audio_path = source_audio_path(recording_doc, REPO_ROOT)
        y = None

        for name, spec in FRONTENDS.items():
            if name == "A0_cqt_fs1":
                mag_aligned = linear_mag(index._features[rid]["cqt_log"][:, :n_cache])
                hz_full = cqt_bin_hz()
            else:
                if y is None:
                    y, _sr = librosa.load(audio_path, sr=SR, mono=True)
                nt, mag_native, hz_full = compute_frontend_native(y, spec)
                mag_aligned = align_to_canonical_grid(nt, mag_native, times)

            r = process_recording_frontend(spec, mag_aligned, hz_full, gt_hz, gt_cents, valid, ttype, dpdt, times, primitives)
            for k_ in acc[name]["rank"]:
                acc[name]["rank"][k_].append(r["rank"][k_])
            for k_ in acc[name]["motion"]:
                acc[name]["motion"][k_].append(r["motion"][k_])
            for tt in range(4):
                acc[name]["continuity"][tt].extend(r["continuity"][tt])
            acc[name]["turn"]["tt"].extend(r["turn"]["tt"])
            acc[name]["turn"]["before"].extend(r["turn"]["before"])
            acc[name]["turn"]["after"].extend(r["turn"]["after"])
            acc[name]["causal"]["tt"].append(r["causal"]["tt"])
            acc[name]["causal"]["cat"].append(r["causal"]["cat"])
        print(f"[{i+1}/{len(index.lanes)}] {rid} done ({time.time()-t_start:.1f}s elapsed)", flush=True)

    out = {name: summarize(acc[name]) for name in FRONTENDS}
    write_json(AUDIT_DIR / "frontend_bakeoff.json", out)
    print(f"total {time.time()-t_start:.1f}s")
    for name in FRONTENDS:
        print(f"\n=== {name} ===")
        for t, v in out[name]["rank_by_type"].items():
            print(" ", t, v)


def summarize(a: dict) -> dict:
    tt = np.concatenate(a["rank"]["tt"]) if a["rank"]["tt"] else np.array([], dtype=np.int8)
    dpdt = np.concatenate(a["rank"]["dpdt"]) if a["rank"]["dpdt"] else np.array([], dtype=np.int8)
    near50 = np.concatenate(a["rank"]["near50"]) if a["rank"]["near50"] else np.array([], dtype=bool)
    near100 = np.concatenate(a["rank"]["near100"]) if a["rank"]["near100"] else np.array([], dtype=bool)
    rank = np.concatenate(a["rank"]["rank"]) if a["rank"]["rank"] else np.array([])
    norm_rank = np.concatenate(a["rank"]["norm_rank"]) if a["rank"]["norm_rank"] else np.array([])

    def rk(mask):
        v, nv = rank[mask], norm_rank[mask]
        return {"n": int(mask.sum()), "median_rank": float(np.median(v)) if len(v) else None,
                "mean_rank": float(np.mean(v)) if len(v) else None,
                "median_norm_rank_pct": float(100 * np.median(nv)) if len(nv) else None}

    rank_by_type = {TYPE_NAMES[t]: rk(tt == t) for t in range(4)}
    rank_by_dpdt = {DPDT_NAMES[b]: rk(dpdt == b) for b in range(4)}
    rank_by_boundary50 = {"near": rk(near50), "away": rk(~near50)}
    rank_by_boundary100 = {"near": rk(near100), "away": rk(~near100)}

    mtt = np.concatenate(a["motion"]["tt"]) if a["motion"]["tt"] else np.array([], dtype=np.int8)
    m_a = np.concatenate(a["motion"]["m_a"]) if a["motion"]["m_a"] else np.array([])
    beats_stat = np.concatenate(a["motion"]["beats_stat"]) if a["motion"]["beats_stat"] else np.array([], dtype=bool)
    beats_half = np.concatenate(a["motion"]["beats_half"]) if a["motion"]["beats_half"] else np.array([], dtype=bool)

    def motion_summ(t):
        msk = mtt == t
        v = m_a[msk]
        return {"n": int(msk.sum()), "median": float(np.median(v)) if len(v) else None,
                "frac_positive": float(np.mean(v > 0)) if len(v) else None,
                "frac_gt_beats_stationary": float(np.mean(beats_stat[msk])) if msk.sum() else None,
                "frac_gt_beats_half_speed": float(np.mean(beats_half[msk])) if msk.sum() else None}

    def run_summ(runs):
        if not runs:
            return {"n_runs": 0}
        arr = np.array(runs)
        return {"n_runs": len(arr), "median_frames": float(np.median(arr)), "median_ms": float(np.median(arr) * 10),
                "p90_frames": float(np.percentile(arr, 90))}

    turn_tt = np.array(a["turn"]["tt"])
    turn_before = np.array(a["turn"]["before"])
    turn_after = np.array(a["turn"]["after"])

    def turn_summ(t):
        msk = turn_tt == t
        if msk.sum() == 0:
            return {"n": 0, "mean_rank_before": None, "mean_rank_after": None}
        return {"n": int(msk.sum()), "mean_rank_before": float(np.mean(turn_before[msk])),
                "mean_rank_after": float(np.mean(turn_after[msk]))}

    ctt = np.concatenate(a["causal"]["tt"]) if a["causal"]["tt"] else np.array([], dtype=np.int8)
    ccat = np.concatenate(a["causal"]["cat"]) if a["causal"]["cat"] else np.array([], dtype=np.int8)
    cat_names = {0: "strong", 1: "weak", 2: "ambiguous", 3: "sub_resolution_movement"}

    def causal_summ(t):
        msk = ctt == t
        total = int(msk.sum())
        if total == 0:
            return {"n": 0}
        out = {"n": total}
        for ci, cn in cat_names.items():
            out[cn] = int((ccat[msk] == ci).sum())
        return out

    return {
        "rank_by_type": rank_by_type, "rank_by_dpdt": rank_by_dpdt,
        "rank_by_boundary_50ms": rank_by_boundary50, "rank_by_boundary_100ms": rank_by_boundary100,
        "motion_contrast": {TYPE_NAMES[t]: motion_summ(t) for t in range(4)},
        "continuity": {TYPE_NAMES[t]: run_summ(v) for t, v in a["continuity"].items()},
        "turning_point_response": {TYPE_NAMES[t]: turn_summ(t) for t in range(4)},
        "causal_proxy": {TYPE_NAMES[t]: causal_summ(t) for t in range(4)},
    }


if __name__ == "__main__":
    main()
