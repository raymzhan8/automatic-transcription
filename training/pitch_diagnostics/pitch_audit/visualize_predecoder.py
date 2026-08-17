"""Step 19 section 19: a representative (not exhaustive) set of predecoder
visualizations -- raw acoustic evidence A (mag, restricted to the shared
candidate window for display), fused salience S=C, GT pitch, D0 decoded
pitch, and trajectory type, at windows selected during a single pass over
iter_recording_records() using the same rank/threshold conventions as
predecoder_audit.py. Each case is picked by an extremum of a diagnostic
already computed in the main audit, not cherry-picked post hoc for a
particular narrative.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.pitch_diagnostics.pitch_audit.common import AUDIT_DIR, NATIVE_HOP_S, both_valid_mask, delta_at_offset  # noqa: E402
from training.pitch_diagnostics.pitch_audit.predecoder_common import iter_recording_records  # noqa: E402

CAND_LO, CAND_HI = 34, 244
N_CAND = CAND_HI - CAND_LO
A_RANK_WEAK_THRESHOLD = 20
FIG_DIR = AUDIT_DIR / "figures"
WINDOW_S = 4.0
TYPE_COLORS = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868", 3: "#C44E52", -1: "#DDDDDD"}


def _rank_of_bin(col, bin_idx):
    if bin_idx < 0 or bin_idx >= len(col):
        return len(col) + 1
    return int((col > col[bin_idx]).sum()) + 1


def plot_case(rec, ti, out_name, title):
    times = rec["times"]
    idx_lo = int(max(0, ti - WINDOW_S / 2 / NATIVE_HOP_S))
    idx_hi = int(min(rec["n"], ti + WINDOW_S / 2 / NATIVE_HOP_S))
    sl = slice(idx_lo, idx_hi)
    t = times[sl]
    mag_cand = rec["mag"][CAND_LO:CAND_HI, sl]
    s_win = rec["fused_probs"][:, sl]
    valid = rec["valid"][sl]
    gt = np.where(valid, rec["gt_cents_rel"][sl], np.nan)
    d0 = np.where(valid, rec["d0_cents_rel"][sl], np.nan)
    tt = np.where(valid, rec["trajectory_type"][sl], -1)

    fig, axes = plt.subplots(4, 1, figsize=(9, 6.5), sharex=True,
                              gridspec_kw={"height_ratios": [2.2, 2.2, 0.35, 1.8]})
    axes[0].imshow(mag_cand, aspect="auto", origin="lower", extent=[t[0], t[-1], 0, N_CAND], cmap="viridis")
    axes[0].set_ylabel("A (mag)\ncand. bin"); axes[0].set_title(title, fontsize=10)
    axes[1].imshow(s_win, aspect="auto", origin="lower", extent=[t[0], t[-1], 0, N_CAND], cmap="magma")
    axes[1].set_ylabel("S=C (fused\nprobs)")

    for k in np.unique(tt):
        m = tt == k
        axes[2].scatter(t[m], np.zeros(m.sum()), c=TYPE_COLORS.get(int(k), "#888"), s=8, marker="s")
    axes[2].set_yticks([])

    gt_cand_bin = np.round(rec["gt_bin_abs"][sl]).astype(np.float64) - CAND_LO
    axes[0].plot(t, gt_cand_bin, color="cyan", lw=1.0, alpha=0.8, label="GT (cand. bin)")
    axes[1].plot(t, gt_cand_bin, color="cyan", lw=1.0, alpha=0.8)
    axes[0].legend(fontsize=7, loc="upper right")

    axes[3].plot(t, gt, color="black", lw=1.4, label="GT pitch", zorder=3)
    axes[3].plot(t, d0, color="tab:blue", lw=1.0, alpha=0.85, label="D0 (framewise)", zorder=2)
    axes[3].set_ylabel("cents\n(tonic-rel.)"); axes[3].set_xlabel("time (s)")
    axes[3].legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / out_name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_name}")


def main() -> None:
    # trackers: (score, rid, ti, rec-snapshot) -- rec captured lazily via a
    # small window copy so we don't hold every recording's full arrays.
    best = {}  # name -> (score, snapshot_dict)

    def consider(name, score, rec, ti, higher_is_better=True):
        cur = best.get(name)
        if cur is not None:
            if higher_is_better and score <= cur[0]:
                return
            if not higher_is_better and score >= cur[0]:
                return
        best[name] = (score, {
            "times": rec["times"], "valid": rec["valid"], "n": rec["n"],
            "mag": rec["mag"], "fused_probs": rec["fused_probs"],
            "gt_bin_abs": rec["gt_bin_abs"], "gt_cents_rel": rec["gt_cents_rel"],
            "d0_cents_rel": rec["d0_cents_rel"], "trajectory_type": rec["trajectory_type"],
            "recording_id": rec["recording_id"],
        }, ti)

    for rec in iter_recording_records():
        valid = rec["valid"]
        n = rec["n"]
        tt_arr = rec["trajectory_type"]
        gt_vel = rec["dp_dt_cents_s"]
        d0_delta1 = delta_at_offset(rec["d0_cents_rel"], 1)
        bv1 = both_valid_mask(valid, 1)
        moving_fast = bv1 & (np.abs(gt_vel) > 300.0)
        zero_delta = bv1 & (np.abs(gt_vel) > 100.0) & (np.abs(d0_delta1) < 1e-6)

        mag = rec["mag"]
        fused = rec["fused_probs"]
        gt_bin_abs = rec["gt_bin_abs"]

        # 1. successful moving ridge: T1/T2/T3, fast GT motion, D0 tracks closely (small err)
        d0_err = np.abs(delta_at_offset(rec["d0_cents_rel"], 5) - delta_at_offset(rec["gt_cents_rel"], 5))
        m = moving_fast & np.isin(tt_arr, [1, 2, 3]) & np.isfinite(d0_err)
        if m.any():
            i = int(np.argmin(np.where(m, d0_err, np.inf)))
            consider("successful_moving_ridge", -d0_err[i], rec, i, higher_is_better=True)

        # 2. GT absent already in A: worst (largest) acoustic rank among zero-delta failure frames
        for t in np.flatnonzero(zero_delta):
            gb = int(round(gt_bin_abs[t]))
            if gb < 0 or gb >= mag.shape[0]:
                continue
            a_rank = _rank_of_bin(mag[:, t], gb)
            consider("acoustic_absent", a_rank, rec, t, higher_is_better=True)

        # 3. GT present in S but wrong candidate selected (present_but_not_selected-like):
        #    zero-delta failure, A rank ok, S rank ok, but D0 still flat -> pick best example
        for t in np.flatnonzero(zero_delta):
            gb = int(round(gt_bin_abs[t]))
            gbc = gb - CAND_LO
            if not (0 <= gbc < N_CAND):
                continue
            a_ok = _rank_of_bin(mag[:, t], gb) <= A_RANK_WEAK_THRESHOLD
            s_rank = _rank_of_bin(fused[:, t], gbc)
            if a_ok and s_rank <= 5:
                consider("present_not_selected", -s_rank, rec, t, higher_is_better=True)

        # 4. T3 turning point: local extremum of GT cents within a T3 segment
        gt_c = rec["gt_cents_rel"]
        for t in np.flatnonzero(valid & (tt_arr == 3)):
            if t < 5 or t >= n - 5:
                continue
            if not (valid[t - 5:t + 5].all()):
                continue
            seg = gt_c[t - 5:t + 5]
            if np.isnan(seg).any():
                continue
            curvature = abs(seg[-1] + seg[0] - 2 * seg[len(seg) // 2])
            consider("t3_turning_point", curvature, rec, t, higher_is_better=True)

        # 5. T0 control: long stable run, high A-rank confidence (small rank)
        for t in np.flatnonzero(valid & (tt_arr == 0)):
            gb = int(round(gt_bin_abs[t]))
            if gb < 0 or gb >= mag.shape[0]:
                continue
            a_rank = _rank_of_bin(mag[:, t], gb)
            consider("t0_control", -a_rank, rec, t, higher_is_better=True)

    titles = {
        "successful_moving_ridge": "Successful moving ridge: D0 tracks fast GT motion (T1-T3)",
        "acoustic_absent": "GT already weak/absent in acoustic evidence A (zero-delta failure)",
        "present_not_selected": "GT present with strong A and S rank, D0 still flat (rare case)",
        "t3_turning_point": "T3 turning point",
        "t0_control": "T0 control: stable pitch, strong acoustic rank",
    }
    for name, (score, snap, ti) in best.items():
        out_name = f"predecoder_{name}.png"
        plot_case(snap, ti, out_name, f"{titles[name]}\n{snap['recording_id']} (score={score:.2f})")


if __name__ == "__main__":
    main()
