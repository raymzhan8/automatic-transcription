"""Step 16 section 18: representative diagnostic visualizations. A modest,
non-exhaustive set covering the clearest, best-evidenced findings from this
audit (not cherry-picked for one hypothesis) -- CQT, GT/estimated pitch,
GT/estimated local velocity, trajectory type.
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

from training.framewise_dataset import PRIMARY_LANE, RecordingLaneIndex  # noqa: E402
from training.pitch_diagnostics.common import octave_adjusted_error  # noqa: E402
from training.pitch_diagnostics.pitch_audit.common import (  # noqa: E402
    AUDIT_DIR, NATIVE_HOP_S, OUTLIER_RECORDING, build_bundles, delta_at_offset,
)

FIG_DIR = AUDIT_DIR / "figures"
TYPE_COLORS = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868", 3: "#C44E52", -1: "#DDDDDD"}
WINDOW_S = 6.0


def strip(ax, types, times, label):
    for t in np.unique(types):
        mask = types == t
        ax.scatter(times[mask], np.zeros(mask.sum()), c=TYPE_COLORS.get(int(t), "#888"), s=8, marker="s")
    ax.set_yticks([]); ax.set_ylabel(label, rotation=0, labelpad=30, va="center", fontsize=9)
    ax.set_xlim(times[0], times[-1])


def plot_window(b, cqt, center_t, out_name, title):
    hop_s = NATIVE_HOP_S
    times = b["times"]
    idx_lo = int(max(0, (center_t - WINDOW_S / 2) / hop_s))
    idx_hi = int(min(len(times), (center_t + WINDOW_S / 2) / hop_s))
    sl = slice(idx_lo, idx_hi)
    t = times[sl]

    fig, axes = plt.subplots(4, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [2.6, 0.4, 1.4, 1.4]})
    axes[0].imshow(cqt[:, sl], aspect="auto", origin="lower", extent=[t[0], t[-1], 0, cqt.shape[0]], cmap="magma")
    axes[0].set_ylabel("CQT bin"); axes[0].set_title(title, fontsize=10)

    valid_local = b["valid"][sl]
    strip(axes[1], np.where(valid_local, b["trajectory_type"][sl], -1), t, "type")

    axes[2].plot(t, np.where(valid_local, b["gt_cents"][sl], np.nan), color="black", lw=1.3, label="GT pitch")
    axes[2].plot(t, b["est_cents"][sl], color="tab:red", lw=0.9, alpha=0.85, label="Fused+D3 estimated")
    axes[2].set_ylabel("cents\n(tonic-rel.)"); axes[2].legend(fontsize=7, loc="upper right")

    d_gt = delta_at_offset(b["gt_cents"], 1)[sl] / NATIVE_HOP_S
    d_est = delta_at_offset(b["est_cents"], 1)[sl] / NATIVE_HOP_S
    axes[3].plot(t, np.where(valid_local, d_gt, np.nan), color="black", lw=1.0, label="GT velocity")
    axes[3].plot(t, d_est, color="tab:red", lw=0.8, alpha=0.85, label="est. velocity")
    axes[3].axhline(0, color="gray", lw=0.5)
    axes[3].set_ylabel("cents/s"); axes[3].set_xlabel("time (s)"); axes[3].legend(fontsize=7, loc="upper right")

    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c, label=f"T{k}" if k >= 0 else "invalid", markersize=8)
               for k, c in TYPE_COLORS.items() if k >= 0]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / out_name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_name}")


def main() -> None:
    bundles = build_bundles()
    index = RecordingLaneIndex.build(REPO_ROOT)
    by_rid = {b["recording_id"]: b for b in bundles}

    # 1. Staircase/oversmoothing example: highest-50ms-motion-error T2/T3 window
    best = None
    for b in bundles:
        d_est = delta_at_offset(b["est_cents"], 5); d_gt = delta_at_offset(b["gt_cents"], 5)
        err = np.abs(d_est - d_gt)
        m = b["valid"] & np.isfinite(err) & np.isin(b["trajectory_type"], [2, 3])
        if not m.any():
            continue
        i = int(np.argmax(np.where(m, err, -1)))
        if best is None or err[i] > best[2]:
            best = (b["recording_id"], b["times"][i], err[i])
    if best:
        rid, t_center, _ = best
        b = by_rid[rid]
        cqt = index._features[rid]["cqt_log"]
        plot_window(b, cqt, t_center, "staircase_oversmoothing_example.png", f"Oversmoothing/staircase example (largest T2/T3 50ms motion error)\n{rid}")

    # 2. Octave-transition example
    for b in bundles:
        _err, k = octave_adjusted_error(b["est_cents"], b["gt_cents"])
        k = np.nan_to_num(k, nan=0.0)
        trans = np.zeros(len(k), dtype=bool)
        trans[1:] = b["valid"][1:] & b["valid"][:-1] & (k[1:] != k[:-1])
        if trans.any():
            i = int(np.flatnonzero(trans)[0])
            cqt = index._features[b["recording_id"]]["cqt_log"]
            plot_window(b, cqt, b["times"][i], "octave_transition_example.png", f"Octave-transition example\n{b['recording_id']}")
            break

    # 3. Good T1 tracking example: lowest 50ms motion error among T1-heavy windows
    best_good = None
    for b in bundles:
        d_est = delta_at_offset(b["est_cents"], 5); d_gt = delta_at_offset(b["gt_cents"], 5)
        err = np.abs(d_est - d_gt)
        m = b["valid"] & np.isfinite(err) & (b["trajectory_type"] == 1) & (np.abs(d_gt) > 50)
        if not m.any():
            continue
        i = int(np.argmin(np.where(m, err, np.inf)))
        if best_good is None or err[i] < best_good[2]:
            best_good = (b["recording_id"], b["times"][i], err[i])
    if best_good:
        rid, t_center, _ = best_good
        b = by_rid[rid]
        cqt = index._features[rid]["cqt_log"]
        plot_window(b, cqt, t_center, "good_t1_tracking_example.png", f"Good T1 tracking example\n{rid}")

    # 4. The 692ed7e6 outlier recording -- static pitch, T1-heavy
    if OUTLIER_RECORDING in by_rid:
        b = by_rid[OUTLIER_RECORDING]
        valid_idx = np.flatnonzero(b["valid"])
        t_center = float(b["times"][valid_idx[len(valid_idx) // 2]]) if len(valid_idx) else b["duration_s"] / 2
        cqt = index._features[OUTLIER_RECORDING]["cqt_log"]
        plot_window(b, cqt, t_center, "outlier_recording_692ed7e6.png", f"Outlier recording 692ed7e6... (92% T1, 0% T2/T3, near-static GT)\n{OUTLIER_RECORDING}")


if __name__ == "__main__":
    main()
