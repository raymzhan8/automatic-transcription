"""Step 17 section 12: representative salience-path visualizations, D0 vs
D1 vs GT overlaid, at the same time/frequency coordinates. A modest,
non-exhaustive representative set (not cherry-picked for one hypothesis).
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

from training.framewise_dataset import RecordingLaneIndex  # noqa: E402
from training.pitch_diagnostics.pitch_audit.common import AUDIT_DIR, NATIVE_HOP_S, build_bundles, delta_at_offset  # noqa: E402

FIG_DIR = AUDIT_DIR / "figures"
TYPE_COLORS = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868", 3: "#C44E52", -1: "#DDDDDD"}
WINDOW_S = 6.0


def strip(ax, types, times, label):
    for t in np.unique(types):
        mask = types == t
        ax.scatter(times[mask], np.zeros(mask.sum()), c=TYPE_COLORS.get(int(t), "#888"), s=8, marker="s")
    ax.set_yticks([]); ax.set_ylabel(label, rotation=0, labelpad=30, va="center", fontsize=9)
    ax.set_xlim(times[0], times[-1])


def plot_window(b0, b1, cqt, center_t, out_name, title):
    times = b0["times"]
    idx_lo = int(max(0, (center_t - WINDOW_S / 2) / NATIVE_HOP_S))
    idx_hi = int(min(len(times), (center_t + WINDOW_S / 2) / NATIVE_HOP_S))
    sl = slice(idx_lo, idx_hi)
    t = times[sl]

    fig, axes = plt.subplots(3, 1, figsize=(10, 5.5), sharex=True, gridspec_kw={"height_ratios": [2.6, 0.4, 1.8]})
    axes[0].imshow(cqt[:, sl], aspect="auto", origin="lower", extent=[t[0], t[-1], 0, cqt.shape[0]], cmap="magma")
    axes[0].set_ylabel("CQT bin"); axes[0].set_title(title, fontsize=10)

    valid_local = b0["valid"][sl]
    strip(axes[1], np.where(valid_local, b0["trajectory_type"][sl], -1), t, "type")

    axes[2].plot(t, np.where(valid_local, b0["gt_cents"][sl], np.nan), color="black", lw=1.4, label="GT pitch", zorder=3)
    axes[2].plot(t, b0["est_cents"][sl], color="tab:blue", lw=1.0, alpha=0.85, label="D0 framewise (no Viterbi)", zorder=2)
    axes[2].plot(t, b1["est_cents"][sl], color="tab:red", lw=1.0, alpha=0.85, label="D1 Viterbi (current system)", zorder=1)
    axes[2].set_ylabel("cents\n(tonic-rel.)"); axes[2].set_xlabel("time (s)")
    axes[2].legend(fontsize=7, loc="upper right")

    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c, label=f"T{k}" if k >= 0 else "invalid", markersize=8)
               for k, c in TYPE_COLORS.items() if k >= 0]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / out_name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_name}")


def main() -> None:
    b0_list = build_bundles(variant="D0")
    b1_list = build_bundles(variant="D1")
    b0_by_rid = {b["recording_id"]: b for b in b0_list}
    b1_by_rid = {b["recording_id"]: b for b in b1_list}
    index = RecordingLaneIndex.build(REPO_ROOT)

    jobs = []

    # 1. D0 tracks a GT bend, D1 flattens it: largest (D1_err - D0_err) at 50ms on T2/T3
    best = None
    for rid, b0 in b0_by_rid.items():
        b1 = b1_by_rid[rid]
        d_gt = delta_at_offset(b0["gt_cents"], 5)
        d0_err = np.abs(delta_at_offset(b0["est_cents"], 5) - d_gt)
        d1_err = np.abs(delta_at_offset(b1["est_cents"], 5) - d_gt)
        gain = d1_err - d0_err  # positive: D0 better than D1
        m = b0["valid"] & np.isfinite(gain) & np.isin(b0["trajectory_type"], [2, 3])
        if not m.any():
            continue
        i = int(np.argmax(np.where(m, gain, -np.inf)))
        if best is None or gain[i] > best[2]:
            best = (rid, b0["times"][i], gain[i])
    if best:
        rid, tc, _ = best
        jobs.append((rid, tc, "d0_tracks_d1_flattens.png", "D0 tracks a GT bend, D1 (Viterbi) flattens it"))

    # 2. D0 noisy, D1 correctly stabilizes: largest (D0_err - D1_err) in T0
    best2 = None
    for rid, b0 in b0_by_rid.items():
        b1 = b1_by_rid[rid]
        d_gt = delta_at_offset(b0["gt_cents"], 5)
        d0_err = np.abs(delta_at_offset(b0["est_cents"], 5) - d_gt)
        d1_err = np.abs(delta_at_offset(b1["est_cents"], 5) - d_gt)
        gain = d0_err - d1_err  # positive: D1 better than D0
        m = b0["valid"] & np.isfinite(gain) & (b0["trajectory_type"] == 0)
        if not m.any():
            continue
        i = int(np.argmax(np.where(m, gain, -np.inf)))
        if best2 is None or gain[i] > best2[2]:
            best2 = (rid, b0["times"][i], gain[i])
    if best2:
        rid, tc, _ = best2
        jobs.append((rid, tc, "d1_stabilizes_d0_noise.png", "D0 is noisy in a flat region, D1 (Viterbi) correctly stabilizes it"))

    # 3. Both miss GT motion: largest min(D0_err, D1_err) on T2/T3
    best3 = None
    for rid, b0 in b0_by_rid.items():
        b1 = b1_by_rid[rid]
        d_gt = delta_at_offset(b0["gt_cents"], 5)
        d0_err = np.abs(delta_at_offset(b0["est_cents"], 5) - d_gt)
        d1_err = np.abs(delta_at_offset(b1["est_cents"], 5) - d_gt)
        both = np.minimum(d0_err, d1_err)
        m = b0["valid"] & np.isfinite(both) & np.isin(b0["trajectory_type"], [2, 3])
        if not m.any():
            continue
        i = int(np.argmax(np.where(m, both, -np.inf)))
        if best3 is None or both[i] > best3[2]:
            best3 = (rid, b0["times"][i], both[i])
    if best3:
        rid, tc, _ = best3
        jobs.append((rid, tc, "both_miss_motion.png", "Both D0 and D1 miss the same GT motion (T2/T3)"))

    for rid, tc, out_name, title in jobs:
        cqt = index._features[rid]["cqt_log"]
        plot_window(b0_by_rid[rid], b1_by_rid[rid], cqt, tc, out_name, f"{title}\n{rid}")


if __name__ == "__main__":
    main()
