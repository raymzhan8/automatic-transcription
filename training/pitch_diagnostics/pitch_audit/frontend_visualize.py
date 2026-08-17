"""Step 20 section 17: matched A0-vs-challenger visualizations, reusing the
exact real-data windows Step 19 already selected (predecoder_*.png) so the
comparison is apples-to-apples, not a new cherry-picked window. A0 uses the
cached production CQT; the challenger is computed fresh from source audio
for just these three recordings (cheap -- seconds each)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.canonical.visualize_targets import source_audio_path  # noqa: E402
from training.features import SR  # noqa: E402
from training.framewise_dataset import PRIMARY_LANE, RecordingLaneIndex  # noqa: E402
from training.normalization import log2_hz_to_cents  # noqa: E402
from training.pitch_diagnostics.common import linear_mag  # noqa: E402
from training.pitch_diagnostics.pitch_audit.common import AUDIT_DIR  # noqa: E402
from training.pitch_diagnostics.pitch_audit.frontends import (  # noqa: E402
    FRONTENDS, align_to_canonical_grid, compute_frontend_native, cqt_bin_hz,
)

FIG_DIR = AUDIT_DIR / "figures"
TYPE_COLORS = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868", 3: "#C44E52", -1: "#DDDDDD"}
CHALLENGER = "A1a_cqt_fs0.5"

CASES = [
    ("6824de49abc4705438ce918b", 194.3, 198.2, "GT weak/absent in A (Step 19's acoustic_absent case)"),
    ("645ff354deeaf2d1e33b3c44", 91.2, 95.2, "T3 turning-point segment"),
    ("6503e36cd9ff49d3988d0b40", 2.7, 6.7, "T0 control"),
]


def plot_case(rid, t0, t1, title, index):
    lane = next(x for x in index.lanes if x.recording_id == rid)
    frames = index._frames[(rid, PRIMARY_LANE)]
    n_cache = min(index._features[rid]["cqt_log"].shape[1], lane.n_frames, len(frames["frame_time_s"]))
    times = frames["frame_time_s"][:n_cache]
    valid = frames["valid_target"][:n_cache] & (times < lane.duration_s)
    tonic_hz = lane.fundamental_hz
    gt_log2 = frames["pitch_log2_hz"][:n_cache].astype(np.float64)
    gt_cents = log2_hz_to_cents(gt_log2, tonic_hz)
    ttype = frames["trajectory_type"][:n_cache]

    mag_a0 = linear_mag(index._features[rid]["cqt_log"][:, :n_cache])
    hz_a0 = cqt_bin_hz()

    rec_path = REPO_ROOT / "output" / "canonical" / "v1" / "recordings" / f"{rid}.json"
    recording_doc = json.loads(rec_path.read_text(encoding="utf-8"))
    audio_path = source_audio_path(recording_doc, REPO_ROOT)
    y, _sr = librosa.load(audio_path, sr=SR, mono=True)
    nt, mag_native, hz_ch = compute_frontend_native(y, FRONTENDS[CHALLENGER])
    mag_ch = align_to_canonical_grid(nt, mag_native, times)

    sl = (times >= t0) & (times <= t1)
    t = times[sl]
    gt = np.where(valid[sl], gt_cents[sl], np.nan)
    tt = np.where(valid[sl], ttype[sl], -1)
    gt_bin_a0 = np.interp(gt_log2[sl], np.log2(hz_a0), np.arange(len(hz_a0)))
    gt_bin_ch = np.interp(gt_log2[sl], np.log2(hz_ch), np.arange(len(hz_ch)))

    # restrict display to a plausible bin window around GT for readability
    lo0, hi0 = int(np.nanmin(gt_bin_a0)) - 60, int(np.nanmax(gt_bin_a0)) + 60
    lo0, hi0 = max(0, lo0), min(mag_a0.shape[0], hi0)
    frac = hi0 - lo0
    lo_c = int(lo0 * mag_ch.shape[0] / mag_a0.shape[0])
    hi_c = int(hi0 * mag_ch.shape[0] / mag_a0.shape[0])

    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [2, 2, 0.35]})
    axes[0].imshow(mag_a0[lo0:hi0][:, sl], aspect="auto", origin="lower", extent=[t[0], t[-1], lo0, hi0], cmap="viridis")
    axes[0].plot(t, gt_bin_a0, color="cyan", lw=1.0, alpha=0.85, label="GT")
    axes[0].set_ylabel("A0 (fs=1)\nbin"); axes[0].legend(fontsize=7, loc="upper right")
    axes[0].set_title(f"{title}\n{rid}", fontsize=10)

    axes[1].imshow(mag_ch[lo_c:hi_c][:, sl], aspect="auto", origin="lower", extent=[t[0], t[-1], lo_c, hi_c], cmap="viridis")
    axes[1].plot(t, gt_bin_ch, color="cyan", lw=1.0, alpha=0.85)
    axes[1].set_ylabel(f"{CHALLENGER}\nbin")

    for k in np.unique(tt):
        m = tt == k
        axes[2].scatter(t[m], np.zeros(m.sum()), c=TYPE_COLORS.get(int(k), "#888"), s=8, marker="s")
    axes[2].set_yticks([]); axes[2].set_xlabel("time (s)")

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"frontend_compare_{rid}.png"
    fig.savefig(FIG_DIR / out_name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_name}")


def main():
    index = RecordingLaneIndex.build(REPO_ROOT)
    for rid, t0, t1, title in CASES:
        plot_case(rid, t0, t1, title, index)


if __name__ == "__main__":
    main()
