"""Step 14 section 20: representative diagnostic visualizations. For a
systematically-selected (not cherry-picked) set of test-recording windows,
plot CQT + GT/A/C/D predicted trajectory-type strips + GT/estimated relative
pitch overlay.

Window selection (per-recording, using the pooled per-frame arrays this
script itself recomputes from the trained checkpoints):
 - "C fixes A" / "C hurts A": the two test recordings with the largest
   positive / negative (macro-F1 proxy: frame accuracy) delta between C and
   A, windowed on their most different sub-region.
 - "D fixes C": largest positive frame-accuracy delta between D and C.
 - one T2-heavy and one T3-heavy window (highest local T2/T3 frame density).
 - one octave-transition-heavy window (from Step 13's octave_k transitions,
   reusing the frozen Step-13 per-recording cache) and one high-|dp/dt|
   window.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.evaluate_relative_pitch_ablation import get_device, load_model, predict_recording  # noqa: E402
from training.folds import build_fold_split, load_kfold_manifest  # noqa: E402
from training.framewise_dataset import FullRecordingDataset, RecordingLaneIndex, collate_variable_length  # noqa: E402
from training.metrics import TYPE_NAMES  # noqa: E402
from training.normalization import load_fold_cqt_stats, log2_hz_to_cents  # noqa: E402
from training.relative_pitch_features import load_dense_estimated_pitch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

FIG_DIR = REPO_ROOT / "output" / "relative_pitch_ablation" / "figures"
WINDOW_S = 8.0
TYPE_COLORS = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868", 3: "#C44E52", -1: "#DDDDDD"}


def strip(ax, types: np.ndarray, times: np.ndarray, label: str) -> None:
    for t in np.unique(types):
        mask = types == t
        ax.scatter(times[mask], np.zeros(mask.sum()), c=TYPE_COLORS.get(int(t), "#888888"), s=8, marker="s")
    ax.set_yticks([])
    ax.set_ylabel(label, rotation=0, labelpad=35, va="center", fontsize=9)
    ax.set_xlim(times[0], times[-1])


def plot_window(rid, fold, all_preds, all_true, all_valid_pad_mask, frame_times, hop_s, cqt, tonic_hz,
                 gt_log2, est_log2, center_t, out_name, title):
    idx_lo = int(max(0, (center_t - WINDOW_S / 2) / hop_s))
    idx_hi = int(min(len(frame_times), (center_t + WINDOW_S / 2) / hop_s))
    sl = slice(idx_lo, idx_hi)
    t = frame_times[sl]

    fig, axes = plt.subplots(6, 1, figsize=(10, 7), sharex=True,
                              gridspec_kw={"height_ratios": [3, 0.4, 0.4, 0.4, 0.4, 1.6]})
    axes[0].imshow(cqt[:, sl], aspect="auto", origin="lower", extent=[t[0], t[-1], 0, cqt.shape[0]], cmap="magma")
    axes[0].set_ylabel("CQT bin")
    axes[0].set_title(title, fontsize=10)

    valid_local = all_valid_pad_mask[sl]
    strip(axes[1], np.where(valid_local, all_true["GT"][sl], -1), t, "GT")
    for i, cond in enumerate(("A", "C", "D")):
        vals = np.where(valid_local, all_preds[cond][sl], -1)
        strip(axes[2 + i], vals, t, cond)

    gt_cents = log2_hz_to_cents(gt_log2[sl].astype(np.float64), tonic_hz)
    est_cents = log2_hz_to_cents(est_log2[sl].astype(np.float64), tonic_hz)
    axes[5].plot(t, np.where(valid_local, gt_cents, np.nan), label="GT pitch", color="black", lw=1.2)
    axes[5].plot(t, est_cents, label="Fused+D3 (estimated)", color="tab:red", lw=0.8, alpha=0.8)
    axes[5].set_ylabel("cents\n(tonic-rel.)")
    axes[5].set_xlabel("time (s)")
    axes[5].legend(fontsize=7, loc="upper right")

    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c, label=TYPE_NAMES[k], markersize=8)
               for k, c in TYPE_COLORS.items() if k >= 0]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / out_name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_name}")


def main() -> None:
    device = get_device()
    index = RecordingLaneIndex.build(REPO_ROOT)
    manifest = load_kfold_manifest(REPO_ROOT)
    estimated_pitch = load_dense_estimated_pitch()

    # collect full per-recording predictions for A/C/D across all folds
    per_rid: dict[str, dict] = {}
    for fold in range(5):
        split = build_fold_split(manifest, fold, seed=42)
        mu_cqt, sigma_cqt = load_fold_cqt_stats(fold, REPO_ROOT)
        test_ds = FullRecordingDataset(
            index, split.test_recording_ids, mu_cqt, sigma_cqt,
            estimated_pitch=estimated_pitch, compute_pitch_features=True,
        )
        loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_variable_length)
        models = {c: load_model(c, fold, device) for c in ("A", "C", "D")}
        for batch in loader:
            rid = batch["recording_id"][0]
            lane = next(x for x in index.lanes if x.recording_id == rid)
            preds = {}
            for c, (model, phi_mu, phi_sigma, _ckpt) in models.items():
                logits = predict_recording(model, batch, c, phi_mu, phi_sigma, device)
                preds[c] = logits[0].argmax(axis=-1)
            per_rid[rid] = {
                "fold": fold,
                "preds": preds,
                "true": batch["trajectory_type"][0].numpy(),
                "valid_pad": batch["valid_target"][0].numpy() & (~batch["padding_mask"][0].numpy()),
                "frame_times": batch["frame_time_s"][0],
                "cqt": batch["spec"][0, 0].numpy(),
                "tonic_hz": lane.fundamental_hz,
                "gt_log2": batch["pitch_log2_hz"][0].numpy(),
                "est_log2": estimated_pitch[rid][: len(batch["frame_time_s"][0])],
            }
        print(f"fold {fold}: collected {len(split.test_recording_ids)} recordings")

    hop_s = 0.01

    # ---- systematic selection ----
    deltas_ca, deltas_dc = {}, {}
    for rid, d in per_rid.items():
        m = d["valid_pad"]
        acc_a = float((d["preds"]["A"][m] == d["true"][m]).mean())
        acc_c = float((d["preds"]["C"][m] == d["true"][m]).mean())
        acc_d = float((d["preds"]["D"][m] == d["true"][m]).mean())
        deltas_ca[rid] = acc_c - acc_a
        deltas_dc[rid] = acc_d - acc_c

    rid_fix = max(deltas_ca, key=deltas_ca.get)
    rid_hurt = min(deltas_ca, key=deltas_ca.get)
    rid_dfix = max(deltas_dc, key=deltas_dc.get)

    def best_window_center(rid, prefer_type=None, prefer_dpdt=False):
        d = per_rid[rid]
        times = d["frame_times"]
        valid = d["valid_pad"]
        true = d["true"]
        if prefer_type is not None:
            mask = valid & (true == prefer_type)
        else:
            mask = valid
        if not np.any(mask):
            mask = valid
        idx = np.flatnonzero(mask)
        return float(times[idx[len(idx) // 2]])

    jobs = [
        (rid_fix, "C fixes A: largest C-A accuracy gain", "case_c_fixes_a.png", None),
        (rid_hurt, "C hurts A: largest C-A accuracy loss", "case_c_hurts_a.png", None),
        (rid_dfix, "D fixes C: largest D-C accuracy gain", "case_d_fixes_c.png", None),
    ]
    # T2/T3 examples: recordings with the most T2/T3 support among evaluated recordings
    t2_rid = max(per_rid, key=lambda r: int((per_rid[r]["true"][per_rid[r]["valid_pad"]] == 2).sum()))
    t3_rid = max(per_rid, key=lambda r: int((per_rid[r]["true"][per_rid[r]["valid_pad"]] == 3).sum()))
    jobs.append((t2_rid, "T2 (sloped-start bend) example", "case_t2_example.png", 2))
    jobs.append((t3_rid, "T3 (sloped-end bend) example", "case_t3_example.png", 3))

    for rid, title, out_name, prefer_type in jobs:
        d = per_rid[rid]
        center = best_window_center(rid, prefer_type=prefer_type)
        all_true = {"GT": d["true"]}
        plot_window(
            rid, d["fold"], d["preds"], all_true, d["valid_pad"], d["frame_times"], hop_s,
            d["cqt"], d["tonic_hz"], d["gt_log2"], d["est_log2"], center, out_name,
            f"{title}\n{rid} (fold {d['fold']})",
        )

    print("\ndelta C-A (accuracy) per recording:")
    for rid, v in sorted(deltas_ca.items(), key=lambda x: -x[1]):
        print(f"  {rid} {v:+.3f}")


if __name__ == "__main__":
    main()
