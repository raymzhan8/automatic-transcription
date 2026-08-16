"""Step 11 §7/§18/§26: CQT + GT pitch + salience-map visualizations.

Currently covers the tiny-overfit sanity check (§18): target-vs-predicted
salience for a handful of frames/windows, for both the local-only control and
harmonic-aware models, using their tiny-overfit checkpoints. The fuller
HPS-vs-learned, per-class/fold representative-window gallery (§26) is a
separate, later step once full 5-fold models exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.folds import prepare_fold  # noqa: E402
from training.framewise_dataset import FramewiseExcerptDataset, RecordingLaneIndex, collate_excerpts  # noqa: E402
from training.normalization import load_fold_cqt_stats, log2_hz_to_cents  # noqa: E402
from training.pitch_diagnostics.common import OUT_DIR  # noqa: E402
from training.pitch_diagnostics.salience_common import candidate_hz, load_or_compute_candidate_range  # noqa: E402
from training.pitch_diagnostics.salience_models import HarmonicSalienceModel  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def plot_tiny_overfit_examples(run_name: str, harmonic_ks: tuple[int, ...], hidden: int, out_name: str) -> Path:
    repo_root = REPO_ROOT
    split, _ = prepare_fold(repo_root, 0, seed=42)
    index = RecordingLaneIndex.build(repo_root)
    cr = load_or_compute_candidate_range(repo_root, index)
    lo, hi = cr["candidate_lo_bin"], cr["candidate_hi_bin"]
    cand_hz = candidate_hz(lo, hi)

    mu, sigma = load_fold_cqt_stats(0, repo_root)
    ds = FramewiseExcerptDataset(index, split.train_recording_ids, mu, sigma, seed=42, excerpts_per_epoch=32, cache_excerpts=32)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_excerpts)
    batch = next(iter(loader))

    ckpt = torch.load(OUT_DIR / "runs" / run_name / "fold_0" / "best.pt", map_location="cpu")
    model = HarmonicSalienceModel(candidate_lo_bin=lo, candidate_hi_bin=hi, harmonic_ks=harmonic_ks, hidden=hidden)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    valid = batch["valid_target"]
    pad = batch["padding_mask"]
    mask = (~pad) & valid
    with torch.no_grad():
        logits = model(batch["spec"])
        probs = torch.softmax(logits, dim=1).numpy()  # [B,F_cand,T]

    b = 0
    mask_np_full = mask[b].numpy()
    # Find the longest contiguous run of valid frames so the plotted window
    # actually shows GT pitch continuity, not a window dominated by gaps.
    padded_mask = np.concatenate([[False], mask_np_full, [False]])
    edges = np.diff(padded_mask.astype(np.int8))
    run_starts = np.flatnonzero(edges == 1)
    run_ends = np.flatnonzero(edges == -1)
    run_lengths = run_ends - run_starts
    best_run = int(np.argmax(run_lengths))
    run_start, run_len = int(run_starts[best_run]), int(run_lengths[best_run])
    center = run_start + run_len // 2
    half_win = 75
    t0 = max(0, center - half_win)
    t1 = min(batch["spec"].shape[-1], center + half_win)

    true_log2 = batch["pitch_log2_hz"][b, t0:t1].numpy()
    true_cents_abs = 1200.0 * true_log2
    cand_cents_abs = 1200.0 * np.log2(cand_hz)
    win_probs = probs[b, :, t0:t1]
    win_mask = mask[b, t0:t1].numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    extent = [t0, t1, cand_cents_abs[0], cand_cents_abs[-1]]
    ax.imshow(win_probs, aspect="auto", origin="lower", extent=extent, cmap="magma")
    t_axis = np.arange(t0, t1)
    plot_true = np.where(win_mask, true_cents_abs, np.nan)
    ax.plot(t_axis, plot_true, color="cyan", linewidth=1.5, label="GT pitch (abs cents)")
    argmax_idx = win_probs.argmax(axis=0)
    pred_cents = cand_cents_abs[argmax_idx]
    plot_pred = np.where(win_mask, pred_cents, np.nan)
    ax.plot(t_axis, plot_pred, color="lime", linewidth=1.0, linestyle="--", label="argmax pred")
    ax.set_xlabel("frame")
    ax.set_ylabel("absolute cents (1200*log2 Hz)")
    ax.set_title(f"{run_name}: tiny-overfit predicted salience P(f|t) vs GT pitch")
    ax.legend(loc="upper right")
    out_path = OUT_DIR / "figures" / "salience_overlays" / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_hps_vs_learned_window(
    rec_id: str,
    fold: int,
    t0: int,
    t1: int,
    out_name: str,
    *,
    title_suffix: str = "",
) -> Path:
    """§26: CQT + GT + deterministic-HPS-salience + learned-harmonic-salience + both predicted pitches, one window."""
    import numpy as np

    from training.normalization import load_fold_cqt_stats, log2_hz_to_cents, normalize_cqt
    from training.pitch_diagnostics.common import PRIMARY_LANE, linear_mag
    from training.pitch_diagnostics.hps_salience import HPS_HARMONICS, hps_salience_probs
    from training.pitch_diagnostics.train_salience import VARIANT_CONFIG

    repo_root = REPO_ROOT
    index = RecordingLaneIndex.build(repo_root)
    cr = load_or_compute_candidate_range(repo_root, index)
    lo, hi = cr["candidate_lo_bin"], cr["candidate_hi_bin"]
    cand_hz_arr = candidate_hz(lo, hi)
    cand_cents_abs = 1200.0 * np.log2(cand_hz_arr)

    lane = next(x for x in index.lanes if x.recording_id == rec_id)
    frames = index._frames[(rec_id, PRIMARY_LANE)]
    cqt_log = index._features[rec_id]["cqt_log"]
    n = min(cqt_log.shape[1], lane.n_frames)
    cqt_log = cqt_log[:, :n]
    mag = linear_mag(cqt_log)
    t1 = min(t1, n)

    hps_probs = hps_salience_probs(mag[:, t0:t1], lo, hi, k_list=HPS_HARMONICS)

    mu, sigma = load_fold_cqt_stats(fold, repo_root)
    spec = normalize_cqt(cqt_log, mu, sigma).astype(np.float32)
    spec_t = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)
    cfg = VARIANT_CONFIG["harmonic"]
    model = HarmonicSalienceModel(candidate_lo_bin=lo, candidate_hi_bin=hi, harmonic_ks=cfg["harmonic_ks"], hidden=cfg["hidden"])
    ckpt = torch.load(OUT_DIR / "runs" / "harmonic_salience_abs" / f"fold_{fold}" / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        logits = model(spec_t)
        learned_probs_full = torch.softmax(logits, dim=1)[0].numpy()
    learned_probs = learned_probs_full[:, t0:t1]

    true_log2 = frames["pitch_log2_hz"][t0:t1].astype(np.float64)
    true_cents_abs = 1200.0 * true_log2
    valid = frames["valid_target"][t0:t1] & (frames["frame_time_s"][t0:t1] < lane.duration_s)

    hps_pred_idx = hps_probs.argmax(axis=0)
    hps_pred_cents = cand_cents_abs[hps_pred_idx]
    learned_pred_idx = learned_probs.argmax(axis=0)
    learned_pred_cents = cand_cents_abs[learned_pred_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    t_axis = np.arange(t0, t1)
    extent = [t0, t1, cand_cents_abs[0], cand_cents_abs[-1]]
    plot_true = np.where(valid, true_cents_abs, np.nan)
    for ax, probs, pred, label in (
        (axes[0], hps_probs, hps_pred_cents, "deterministic HPS salience"),
        (axes[1], learned_probs, learned_pred_cents, "learned harmonic salience"),
    ):
        ax.imshow(probs, aspect="auto", origin="lower", extent=extent, cmap="magma")
        ax.plot(t_axis, plot_true, color="cyan", linewidth=1.5, label="GT pitch")
        ax.plot(t_axis, np.where(valid, pred, np.nan), color="lime", linewidth=1.0, linestyle="--", label="argmax pred")
        ax.set_xlabel("frame")
        ax.set_title(label)
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("absolute cents (1200*log2 Hz)")
    fig.suptitle(f"{rec_id} fold {fold} frames [{t0},{t1}) {title_suffix}")
    out_path = OUT_DIR / "figures" / "salience_overlays" / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def _find_type_window(index: RecordingLaneIndex, rec_id: str, want_type: int, min_len: int = 40) -> tuple[int, int] | None:
    """First contiguous run of `min_len`+ valid frames of trajectory_type==want_type."""
    from training.pitch_diagnostics.common import PRIMARY_LANE

    frames = index._frames[(rec_id, PRIMARY_LANE)]
    lane = next(x for x in index.lanes if x.recording_id == rec_id)
    n = min(index._features[rec_id]["cqt_log"].shape[1], lane.n_frames)
    valid = frames["valid_target"][:n] & (frames["frame_time_s"][:n] < lane.duration_s)
    is_type = valid & (frames["trajectory_type"][:n] == want_type)
    padded = np.concatenate([[False], is_type, [False]])
    edges = np.diff(padded.astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if len(starts) == 0:
        return None
    lengths = ends - starts
    best = int(np.argmax(lengths))
    if lengths[best] < min_len:
        best = int(np.argmax(lengths))  # take the longest available even if short
    t0, t1 = int(starts[best]), int(ends[best])
    if t1 - t0 > 150:
        c = (t0 + t1) // 2
        t0, t1 = c - 75, c + 75
    return t0, t1


def _find_weak_fundamental_window(index: RecordingLaneIndex, rec_id: str, lo_bin: int, hi_bin: int, min_len: int = 40) -> tuple[int, int] | None:
    """Window where target/frame-max CQT energy ratio is small (fundamental spectrally weak)."""
    from training.pitch_diagnostics.common import PRIMARY_LANE, bin_from_hz, clip_bin, linear_mag

    frames = index._frames[(rec_id, PRIMARY_LANE)]
    lane = next(x for x in index.lanes if x.recording_id == rec_id)
    cqt_log = index._features[rec_id]["cqt_log"]
    n = min(cqt_log.shape[1], lane.n_frames)
    mag = linear_mag(cqt_log[:, :n])
    valid = frames["valid_target"][:n] & (frames["frame_time_s"][:n] < lane.duration_s)
    target_bin = clip_bin(bin_from_hz(2.0 ** frames["pitch_log2_hz"][:n]))
    frame_max = mag.max(axis=0)
    target_energy = mag[target_bin, np.arange(n)]
    ratio = np.where(frame_max > 0, target_energy / np.maximum(frame_max, 1e-12), 1.0)
    weak = valid & (ratio < 0.15)
    padded = np.concatenate([[False], weak, [False]])
    edges = np.diff(padded.astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if len(starts) == 0:
        return None
    lengths = ends - starts
    best = int(np.argmax(lengths))
    t0, t1 = int(starts[best]), int(ends[best])
    if t1 - t0 < min_len:
        c = (t0 + t1) // 2
        t0, t1 = max(0, c - min_len // 2), c + min_len // 2
    if t1 - t0 > 150:
        c = (t0 + t1) // 2
        t0, t1 = c - 75, c + 75
    return t0, t1


def plot_representative_gallery() -> list[Path]:
    """§26: HPS-vs-learned windows across success/failure quadrants, T0-T3, fold 3/4, weak fundamental."""
    import json

    repo_root = REPO_ROOT
    index = RecordingLaneIndex.build(repo_root)
    cr = load_or_compute_candidate_range(repo_root, index)
    lo, hi = cr["candidate_lo_bin"], cr["candidate_hi_bin"]

    per_rec = json.loads((OUT_DIR / "harmonic_salience_per_recording.json").read_text())
    entries = sorted(per_rec.items(), key=lambda kv: -kv[1]["delta_mae"])

    out_paths: list[Path] = []

    def _win_for(rec_id: str, fold: int, min_len: int = 150) -> tuple[int, int]:
        lane = next(x for x in index.lanes if x.recording_id == rec_id)
        n = min(index._features[rec_id]["cqt_log"].shape[1], lane.n_frames)
        c = n // 2
        return max(0, c - min_len // 2), min(n, c + min_len // 2)

    # HPS fails / model succeeds: largest positive delta (hps_mae - harmonic_mae)
    rid, e = entries[0]
    t0, t1 = _win_for(rid, e["fold"])
    out_paths.append(plot_hps_vs_learned_window(rid, e["fold"], t0, t1, "gallery_hps_fails_model_succeeds.png",
                                                  title_suffix=f"HPS_fails/model_succeeds delta={e['delta_mae']:.0f}c"))

    # HPS succeeds / model fails: most negative delta
    rid, e = entries[-1]
    t0, t1 = _win_for(rid, e["fold"])
    out_paths.append(plot_hps_vs_learned_window(rid, e["fold"], t0, t1, "gallery_hps_succeeds_model_fails.png",
                                                  title_suffix=f"HPS_succeeds/model_fails delta={e['delta_mae']:.0f}c"))

    # both succeed: low hps_mae and low harmonic_mae
    both_ok = sorted(entries, key=lambda kv: kv[1]["hps_mae"] + kv[1]["harmonic_mae"])
    rid, e = both_ok[0]
    t0, t1 = _win_for(rid, e["fold"])
    out_paths.append(plot_hps_vs_learned_window(rid, e["fold"], t0, t1, "gallery_both_succeed.png",
                                                  title_suffix=f"both_succeed hps={e['hps_mae']:.0f}c model={e['harmonic_mae']:.0f}c"))

    # both fail: high hps_mae and high harmonic_mae
    rid, e = both_ok[-1]
    t0, t1 = _win_for(rid, e["fold"])
    out_paths.append(plot_hps_vs_learned_window(rid, e["fold"], t0, t1, "gallery_both_fail.png",
                                                  title_suffix=f"both_fail hps={e['hps_mae']:.0f}c model={e['harmonic_mae']:.0f}c"))

    # fold 3 and fold 4 representative (moderate delta, not extreme)
    for target_fold in (3, 4):
        fold_entries = [kv for kv in entries if kv[1]["fold"] == target_fold]
        if not fold_entries:
            continue
        fold_entries.sort(key=lambda kv: abs(kv[1]["delta_mae"] - np.median([v["delta_mae"] for _, v in fold_entries])))
        rid, e = fold_entries[0]
        t0, t1 = _win_for(rid, e["fold"])
        out_paths.append(plot_hps_vs_learned_window(rid, e["fold"], t0, t1, f"gallery_fold{target_fold}.png",
                                                      title_suffix=f"fold{target_fold} representative delta={e['delta_mae']:.0f}c"))

    # T0-T3: scan the "both succeed" recording (has richest musical content) for each type's window
    src_rid, src_e = both_ok[0]
    for ttype in (0, 1, 2, 3):
        win = _find_type_window(index, src_rid, ttype)
        if win is None:
            continue
        out_paths.append(plot_hps_vs_learned_window(src_rid, src_e["fold"], win[0], win[1], f"gallery_type_T{ttype}.png",
                                                      title_suffix=f"T{ttype} example"))

    # weak-fundamental example: scan the HPS-fails/model-succeeds recording, since that's
    # exactly the failure mode weak fundamentals produce
    weak_rid, weak_e = entries[0]
    weak_win = _find_weak_fundamental_window(index, weak_rid, lo, hi)
    if weak_win is not None:
        out_paths.append(plot_hps_vs_learned_window(weak_rid, weak_e["fold"], weak_win[0], weak_win[1],
                                                      "gallery_weak_fundamental.png", title_suffix="weak-fundamental example"))

    return out_paths


def main() -> None:
    p1 = plot_tiny_overfit_examples("tiny_overfit_local_debug_highlr", (1,), 24, "tiny_overfit_local_salience.png")
    p2 = plot_tiny_overfit_examples("tiny_overfit_harmonic_debug_highlr", (1, 2, 3, 4), 16, "tiny_overfit_harmonic_salience.png")
    print("wrote", p1)
    print("wrote", p2)
    for p in plot_representative_gallery():
        print("wrote", p)


if __name__ == "__main__":
    main()
