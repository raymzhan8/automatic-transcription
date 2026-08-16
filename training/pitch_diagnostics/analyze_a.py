"""Diagnostic A: target visibility, overlays, harmonics, octave, distributions, drone."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.features import N_BINS  # noqa: E402
from training.folds import build_fold_split, load_kfold_manifest  # noqa: E402
from training.framewise_dataset import RecordingLaneIndex  # noqa: E402
from training.metrics import pitch_error_metrics  # noqa: E402
from training.normalization import log2_hz_to_cents  # noqa: E402
from training.pitch_diagnostics.common import (  # noqa: E402
    B1_RUN,
    C_RUN,
    DRONE_CENTS,
    OUT_DIR,
    PRIMARY_LANE,
    bin_from_hz,
    cents_from_hz,
    clip_bin,
    hz_from_cents,
    linear_mag,
    octave_adjusted_error,
    summarize_array,
    write_json,
)

NEIGHBOR = 2


def _valid_mask(index: RecordingLaneIndex, rec_id: str, duration_s: float) -> np.ndarray:
    frames = index._frames[(rec_id, PRIMARY_LANE)]
    times = frames["frame_time_s"]
    return frames["valid_target"] & (times < duration_s)


def _frame_energy_stats(
    mag: np.ndarray,
    target_hz: np.ndarray,
    tonic_hz: float,
) -> dict[str, np.ndarray]:
    target_bin = clip_bin(bin_from_hz(target_hz))
    tonic_bin = int(clip_bin(bin_from_hz(tonic_hz)))
    f_bins, n_t = mag.shape
    max_e = mag.max(axis=0)
    tonic_e = mag[tonic_bin]
    target_e = np.empty(n_t, dtype=np.float64)
    neigh_e = np.empty(n_t, dtype=np.float64)
    rank = np.empty(n_t, dtype=np.int64)
    harm = {k: np.full(n_t, np.nan) for k in (1, 2, 3, 4)}
    for t in range(n_t):
        tb = int(target_bin[t])
        col = mag[:, t]
        target_e[t] = col[tb]
        lo = max(0, tb - NEIGHBOR)
        hi = min(f_bins, tb + NEIGHBOR + 1)
        neigh_e[t] = col[lo:hi].max()
        rank[t] = int((col > col[tb]).sum()) + 1
        f0 = float(target_hz[t])
        for k in (1, 2, 3, 4):
            b = bin_from_hz(k * f0)
            if 0 <= float(b) < f_bins:
                harm[k][t] = col[int(np.clip(np.round(b), 0, f_bins - 1))]
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "target_over_max": target_e / np.maximum(max_e, 1e-12),
            "neigh_over_max": neigh_e / np.maximum(max_e, 1e-12),
            "target_over_tonic": target_e / np.maximum(tonic_e, 1e-12),
            "rank": rank.astype(np.float64),
            "harm_1": harm[1],
            "harm_2": harm[2],
            "harm_3": harm[3],
            "harm_4": harm[4],
            "max_e": max_e,
            "target_e": target_e,
            "tonic_e": tonic_e,
        }


def _load_contour(run_dir: Path, fold: int, rec_id: str) -> dict[str, np.ndarray] | None:
    path = run_dir / f"fold_{fold}" / "eval" / "pitch_contours" / f"{rec_id}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    if "pred_cents" not in data.files:
        return None
    return {k: data[k] for k in data.files}


def _drone_fractions(pred: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for center in DRONE_CENTS:
        name = f"tonic{int(center):+d}" if center else "tonic"
        d = np.abs(pred - center)
        out[name] = {
            "pct_within_25": float((d <= 25).mean()) if len(pred) else 0.0,
            "pct_within_50": float((d <= 50).mean()) if len(pred) else 0.0,
        }
    return out


def _overlay(
    cqt_log: np.ndarray,
    times: np.ndarray,
    target_hz: np.ndarray,
    valid: np.ndarray,
    traj: np.ndarray,
    out_path: Path,
    *,
    title: str,
    t0: float,
    window_s: float = 8.0,
    pred_hz: np.ndarray | None = None,
) -> None:
    t1 = t0 + window_s
    mask = (times >= t0) & (times <= t1)
    if not np.any(mask):
        return
    fig, ax = plt.subplots(figsize=(14, 4.5))
    spec = cqt_log[:, mask]
    ax.imshow(
        spec,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=(t0, min(t1, float(times[mask][-1])), 0, N_BINS - 1),
    )
    tb = clip_bin(bin_from_hz(target_hz[mask]))
    ax.plot(times[mask], tb, color="white", linewidth=1.4, label="GT pitch")
    if pred_hz is not None:
        pb = clip_bin(bin_from_hz(pred_hz[mask]))
        ax.plot(times[mask], pb, color="cyan", linewidth=1.0, alpha=0.9, label="pred")
    ax.fill_between(
        times[mask], 0, 1, where=valid[mask], transform=ax.get_xaxis_transform(),
        alpha=0.08, color="lime",
    )
    ax.set_ylabel("CQT bin")
    ax.set_xlabel("time (s)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUT_DIR / "figures" / "overlays"
    hist_dir = OUT_DIR / "figures" / "histograms"
    index = RecordingLaneIndex.build(REPO_ROOT)
    manifest = load_kfold_manifest(REPO_ROOT)

    rec_stats: dict[str, Any] = {}
    all_vis = defaultdict(list)
    type_vis: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    split_cents: dict[int, dict[str, list]] = {i: {"train": [], "val": [], "test": []} for i in range(5)}
    tonics: dict[str, float] = {}
    rec_to_fold_test: dict[str, int] = {}
    for fold in range(5):
        split = build_fold_split(manifest, fold, seed=42)
        for rid in split.test_recording_ids:
            rec_to_fold_test[rid] = fold
        for name, ids in (
            ("train", split.train_recording_ids),
            ("val", split.val_recording_ids),
            ("test", split.test_recording_ids),
        ):
            for rid in ids:
                lane = next(x for x in index.lanes if x.recording_id == rid)
                frames = index._frames[(rid, PRIMARY_LANE)]
                mask = _valid_mask(index, rid, lane.duration_s)
                cents = log2_hz_to_cents(frames["pitch_log2_hz"], lane.fundamental_hz)
                split_cents[fold][name].append(np.asarray(cents)[mask])

    b1_mae_by_rec: dict[str, float] = {}
    for fold in range(5):
        summary_path = B1_RUN / f"fold_{fold}" / "eval" / "eval_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        for rid, block in (summary.get("per_recording") or {}).items():
            if block.get("pitch_mae_cents") is not None:
                b1_mae_by_rec[rid] = float(block["pitch_mae_cents"])

    octave_rows = []
    drone_rows = []
    overlay_jobs: list[dict[str, Any]] = []

    for lane in index.lanes:
        rid = lane.recording_id
        frames = index._frames[(rid, PRIMARY_LANE)]
        cqt = index._features[rid]["cqt_log"]
        times = frames["frame_time_s"]
        n = min(cqt.shape[1], len(times), lane.n_frames)
        cqt = cqt[:, :n]
        times = times[:n]
        valid = frames["valid_target"][:n] & (times < lane.duration_s)
        traj = frames["trajectory_type"][:n]
        pitch_log2 = frames["pitch_log2_hz"][:n]
        cents = np.asarray(log2_hz_to_cents(pitch_log2, lane.fundamental_hz), dtype=np.float64)
        target_hz = np.asarray(hz_from_cents(cents, lane.fundamental_hz), dtype=np.float64)
        mag = linear_mag(cqt)
        vis = _frame_energy_stats(mag[:, valid], target_hz[valid], lane.fundamental_hz)
        tonics[rid] = lane.fundamental_hz

        rec_block = {
            "fundamental_hz": lane.fundamental_hz,
            "n_valid": int(valid.sum()),
            "cents": summarize_array(cents[valid]),
            "visibility": {k: summarize_array(vis[k]) for k in (
                "target_over_max", "neigh_over_max", "target_over_tonic", "rank"
            )},
            "harmonics": {
                f"{k}f_over_f": summarize_array(vis[f"harm_{k}"] / np.maximum(vis["harm_1"], 1e-12))
                for k in (2, 3, 4)
            },
            "b1_pitch_mae_cents": b1_mae_by_rec.get(rid),
        }
        rec_stats[rid] = rec_block
        for k in ("target_over_max", "neigh_over_max", "target_over_tonic", "rank"):
            all_vis[k].append(vis[k])
        types_v = traj[valid]
        for t in range(4):
            m = types_v == t
            if np.any(m):
                for k in ("target_over_max", "rank"):
                    type_vis[t][k].append(vis[k][m])

        fold = rec_to_fold_test.get(rid)
        if fold is not None:
            for run_name, run_dir in (("b1", B1_RUN), ("c", C_RUN)):
                contour = _load_contour(run_dir, fold, rid)
                if contour is None:
                    continue
                em = (~np.asarray(contour.get("padding_mask", np.zeros(n, dtype=bool)))) & np.asarray(
                    contour["valid_target"]
                )
                pred = np.asarray(contour["pred_cents"])
                true = np.asarray(contour["true_cents"])
                if em.shape != pred.shape:
                    mlen = min(len(em), len(pred), len(true))
                    em, pred, true = em[:mlen], pred[:mlen], true[:mlen]
                pred_v, true_v = pred[em], true[em]
                raw = np.abs(pred_v - true_v)
                adj, ks = octave_adjusted_error(pred_v, true_v)
                octave_rows.append(
                    {
                        "recording_id": rid,
                        "fold": fold,
                        "model": run_name,
                        "raw": pitch_error_metrics(pred_v, true_v),
                        "octave_adjusted_mae": float(adj.mean()) if len(adj) else None,
                        "octave_k_hist": {str(k): int((ks == k).sum()) for k in (-2, -1, 0, 1, 2)},
                    }
                )
                drone_rows.append(
                    {
                        "recording_id": rid,
                        "fold": fold,
                        "model": run_name,
                        "pred": _drone_fractions(pred_v),
                        "gt": _drone_fractions(true_v),
                    }
                )
                hist_dir.mkdir(parents=True, exist_ok=True)
                fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
                axes[0].hist(true_v, bins=40, color="black", alpha=0.7)
                axes[0].set_title(f"{rid[:8]} GT cents")
                axes[1].hist(pred_v, bins=40, color="tab:blue", alpha=0.7)
                axes[1].set_title(f"{run_name} pred cents")
                for ax in axes:
                    ax.set_xlabel("cents vs tonic")
                fig.tight_layout()
                fig.savefig(hist_dir / f"fold{fold}_{run_name}_{rid}_cents_hist.png", dpi=100)
                plt.close(fig)

        overlay_jobs.append(
            {
                "rid": rid,
                "cqt": cqt,
                "times": times,
                "target_hz": target_hz,
                "valid": valid,
                "traj": traj,
                "cents": cents,
                "vis_max": vis["target_over_max"],
                "tonic_ratio": vis["target_over_tonic"],
            }
        )

    # Overlays: types, high/low pitch, drone-heavy, clean, best/worst B1
    def _window_for_mask(times: np.ndarray, mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        return float(times[np.where(mask)[0][0]])

    ranked = sorted(b1_mae_by_rec.items(), key=lambda kv: kv[1])
    best_ids = {r for r, _ in ranked[:2]} if ranked else set()
    worst_ids = {r for r, _ in ranked[-2:]} if ranked else set()

    type_done = set()
    special_done = {"high": False, "low": False, "drone": False, "clean": False}
    for job in overlay_jobs:
        rid = job["rid"]
        times, traj, valid = job["times"], job["traj"], job["valid"]
        cents = job["cents"]
        for t in range(4):
            if t in type_done:
                continue
            m = valid & (traj == t)
            if m.sum() < 50:
                continue
            t0 = _window_for_mask(times, m)
            _overlay(
                job["cqt"], times, job["target_hz"], valid, traj,
                fig_dir / f"type_T{t}_{rid}.png",
                title=f"T{t}  {rid}", t0=t0,
            )
            type_done.add(t)
        if not special_done["high"] and np.any(valid):
            thr = np.quantile(cents[valid], 0.9)
            m = valid & (cents >= thr)
            if m.sum() > 30:
                _overlay(job["cqt"], times, job["target_hz"], valid, traj,
                         fig_dir / f"high_pitch_{rid}.png", title=f"high pitch {rid}",
                         t0=_window_for_mask(times, m))
                special_done["high"] = True
        if not special_done["low"] and np.any(valid):
            thr = np.quantile(cents[valid], 0.1)
            m = valid & (cents <= thr)
            if m.sum() > 30:
                _overlay(job["cqt"], times, job["target_hz"], valid, traj,
                         fig_dir / f"low_pitch_{rid}.png", title=f"low pitch {rid}",
                         t0=_window_for_mask(times, m))
                special_done["low"] = True
        if rid in best_ids:
            _overlay(job["cqt"], times, job["target_hz"], valid, traj,
                     fig_dir / f"b1_good_{rid}.png", title=f"B1 good {rid}", t0=0.0)
        if rid in worst_ids:
            _overlay(job["cqt"], times, job["target_hz"], valid, traj,
                     fig_dir / f"b1_bad_{rid}.png", title=f"B1 bad {rid}", t0=0.0)

    vis_ranked = sorted(
        rec_stats.items(),
        key=lambda kv: (kv[1]["visibility"]["target_over_max"] or {}).get("mean") or 0.0,
    )
    if vis_ranked:
        weak_id = vis_ranked[0][0]
        clean_id = vis_ranked[-1][0]
        by_id = {j["rid"]: j for j in overlay_jobs}
        for tag, rid in (("drone_heavy", weak_id), ("clean", clean_id)):
            job = by_id[rid]
            _overlay(
                job["cqt"], job["times"], job["target_hz"], job["valid"], job["traj"],
                fig_dir / f"{tag}_{rid}.png", title=f"{tag} {rid}", t0=0.0,
            )

    dist_folds = []
    for fold in range(5):
        block = {"fold": fold, "splits": {}, "tonics_hz": {}}
        split = build_fold_split(manifest, fold, seed=42)
        for name in ("train", "val", "test"):
            arr = np.concatenate(split_cents[fold][name]) if split_cents[fold][name] else np.array([])
            block["splits"][name] = summarize_array(arr)
        ids = {
            "train": split.train_recording_ids,
            "val": split.val_recording_ids,
            "test": split.test_recording_ids,
        }
        for name, rids in ids.items():
            block["tonics_hz"][name] = {rid: tonics[rid] for rid in rids if rid in tonics}
        try:
            from scipy.stats import wasserstein_distance

            tr = np.concatenate(split_cents[fold]["train"])
            te = np.concatenate(split_cents[fold]["test"])
            block["wasserstein_train_test"] = float(wasserstein_distance(tr, te))
        except Exception:
            block["wasserstein_train_test"] = None
        dist_folds.append(block)

    overall_vis = {k: summarize_array(np.concatenate(v)) for k, v in all_vis.items() if v}
    by_type = {
        f"T{t}": {k: summarize_array(np.concatenate(vs)) for k, vs in type_vis[t].items() if vs}
        for t in range(4)
        if type_vis[t]
    }
    good_ids = set(best_ids)
    bad_ids = set(worst_ids)
    good_vis = []
    bad_vis = []
    for rid, block in rec_stats.items():
        mean_r = (block["visibility"]["target_over_max"] or {}).get("mean")
        if mean_r is None:
            continue
        if rid in good_ids:
            good_vis.append(mean_r)
        if rid in bad_ids:
            bad_vis.append(mean_r)

    write_json(OUT_DIR / "visibility.json", {
        "overall": overall_vis,
        "by_type": by_type,
        "by_recording": rec_stats,
        "b1_good_recordings_target_over_max_mean": good_vis,
        "b1_bad_recordings_target_over_max_mean": bad_vis,
        "neighbor_bins": NEIGHBOR,
    })
    write_json(OUT_DIR / "octave_errors.json", {"records": octave_rows})
    write_json(OUT_DIR / "drone_lock.json", {"records": drone_rows})
    write_json(OUT_DIR / "pitch_distributions.json", {
        "folds": dist_folds,
        "tonics_hz": tonics,
    })
    print("Diagnostic A written under", OUT_DIR)


if __name__ == "__main__":
    main()
