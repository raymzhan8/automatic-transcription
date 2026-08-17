"""Step 15 sections 12-19: evaluate P0-P3 on full held-out test recordings
(continuous framewise inference), primary comparison table, per-class
metrics, trajectory-aggregated metrics, duration/|dp/dt|/salience-
uncertainty stratification, per-recording comparison, oracle-gap numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.folds import build_fold_split, load_kfold_manifest  # noqa: E402
from training.framewise_dataset import FullRecordingDataset, RecordingLaneIndex, collate_variable_length  # noqa: E402
from training.framewise_models import (  # noqa: E402
    FramewiseConditionalTCNModel, PitchMotionEncoder, PitchOnlyClassifier, SalienceMotionEncoder,
)
from training.metrics import (  # noqa: E402
    DURATION_S_BUCKETS, TYPE_NAMES, aggregate_trajectory_predictions, duration_s_per_frame,
    frame_metrics, metrics_by_buckets, trajectory_metrics,
)
from training.normalization import load_fold_cqt_stats  # noqa: E402
from training.pitch_diagnostics.common import bin_from_hz  # noqa: E402
from training.pitch_diagnostics.relative_pitch.dense_relative_salience import HALF_W, W_BINS
from training.pitch_diagnostics.relative_pitch.dense_relative_salience import build as build_relative_salience  # noqa: E402
from training.relative_pitch_features import PITCH_FEATURE_DIM, load_dense_estimated_pitch  # noqa: E402
from training.train_pitch_motion_ablation import CONDITIONS, forward, get_pitch_input  # noqa: E402

RUN_DIR = REPO_ROOT / "output" / "pitch_motion_ablation"
OUT_PATH = RUN_DIR / "evaluation_result.json"
DPDT_BUCKET_EDGES = (0.0, 100.0, 400.0, 1000.0, np.inf)
DPDT_BUCKET_NAMES = ("0-100c/s", "100-400c/s", "400-1000c/s", ">1000c/s")


def get_device() -> torch.device:
    return torch.device("cpu")  # long full-recording sequences; MPS conv2d fails past 65536 "channels"


def build_model(condition: str) -> torch.nn.Module:
    if condition == "P0":
        return FramewiseConditionalTCNModel(use_audio=False, use_pitch=True, pitch_dim=PITCH_FEATURE_DIM)
    if condition in ("P1", "P3"):
        return PitchOnlyClassifier(PitchMotionEncoder(in_channels=1, hidden=32))
    if condition == "P2":
        return PitchOnlyClassifier(SalienceMotionEncoder(hidden=32))
    raise ValueError(condition)


def load_model(condition: str, fold: int, device: torch.device):
    ckpt = torch.load(RUN_DIR / f"condition_{condition}" / f"fold_{fold}" / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(condition).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt.get("phi_mu"), ckpt.get("phi_sigma")


@torch.no_grad()
def predict_recording(model, batch, condition, phi_mu, phi_sigma, device) -> np.ndarray:
    pitch_input = get_pitch_input(batch, condition, phi_mu, phi_sigma, device)
    logits = forward(model, condition, pitch_input)
    return logits.cpu().numpy()[0]  # [T, 4]


def salience_uncertainty(rel_sal_col: np.ndarray) -> tuple[float, float]:
    p = np.maximum(rel_sal_col, 1e-12)
    p_norm = p / p.sum()
    entropy = float(-(p_norm * np.log(p_norm)).sum())
    sorted_vals = np.sort(rel_sal_col)[::-1]
    margin = float(sorted_vals[0] - sorted_vals[1])
    return entropy, margin


def gt_rank_in_window(rel_sal_col: np.ndarray, gt_offset_bin: int) -> int:
    if gt_offset_bin < 0 or gt_offset_bin >= W_BINS:
        return W_BINS + 1  # GT outside the window entirely
    order = np.argsort(-rel_sal_col)
    return int(np.where(order == gt_offset_bin)[0][0]) + 1


def main() -> None:
    device = get_device()
    index = RecordingLaneIndex.build(REPO_ROOT)
    manifest = load_kfold_manifest(REPO_ROOT)
    estimated_pitch = load_dense_estimated_pitch()
    relative_salience = build_relative_salience()

    pooled = {c: {"pred": [], "true": []} for c in CONDITIONS}
    by_fold_pooled = {c: {f: {"pred": [], "true": []} for f in range(5)} for c in CONDITIONS}
    dpdt_all, dur_all = {c: [] for c in CONDITIONS}, {c: [] for c in CONDITIONS}
    per_recording: dict[str, dict[str, dict]] = {c: {} for c in CONDITIONS}
    traj_results: dict[str, list[dict]] = {c: [] for c in CONDITIONS}
    sal_uncertainty_rows: list[dict] = []

    rng = np.random.default_rng(0)

    for fold in range(5):
        split = build_fold_split(manifest, fold, seed=42)
        mu_cqt, sigma_cqt = load_fold_cqt_stats(fold, REPO_ROOT)
        test_ds = FullRecordingDataset(
            index, split.test_recording_ids, mu_cqt, sigma_cqt,
            estimated_pitch=estimated_pitch, compute_pitch_features=True,
            relative_salience=relative_salience,
        )
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_variable_length)
        models = {c: load_model(c, fold, device) for c in CONDITIONS}

        for batch in test_loader:
            rid = batch["recording_id"][0]
            valid = batch["valid_target"][0].numpy()
            pad = batch["padding_mask"][0].numpy()
            mask = valid & (~pad)
            lane = next(x for x in index.lanes if x.recording_id == rid)
            frames = index._frames[(rid, lane.lane_id)]
            n = len(mask)
            dpdt_full = np.abs(frames["dp_dt_log2_hz_per_s"][:n] * 1200.0)
            prims = index.primitives_for_recording(rid)
            dur_full = duration_s_per_frame(batch["frame_time_s"][0][:n], prims)
            true_full = batch["trajectory_type"][0].numpy()

            logits_by_c = {}
            for c, (model, phi_mu, phi_sigma) in models.items():
                logits = predict_recording(model, batch, c, phi_mu, phi_sigma, device)
                logits_by_c[c] = logits
                pred = logits.argmax(axis=-1)[mask]
                true = true_full[mask]
                pooled[c]["pred"].append(pred); pooled[c]["true"].append(true)
                by_fold_pooled[c][fold]["pred"].append(pred); by_fold_pooled[c][fold]["true"].append(true)
                dpdt_all[c].append(dpdt_full[mask]); dur_all[c].append(dur_full[mask])
                per_recording[c][rid] = {"fold": fold, **frame_metrics(pred, true)}
                results, _hist = aggregate_trajectory_predictions(batch["frame_time_s"][0], logits, pad, prims)
                traj_results[c].extend(results)

            # salience-uncertainty stratified rows (P1 vs P2), evaluation metadata only, subsampled
            sal = batch["relative_salience"][0, 0].numpy()  # [W_BINS, T]
            gt_bin = bin_from_hz(np.exp2(batch["pitch_log2_hz"][0].numpy()[:n]))
            ref_bin = bin_from_hz(np.exp2(estimated_pitch[rid][:n]))
            gt_offset = np.rint(gt_bin - ref_bin).astype(np.int64) + HALF_W
            idxs = np.flatnonzero(mask)
            if len(idxs) > 3000:
                idxs = rng.choice(idxs, size=3000, replace=False)
            p1_pred = logits_by_c["P1"].argmax(axis=-1)
            p2_pred = logits_by_c["P2"].argmax(axis=-1)
            for t in idxs:
                ent, margin = salience_uncertainty(sal[:, t])
                rank = gt_rank_in_window(sal[:, t], int(gt_offset[t]))
                sal_uncertainty_rows.append({
                    "entropy": ent, "margin": margin, "gt_rank": rank,
                    "p1_correct": bool(p1_pred[t] == true_full[t]),
                    "p2_correct": bool(p2_pred[t] == true_full[t]),
                })
        print(f"fold {fold} done ({len(split.test_recording_ids)} recordings)")

    pooled_summary, by_dpdt, by_duration, per_fold_macro_f1 = {}, {}, {}, {c: {} for c in CONDITIONS}
    for c in CONDITIONS:
        pred = np.concatenate(pooled[c]["pred"]); true = np.concatenate(pooled[c]["true"])
        pooled_summary[c] = frame_metrics(pred, true)
        dpdt = np.concatenate(dpdt_all[c]); dur = np.concatenate(dur_all[c])
        by_dpdt[c] = metrics_by_buckets(pred, true, dpdt, tuple(zip(DPDT_BUCKET_NAMES, DPDT_BUCKET_EDGES[:-1], DPDT_BUCKET_EDGES[1:])))
        by_duration[c] = metrics_by_buckets(pred, true, dur, DURATION_S_BUCKETS)
        for f in range(5):
            fp = np.concatenate(by_fold_pooled[c][f]["pred"]); ft = np.concatenate(by_fold_pooled[c][f]["true"])
            per_fold_macro_f1[c][f] = frame_metrics(fp, ft)["macro_f1"]

    traj_summary = {c: trajectory_metrics(traj_results[c]) for c in CONDITIONS}
    fold_deltas = {
        "P1_minus_P0": {f: per_fold_macro_f1["P1"][f] - per_fold_macro_f1["P0"][f] for f in range(5)},
        "P2_minus_P1": {f: per_fold_macro_f1["P2"][f] - per_fold_macro_f1["P1"][f] for f in range(5)},
        "P3_minus_P1": {f: per_fold_macro_f1["P3"][f] - per_fold_macro_f1["P1"][f] for f in range(5)},
    }
    grouped_mean = {
        c: {"mean_macro_f1": float(np.mean(list(per_fold_macro_f1[c].values()))),
            "std_macro_f1": float(np.std(list(per_fold_macro_f1[c].values())))}
        for c in CONDITIONS
    }

    out = {
        "pooled_summary": pooled_summary,
        "per_fold_macro_f1": per_fold_macro_f1,
        "fold_deltas": fold_deltas,
        "grouped_mean": grouped_mean,
        "by_dpdt_bucket": by_dpdt,
        "by_duration_bucket": by_duration,
        "per_recording": per_recording,
        "trajectory_aggregated": traj_summary,
        "salience_uncertainty_rows": sal_uncertainty_rows,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))

    print("\n=== Step 15 pooled comparison table ===")
    print(f"{'Condition':6s} {'FrameAcc':>9s} {'MacroF1':>8s} " + " ".join(f"{TYPE_NAMES[i]}F1".rjust(7) for i in range(4)))
    for c in CONDITIONS:
        s = pooled_summary[c]
        row = " ".join(f"{s['per_class'][TYPE_NAMES[i]]['f1']:.3f}".rjust(7) for i in range(4))
        print(f"{c:6s} {s['accuracy']:9.3f} {s['macro_f1']:8.3f} {row}")
    print("\ntrajectory-aggregated macro F1:", {c: round(traj_summary[c]["macro_f1"], 3) for c in CONDITIONS})
    print("fold deltas P1-P0:", fold_deltas["P1_minus_P0"])
    print("fold deltas P2-P1:", fold_deltas["P2_minus_P1"])
    print("fold deltas P3-P1:", fold_deltas["P3_minus_P1"])


if __name__ == "__main__":
    main()
