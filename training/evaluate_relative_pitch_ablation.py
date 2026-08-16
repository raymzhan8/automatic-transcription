"""Step 14 sections 14-17: evaluate the trained A/B/C/D checkpoints on full
held-out test recordings (continuous framewise inference, no oracle
primitive boundaries at inference -- spec section 7), and produce the
primary comparison table, per-class metrics, fold-by-fold C-A/D-A deltas,
per-type/|dp/dt|/duration/per-recording breakdowns, and the oracle-gap
verdict.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.folds import build_fold_split, load_kfold_manifest  # noqa: E402
from training.framewise_dataset import FullRecordingDataset, RecordingLaneIndex, collate_variable_length  # noqa: E402
from training.framewise_models import FramewiseConditionalTCNModel  # noqa: E402
from training.metrics import (  # noqa: E402
    DURATION_S_BUCKETS, TYPE_NAMES, duration_s_per_frame, frame_metrics, metrics_by_buckets,
)
from training.normalization import load_fold_cqt_stats  # noqa: E402
from training.relative_pitch_features import PITCH_FEATURE_DIM, load_dense_estimated_pitch, standardize_phi  # noqa: E402
from training.train_relative_pitch_ablation import CONDITIONS, PITCH_SOURCE, USE_AUDIO, USE_PITCH  # noqa: E402

RUN_DIR = REPO_ROOT / "output" / "relative_pitch_ablation"
OUT_PATH = REPO_ROOT / "output" / "relative_pitch_ablation" / "evaluation_result.json"
DPDT_BUCKET_EDGES = (0.0, 100.0, 400.0, 1000.0, np.inf)
DPDT_BUCKET_NAMES = ("0-100c/s", "100-400c/s", "400-1000c/s", ">1000c/s")


def get_device() -> torch.device:
    # CPU only: MPS conv2d fails on very long full-recording sequences
    # (up to ~220k frames here, vs 400-frame training excerpts) with
    # "Output channels > 65536 not supported at the MPS device."
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(condition: str, fold: int, device: torch.device):
    ckpt = torch.load(
        RUN_DIR / f"condition_{condition}" / f"fold_{fold}" / "best.pt", map_location="cpu", weights_only=False
    )
    model = FramewiseConditionalTCNModel(
        use_audio=USE_AUDIO[condition], use_pitch=USE_PITCH[condition], pitch_dim=PITCH_FEATURE_DIM,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    phi_mu, phi_sigma = ckpt.get("phi_mu"), ckpt.get("phi_sigma")
    return model, phi_mu, phi_sigma, ckpt


@torch.no_grad()
def predict_recording(model, batch, condition: str, phi_mu, phi_sigma, device) -> np.ndarray:
    spec = batch["spec"].to(device) if USE_AUDIO[condition] else None
    pitch_feat = None
    if USE_PITCH[condition]:
        key = "phi_estimated" if PITCH_SOURCE[condition] == "estimated" else "phi_oracle"
        raw = batch[key].numpy()
        std = standardize_phi(raw.reshape(-1, PITCH_FEATURE_DIM), phi_mu, phi_sigma).reshape(raw.shape)
        pitch_feat = torch.from_numpy(std).to(device)
    logits = model(spec, pitch_feat)
    return logits.cpu().numpy()


def main() -> None:
    device = get_device()
    index = RecordingLaneIndex.build(REPO_ROOT)
    manifest = load_kfold_manifest(REPO_ROOT)
    estimated_pitch = load_dense_estimated_pitch()

    # per-condition pooled frame records
    pooled: dict[str, dict[str, list]] = {c: {"pred": [], "true": [], "dpdt": [], "dur": [], "rid": [], "fold": []} for c in CONDITIONS}
    per_recording: dict[str, dict[str, dict]] = {c: {} for c in CONDITIONS}
    per_fold_macro_f1: dict[str, dict[int, float]] = {c: {} for c in CONDITIONS}

    for fold in range(5):
        split = build_fold_split(manifest, fold, seed=42)
        mu_cqt, sigma_cqt = load_fold_cqt_stats(fold, REPO_ROOT)
        test_ds = FullRecordingDataset(
            index, split.test_recording_ids, mu_cqt, sigma_cqt,
            estimated_pitch=estimated_pitch, compute_pitch_features=True,
        )
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_variable_length)

        for condition in CONDITIONS:
            model, phi_mu, phi_sigma, _ckpt = load_model(condition, fold, device)
            fold_pred, fold_true = [], []
            for batch in test_loader:
                logits = predict_recording(model, batch, condition, phi_mu, phi_sigma, device)
                rid = batch["recording_id"][0]
                valid = batch["valid_target"][0].numpy()
                pad = batch["padding_mask"][0].numpy()
                mask = valid & (~pad)
                pred = logits[0].argmax(axis=-1)[mask]
                true = batch["trajectory_type"][0].numpy()[mask]
                lane = next(x for x in index.lanes if x.recording_id == rid)
                frames = index._frames[(rid, lane.lane_id)]
                dpdt = np.abs(frames["dp_dt_log2_hz_per_s"][: len(mask)][mask] * 1200.0)
                prims = index.primitives_for_recording(rid)
                dur = duration_s_per_frame(batch["frame_time_s"][0][: len(mask)], prims)[mask]

                pooled[condition]["pred"].append(pred)
                pooled[condition]["true"].append(true)
                pooled[condition]["dpdt"].append(dpdt)
                pooled[condition]["dur"].append(dur)
                pooled[condition]["rid"].append(np.full(mask.sum(), rid))
                pooled[condition]["fold"].append(np.full(mask.sum(), fold))
                per_recording[condition][rid] = {"fold": fold, **frame_metrics(pred, true)}
                fold_pred.append(pred)
                fold_true.append(true)
            fp = np.concatenate(fold_pred) if fold_pred else np.array([])
            ft = np.concatenate(fold_true) if fold_true else np.array([])
            per_fold_macro_f1[condition][fold] = frame_metrics(fp, ft)["macro_f1"]
            print(f"condition {condition} fold {fold}: macro_f1={per_fold_macro_f1[condition][fold]:.3f}")

    pooled_summary = {}
    by_type = {}
    by_dpdt = {}
    by_duration = {}
    for condition in CONDITIONS:
        pred = np.concatenate(pooled[condition]["pred"])
        true = np.concatenate(pooled[condition]["true"])
        dpdt = np.concatenate(pooled[condition]["dpdt"])
        dur = np.concatenate(pooled[condition]["dur"])
        pooled_summary[condition] = frame_metrics(pred, true)
        by_type[condition] = pooled_summary[condition]["per_class"]
        by_dpdt[condition] = metrics_by_buckets(
            pred, true, dpdt, tuple(zip(DPDT_BUCKET_NAMES, DPDT_BUCKET_EDGES[:-1], DPDT_BUCKET_EDGES[1:]))
        )
        by_duration[condition] = metrics_by_buckets(pred, true, dur, DURATION_S_BUCKETS)

    fold_deltas = {
        "C_minus_A": {f: per_fold_macro_f1["C"][f] - per_fold_macro_f1["A"][f] for f in range(5)},
        "D_minus_A": {f: per_fold_macro_f1["D"][f] - per_fold_macro_f1["A"][f] for f in range(5)},
        "D_minus_C": {f: per_fold_macro_f1["D"][f] - per_fold_macro_f1["C"][f] for f in range(5)},
    }
    grouped_mean = {
        c: {"mean_macro_f1": float(np.mean(list(per_fold_macro_f1[c].values()))),
            "std_macro_f1": float(np.std(list(per_fold_macro_f1[c].values())))}
        for c in CONDITIONS
    }

    a_f1 = pooled_summary["A"]["macro_f1"]
    c_f1 = pooled_summary["C"]["macro_f1"]
    d_f1 = pooled_summary["D"]["macro_f1"]
    out = {
        "pooled_summary": pooled_summary,
        "per_fold_macro_f1": per_fold_macro_f1,
        "grouped_mean": grouped_mean,
        "fold_deltas": fold_deltas,
        "by_type": by_type,
        "by_dpdt_bucket": by_dpdt,
        "by_duration_bucket": by_duration,
        "per_recording": per_recording,
        "oracle_gap_pooled": {"A": a_f1, "C": c_f1, "D": d_f1, "C_minus_A": c_f1 - a_f1, "D_minus_C": d_f1 - c_f1, "D_minus_A": d_f1 - a_f1},
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))

    print("\n=== Step 14 pooled comparison table ===")
    print(f"{'Condition':10s} {'FrameAcc':>9s} {'MacroF1':>8s} " + " ".join(f"{TYPE_NAMES[i]}F1".rjust(7) for i in range(4)))
    for c in CONDITIONS:
        s = pooled_summary[c]
        row = " ".join(f"{s['per_class'][TYPE_NAMES[i]]['f1']:.3f}".rjust(7) for i in range(4))
        print(f"{c:10s} {s['accuracy']:9.3f} {s['macro_f1']:8.3f} {row}")
    print("\nfold deltas (macro F1): C-A", fold_deltas["C_minus_A"], "\nD-A", fold_deltas["D_minus_A"])
    print(f"\noracle gap: A={a_f1:.3f} C={c_f1:.3f} D={d_f1:.3f}  C-A={c_f1-a_f1:+.3f}  D-C={d_f1-c_f1:+.3f}  D-A={d_f1-a_f1:+.3f}")


if __name__ == "__main__":
    main()
