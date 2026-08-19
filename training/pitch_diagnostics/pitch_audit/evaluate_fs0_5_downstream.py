"""Step 20 Phase B: pooled test-set evaluation of the P0 architecture
retrained on the fs=0.5 Fused+D3 dense pitch path, for direct comparison
against the frozen fs=1 P0 result (pooled macro F1 0.338, grouped mean
0.348 -- docs/step_15_learned_pitch_motion.md) and the oracle P3 ceiling
(0.771 / 0.778). Mirrors evaluate_d0_downstream.py's structure exactly."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.folds import build_fold_split, load_kfold_manifest  # noqa: E402
from training.framewise_dataset import FullRecordingDataset, RecordingLaneIndex, collate_variable_length  # noqa: E402
from training.framewise_models import FramewiseConditionalTCNModel  # noqa: E402
from training.metrics import frame_metrics  # noqa: E402
from training.normalization import load_fold_cqt_stats  # noqa: E402
from training.pitch_diagnostics.pitch_audit.phase_b_fs0_5 import load_dense_estimated_pitch_fs0_5  # noqa: E402
from training.relative_pitch_features import PITCH_FEATURE_DIM, standardize_phi  # noqa: E402

RUN_DIR = REPO_ROOT / "output" / "pitch_motion_ablation"


def main() -> None:
    device = torch.device("cpu")
    index = RecordingLaneIndex.build(REPO_ROOT)
    manifest = load_kfold_manifest(REPO_ROOT)
    fs05_pitch = load_dense_estimated_pitch_fs0_5()

    pooled_pred, pooled_true = [], []
    per_fold_f1 = {}
    for fold in range(5):
        split = build_fold_split(manifest, fold, seed=42)
        mu_cqt, sigma_cqt = load_fold_cqt_stats(fold, REPO_ROOT)
        ckpt = torch.load(RUN_DIR / "condition_P0_fs0.5" / f"fold_{fold}" / "best.pt", map_location="cpu", weights_only=False)
        model = FramewiseConditionalTCNModel(use_audio=False, use_pitch=True, pitch_dim=PITCH_FEATURE_DIM).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        phi_mu, phi_sigma = ckpt["phi_mu"], ckpt["phi_sigma"]

        test_ds = FullRecordingDataset(
            index, split.test_recording_ids, mu_cqt, sigma_cqt,
            estimated_pitch=fs05_pitch, compute_pitch_features=True,
        )
        loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_variable_length)
        fold_pred, fold_true = [], []
        with torch.no_grad():
            for batch in loader:
                raw = batch["phi_estimated"].numpy()
                std = standardize_phi(raw.reshape(-1, PITCH_FEATURE_DIM), phi_mu, phi_sigma).reshape(raw.shape)
                x = torch.from_numpy(std).to(device)
                logits = model(None, x)[0].numpy()
                valid = batch["valid_target"][0].numpy()
                pad = batch["padding_mask"][0].numpy()
                mask = valid & (~pad)
                p, t = logits.argmax(axis=-1)[mask], batch["trajectory_type"][0].numpy()[mask]
                pooled_pred.append(p); pooled_true.append(t)
                fold_pred.append(p); fold_true.append(t)
        fold_m = frame_metrics(np.concatenate(fold_pred), np.concatenate(fold_true))
        per_fold_f1[fold] = fold_m["macro_f1"]
        print(f"fold {fold} done: macro_f1={fold_m['macro_f1']:.4f}")

    pred = np.concatenate(pooled_pred); true = np.concatenate(pooled_true)
    m = frame_metrics(pred, true)
    grouped_mean = float(np.mean(list(per_fold_f1.values())))
    grouped_std = float(np.std(list(per_fold_f1.values())))
    print("\n=== fs=0.5-trained P0 pooled test metrics ===")
    print("accuracy", m["accuracy"], "macro_f1", m["macro_f1"])
    print(f"grouped mean {grouped_mean:.4f} +/- {grouped_std:.4f}")
    for k, v in m["per_class"].items():
        print(k, v)

    out = {**m, "per_fold_macro_f1": per_fold_f1, "grouped_mean_macro_f1": grouped_mean, "grouped_std_macro_f1": grouped_std}
    out_path = RUN_DIR / "condition_P0_fs0.5" / "pooled_test_evaluation.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
