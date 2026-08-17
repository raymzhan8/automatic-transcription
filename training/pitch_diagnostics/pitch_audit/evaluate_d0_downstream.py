"""Step 17 section 14: pooled test-set evaluation of the P0 architecture
retrained on D0 (framewise argmax) pitch, for direct comparison against
D1's existing Step 15 P0 result and D2's Step 15 P3 result."""

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
from training.pitch_diagnostics.relative_pitch.dense_framewise_argmax_path import build as build_d0  # noqa: E402
from training.relative_pitch_features import PITCH_FEATURE_DIM, standardize_phi  # noqa: E402

RUN_DIR = REPO_ROOT / "output" / "pitch_motion_ablation"


def main() -> None:
    device = torch.device("cpu")
    index = RecordingLaneIndex.build(REPO_ROOT)
    manifest = load_kfold_manifest(REPO_ROOT)
    d0_pitch = build_d0()

    pooled_pred, pooled_true = [], []
    for fold in range(5):
        split = build_fold_split(manifest, fold, seed=42)
        mu_cqt, sigma_cqt = load_fold_cqt_stats(fold, REPO_ROOT)
        ckpt = torch.load(RUN_DIR / "condition_P0_D0" / f"fold_{fold}" / "best.pt", map_location="cpu", weights_only=False)
        model = FramewiseConditionalTCNModel(use_audio=False, use_pitch=True, pitch_dim=PITCH_FEATURE_DIM).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        phi_mu, phi_sigma = ckpt["phi_mu"], ckpt["phi_sigma"]

        test_ds = FullRecordingDataset(
            index, split.test_recording_ids, mu_cqt, sigma_cqt,
            estimated_pitch=d0_pitch, compute_pitch_features=True,
        )
        loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_variable_length)
        with torch.no_grad():
            for batch in loader:
                raw = batch["phi_estimated"].numpy()
                std = standardize_phi(raw.reshape(-1, PITCH_FEATURE_DIM), phi_mu, phi_sigma).reshape(raw.shape)
                x = torch.from_numpy(std).to(device)
                logits = model(None, x)[0].numpy()
                valid = batch["valid_target"][0].numpy()
                pad = batch["padding_mask"][0].numpy()
                mask = valid & (~pad)
                pooled_pred.append(logits.argmax(axis=-1)[mask])
                pooled_true.append(batch["trajectory_type"][0].numpy()[mask])
        print(f"fold {fold} done")

    pred = np.concatenate(pooled_pred); true = np.concatenate(pooled_true)
    m = frame_metrics(pred, true)
    print("\n=== D0-trained P0 pooled test metrics ===")
    print("accuracy", m["accuracy"], "macro_f1", m["macro_f1"])
    for k, v in m["per_class"].items():
        print(k, v)

    out_path = RUN_DIR / "condition_P0_D0" / "pooled_test_evaluation.json"
    out_path.write_text(json.dumps(m, indent=2))
    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
