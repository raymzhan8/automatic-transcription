"""Step 21 sections 8-11: pooled + per-fold + per-recording + per-class
held-out TEST evaluation of the P0 architecture retrained on CREPE
(Step 21's frozen default pitch source), for direct comparison against the
existing D1 and oracle (P3) results already saved in
output/pitch_motion_ablation/evaluation_result.json (Step 15's evaluation
harness) -- those baselines are reused as-is, not retrained, per spec
section 8 ("use the existing validated results where possible")."""

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
from training.pitch_diagnostics.relative_pitch.dense_crepe_path import build as build_crepe  # noqa: E402
from training.relative_pitch_features import PITCH_FEATURE_DIM, standardize_phi  # noqa: E402

RUN_DIR = REPO_ROOT / "output" / "pitch_motion_ablation"
D1_ORACLE_RESULT = RUN_DIR / "evaluation_result.json"
OUT_PATH = RUN_DIR / "condition_P0_CREPE" / "pooled_test_evaluation.json"


def main() -> None:
    device = torch.device("cpu")
    index = RecordingLaneIndex.build(REPO_ROOT)
    manifest = load_kfold_manifest(REPO_ROOT)
    crepe_pitch = build_crepe()

    pooled_pred, pooled_true = [], []
    per_fold_macro_f1: dict[int, float] = {}
    per_recording: dict[str, dict] = {}

    for fold in range(5):
        split = build_fold_split(manifest, fold, seed=42)
        mu_cqt, sigma_cqt = load_fold_cqt_stats(fold, REPO_ROOT)
        ckpt = torch.load(RUN_DIR / "condition_P0_CREPE" / f"fold_{fold}" / "best.pt", map_location="cpu", weights_only=False)
        model = FramewiseConditionalTCNModel(use_audio=False, use_pitch=True, pitch_dim=PITCH_FEATURE_DIM).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        phi_mu, phi_sigma = ckpt["phi_mu"], ckpt["phi_sigma"]

        test_ds = FullRecordingDataset(
            index, split.test_recording_ids, mu_cqt, sigma_cqt,
            estimated_pitch=crepe_pitch, compute_pitch_features=True,
        )
        loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_variable_length)
        fold_pred, fold_true = [], []
        with torch.no_grad():
            for batch in loader:
                rid = batch["recording_id"][0]
                raw = batch["phi_estimated"].numpy()
                std = standardize_phi(raw.reshape(-1, PITCH_FEATURE_DIM), phi_mu, phi_sigma).reshape(raw.shape)
                x = torch.from_numpy(std).to(device)
                logits = model(None, x)[0].numpy()
                valid = batch["valid_target"][0].numpy()
                pad = batch["padding_mask"][0].numpy()
                mask = valid & (~pad)
                pred = logits.argmax(axis=-1)[mask]
                true = batch["trajectory_type"][0].numpy()[mask]
                pooled_pred.append(pred); pooled_true.append(true)
                fold_pred.append(pred); fold_true.append(true)
                per_recording[rid] = {"fold": fold, **frame_metrics(pred, true)}
        fp = np.concatenate(fold_pred); ft = np.concatenate(fold_true)
        per_fold_macro_f1[fold] = frame_metrics(fp, ft)["macro_f1"]
        print(f"fold {fold} done ({len(split.test_recording_ids)} recordings), macro_f1={per_fold_macro_f1[fold]:.4f}")

    pred = np.concatenate(pooled_pred); true = np.concatenate(pooled_true)
    pooled_summary = frame_metrics(pred, true)
    grouped_mean = float(np.mean(list(per_fold_macro_f1.values())))
    grouped_std = float(np.std(list(per_fold_macro_f1.values())))

    out = {
        "condition": "P0_CREPE",
        "pooled_summary": pooled_summary,
        "per_fold_macro_f1": per_fold_macro_f1,
        "grouped_mean_macro_f1": grouped_mean,
        "grouped_std_macro_f1": grouped_std,
        "per_recording": per_recording,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")

    print("\n=== CREPE-trained P0 pooled test metrics ===")
    print("accuracy", pooled_summary["accuracy"], "macro_f1", pooled_summary["macro_f1"])
    for k, v in pooled_summary["per_class"].items():
        print(k, v)
    print("\ngrouped 5-fold mean +/- std:", grouped_mean, "+/-", grouped_std)
    print("per-fold macro F1:", per_fold_macro_f1)
    print(f"\nsaved to {OUT_PATH}")

    if D1_ORACLE_RESULT.exists():
        d1_oracle = json.loads(D1_ORACLE_RESULT.read_text())
        print("\n=== reference (existing, not retrained) ===")
        print("D1 pooled macro_f1:", d1_oracle["pooled_summary"]["P0"]["macro_f1"],
              "grouped mean:", d1_oracle["grouped_mean"]["P0"])
        print("Oracle (P3) pooled macro_f1:", d1_oracle["pooled_summary"]["P3"]["macro_f1"],
              "grouped mean:", d1_oracle["grouped_mean"]["P3"])


if __name__ == "__main__":
    main()
