"""Step 18: pooled + per-fold + per-recording test-set evaluation of the P0
architecture trained on each of the four Step 17 lambda-sweep pitch paths
(L0=D0/0x, L25=0.25x, L50=0.5x, L100=D1/1.0x), for the central four-lambda
comparison table.
"""

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
from training.relative_pitch_features import PITCH_FEATURE_DIM, load_dense_estimated_pitch, standardize_phi  # noqa: E402

RUN_DIR = REPO_ROOT / "output" / "pitch_motion_ablation"

LAMBDA_LABELS = {
    "L0": ("D0", "condition_P0_D0"),
    "L25": ("0.25x", "condition_P0_0.25x"),
    "L50": ("0.5x", "condition_P0_0.5x"),
    "L100": ("D1", "condition_P0"),  # Step 15's original D1-trained P0
}


def _load_pitch(variant: str) -> dict[str, np.ndarray]:
    if variant == "D1":
        return load_dense_estimated_pitch()
    if variant == "D0":
        from training.pitch_diagnostics.relative_pitch.dense_framewise_argmax_path import build as build_d0
        return build_d0()
    from training.pitch_diagnostics.relative_pitch.dense_lambda_sweep_path import build as build_sweep
    return build_sweep(float(variant.rstrip("x")))


def evaluate_one(label: str) -> dict:
    variant, run_subdir = LAMBDA_LABELS[label]
    pitch = _load_pitch(variant)
    device = torch.device("cpu")
    index = RecordingLaneIndex.build(REPO_ROOT)
    manifest = load_kfold_manifest(REPO_ROOT)

    per_fold_pred, per_fold_true = {}, {}
    per_recording = {}
    for fold in range(5):
        split = build_fold_split(manifest, fold, seed=42)
        mu_cqt, sigma_cqt = load_fold_cqt_stats(fold, REPO_ROOT)
        ckpt = torch.load(RUN_DIR / run_subdir / f"fold_{fold}" / "best.pt", map_location="cpu", weights_only=False)
        model = FramewiseConditionalTCNModel(use_audio=False, use_pitch=True, pitch_dim=PITCH_FEATURE_DIM).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        phi_mu, phi_sigma = ckpt["phi_mu"], ckpt["phi_sigma"]

        test_ds = FullRecordingDataset(
            index, split.test_recording_ids, mu_cqt, sigma_cqt,
            estimated_pitch=pitch, compute_pitch_features=True,
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
                pred = logits.argmax(axis=-1)[mask]
                true = batch["trajectory_type"][0].numpy()[mask]
                fold_pred.append(pred); fold_true.append(true)
                rid = batch["recording_id"][0]
                per_recording[rid] = {"fold": fold, **frame_metrics(pred, true)}
        per_fold_pred[fold] = np.concatenate(fold_pred)
        per_fold_true[fold] = np.concatenate(fold_true)
        print(f"  [{label}] fold {fold} done")

    per_fold_metrics = {f: frame_metrics(per_fold_pred[f], per_fold_true[f]) for f in range(5)}
    pooled_pred = np.concatenate(list(per_fold_pred.values()))
    pooled_true = np.concatenate(list(per_fold_true.values()))
    pooled = frame_metrics(pooled_pred, pooled_true)

    fold_f1 = [per_fold_metrics[f]["macro_f1"] for f in range(5)]
    return {
        "label": label, "variant": variant,
        "pooled": pooled,
        "per_fold": {str(f): per_fold_metrics[f] for f in range(5)},
        "grouped_mean_macro_f1": float(np.mean(fold_f1)),
        "grouped_std_macro_f1": float(np.std(fold_f1)),
        "per_recording": per_recording,
    }


def l100_from_step15_record() -> dict:
    """L100/D1's checkpoints (output/pitch_motion_ablation/condition_P0/fold_*/
    best.pt) were inadvertently overwritten and renamed away by Step 17/18's
    variant-training runs, which all stage through the same shared
    `condition_P0/` directory before relabeling it (a real bug, not fixed
    retroactively here -- see the Step 18 report). The checkpoint files are
    gone, but Step 15's complete evaluation output was saved separately and
    is unaffected; reused verbatim rather than re-deriving numbers that
    already exist and were already cross-checked in Step 17's report."""
    d = json.loads((RUN_DIR / "evaluation_result.json").read_text())
    return {
        "label": "L100", "variant": "D1",
        "pooled": d["pooled_summary"]["P0"],
        "per_fold": {f: {"macro_f1": v} for f, v in d["per_fold_macro_f1"]["P0"].items()},
        "grouped_mean_macro_f1": d["grouped_mean"]["P0"]["mean_macro_f1"],
        "grouped_std_macro_f1": d["grouped_mean"]["P0"]["std_macro_f1"],
        "per_recording": d["per_recording"]["P0"],
    }


def main() -> None:
    all_results = {}
    for label in ("L0", "L25", "L50"):
        print(f"=== evaluating {label} ({LAMBDA_LABELS[label][0]}) ===")
        all_results[label] = evaluate_one(label)
    print("=== L100 (D1): reusing Step 15's saved evaluation (checkpoint no longer on disk, see report) ===")
    all_results["L100"] = l100_from_step15_record()

    out_path = RUN_DIR / "step18_lambda_comparison.json"
    out_path.write_text(json.dumps(all_results, indent=2))

    print("\n=== central comparison table ===")
    print(f"{'lambda':8s} {'pooled_F1':>10s} {'grouped_mean':>13s} {'T0':>6s} {'T1':>6s} {'T2':>6s} {'T3':>6s}")
    for label, res in all_results.items():
        p = res["pooled"]
        row = " ".join(f"{p['per_class'][t]['f1']:.3f}" for t in ("T0", "T1", "T2", "T3"))
        print(f"{label:8s} {p['macro_f1']:10.4f} {res['grouped_mean_macro_f1']:13.4f} {row}")


if __name__ == "__main__":
    main()
