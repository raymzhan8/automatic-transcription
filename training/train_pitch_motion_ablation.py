"""Step 15: P0-P3 pitch/salience-only trajectory-classification ablation.
Pitch/salience evidence only -- no audio/CQT branch, no fusion, no register
decoding, same unweighted-CE framewise task and grouped folds as Step 14.

P0: reproduction of Step 14 condition B (Fused+D3 -> fixed phi(10/50/100/200ms) -> shared TCN).
P1: Fused+D3 -> dense octave-unwrapped frame-to-frame delta (phi offset=1 only) -> PitchMotionEncoder.
P2: frozen fused salience, windowed/register-recentered (dense_relative_salience.py) -> SalienceMotionEncoder.
P3: identical to P1, source = GT parametric pitch.

Same excerpt sampler / grouped 5-fold splits / valid-frame CE loss as
Step 14; training protocol (default 50 epochs, patience 10) is IDENTICAL
across all four conditions (spec section 3-4).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.folds import prepare_fold  # noqa: E402
from training.framewise_dataset import FramewiseExcerptDataset, RecordingLaneIndex, collate_excerpts  # noqa: E402
from training.framewise_models import (  # noqa: E402
    FramewiseConditionalTCNModel, PitchMotionEncoder, PitchOnlyClassifier, SalienceMotionEncoder, count_params,
)
from training.losses import FramewiseTypeLoss  # noqa: E402
from training.metrics import collect_frame_predictions, frame_metrics  # noqa: E402
from training.normalization import load_fold_cqt_stats, log2_hz_to_cents  # noqa: E402
from training.pitch_diagnostics.relative_pitch.dense_relative_salience import build as build_relative_salience  # noqa: E402
from training.relative_pitch_features import (  # noqa: E402
    PITCH_FEATURE_DIM, compute_phi, load_dense_estimated_pitch, phi_stats, standardize_phi,
)
from training.train_relative_pitch_ablation import fold_phi_stats  # noqa: E402

CONDITIONS = ("P0", "P1", "P2", "P3")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", choices=CONDITIONS, required=True)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--all-folds", action="store_true")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output" / "pitch_motion_ablation")
    p.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    p.add_argument("--tiny-overfit", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--excerpts-per-epoch", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--manifest-name", default="grouped_kfold_k5_seed42.json")
    p.add_argument("--pitch-variant", choices=("D1", "D0", "0.25x", "0.5x"), default="D1",
                    help="Step 17/18: D1 (default, Fused+D3/1.0x), D0 (framewise argmax/0x), "
                         "or 0.25x/0.5x (Step 17's movement-cost sweep points)")
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(condition: str) -> torch.nn.Module:
    if condition == "P0":
        return FramewiseConditionalTCNModel(use_audio=False, use_pitch=True, pitch_dim=PITCH_FEATURE_DIM)
    if condition in ("P1", "P3"):
        return PitchOnlyClassifier(PitchMotionEncoder(in_channels=1, hidden=32))
    if condition == "P2":
        return PitchOnlyClassifier(SalienceMotionEncoder(hidden=32))
    raise ValueError(condition)


def get_pitch_input(batch: dict, condition: str, phi_mu, phi_sigma, device: torch.device) -> torch.Tensor:
    """Returns the model's input tensor for the given condition, already
    device-placed and (for P0/P1/P3) standardized with train-only stats."""
    if condition == "P0":
        raw = batch["phi_estimated"].numpy()
        std = standardize_phi(raw.reshape(-1, PITCH_FEATURE_DIM), phi_mu, phi_sigma).reshape(raw.shape)
        return torch.from_numpy(std).to(device)  # [B, T, 4] -> model transposes internally
    if condition == "P1":
        raw = batch["phi_estimated"][..., 0:1].numpy()
        std = standardize_phi(raw.reshape(-1, 1), phi_mu, phi_sigma).reshape(raw.shape)
        return torch.from_numpy(std).transpose(1, 2).to(device)  # [B, 1, T]
    if condition == "P3":
        raw = batch["phi_oracle"][..., 0:1].numpy()
        std = standardize_phi(raw.reshape(-1, 1), phi_mu, phi_sigma).reshape(raw.shape)
        return torch.from_numpy(std).transpose(1, 2).to(device)  # [B, 1, T]
    if condition == "P2":
        return batch["relative_salience"].to(device)  # [B, 1, W, T]
    raise ValueError(condition)


def forward(model: torch.nn.Module, condition: str, pitch_input: torch.Tensor) -> torch.Tensor:
    if condition == "P0":
        return model(None, pitch_input)
    return model(pitch_input)


def run_epoch(model, loader, criterion, optimizer, device, *, condition, phi_mu, phi_sigma, grad_clip) -> dict[str, Any]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, n_batches = 0.0, 0
    all_pred, all_label = [], []

    for batch in loader:
        traj_type = batch["trajectory_type"].to(device)
        valid = batch["valid_target"].to(device)
        pad = batch["padding_mask"].to(device)
        pitch_input = get_pitch_input(batch, condition, phi_mu, phi_sigma, device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        logits = forward(model, condition, pitch_input)
        loss = criterion(logits, traj_type, valid, pad)
        if train_mode:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1
        pred, label = collect_frame_predictions(
            logits.detach().cpu(), batch["trajectory_type"], batch["valid_target"], batch["padding_mask"]
        )
        all_pred.append(pred)
        all_label.append(label)

    y_pred = np.concatenate(all_pred) if all_pred else np.array([])
    y_true = np.concatenate(all_label) if all_label else np.array([])
    metrics = frame_metrics(y_pred, y_true)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def fold_pitch_stats_for(condition: str, train_ids, index, estimated_pitch) -> tuple[np.ndarray, np.ndarray]:
    if condition == "P0":
        return fold_phi_stats(train_ids, index, estimated_pitch, "estimated")
    if condition == "P1":
        mu, sigma = fold_phi_stats(train_ids, index, estimated_pitch, "estimated")
        return mu[:1], sigma[:1]
    if condition == "P3":
        mu, sigma = fold_phi_stats(train_ids, index, None, "oracle")
        return mu[:1], sigma[:1]
    return None, None  # P2: no scalar standardization (salience already a bounded probability window)


def train_fold(condition, fold_index, *, repo_root, output_dir, tiny_overfit, max_epochs, patience,
                excerpts_per_epoch, batch_size, seed, manifest_name, estimated_pitch_override=None) -> dict[str, Any]:
    """Step 17: `estimated_pitch_override` lets condition P0 be retrained on
    a different pitch-path cache (e.g. D0, the framewise-argmax decode)
    instead of the default D1/Fused+D3 -- same architecture, folds, seeds,
    budget, sampler, loss, and normalization protocol (fold-wise phi stats
    are re-derived from whichever source is actually used), per Step 17
    spec section 14's fallback when frozen-model evaluation would be
    invalid across two very differently-distributed pitch sources."""
    set_seed(seed + fold_index)
    device = get_device()

    split, fold_summary = prepare_fold(repo_root, fold_index, manifest_name=manifest_name, seed=seed)
    mu_cqt, sigma_cqt = load_fold_cqt_stats(fold_index, repo_root)
    index = RecordingLaneIndex.build(repo_root)
    estimated_pitch = estimated_pitch_override if estimated_pitch_override is not None else load_dense_estimated_pitch()
    relative_salience = build_relative_salience() if condition == "P2" else None

    phi_mu, phi_sigma = fold_pitch_stats_for(condition, split.train_recording_ids, index, estimated_pitch)

    train_ids = split.train_recording_ids[:1] if tiny_overfit > 0 else split.train_recording_ids
    cache_excerpts = tiny_overfit if tiny_overfit > 0 else None
    ds_kwargs = dict(
        estimated_pitch=estimated_pitch, compute_pitch_features=True,
        relative_salience=relative_salience,
    )
    train_ds = FramewiseExcerptDataset(
        index, train_ids, mu_cqt, sigma_cqt, seed=seed + fold_index,
        excerpts_per_epoch=excerpts_per_epoch if tiny_overfit == 0 else tiny_overfit,
        cache_excerpts=cache_excerpts, **ds_kwargs,
    )
    val_ds = FramewiseExcerptDataset(
        index, split.val_recording_ids, mu_cqt, sigma_cqt, seed=seed + fold_index + 1000,
        excerpts_per_epoch=max(len(split.val_recording_ids) * 20, 64), **ds_kwargs,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_excerpts)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_excerpts)

    model = build_model(condition).to(device)
    n_params = count_params(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = FramewiseTypeLoss()

    run_dir = output_dir / f"condition_{condition}" / f"fold_{fold_index}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "fold_summary.json").write_text(json.dumps(fold_summary, indent=2) + "\n")
    (run_dir / "config.json").write_text(json.dumps({
        "condition": condition, "n_params": n_params, "tiny_overfit": tiny_overfit,
        "max_epochs": max_epochs, "patience": patience, "excerpts_per_epoch": excerpts_per_epoch,
        "batch_size": batch_size, "seed": seed,
        "phi_mu": phi_mu.tolist() if phi_mu is not None else None,
        "phi_sigma": phi_sigma.tolist() if phi_sigma is not None else None,
    }, indent=2) + "\n")

    csv_path = run_dir / "train_log.csv"
    fields = ["epoch", "train_loss", "train_acc", "train_macro_f1", "val_loss", "val_acc", "val_macro_f1", "elapsed_s"]
    best_f1, best_epoch, stale = -1.0, -1, 0
    history: list[dict[str, Any]] = []

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        t0 = time.time()
        for epoch in range(1, max_epochs + 1):
            train_m = run_epoch(model, train_loader, criterion, optimizer, device,
                                 condition=condition, phi_mu=phi_mu, phi_sigma=phi_sigma, grad_clip=1.0)
            val_m = run_epoch(model, val_loader, criterion, None, device,
                               condition=condition, phi_mu=phi_mu, phi_sigma=phi_sigma, grad_clip=0.0)
            row = {
                "epoch": epoch, "train_loss": train_m["loss"], "train_acc": train_m["accuracy"],
                "train_macro_f1": train_m["macro_f1"], "val_loss": val_m["loss"],
                "val_acc": val_m["accuracy"], "val_macro_f1": val_m["macro_f1"],
                "elapsed_s": time.time() - t0,
            }
            writer.writerow(row)
            history.append({"epoch": epoch, "train": train_m, "val": val_m})
            print(f"[{condition}] fold {fold_index} epoch {epoch}: "
                  f"train f1={train_m['macro_f1']:.3f} acc={train_m['accuracy']:.3f}  "
                  f"val f1={val_m['macro_f1']:.3f} acc={val_m['accuracy']:.3f}")

            if val_m["macro_f1"] > best_f1:
                best_f1, best_epoch, stale = val_m["macro_f1"], epoch, 0
                torch.save({
                    "model_state": model.state_dict(), "condition": condition, "fold_index": fold_index,
                    "epoch": epoch, "val_macro_f1": best_f1, "n_params": n_params, "seed": seed,
                    "phi_mu": phi_mu, "phi_sigma": phi_sigma,
                }, run_dir / "best.pt")
            else:
                stale += 1
                if stale >= patience and tiny_overfit == 0:
                    print(f"early stop at epoch {epoch}")
                    break

    result = {
        "condition": condition, "fold_index": fold_index, "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1, "n_params": n_params,
        "run_dir": str(run_dir.relative_to(repo_root)), "history": history,
    }
    (run_dir / "train_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else args.repo_root / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds = list(range(5)) if args.all_folds else [args.fold if args.fold is not None else 0]

    estimated_pitch_override = None
    condition_label = args.condition
    if args.pitch_variant != "D1":
        if args.condition != "P0":
            raise ValueError("--pitch-variant other than D1 is only meaningful with --condition P0 (Step 17/18's downstream test)")
        if args.pitch_variant == "D0":
            from training.pitch_diagnostics.relative_pitch.dense_framewise_argmax_path import build as build_variant
            estimated_pitch_override = build_variant()
        else:
            from training.pitch_diagnostics.relative_pitch.dense_lambda_sweep_path import build as build_sweep
            estimated_pitch_override = build_sweep(float(args.pitch_variant.rstrip("x")))
        condition_label = f"P0_{args.pitch_variant}"  # separate run_dir from the existing D1-trained P0

    results = []
    for fold in folds:
        print(f"\n=== condition {condition_label} fold {fold} ===")
        results.append(train_fold(
            args.condition, fold, repo_root=args.repo_root, output_dir=args.output_dir,
            tiny_overfit=args.tiny_overfit, max_epochs=args.max_epochs, patience=args.patience,
            excerpts_per_epoch=args.excerpts_per_epoch, batch_size=args.batch_size,
            seed=args.seed, manifest_name=args.manifest_name,
            estimated_pitch_override=estimated_pitch_override,
        ))
        if args.pitch_variant != "D1":
            # train_fold writes under condition_P0/; move/relabel to condition_{label}/ for this fold
            src = args.output_dir / "condition_P0" / f"fold_{fold}"
            dst = args.output_dir / f"condition_{condition_label}" / f"fold_{fold}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                import shutil
                shutil.rmtree(dst)
            src.rename(dst)

    summary = {
        "condition": condition_label, "timestamp": datetime.now(timezone.utc).isoformat(),
        "folds": results, "mean_val_macro_f1": float(np.mean([r["best_val_macro_f1"] for r in results])),
        "std_val_macro_f1": float(np.std([r["best_val_macro_f1"] for r in results])),
    }
    out = args.output_dir / f"condition_{condition_label}" / "cv_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "folds"}, indent=2))


if __name__ == "__main__":
    main()
