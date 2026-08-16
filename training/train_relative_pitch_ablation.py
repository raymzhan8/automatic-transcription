"""Step 14: A/B/C/D relative-pitch feature ablation for framewise T0-T3
trajectory classification. Type-only (no pitch regression head, unlike
B0/B1/C) -- a separate, leaner script from train_framewise.py so that
script and its frozen B0/B1/C results are untouched.

Conditions (spec section 1):
  A: audio only               (FramewiseConditionalTCNModel(use_audio=True, use_pitch=False))
  B: estimated relative pitch only (use_audio=False, use_pitch=True, source=Fused+D3)
  C: audio + estimated relative pitch (use_audio=True, use_pitch=True, source=Fused+D3)
  D: audio + oracle relative pitch    (use_audio=True, use_pitch=True, source=GT)

Same TCN + type_head across all four (framewise_models.FramewiseConditionalTCNModel);
same excerpt sampler, same grouped folds, same unweighted CE loss on
valid & non-padding frames only (spec sections 3, 7-9).
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
from training.framewise_dataset import (  # noqa: E402
    FramewiseExcerptDataset,
    RecordingLaneIndex,
    collate_excerpts,
)
from training.framewise_models import FramewiseConditionalTCNModel, count_params  # noqa: E402
from training.losses import FramewiseTypeLoss  # noqa: E402
from training.metrics import collect_frame_predictions, frame_metrics  # noqa: E402
from training.normalization import load_fold_cqt_stats, log2_hz_to_cents  # noqa: E402
from training.relative_pitch_features import (  # noqa: E402
    PITCH_FEATURE_DIM, compute_phi, load_dense_estimated_pitch, phi_stats, standardize_phi,
)

CONDITIONS = ("A", "B", "C", "D")
PITCH_SOURCE = {"A": None, "B": "estimated", "C": "estimated", "D": "oracle"}
USE_AUDIO = {"A": True, "B": False, "C": True, "D": True}
USE_PITCH = {"A": False, "B": True, "C": True, "D": True}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", choices=CONDITIONS, required=True)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--all-folds", action="store_true")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output" / "relative_pitch_ablation")
    p.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    p.add_argument("--tiny-overfit", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--excerpts-per-epoch", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--manifest-name", default="grouped_kfold_k5_seed42.json")
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


def fold_phi_stats(
    recording_ids: list[str],
    index: RecordingLaneIndex,
    estimated_pitch: dict[str, np.ndarray] | None,
    source: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Train-recordings-only mean/std for the pitch-feature source actually
    used by this condition (spec section 10)."""
    phis, valids = [], []
    for rid in recording_ids:
        lane = next(x for x in index.lanes if x.recording_id == rid)
        frames = index._frames[(rid, lane.lane_id)]
        valid = frames["valid_target"]
        if source == "oracle":
            cents = log2_hz_to_cents(frames["pitch_log2_hz"].astype(np.float64), lane.fundamental_hz)
        else:
            cents = log2_hz_to_cents(estimated_pitch[rid].astype(np.float64), lane.fundamental_hz)
        phis.append(compute_phi(cents, valid))
        valids.append(valid)
    return phi_stats(phis, valids)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    condition: str,
    phi_mu: np.ndarray | None,
    phi_sigma: np.ndarray | None,
    grad_clip: float,
) -> dict[str, Any]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, n_batches = 0.0, 0
    all_pred, all_label = [], []
    source = PITCH_SOURCE[condition]

    for batch in loader:
        spec = batch["spec"].to(device) if USE_AUDIO[condition] else None
        traj_type = batch["trajectory_type"].to(device)
        valid = batch["valid_target"].to(device)
        pad = batch["padding_mask"].to(device)

        pitch_feat = None
        if USE_PITCH[condition]:
            key = "phi_estimated" if source == "estimated" else "phi_oracle"
            raw = batch[key].numpy()
            std = standardize_phi(raw.reshape(-1, PITCH_FEATURE_DIM), phi_mu, phi_sigma).reshape(raw.shape)
            pitch_feat = torch.from_numpy(std).to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        logits = model(spec, pitch_feat)
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


def train_fold(
    condition: str,
    fold_index: int,
    *,
    repo_root: Path,
    output_dir: Path,
    tiny_overfit: int,
    max_epochs: int,
    patience: int,
    excerpts_per_epoch: int,
    batch_size: int,
    seed: int,
    manifest_name: str,
) -> dict[str, Any]:
    set_seed(seed + fold_index)
    device = get_device()

    split, fold_summary = prepare_fold(repo_root, fold_index, manifest_name=manifest_name, seed=seed)
    mu_cqt, sigma_cqt = load_fold_cqt_stats(fold_index, repo_root)
    index = RecordingLaneIndex.build(repo_root)
    estimated_pitch = load_dense_estimated_pitch() if USE_PITCH[condition] else None

    phi_mu = phi_sigma = None
    if USE_PITCH[condition]:
        phi_mu, phi_sigma = fold_phi_stats(split.train_recording_ids, index, estimated_pitch, PITCH_SOURCE[condition])

    train_ids = split.train_recording_ids[:1] if tiny_overfit > 0 else split.train_recording_ids
    cache_excerpts = tiny_overfit if tiny_overfit > 0 else None
    ds_kwargs = dict(estimated_pitch=estimated_pitch, compute_pitch_features=USE_PITCH[condition])
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

    model = FramewiseConditionalTCNModel(
        use_audio=USE_AUDIO[condition], use_pitch=USE_PITCH[condition], pitch_dim=PITCH_FEATURE_DIM,
    ).to(device)
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

    results = []
    for fold in folds:
        print(f"\n=== condition {args.condition} fold {fold} ===")
        results.append(train_fold(
            args.condition, fold, repo_root=args.repo_root, output_dir=args.output_dir,
            tiny_overfit=args.tiny_overfit, max_epochs=args.max_epochs, patience=args.patience,
            excerpts_per_epoch=args.excerpts_per_epoch, batch_size=args.batch_size,
            seed=args.seed, manifest_name=args.manifest_name,
        ))

    summary = {
        "condition": args.condition, "timestamp": datetime.now(timezone.utc).isoformat(),
        "folds": results, "mean_val_macro_f1": float(np.mean([r["best_val_macro_f1"] for r in results])),
        "std_val_macro_f1": float(np.std([r["best_val_macro_f1"] for r in results])),
    }
    out = args.output_dir / f"condition_{args.condition}" / "cv_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "folds"}, indent=2))


if __name__ == "__main__":
    main()
