"""Train framewise trajectory models (Experiments B0 / B1 / C)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
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
from training.framewise_models import build_model, count_params  # noqa: E402
from training.losses import FramewiseMultitaskLoss, FramewiseTypeLoss  # noqa: E402
from training.metrics import collect_frame_predictions, frame_metrics, pitch_error_metrics  # noqa: E402
from training.normalization import denormalize_pitch, load_fold_cqt_stats, load_fold_pitch_stats  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "training" / "configs" / "b0_tcn_type_only.json",
    )
    parser.add_argument("--fold", type=int, default=None, help="Single fold index (0-4)")
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output" / "framewise_runs")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--tiny-overfit", type=int, default=0, help="If >0, train on N excerpts only")
    parser.add_argument("--label-shuffle", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def expected_param_count(config: dict[str, Any]) -> int | None:
    task = config.get("task")
    arch = config.get("architecture", "tcn")
    if task == "type_only" and arch == "tcn":
        return 434_500
    if task == "type_and_pitch" and arch == "tcn":
        return 434_629
    if task == "type_and_pitch" and arch == "bigru":
        return 305_221
    return None


def _shared_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    if hasattr(model, "freq_cnn"):
        params.extend(list(model.freq_cnn.parameters()))
    if hasattr(model, "tcn"):
        params.extend(list(model.tcn.parameters()))
    if hasattr(model, "gru"):
        params.extend(list(model.gru.parameters()))
    return params


def _flatten_grads(grads: tuple[torch.Tensor | None, ...]) -> torch.Tensor:
    pieces = [g.reshape(-1) for g in grads if g is not None]
    if not pieces:
        return torch.zeros(1)
    return torch.cat(pieces)


def shared_grad_diagnostics(
    model: torch.nn.Module,
    type_loss: torch.Tensor,
    pitch_loss: torch.Tensor,
) -> dict[str, float]:
    shared = _shared_params(model)
    if not shared:
        return {}
    g_type = torch.autograd.grad(type_loss, shared, retain_graph=True, allow_unused=True)
    g_pitch = torch.autograd.grad(pitch_loss, shared, retain_graph=True, allow_unused=True)
    vt = _flatten_grads(g_type)
    vp = _flatten_grads(g_pitch)
    nt = float(vt.norm().item())
    np_ = float(vp.norm().item())
    denom = max(nt * np_, 1e-12)
    cos = float(torch.dot(vt, vp).item() / denom)
    return {
        "grad_shared_from_type": nt,
        "grad_shared_from_pitch": np_,
        "grad_cosine": cos,
    }


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    grad_clip: float,
    label_shuffle: bool = False,
    predict_pitch: bool = False,
    mu_pitch: float | None = None,
    sigma_pitch: float | None = None,
    log_grad_once: bool = False,
) -> dict[str, Any]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_type_loss = 0.0
    total_pitch_loss = 0.0
    n_batches = 0
    all_pred, all_label = [], []
    pitch_pred_cents: list[np.ndarray] = []
    pitch_true_cents: list[np.ndarray] = []
    grad_diag: dict[str, float] | None = None

    for batch in loader:
        spec = batch["spec"].to(device)
        traj_type = batch["trajectory_type"].to(device)
        valid = batch["valid_target"].to(device)
        pad = batch["padding_mask"].to(device)

        if label_shuffle and train_mode:
            flat_valid = valid.view(-1)
            flat_types = traj_type.view(-1)
            valid_types = flat_types[flat_valid]
            perm = valid_types[torch.randperm(len(valid_types))]
            flat_types[flat_valid] = perm
            traj_type = flat_types.view_as(traj_type)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        if "lengths" in batch:
            lengths = batch["lengths"].to(device).clamp(min=1)
        else:
            lengths = (~pad).sum(dim=1).clamp(min=1)
        if hasattr(model, "gru"):
            outputs = model(spec, lengths)
        else:
            outputs = model(spec)
        if predict_pitch:
            logits, pitch_pred = outputs
            pitch_target = batch["pitch_standardized"].to(device)
            loss, parts = criterion(
                logits, pitch_pred, traj_type, pitch_target, valid, pad
            )
            type_l = parts["type_loss"]
            pitch_l = parts["pitch_loss"]
            if log_grad_once and train_mode and grad_diag is None:
                try:
                    grad_diag = shared_grad_diagnostics(model, type_l, pitch_l)
                except RuntimeError:
                    grad_diag = {}
        else:
            logits = outputs
            loss = criterion(logits, traj_type, valid, pad)
            type_l = loss
            pitch_l = None

        if train_mode:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += float(loss.item())
        total_type_loss += float(type_l.item())
        if pitch_l is not None:
            total_pitch_loss += float(pitch_l.item())
        n_batches += 1
        pred, label = collect_frame_predictions(
            logits.detach().cpu(),
            batch["trajectory_type"],
            batch["valid_target"],
            batch["padding_mask"],
        )
        all_pred.append(pred)
        all_label.append(label)

        if predict_pitch and mu_pitch is not None and sigma_pitch is not None:
            mask = ((~batch["padding_mask"]) & batch["valid_target"]).numpy()
            pred_c = denormalize_pitch(
                pitch_pred.detach().cpu().numpy(), mu_pitch, sigma_pitch
            )
            true_c = batch["pitch_cents"].numpy()
            pitch_pred_cents.append(pred_c[mask])
            pitch_true_cents.append(true_c[mask])

    y_pred = np.concatenate(all_pred) if all_pred else np.array([])
    y_true = np.concatenate(all_label) if all_label else np.array([])
    metrics = frame_metrics(y_pred, y_true)
    metrics["loss"] = total_loss / max(n_batches, 1)
    metrics["type_loss"] = total_type_loss / max(n_batches, 1)
    if predict_pitch:
        metrics["pitch_loss"] = total_pitch_loss / max(n_batches, 1)
        if pitch_true_cents:
            pc = np.concatenate(pitch_pred_cents)
            tc = np.concatenate(pitch_true_cents)
            metrics.update({f"pitch_{k}": v for k, v in pitch_error_metrics(pc, tc).items()})
        if grad_diag:
            metrics.update(grad_diag)
    return metrics


@torch.no_grad()
def sampler_audit(loader: DataLoader, index: RecordingLaneIndex) -> dict[str, Any]:
    rec_counts: Counter[str] = Counter()
    class_counts: Counter[int] = Counter()
    t1_prov: Counter[str] = Counter()
    valid_per_rec: Counter[str] = Counter()
    n_valid = 0
    n_total = 0

    for batch in loader:
        for i, rid in enumerate(batch["recording_id"]):
            rec_counts[rid] += 1
            valid = batch["valid_target"][i].numpy()
            types = batch["trajectory_type"][i].numpy()
            pad = batch["padding_mask"][i].numpy()
            valid_per_rec[rid] += int(valid.sum())
            n_valid += int((valid & (~pad)).sum())
            n_total += int((~pad).sum())
            for t in types[valid]:
                class_counts[int(t)] += 1
                if int(t) == 1:
                    prim_ids = batch["primitive_id"][i]
                    for pid in prim_ids[valid]:
                        prim = index.get_primitive(rid, str(pid))
                        if prim and prim.get("t1_provenance"):
                            t1_prov[prim["t1_provenance"]] += 1

    counts = list(rec_counts.values()) or [0]
    return {
        "excerpts_per_recording": dict(rec_counts),
        "valid_frames_per_recording": dict(valid_per_rec),
        "class_frame_counts": dict(sorted(class_counts.items())),
        "t1_provenance_counts": dict(t1_prov),
        "min_recording_ratio": min(counts) / max(counts) if max(counts) else 0.0,
        "valid_frame_fraction": n_valid / max(n_total, 1),
        "n_excerpts": sum(counts),
    }


def train_fold(
    config: dict[str, Any],
    fold_index: int,
    *,
    repo_root: Path,
    output_dir: Path,
    tiny_overfit: int = 0,
    label_shuffle: bool = False,
    max_epochs: int | None = None,
) -> dict[str, Any]:
    seed = config.get("seed", 42)
    set_seed(seed + fold_index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predict_pitch = config.get("task") == "type_and_pitch"

    split, fold_summary = prepare_fold(
        repo_root,
        fold_index,
        manifest_name=config.get("manifest_name", "grouped_kfold_k5_seed42.json"),
        seed=seed,
    )
    mu, sigma = load_fold_cqt_stats(fold_index, repo_root)
    mu_pitch, sigma_pitch = load_fold_pitch_stats(fold_index, repo_root)
    fold_summary["pitch_mu_cents"] = mu_pitch
    fold_summary["pitch_sigma_cents"] = sigma_pitch
    index = RecordingLaneIndex.build(repo_root)

    excerpts = config.get("excerpts_per_epoch", 512)
    cache_excerpts = tiny_overfit if tiny_overfit > 0 else None
    train_ids = (
        split.train_recording_ids[:1] if tiny_overfit > 0 else split.train_recording_ids
    )
    ds_kwargs = dict(
        mu_pitch=mu_pitch if predict_pitch else None,
        sigma_pitch=sigma_pitch if predict_pitch else None,
    )
    train_ds = FramewiseExcerptDataset(
        index,
        train_ids,
        mu,
        sigma,
        seed=seed + fold_index,
        excerpts_per_epoch=excerpts if tiny_overfit == 0 else tiny_overfit,
        cache_excerpts=cache_excerpts,
        **ds_kwargs,
    )
    val_ds = FramewiseExcerptDataset(
        index,
        split.val_recording_ids,
        mu,
        sigma,
        seed=seed + fold_index + 1000,
        excerpts_per_epoch=max(len(split.val_recording_ids) * 20, 64),
        **ds_kwargs,
    )

    batch_size = config.get("batch_size", 8)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_excerpts,
        num_workers=config.get("num_workers", 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_excerpts,
        num_workers=config.get("num_workers", 0),
    )

    model = build_model(config).to(device)
    n_params = count_params(model)
    expected = expected_param_count(config)
    if expected is not None and n_params != expected:
        raise RuntimeError(f"param count {n_params} != expected {expected}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 1e-3),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    if predict_pitch:
        criterion: torch.nn.Module = FramewiseMultitaskLoss(
            lambda_type=config.get("lambda_type", 1.0),
            lambda_pitch=config.get("lambda_pitch", config.get("pitch_weight", 1.0)),
        )
    else:
        criterion = FramewiseTypeLoss()
    grad_clip = config.get("grad_clip", 1.0)

    output_dir = output_dir if output_dir.is_absolute() else repo_root / output_dir
    run_dir = output_dir / config["experiment"] / f"fold_{fold_index}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "fold_summary.json").write_text(json.dumps(fold_summary, indent=2) + "\n")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    log_path = run_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()
    csv_path = run_dir / "train_log.csv"
    csv_fields = [
        "epoch", "train_loss", "train_type_loss", "train_pitch_loss",
        "train_acc", "train_macro_f1",
        "val_loss", "val_type_loss", "val_pitch_loss",
        "val_acc", "val_macro_f1",
        "val_pitch_mae_cents", "val_pitch_median_ae_cents",
        "grad_shared_from_type", "grad_shared_from_pitch", "grad_cosine",
        "lr", "elapsed_s",
    ]

    best_f1 = -1.0
    best_epoch = -1
    patience = config.get("patience", 10)
    stale = 0
    epochs = max_epochs or config.get("epochs", 50)
    history: list[dict[str, Any]] = []

    audit = sampler_audit(train_loader, index)
    (run_dir / "sampler_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()
        t0 = time.time()
        for epoch in range(1, epochs + 1):
            train_m = run_epoch(
                model, train_loader, criterion, optimizer, device,
                grad_clip=grad_clip, label_shuffle=label_shuffle,
                predict_pitch=predict_pitch, mu_pitch=mu_pitch, sigma_pitch=sigma_pitch,
                log_grad_once=predict_pitch,
            )
            val_m = run_epoch(
                model, val_loader, criterion, None, device, grad_clip=0,
                predict_pitch=predict_pitch, mu_pitch=mu_pitch, sigma_pitch=sigma_pitch,
            )
            lr = optimizer.param_groups[0]["lr"]
            row = {
                "epoch": epoch,
                "train_loss": train_m["loss"],
                "train_type_loss": train_m.get("type_loss"),
                "train_pitch_loss": train_m.get("pitch_loss"),
                "train_acc": train_m["accuracy"],
                "train_macro_f1": train_m["macro_f1"],
                "val_loss": val_m["loss"],
                "val_type_loss": val_m.get("type_loss"),
                "val_pitch_loss": val_m.get("pitch_loss"),
                "val_acc": val_m["accuracy"],
                "val_macro_f1": val_m["macro_f1"],
                "val_pitch_mae_cents": val_m.get("pitch_mae_cents"),
                "val_pitch_median_ae_cents": val_m.get("pitch_median_ae_cents"),
                "grad_shared_from_type": train_m.get("grad_shared_from_type"),
                "grad_shared_from_pitch": train_m.get("grad_shared_from_pitch"),
                "grad_cosine": train_m.get("grad_cosine"),
                "lr": lr,
                "elapsed_s": time.time() - t0,
            }
            writer.writerow(row)
            record = {"epoch": epoch, "train": train_m, "val": val_m, "lr": lr}
            history.append(record)
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")

            extra = ""
            if predict_pitch:
                extra = (
                    f" type_l={train_m.get('type_loss', 0):.3f}"
                    f" pitch_l={train_m.get('pitch_loss', 0):.3f}"
                    f" val_pitch_mae={val_m.get('pitch_mae_cents', 0):.1f}"
                )
            print(
                f"fold {fold_index} epoch {epoch}: "
                f"train f1={train_m['macro_f1']:.3f} val f1={val_m['macro_f1']:.3f}{extra}"
            )

            if val_m["macro_f1"] > best_f1:
                best_f1 = val_m["macro_f1"]
                best_epoch = epoch
                stale = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": config,
                        "fold_index": fold_index,
                        "epoch": epoch,
                        "val_macro_f1": best_f1,
                        "n_params": n_params,
                        "seed": seed,
                        "mu_pitch": mu_pitch,
                        "sigma_pitch": sigma_pitch,
                    },
                    run_dir / "best.pt",
                )
            else:
                stale += 1
                if stale >= patience and tiny_overfit == 0:
                    print(f"early stop at epoch {epoch}")
                    break

    result = {
        "fold_index": fold_index,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "n_params": n_params,
        "run_dir": str(run_dir.relative_to(repo_root)),
        "pitch_mu_cents": mu_pitch,
        "pitch_sigma_cents": sigma_pitch,
        "history": history,
    }
    (run_dir / "train_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else args.repo_root / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    folds = list(range(5)) if args.all_folds else [args.fold if args.fold is not None else 0]
    results = []
    for fold in folds:
        print(f"\n=== fold {fold} ===")
        results.append(
            train_fold(
                config,
                fold,
                repo_root=args.repo_root,
                output_dir=args.output_dir,
                tiny_overfit=args.tiny_overfit,
                label_shuffle=args.label_shuffle,
                max_epochs=args.max_epochs,
            )
        )

    summary = {
        "experiment": config["experiment"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "folds": results,
        "mean_val_macro_f1": float(np.mean([r["best_val_macro_f1"] for r in results])),
        "std_val_macro_f1": float(np.std([r["best_val_macro_f1"] for r in results])),
    }
    out = args.output_dir / config["experiment"] / "cv_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "folds"}, indent=2))


if __name__ == "__main__":
    main()
