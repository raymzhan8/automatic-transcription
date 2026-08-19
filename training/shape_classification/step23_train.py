"""Step 23: generic grouped-5-fold ContourCNN training/eval, reusing Step
22's frozen representation (CREPE, GT boundaries, N=64 phase grid,
q(x)+dq/dx) and frozen ContourCNN architecture/optimizer/budget exactly.
The only experimental variable here is (a) which canonical_type subset is
being classified (4-class main / 2-class T2-vs-T3 / 3-class bend-only) and
(b) how class imbalance is handled during TRAINING ("none" / "sampler" /
"weighted_ce") -- never both at once, never combined with any change to the
pitch representation, architecture, optimizer, LR, epoch budget, patience,
or fold manifest, all inherited unchanged from `cnn_model.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.folds import build_fold_split, load_kfold_manifest  # noqa: E402
from training.shape_classification.cnn_model import (  # noqa: E402
    BATCH_SIZE, MAX_EPOCHS, N_FOLDS, PATIENCE, SEED, ContourCNN, channel_stats, count_params, get_device, standardize,
)
from training.shape_classification.metrics_utils import eval_metrics  # noqa: E402

BalancingMode = str  # "none" | "sampler" | "weighted_ce"


def select_and_remap(
    records: list[dict], source_key: str, input_mode: str, recording_ids: set[str], class_ids: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    id_to_new = {c: i for i, c in enumerate(class_ids)}
    X, y, meta = [], [], []
    for r in records:
        if r["recording_id"] not in recording_ids or r["canonical_type"] not in id_to_new or r[source_key] is None:
            continue
        q = r[source_key]["q"]
        x = q[None, :] if input_mode == "shape_only" else np.stack([q, r[source_key]["v"]])
        X.append(x)
        y.append(id_to_new[r["canonical_type"]])
        meta.append(r)
    if not X:
        n_ch = 1 if input_mode == "shape_only" else 2
        return np.zeros((0, n_ch, 64), dtype=np.float32), np.array([], dtype=np.int64), []
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64), meta


def class_weights_inverse_freq(y_train: np.ndarray, n_classes: int) -> np.ndarray:
    """weight_c ~ 1/n_c, rescaled so the MEAN class weight is 1 (section 3's
    B2 rule -- no exponent search, no alternative formula)."""
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = 1.0 / counts
    w *= n_classes / w.sum()
    return w


def run_fold(
    records: list[dict], source_key: str, input_mode: str, fold_index: int, class_ids: tuple[int, ...],
    label_names: tuple[str, ...], *, balancing: BalancingMode = "none", save_dir: Path | None = None,
    manifest_name: str = "grouped_kfold_k5_seed42.json",
) -> dict[str, Any]:
    torch.manual_seed(SEED + fold_index)
    np.random.seed(SEED + fold_index)
    device = get_device()
    n_classes = len(class_ids)

    manifest = load_kfold_manifest(REPO_ROOT, manifest_name)
    split = build_fold_split(manifest, fold_index, seed=SEED)
    X_train, y_train, _ = select_and_remap(records, source_key, input_mode, set(split.train_recording_ids), class_ids)
    X_val, y_val, _ = select_and_remap(records, source_key, input_mode, set(split.val_recording_ids), class_ids)
    X_test, y_test, meta_test = select_and_remap(records, source_key, input_mode, set(split.test_recording_ids), class_ids)

    mu, sigma = channel_stats(X_train)
    X_train = standardize(X_train, mu, sigma)
    X_val = standardize(X_val, mu, sigma)
    X_test = standardize(X_test, mu, sigma)

    train_class_weights = class_weights_inverse_freq(y_train, n_classes)
    train_class_counts = np.bincount(y_train, minlength=n_classes).tolist()

    in_channels = 1 if input_mode == "shape_only" else 2
    model = ContourCNN(in_channels=in_channels, n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    if balancing == "weighted_ce":
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(train_class_weights, dtype=torch.float32, device=device))
    else:
        criterion = nn.CrossEntropyLoss()

    Xtr = torch.from_numpy(X_train).to(device); ytr = torch.from_numpy(y_train).to(device)
    Xva = torch.from_numpy(X_val).to(device)
    Xte = torch.from_numpy(X_test).to(device)

    n = len(Xtr)
    # section 3's B1: sample-weight per training example inversely
    # proportional to its TRAIN class frequency -- balanced exposure only,
    # nothing else changes.
    sample_weights = None
    if balancing == "sampler":
        per_class_w = class_weights_inverse_freq(y_train, n_classes)
        sample_weights = per_class_w[y_train]

    best_f1, best_epoch, stale, best_state = -1.0, -1, 0, None
    rng = np.random.default_rng(SEED + fold_index)
    torch_gen = torch.Generator().manual_seed(SEED + fold_index)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        if balancing == "sampler":
            probs = sample_weights / sample_weights.sum()
            perm = torch.multinomial(torch.from_numpy(probs), num_samples=n, replacement=True, generator=torch_gen).numpy()
        else:
            perm = rng.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            xb, yb = Xtr[idx], ytr[idx]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xva).argmax(dim=-1).cpu().numpy()
        val_f1 = eval_metrics(val_pred, y_val, label_names)["macro_f1"]

        if val_f1 > best_f1:
            best_f1, best_epoch, stale = val_f1, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(Xte)
        test_probs = torch.softmax(test_logits, dim=-1).cpu().numpy()
        test_pred = test_logits.argmax(dim=-1).cpu().numpy()
    test_metrics = eval_metrics(test_pred, y_test, label_names)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": best_state, "mu": mu, "sigma": sigma, "best_epoch": best_epoch,
                    "in_channels": in_channels, "n_classes": n_classes}, save_dir / "best.pt")

    return {
        "fold": fold_index, "best_epoch": best_epoch, "val_macro_f1": best_f1,
        "test_metrics": test_metrics, "n_train": n, "n_val": len(Xva), "n_test": len(Xte),
        "n_params": count_params(model), "train_class_counts": train_class_counts,
        "train_class_weights": train_class_weights.tolist(),
        "test_pred": test_pred.tolist(), "test_true": y_test.tolist(),
        "test_probs": test_probs.tolist(),
        "test_recording_ids": [r["recording_id"] for r in meta_test],
        "test_primitive_id": [r["primitive_id"] for r in meta_test],
    }


def run_condition(
    records: list[dict], source_key: str, input_mode: str, class_ids: tuple[int, ...], label_names: tuple[str, ...],
    *, balancing: BalancingMode = "none", save_root: Path | None = None, n_folds: int = N_FOLDS,
) -> dict[str, Any]:
    folds = []
    for f in range(n_folds):
        save_dir = (save_root / f"fold_{f}") if save_root is not None else None
        folds.append(run_fold(records, source_key, input_mode, f, class_ids, label_names,
                               balancing=balancing, save_dir=save_dir))
    pooled_pred = np.concatenate([np.array(r["test_pred"]) for r in folds])
    pooled_true = np.concatenate([np.array(r["test_true"]) for r in folds])
    pooled = eval_metrics(pooled_pred, pooled_true, label_names)
    per_fold_f1 = {r["fold"]: r["test_metrics"]["macro_f1"] for r in folds}
    return {
        "source": source_key, "input_mode": input_mode, "balancing": balancing, "label_names": list(label_names),
        "pooled": pooled, "per_fold_macro_f1": per_fold_f1,
        "grouped_mean_macro_f1": float(np.mean(list(per_fold_f1.values()))),
        "grouped_std_macro_f1": float(np.std(list(per_fold_f1.values()))),
        "folds": folds,
    }
