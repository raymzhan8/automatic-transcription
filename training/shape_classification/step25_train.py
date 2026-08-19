"""Step 25 sections 6-7: F1 (template evidence only, `Linear(4->4)`) and F2
(Step 22/23's `ContourCNN` + Step 25's normalized template evidence `z`,
concatenated immediately before the final linear layer). Same balanced-
sampler / unweighted-CE / optimizer / epoch-budget / patience / grouped-
fold protocol as Step 23's B1 -- F0 itself is just `step23_train.run_condition`
with `balancing="sampler"`, reused unchanged, not reimplemented here.
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
from training.shape_classification.step23_train import class_weights_inverse_freq  # noqa: E402

FOUR_CLASS_NAMES = ("Fixed", "Cosine", "Sloped-start", "Sloped-end")
Z_DIM = 4


class TemplateLinear(nn.Module):
    """F1: the simplest reasonable classifier over the 4-d normalized
    template-evidence vector alone -- no hidden layer."""

    def __init__(self, in_dim: int = Z_DIM, n_classes: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)


def select_data(records: list[dict], z_lookup: dict[str, list[float]], recording_ids: set[str], *,
                 need_contour: bool, source_key: str = "crepe"):
    X_contour, Z, y, meta = [], [], [], []
    for r in records:
        if r["recording_id"] not in recording_ids:
            continue
        z = z_lookup.get(r["primitive_id"])
        if z is None:
            continue
        if need_contour:
            d = r[source_key]
            if d is None:
                continue
            X_contour.append(np.stack([d["q"], d["v"]]))
        Z.append(z)
        y.append(r["canonical_type"])
        meta.append(r)
    y_arr = np.array(y, dtype=np.int64)
    Z_arr = np.array(Z, dtype=np.float32) if Z else np.zeros((0, Z_DIM), dtype=np.float32)
    Xc_arr = np.stack(X_contour).astype(np.float32) if X_contour else np.zeros((0, 2, 64), dtype=np.float32)
    return Xc_arr, Z_arr, y_arr, meta


def _stats(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = arr.mean(axis=0)
    sigma = np.maximum(arr.std(axis=0), 1e-6)
    return mu, sigma


def run_fold(records: list[dict], z_lookup: dict[str, list[float]], fold_index: int, mode: str, *,
             save_dir: Path | None = None, manifest_name: str = "grouped_kfold_k5_seed42.json",
             source_key: str = "crepe") -> dict[str, Any]:
    assert mode in ("template_linear", "fusion")
    need_contour = mode == "fusion"
    torch.manual_seed(SEED + fold_index)
    np.random.seed(SEED + fold_index)
    device = get_device()

    manifest = load_kfold_manifest(REPO_ROOT, manifest_name)
    split = build_fold_split(manifest, fold_index, seed=SEED)
    Xc_tr, Z_tr, y_tr, _ = select_data(records, z_lookup, set(split.train_recording_ids), need_contour=need_contour, source_key=source_key)
    Xc_va, Z_va, y_va, _ = select_data(records, z_lookup, set(split.val_recording_ids), need_contour=need_contour, source_key=source_key)
    Xc_te, Z_te, y_te, meta_te = select_data(records, z_lookup, set(split.test_recording_ids), need_contour=need_contour, source_key=source_key)

    z_mu, z_sigma = _stats(Z_tr)
    Z_tr = (Z_tr - z_mu) / z_sigma
    Z_va = (Z_va - z_mu) / z_sigma
    Z_te = (Z_te - z_mu) / z_sigma

    c_mu = c_sigma = None
    if need_contour:
        c_mu, c_sigma = channel_stats(Xc_tr)
        Xc_tr = standardize(Xc_tr, c_mu, c_sigma)
        Xc_va = standardize(Xc_va, c_mu, c_sigma)
        Xc_te = standardize(Xc_te, c_mu, c_sigma)
        model: nn.Module = ContourCNN(in_channels=2, n_classes=4, extra_dim=Z_DIM).to(device)
    else:
        model = TemplateLinear().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()  # unweighted -- Step 23 B1's protocol is sampler-only balancing

    Ztr_t = torch.from_numpy(Z_tr.astype(np.float32)).to(device)
    ytr_t = torch.from_numpy(y_tr).to(device)
    Zva_t = torch.from_numpy(Z_va.astype(np.float32)).to(device)
    Zte_t = torch.from_numpy(Z_te.astype(np.float32)).to(device)
    Xctr_t = torch.from_numpy(Xc_tr).to(device) if need_contour else None
    Xcva_t = torch.from_numpy(Xc_va).to(device) if need_contour else None
    Xcte_t = torch.from_numpy(Xc_te).to(device) if need_contour else None

    def fwd(model, xc, z):
        return model(xc, z) if need_contour else model(z)

    n = len(y_tr)
    per_class_w = class_weights_inverse_freq(y_tr, 4)
    sample_weights = per_class_w[y_tr]
    probs = sample_weights / sample_weights.sum()
    torch_gen = torch.Generator().manual_seed(SEED + fold_index)

    best_f1, best_epoch, stale, best_state = -1.0, -1, 0, None
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.multinomial(torch.from_numpy(probs), num_samples=n, replacement=True, generator=torch_gen).numpy()
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            xc_b = Xctr_t[idx] if need_contour else None
            logits = fwd(model, xc_b, Ztr_t[idx])
            loss = criterion(logits, ytr_t[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = fwd(model, Xcva_t, Zva_t).argmax(dim=-1).cpu().numpy()
        val_f1 = eval_metrics(val_pred, y_va, FOUR_CLASS_NAMES)["macro_f1"]
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
        test_logits = fwd(model, Xcte_t, Zte_t)
        test_probs = torch.softmax(test_logits, dim=-1).cpu().numpy()
        test_pred = test_logits.argmax(dim=-1).cpu().numpy()
        z_zeroed_pred = None
        if need_contour:
            z_zeroed_logits = model(Xcte_t, torch.zeros_like(Zte_t))
            z_zeroed_pred = z_zeroed_logits.argmax(dim=-1).cpu().numpy().tolist()

    test_metrics = eval_metrics(test_pred, y_te, FOUR_CLASS_NAMES)
    head = model.head if hasattr(model, "head") else model.linear
    head_weight = head.weight.detach().cpu().numpy().tolist()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        state = {"model_state": best_state, "z_mu": z_mu, "z_sigma": z_sigma, "best_epoch": best_epoch}
        if need_contour:
            state.update({"c_mu": c_mu, "c_sigma": c_sigma})
        torch.save(state, save_dir / "best.pt")

    return {
        "fold": fold_index, "mode": mode, "best_epoch": best_epoch, "val_macro_f1": best_f1,
        "test_metrics": test_metrics, "n_train": n, "n_test": len(y_te), "n_params": count_params(model),
        "test_pred": test_pred.tolist(), "test_true": y_te.tolist(), "test_probs": test_probs.tolist(),
        "test_recording_ids": [r["recording_id"] for r in meta_te],
        "test_primitive_id": [r["primitive_id"] for r in meta_te],
        "z_zeroed_test_pred": z_zeroed_pred, "head_weight": head_weight,
    }


def run_condition(records: list[dict], z_lookup: dict[str, list[float]], mode: str, *,
                   save_root: Path | None = None, n_folds: int = N_FOLDS, source_key: str = "crepe") -> dict[str, Any]:
    folds = []
    for f in range(n_folds):
        save_dir = (save_root / f"fold_{f}") if save_root is not None else None
        folds.append(run_fold(records, z_lookup, f, mode, save_dir=save_dir, source_key=source_key))
    pooled_pred = np.concatenate([np.array(r["test_pred"]) for r in folds])
    pooled_true = np.concatenate([np.array(r["test_true"]) for r in folds])
    pooled = eval_metrics(pooled_pred, pooled_true, FOUR_CLASS_NAMES)
    per_fold_f1 = {r["fold"]: r["test_metrics"]["macro_f1"] for r in folds}
    return {
        "mode": mode, "pooled": pooled, "per_fold_macro_f1": per_fold_f1,
        "grouped_mean_macro_f1": float(np.mean(list(per_fold_f1.values()))),
        "grouped_std_macro_f1": float(np.std(list(per_fold_f1.values()))),
        "n_params": folds[0]["n_params"], "folds": folds,
    }
