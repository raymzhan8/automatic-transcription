"""Step 29: S2 trainer (BiGRU context model, S2-context and S2-center-only
conditions). Reuses Step 28's exact triplet construction (`build_triplet_arrays`,
`_channel_stats_contour`/`_standardize_contour`/`_audio_stats`/`_standardize_audio`)
unchanged -- same missing-neighbor zero+mask mechanism, same TRAIN-only
normalization. Same grouped-fold protocol, class-balanced sampler, unweighted
CE, optimizer/epoch budget/patience/seed as Steps 23-28. Encoders trainable
end-to-end, matching Step 28 C1 (section 10's explicit instruction to match,
not diverge)."""

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
from training.shape_classification.cnn_model import BATCH_SIZE, MAX_EPOCHS, PATIENCE, SEED, get_device  # noqa: E402
from training.shape_classification.metrics_utils import eval_metrics  # noqa: E402
from training.shape_classification.step23_train import class_weights_inverse_freq  # noqa: E402
from training.shape_classification.step28_train import (  # noqa: E402
    _audio_stats, _channel_stats_contour, _standardize_audio, _standardize_contour, build_triplet_arrays,
)
from training.shape_classification.step29_model import BiGRUContextModel, count_params  # noqa: E402

FOUR_CLASS_NAMES = ("Fixed", "Cosine", "Sloped-start", "Sloped-end")
N_FOLDS = 5


def _standardize_split(d: dict, c_mu, c_sigma, a_mu, a_sigma) -> dict:
    out = dict(d)
    for key in ("center_c", "prev_c", "next_c"):
        out[key] = _standardize_contour(d[key], c_mu, c_sigma)
    for key in ("center_a", "prev_a", "next_a"):
        out[key] = _standardize_audio(d[key], a_mu, a_sigma)
    return out


def _to_device(d: dict, device) -> dict:
    t = {}
    for key in ("center_c", "prev_c", "next_c"):
        t[key] = torch.from_numpy(d[key]).to(device)
    for key in ("center_a", "prev_a", "next_a"):
        t[key] = torch.from_numpy(d[key]).unsqueeze(1).to(device)
    for key in ("prev_m", "next_m"):
        t[key] = torch.from_numpy(d[key]).to(device)
    t["y"] = torch.from_numpy(d["y"]).to(device)
    return t


def _sampler_perm(y_train: np.ndarray, n_classes: int, n: int, torch_gen: torch.Generator) -> np.ndarray:
    per_class_w = class_weights_inverse_freq(y_train, n_classes)
    sample_weights = per_class_w[y_train]
    probs = sample_weights / sample_weights.sum()
    return torch.multinomial(torch.from_numpy(probs), num_samples=n, replacement=True, generator=torch_gen).numpy()


def _forward(model: BiGRUContextModel, t: dict, idx=None, *, center_only: bool = False, swap: bool = False):
    def sel(k):
        return t[k][idx] if idx is not None else t[k]
    prev_c, prev_a, prev_m = sel("prev_c"), sel("prev_a"), sel("prev_m")
    next_c, next_a, next_m = sel("next_c"), sel("next_a"), sel("next_m")
    if swap:
        prev_c, next_c = next_c, prev_c
        prev_a, next_a = next_a, prev_a
        prev_m, next_m = next_m, prev_m
    return model(sel("center_c"), sel("center_a"), prev_c, prev_a, prev_m, next_c, next_a, next_m,
                 center_only=center_only)


def run_fold(
    records: list[dict], audio_lookup: dict[str, np.ndarray], neighbor_map: dict, condition: str, fold_index: int, *,
    save_dir: Path | None = None, manifest_name: str = "grouped_kfold_k5_seed42.json",
) -> dict[str, Any]:
    assert condition in ("S2_center_only", "S2_context")
    center_only = condition == "S2_center_only"
    torch.manual_seed(SEED + fold_index)
    np.random.seed(SEED + fold_index)
    device = get_device()
    n_classes = 4

    manifest = load_kfold_manifest(REPO_ROOT, manifest_name)
    split = build_fold_split(manifest, fold_index, seed=SEED)
    d_tr = build_triplet_arrays(records, audio_lookup, neighbor_map, set(split.train_recording_ids))
    d_va = build_triplet_arrays(records, audio_lookup, neighbor_map, set(split.val_recording_ids))
    d_te = build_triplet_arrays(records, audio_lookup, neighbor_map, set(split.test_recording_ids))

    c_mu, c_sigma = _channel_stats_contour(d_tr["center_c"])
    a_mu, a_sigma = _audio_stats(d_tr["center_a"])
    d_tr = _standardize_split(d_tr, c_mu, c_sigma, a_mu, a_sigma)
    d_va = _standardize_split(d_va, c_mu, c_sigma, a_mu, a_sigma)
    d_te = _standardize_split(d_te, c_mu, c_sigma, a_mu, a_sigma)

    t_tr, t_va, t_te = _to_device(d_tr, device), _to_device(d_va, device), _to_device(d_te, device)
    ytr = d_tr["y"]

    model = BiGRUContextModel(n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    n = len(ytr)
    torch_gen = torch.Generator().manual_seed(SEED + fold_index)
    best_f1, best_epoch, stale, best_state = -1.0, -1, 0, None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = _sampler_perm(ytr, n_classes, n, torch_gen)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits = _forward(model, t_tr, idx, center_only=center_only)
            loss = criterion(logits, t_tr["y"][idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = _forward(model, t_va, center_only=center_only).argmax(dim=-1).cpu().numpy()
        val_f1 = eval_metrics(val_pred, d_va["y"], FOUR_CLASS_NAMES)["macro_f1"]
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
        test_logits = _forward(model, t_te, center_only=center_only)
        test_pred = test_logits.argmax(dim=-1).cpu().numpy()
        test_probs = torch.softmax(test_logits, dim=-1).cpu().numpy()
        # section 19: temporal-order swap diagnostic (S2-context only), same weights, no retraining
        swap_pred = None
        if not center_only:
            swap_pred = _forward(model, t_te, center_only=False, swap=True).argmax(dim=-1).cpu().numpy().tolist()
    test_metrics = eval_metrics(test_pred, d_te["y"], FOUR_CLASS_NAMES)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": best_state, "c_mu": c_mu, "c_sigma": c_sigma, "a_mu": a_mu, "a_sigma": a_sigma,
                    "best_epoch": best_epoch}, save_dir / "best.pt")

    return {
        "fold": fold_index, "condition": condition, "best_epoch": best_epoch, "val_macro_f1": best_f1,
        "test_metrics": test_metrics, "n_train": n, "n_val": len(d_va["y"]), "n_test": len(d_te["y"]),
        "n_params": count_params(model),
        "test_pred": test_pred.tolist(), "test_true": d_te["y"].tolist(), "test_probs": test_probs.tolist(),
        "test_recording_ids": [r["recording_id"] for r in d_te["meta"]],
        "test_primitive_id": [r["primitive_id"] for r in d_te["meta"]],
        "swap_test_pred": swap_pred,
    }


def run_condition(
    records: list[dict], audio_lookup: dict[str, np.ndarray], neighbor_map: dict, condition: str, *,
    save_root: Path | None = None, n_folds: int = N_FOLDS,
) -> dict[str, Any]:
    folds = []
    for f in range(n_folds):
        save_dir = (save_root / f"fold_{f}") if save_root is not None else None
        folds.append(run_fold(records, audio_lookup, neighbor_map, condition, f, save_dir=save_dir))
    pooled_pred = np.concatenate([np.array(f["test_pred"]) for f in folds])
    pooled_true = np.concatenate([np.array(f["test_true"]) for f in folds])
    pooled = eval_metrics(pooled_pred, pooled_true, FOUR_CLASS_NAMES)
    per_fold_f1 = {f["fold"]: f["test_metrics"]["macro_f1"] for f in folds}
    return {
        "condition": condition, "pooled": pooled, "per_fold_macro_f1": per_fold_f1,
        "grouped_mean_macro_f1": float(np.mean(list(per_fold_f1.values()))),
        "grouped_std_macro_f1": float(np.std(list(per_fold_f1.values()))),
        "n_params": folds[0]["n_params"], "folds": folds,
    }
