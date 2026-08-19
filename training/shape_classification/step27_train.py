"""Step 27: L1 trainer. Identical protocol to Step 26's L0/A2 fusion trainer
(`step26_train.py::run_fold_fusion`) -- same data selection, same TRAIN-only
per-bin audio normalization, same class-balanced sampler, same unweighted
CE, same optimizer/epoch budget/patience/seed/grouped folds, same encoders
trainable end-to-end. The only difference is the model class."""

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
from training.shape_classification.step26_train import (  # noqa: E402
    _channel_stats_contour, _standardize_contour, audio_channel_stats, select_audio_and_contour, standardize_audio,
)
from training.shape_classification.step27_model import NonlinearFusionModel, count_params  # noqa: E402

FOUR_CLASS_NAMES = ("Fixed", "Cosine", "Sloped-start", "Sloped-end")
N_FOLDS = 5


def _sampler_perm(y_train: np.ndarray, n_classes: int, n: int, torch_gen: torch.Generator) -> np.ndarray:
    per_class_w = class_weights_inverse_freq(y_train, n_classes)
    sample_weights = per_class_w[y_train]
    probs = sample_weights / sample_weights.sum()
    return torch.multinomial(torch.from_numpy(probs), num_samples=n, replacement=True, generator=torch_gen).numpy()


def run_fold(
    records: list[dict], audio_lookup: dict[str, np.ndarray], fold_index: int, *,
    contour_source: str = "crepe", save_dir: Path | None = None,
    manifest_name: str = "grouped_kfold_k5_seed42.json",
) -> dict[str, Any]:
    torch.manual_seed(SEED + fold_index)
    np.random.seed(SEED + fold_index)
    device = get_device()
    n_classes = 4

    manifest = load_kfold_manifest(REPO_ROOT, manifest_name)
    split = build_fold_split(manifest, fold_index, seed=SEED)
    Xa_tr, Xc_tr, ytr, _ = select_audio_and_contour(records, audio_lookup, set(split.train_recording_ids), contour_source=contour_source)
    Xa_va, Xc_va, yva, _ = select_audio_and_contour(records, audio_lookup, set(split.val_recording_ids), contour_source=contour_source)
    Xa_te, Xc_te, yte, meta_te = select_audio_and_contour(records, audio_lookup, set(split.test_recording_ids), contour_source=contour_source)

    a_mu, a_sigma = audio_channel_stats(Xa_tr)
    Xa_tr = standardize_audio(Xa_tr, a_mu, a_sigma)
    Xa_va = standardize_audio(Xa_va, a_mu, a_sigma)
    Xa_te = standardize_audio(Xa_te, a_mu, a_sigma)

    c_mu, c_sigma = _channel_stats_contour(Xc_tr)
    Xc_tr = _standardize_contour(Xc_tr, c_mu, c_sigma)
    Xc_va = _standardize_contour(Xc_va, c_mu, c_sigma)
    Xc_te = _standardize_contour(Xc_te, c_mu, c_sigma)

    model = NonlinearFusionModel(n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    Xatr_t = torch.from_numpy(Xa_tr).unsqueeze(1).to(device)
    Xctr_t = torch.from_numpy(Xc_tr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    Xava_t = torch.from_numpy(Xa_va).unsqueeze(1).to(device)
    Xcva_t = torch.from_numpy(Xc_va).to(device)
    Xate_t = torch.from_numpy(Xa_te).unsqueeze(1).to(device)
    Xcte_t = torch.from_numpy(Xc_te).to(device)

    n = len(ytr)
    torch_gen = torch.Generator().manual_seed(SEED + fold_index)
    best_f1, best_epoch, stale, best_state = -1.0, -1, 0, None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = _sampler_perm(ytr, n_classes, n, torch_gen)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits, _z = model(Xctr_t[idx], Xatr_t[idx])
            loss = criterion(logits, ytr_t[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(Xcva_t, Xava_t)
            val_pred = val_logits.argmax(dim=-1).cpu().numpy()
        val_f1 = eval_metrics(val_pred, yva, FOUR_CLASS_NAMES)["macro_f1"]
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
        test_logits, test_z = model(Xcte_t, Xate_t)
        test_pred = test_logits.argmax(dim=-1).cpu().numpy()
        test_probs = torch.softmax(test_logits, dim=-1).cpu().numpy()
        audio_zeroed_pred = model(Xcte_t, Xate_t, zero_audio=True)[0].argmax(dim=-1).cpu().numpy().tolist()
        pitch_zeroed_pred = model(Xcte_t, Xate_t, zero_pitch=True)[0].argmax(dim=-1).cpu().numpy().tolist()
    test_metrics = eval_metrics(test_pred, yte, FOUR_CLASS_NAMES)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": best_state, "a_mu": a_mu, "a_sigma": a_sigma,
                    "c_mu": c_mu, "c_sigma": c_sigma, "best_epoch": best_epoch}, save_dir / "best.pt")

    return {
        "fold": fold_index, "contour_source": contour_source, "best_epoch": best_epoch, "val_macro_f1": best_f1,
        "test_metrics": test_metrics, "n_train": n, "n_val": len(yva), "n_test": len(yte),
        "n_params": count_params(model),
        "test_pred": test_pred.tolist(), "test_true": yte.tolist(), "test_probs": test_probs.tolist(),
        "test_recording_ids": [r["recording_id"] for r in meta_te],
        "test_primitive_id": [r["primitive_id"] for r in meta_te],
        "audio_zeroed_test_pred": audio_zeroed_pred, "pitch_zeroed_test_pred": pitch_zeroed_pred,
        "test_z": test_z.cpu().numpy().tolist(),
    }


def run_condition(
    records: list[dict], audio_lookup: dict[str, np.ndarray], *, save_root: Path | None = None, n_folds: int = N_FOLDS,
) -> dict[str, Any]:
    folds = []
    for f in range(n_folds):
        save_dir = (save_root / f"fold_{f}") if save_root is not None else None
        folds.append(run_fold(records, audio_lookup, f, save_dir=save_dir))
    pooled_pred = np.concatenate([np.array(f["test_pred"]) for f in folds])
    pooled_true = np.concatenate([np.array(f["test_true"]) for f in folds])
    pooled = eval_metrics(pooled_pred, pooled_true, FOUR_CLASS_NAMES)
    per_fold_f1 = {f["fold"]: f["test_metrics"]["macro_f1"] for f in folds}
    return {
        "condition": "L1_nonlinear_fusion", "pooled": pooled, "per_fold_macro_f1": per_fold_f1,
        "grouped_mean_macro_f1": float(np.mean(list(per_fold_f1.values()))),
        "grouped_std_macro_f1": float(np.std(list(per_fold_f1.values()))),
        "n_params": folds[0]["n_params"], "folds": folds,
    }
