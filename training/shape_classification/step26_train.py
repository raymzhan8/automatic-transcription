"""Step 26 sections 5-10: A1 (audio-only) and A2/A4 (CREPE+audio fusion)
trainers. Same grouped-5-fold protocol, optimizer, epoch budget, patience,
and class-balanced-sampler / unweighted-CE objective as Step 23's B1 --
inherited unchanged from `cnn_model.py`/`step23_train.py`, not
reimplemented. A0 itself needs no new code: it is
`step23_train.run_condition(records, "crepe", "shape_velocity", (0,1,2,3),
FOUR_CLASS_NAMES, balancing="sampler")`, called directly from
step26_experiments.py exactly as Step 25's F0 was.

TRAIN-only normalization: acoustic patches are standardized per-frequency-
bin using mean/std computed from the fold's TRAIN primitives only (mirrors
`training/normalization.py::compute_cqt_stats`'s per-bin convention, applied
here to the primitive-patch cache rather than the framewise feature cache).
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
    BATCH_SIZE, MAX_EPOCHS, PATIENCE, SEED, get_device,
)
from training.shape_classification.metrics_utils import eval_metrics  # noqa: E402
from training.shape_classification.step23_train import class_weights_inverse_freq  # noqa: E402
from training.shape_classification.step26_model import AudioOnlyModel, FusionModel, count_params  # noqa: E402

FOUR_CLASS_NAMES = ("Fixed", "Cosine", "Sloped-start", "Sloped-end")
N_FOLDS = 5
N_BINS = 360
N_PHASE_POINTS = 64


def audio_channel_stats(patches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frequency-bin mean/std over (N, time), matching
    `training/normalization.py::compute_cqt_stats`'s convention."""
    mu = patches.mean(axis=(0, 2))
    sigma = np.maximum(patches.std(axis=(0, 2)), 1e-6)
    return mu, sigma


def standardize_audio(patches: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (patches - mu[None, :, None]) / sigma[None, :, None]


def select_audio(
    records: list[dict], audio_lookup: dict[str, np.ndarray], recording_ids: set[str],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    X, y, meta = [], [], []
    for r in records:
        if r["recording_id"] not in recording_ids:
            continue
        patch = audio_lookup.get(r["primitive_id"])
        if patch is None:
            continue
        X.append(patch)
        y.append(r["canonical_type"])
        meta.append(r)
    if not X:
        return np.zeros((0, N_BINS, N_PHASE_POINTS), dtype=np.float32), np.array([], dtype=np.int64), []
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64), meta


def select_audio_and_contour(
    records: list[dict], audio_lookup: dict[str, np.ndarray], recording_ids: set[str], *,
    contour_source: str = "crepe",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    Xa, Xc, y, meta = [], [], [], []
    for r in records:
        if r["recording_id"] not in recording_ids:
            continue
        patch = audio_lookup.get(r["primitive_id"])
        d = r[contour_source]
        if patch is None or d is None:
            continue
        Xa.append(patch)
        Xc.append(np.stack([d["q"], d["v"]]))
        y.append(r["canonical_type"])
        meta.append(r)
    if not Xa:
        return (np.zeros((0, N_BINS, N_PHASE_POINTS), dtype=np.float32), np.zeros((0, 2, N_PHASE_POINTS), dtype=np.float32),
                np.array([], dtype=np.int64), [])
    return np.stack(Xa).astype(np.float32), np.stack(Xc).astype(np.float32), np.array(y, dtype=np.int64), meta


def _channel_stats_contour(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=(0, 2))
    sigma = np.maximum(X.std(axis=(0, 2)), 1e-6)
    return mu, sigma


def _standardize_contour(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (X - mu[None, :, None]) / sigma[None, :, None]


def _sampler_perm(y_train: np.ndarray, n_classes: int, n: int, torch_gen: torch.Generator) -> np.ndarray:
    per_class_w = class_weights_inverse_freq(y_train, n_classes)
    sample_weights = per_class_w[y_train]
    probs = sample_weights / sample_weights.sum()
    return torch.multinomial(torch.from_numpy(probs), num_samples=n, replacement=True, generator=torch_gen).numpy()


def run_fold_audio_only(
    records: list[dict], audio_lookup: dict[str, np.ndarray], fold_index: int, *,
    save_dir: Path | None = None, manifest_name: str = "grouped_kfold_k5_seed42.json",
) -> dict[str, Any]:
    """A1."""
    torch.manual_seed(SEED + fold_index)
    np.random.seed(SEED + fold_index)
    device = get_device()
    n_classes = 4

    manifest = load_kfold_manifest(REPO_ROOT, manifest_name)
    split = build_fold_split(manifest, fold_index, seed=SEED)
    Xtr, ytr, _ = select_audio(records, audio_lookup, set(split.train_recording_ids))
    Xva, yva, _ = select_audio(records, audio_lookup, set(split.val_recording_ids))
    Xte, yte, meta_te = select_audio(records, audio_lookup, set(split.test_recording_ids))

    mu, sigma = audio_channel_stats(Xtr)
    Xtr = standardize_audio(Xtr, mu, sigma)
    Xva = standardize_audio(Xva, mu, sigma)
    Xte = standardize_audio(Xte, mu, sigma)

    model = AudioOnlyModel(n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    Xtr_t = torch.from_numpy(Xtr).unsqueeze(1).to(device)  # [N,1,N_BINS,64]
    ytr_t = torch.from_numpy(ytr).to(device)
    Xva_t = torch.from_numpy(Xva).unsqueeze(1).to(device)
    Xte_t = torch.from_numpy(Xte).unsqueeze(1).to(device)

    n = len(ytr)
    torch_gen = torch.Generator().manual_seed(SEED + fold_index)
    best_f1, best_epoch, stale, best_state = -1.0, -1, 0, None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = _sampler_perm(ytr, n_classes, n, torch_gen)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits = model(Xtr_t[idx])
            loss = criterion(logits, ytr_t[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xva_t).argmax(dim=-1).cpu().numpy()
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
        test_logits = model(Xte_t)
        test_pred = test_logits.argmax(dim=-1).cpu().numpy()
        test_probs = torch.softmax(test_logits, dim=-1).cpu().numpy()
    test_metrics = eval_metrics(test_pred, yte, FOUR_CLASS_NAMES)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": best_state, "mu": mu, "sigma": sigma, "best_epoch": best_epoch},
                   save_dir / "best.pt")

    return {
        "fold": fold_index, "best_epoch": best_epoch, "val_macro_f1": best_f1,
        "test_metrics": test_metrics, "n_train": n, "n_val": len(yva), "n_test": len(yte),
        "n_params": count_params(model),
        "test_pred": test_pred.tolist(), "test_true": yte.tolist(), "test_probs": test_probs.tolist(),
        "test_recording_ids": [r["recording_id"] for r in meta_te],
        "test_primitive_id": [r["primitive_id"] for r in meta_te],
    }


def run_fold_fusion(
    records: list[dict], audio_lookup: dict[str, np.ndarray], fold_index: int, *,
    contour_source: str = "crepe", save_dir: Path | None = None,
    manifest_name: str = "grouped_kfold_k5_seed42.json",
) -> dict[str, Any]:
    """A2 (contour_source='crepe') / A4 (contour_source='oracle')."""
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

    model = FusionModel(n_classes=n_classes).to(device)
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
            logits = model(Xctr_t[idx], Xatr_t[idx])
            loss = criterion(logits, ytr_t[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xcva_t, Xava_t).argmax(dim=-1).cpu().numpy()
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
        test_logits = model(Xcte_t, Xate_t)
        test_pred = test_logits.argmax(dim=-1).cpu().numpy()
        test_probs = torch.softmax(test_logits, dim=-1).cpu().numpy()
        # Section 20: same trained weights, no retraining, both zeroing ablations.
        audio_zeroed_pred = model(Xcte_t, Xate_t, zero_audio=True).argmax(dim=-1).cpu().numpy().tolist()
        pitch_zeroed_pred = model(Xcte_t, Xate_t, zero_pitch=True).argmax(dim=-1).cpu().numpy().tolist()
    test_metrics = eval_metrics(test_pred, yte, FOUR_CLASS_NAMES)
    head_weight = model.head.weight.detach().cpu().numpy()  # [4, PITCH_HIDDEN + AUDIO_HIDDEN]

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
        "head_weight": head_weight.tolist(),
    }


def run_condition_audio_only(
    records: list[dict], audio_lookup: dict[str, np.ndarray], *, save_root: Path | None = None, n_folds: int = N_FOLDS,
) -> dict[str, Any]:
    folds = []
    for f in range(n_folds):
        save_dir = (save_root / f"fold_{f}") if save_root is not None else None
        folds.append(run_fold_audio_only(records, audio_lookup, f, save_dir=save_dir))
    return _pool(folds, "A1_audio_only")


def run_condition_fusion(
    records: list[dict], audio_lookup: dict[str, np.ndarray], *, contour_source: str = "crepe",
    save_root: Path | None = None, n_folds: int = N_FOLDS,
) -> dict[str, Any]:
    folds = []
    for f in range(n_folds):
        save_dir = (save_root / f"fold_{f}") if save_root is not None else None
        folds.append(run_fold_fusion(records, audio_lookup, f, contour_source=contour_source, save_dir=save_dir))
    tag = "A2_crepe_audio" if contour_source == "crepe" else "A4_oracle_audio"
    return _pool(folds, tag)


def _pool(folds: list[dict], tag: str) -> dict[str, Any]:
    pooled_pred = np.concatenate([np.array(f["test_pred"]) for f in folds])
    pooled_true = np.concatenate([np.array(f["test_true"]) for f in folds])
    pooled = eval_metrics(pooled_pred, pooled_true, FOUR_CLASS_NAMES)
    per_fold_f1 = {f["fold"]: f["test_metrics"]["macro_f1"] for f in folds}
    return {
        "condition": tag, "pooled": pooled, "per_fold_macro_f1": per_fold_f1,
        "grouped_mean_macro_f1": float(np.mean(list(per_fold_f1.values()))),
        "grouped_std_macro_f1": float(np.std(list(per_fold_f1.values()))),
        "n_params": folds[0]["n_params"], "folds": folds,
    }
