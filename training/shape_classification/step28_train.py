"""Step 28: C1/C2 trainers. Same protocol as Step 26/27: grouped 5-fold,
TRAIN-only per-bin audio normalization + per-channel contour normalization,
class-balanced sampler, unweighted CE, Adam lr=1e-3/wd=1e-4, batch=32,
max_epochs=100/patience=15, seed=42+fold. Neighbor construction only ever
pairs primitives from the SAME recording (`step28_neighbors.build_neighbor_map`),
so it can never pull data across a grouped-fold split boundary -- verified
directly in step28_experiments.py's leakage check, not just asserted here.
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
from training.shape_classification.cnn_model import BATCH_SIZE, MAX_EPOCHS, PATIENCE, SEED, get_device  # noqa: E402
from training.shape_classification.metrics_utils import eval_metrics  # noqa: E402
from training.shape_classification.step23_train import class_weights_inverse_freq  # noqa: E402
from training.shape_classification.step28_model import C1ContextModel, C2ContextModel, count_params  # noqa: E402
from training.shape_classification.step28_neighbors import build_neighbor_map  # noqa: E402

FOUR_CLASS_NAMES = ("Fixed", "Cosine", "Sloped-start", "Sloped-end")
N_FOLDS = 5
N_BINS = 360
N_PHASE_POINTS = 64


def _contour_xy(r: dict) -> np.ndarray | None:
    d = r["crepe"]
    if d is None:
        return None
    return np.stack([d["q"], d["v"]]).astype(np.float32)


def build_triplet_arrays(
    records: list[dict], audio_lookup: dict[str, np.ndarray], neighbor_map: dict[str, dict],
    recording_ids: set[str],
) -> dict[str, np.ndarray] | dict[str, list]:
    by_pid = {r["primitive_id"]: r for r in records}
    center_c, center_a, prev_c, prev_a, prev_m, next_c, next_a, next_m, y, meta = ([] for _ in range(10))
    zero_c = np.zeros((2, N_PHASE_POINTS), dtype=np.float32)
    zero_a = np.zeros((N_BINS, N_PHASE_POINTS), dtype=np.float32)

    for r in records:
        if r["recording_id"] not in recording_ids:
            continue
        cc = _contour_xy(r)
        ca = audio_lookup.get(r["primitive_id"])
        if cc is None or ca is None:
            continue

        nb = neighbor_map[r["primitive_id"]]
        p_id, n_id = nb["prev"], nb["next"]
        pc, pa, pm = zero_c, zero_a, 0.0
        if p_id is not None and p_id in by_pid:
            prow = by_pid[p_id]
            cc_p, ca_p = _contour_xy(prow), audio_lookup.get(p_id)
            if cc_p is not None and ca_p is not None:
                pc, pa, pm = cc_p, ca_p, 1.0
        nc, na, nm = zero_c, zero_a, 0.0
        if n_id is not None and n_id in by_pid:
            nrow = by_pid[n_id]
            cc_n, ca_n = _contour_xy(nrow), audio_lookup.get(n_id)
            if cc_n is not None and ca_n is not None:
                nc, na, nm = cc_n, ca_n, 1.0

        center_c.append(cc); center_a.append(ca)
        prev_c.append(pc); prev_a.append(pa); prev_m.append(pm)
        next_c.append(nc); next_a.append(na); next_m.append(nm)
        y.append(r["canonical_type"]); meta.append(r)

    if not y:
        z = lambda shape: np.zeros((0, *shape), dtype=np.float32)  # noqa: E731
        return {"center_c": z((2, N_PHASE_POINTS)), "center_a": z((N_BINS, N_PHASE_POINTS)),
                "prev_c": z((2, N_PHASE_POINTS)), "prev_a": z((N_BINS, N_PHASE_POINTS)), "prev_m": np.zeros(0, dtype=np.float32),
                "next_c": z((2, N_PHASE_POINTS)), "next_a": z((N_BINS, N_PHASE_POINTS)), "next_m": np.zeros(0, dtype=np.float32),
                "y": np.zeros(0, dtype=np.int64), "meta": []}
    return {
        "center_c": np.stack(center_c), "center_a": np.stack(center_a),
        "prev_c": np.stack(prev_c), "prev_a": np.stack(prev_a), "prev_m": np.array(prev_m, dtype=np.float32),
        "next_c": np.stack(next_c), "next_a": np.stack(next_a), "next_m": np.array(next_m, dtype=np.float32),
        "y": np.array(y, dtype=np.int64), "meta": meta,
    }


def _channel_stats_contour(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=(0, 2)); sigma = np.maximum(X.std(axis=(0, 2)), 1e-6)
    return mu, sigma


def _standardize_contour(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (X - mu[None, :, None]) / sigma[None, :, None]


def _audio_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=(0, 2)); sigma = np.maximum(X.std(axis=(0, 2)), 1e-6)
    return mu, sigma


def _standardize_audio(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (X - mu[None, :, None]) / sigma[None, :, None]


def _sampler_perm(y_train: np.ndarray, n_classes: int, n: int, torch_gen: torch.Generator) -> np.ndarray:
    per_class_w = class_weights_inverse_freq(y_train, n_classes)
    sample_weights = per_class_w[y_train]
    probs = sample_weights / sample_weights.sum()
    return torch.multinomial(torch.from_numpy(probs), num_samples=n, replacement=True, generator=torch_gen).numpy()


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


def _forward(model, condition: str, t: dict, idx=None):
    def sel(k):
        return t[k][idx] if idx is not None else t[k]
    if condition == "C1":
        return model(sel("center_c"), sel("center_a"), sel("prev_c"), sel("prev_a"), sel("prev_m"),
                     sel("next_c"), sel("next_a"), sel("next_m"))
    return model(sel("center_c"), sel("center_a"), sel("prev_c"), sel("prev_m"), sel("next_c"), sel("next_m"))


def run_fold(
    records: list[dict], audio_lookup: dict[str, np.ndarray], neighbor_map: dict, condition: str, fold_index: int, *,
    save_dir: Path | None = None, manifest_name: str = "grouped_kfold_k5_seed42.json",
) -> dict[str, Any]:
    assert condition in ("C1", "C2")
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

    model = (C1ContextModel if condition == "C1" else C2ContextModel)(n_classes=n_classes).to(device)
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
            logits = _forward(model, condition, t_tr, idx)
            loss = criterion(logits, t_tr["y"][idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = _forward(model, condition, t_va).argmax(dim=-1).cpu().numpy()
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
        test_logits = _forward(model, condition, t_te)
        test_pred = test_logits.argmax(dim=-1).cpu().numpy()
        test_probs = torch.softmax(test_logits, dim=-1).cpu().numpy()
        # section 15: context-position ablation, same trained weights, no retraining
        prev_only_kwargs = dict(t_te); prev_only_kwargs["next_m"] = torch.zeros_like(t_te["next_m"])
        next_only_kwargs = dict(t_te); next_only_kwargs["prev_m"] = torch.zeros_like(t_te["prev_m"])
        prev_only_pred = _forward(model, condition, prev_only_kwargs).argmax(dim=-1).cpu().numpy().tolist()
        next_only_pred = _forward(model, condition, next_only_kwargs).argmax(dim=-1).cpu().numpy().tolist()
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
        "prev_only_test_pred": prev_only_pred, "next_only_test_pred": next_only_pred,
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
