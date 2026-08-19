"""Step 22 sections 9-10: a small 1D CNN trained directly on the 64-point
normalized contour -- input [B, 1, 64] (q(x) only) or, as ONE controlled
ablation, [B, 2, 64] (q(x) + normalized velocity dq/dx). Same architecture,
same training protocol, Oracle vs. CREPE; no architecture search, no audio,
no absolute duration. Grouped 5-fold recording-level splits (section 12),
train-only standardization.
"""

from __future__ import annotations

import json
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
from training.metrics import frame_metrics  # noqa: E402
from training.shape_classification.dataset import OUT_DIR, build  # noqa: E402

N_FOLDS = 5
SEED = 42
MAX_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 32
RUN_DIR = OUT_DIR / "cnn"


class ContourCNN(nn.Module):
    """3 dilated causal-free conv1d blocks + global average pool + linear
    head. ~2-3k params regardless of in_channels in {1, 2}.

    Step 25's F2: `extra_dim>0` concatenates an `extra` feature vector to the
    pooled contour representation immediately before the (now wider) linear
    head -- the only architectural change section 6's fusion condition
    allows. `extra_dim=0` (default) is byte-identical to the original
    Step 22/23 model; existing single-argument `model(x)` call sites are
    unaffected."""

    def __init__(self, in_channels: int = 1, hidden: int = 16, n_classes: int = 4, extra_dim: int = 0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=4, dilation=2), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=8, dilation=4), nn.ReLU(),
        )
        self.extra_dim = extra_dim
        self.head = nn.Linear(hidden + extra_dim, n_classes)

    def forward(self, x: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:  # x: [B, C, 64]
        h = self.net(x)
        pooled = h.mean(dim=-1)
        if self.extra_dim > 0:
            pooled = torch.cat([pooled, extra], dim=-1)
        return self.head(pooled)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_xy(records: list[dict], source_key: str, input_mode: str, recording_ids: set[str]) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for r in records:
        if r["recording_id"] not in recording_ids or r[source_key] is None:
            continue
        q = r[source_key]["q"]
        if input_mode == "shape_only":
            X.append(q[None, :])
        else:
            X.append(np.stack([q, r[source_key]["v"]]))
        y.append(r["canonical_type"])
    if not X:
        n_ch = 1 if input_mode == "shape_only" else 2
        return np.zeros((0, n_ch, 64)), np.array([], dtype=np.int64)
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)


def standardize(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (X - mu[None, :, None]) / sigma[None, :, None]


def channel_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=(0, 2))
    sigma = X.std(axis=(0, 2))
    sigma = np.maximum(sigma, 1e-6)
    return mu, sigma


def run_fold(records, source_key: str, input_mode: str, fold_index: int, *, save: bool = True) -> dict[str, Any]:
    torch.manual_seed(SEED + fold_index)
    np.random.seed(SEED + fold_index)
    device = get_device()

    manifest = load_kfold_manifest(REPO_ROOT)
    split = build_fold_split(manifest, fold_index, seed=SEED)
    X_train, y_train = build_xy(records, source_key, input_mode, set(split.train_recording_ids))
    X_val, y_val = build_xy(records, source_key, input_mode, set(split.val_recording_ids))
    X_test, y_test = build_xy(records, source_key, input_mode, set(split.test_recording_ids))

    test_ids = set(split.test_recording_ids)
    test_selected = [r for r in records if r["recording_id"] in test_ids and r[source_key] is not None]
    test_duration_s = [r["duration_s"] for r in test_selected]
    test_abs_span_cents = [abs(r[source_key]["span_cents"]) for r in test_selected]
    test_primitive_id = [r["primitive_id"] for r in test_selected]

    mu, sigma = channel_stats(X_train)
    X_train = standardize(X_train, mu, sigma)
    X_val = standardize(X_val, mu, sigma)
    X_test = standardize(X_test, mu, sigma)

    in_channels = 1 if input_mode == "shape_only" else 2
    model = ContourCNN(in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    Xtr = torch.from_numpy(X_train).to(device); ytr = torch.from_numpy(y_train).to(device)
    Xva = torch.from_numpy(X_val).to(device)
    Xte = torch.from_numpy(X_test).to(device)

    n = len(Xtr)
    best_f1, best_epoch, stale, best_state = -1.0, -1, 0, None
    rng = np.random.default_rng(SEED + fold_index)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
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
            val_logits = model(Xva)
            val_pred = val_logits.argmax(dim=-1).cpu().numpy()
        val_f1 = frame_metrics(val_pred, y_val)["macro_f1"]

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
        test_pred = model(Xte).argmax(dim=-1).cpu().numpy()
    test_metrics = frame_metrics(test_pred, y_test)

    if save:
        run_dir = RUN_DIR / source_key / input_mode / f"fold_{fold_index}"
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": best_state, "mu": mu, "sigma": sigma,
                    "best_epoch": best_epoch, "in_channels": in_channels}, run_dir / "best.pt")

    return {
        "fold": fold_index, "best_epoch": best_epoch, "val_macro_f1": best_f1,
        "test_metrics": test_metrics, "n_train": n, "n_val": len(Xva), "n_test": len(Xte),
        "n_params": count_params(model),
        "test_pred": test_pred.tolist(), "test_true": y_test.tolist(),
        "test_recording_ids": [r["recording_id"] for r in test_selected],
        "test_duration_s": test_duration_s,
        "test_abs_span_cents": test_abs_span_cents,
        "test_primitive_id": test_primitive_id,
    }


def run_condition(records, source_key: str, input_mode: str) -> dict[str, Any]:
    fold_results = [run_fold(records, source_key, input_mode, f) for f in range(N_FOLDS)]
    pooled_pred = np.concatenate([np.array(f["test_pred"]) for f in fold_results])
    pooled_true = np.concatenate([np.array(f["test_true"]) for f in fold_results])
    pooled = frame_metrics(pooled_pred, pooled_true)
    per_fold_f1 = {f["fold"]: f["test_metrics"]["macro_f1"] for f in fold_results}
    return {
        "source": source_key, "input_mode": input_mode,
        "pooled": pooled, "per_fold_macro_f1": per_fold_f1,
        "grouped_mean_macro_f1": float(np.mean(list(per_fold_f1.values()))),
        "grouped_std_macro_f1": float(np.std(list(per_fold_f1.values()))),
        "folds": fold_results,
    }


def _print(result: dict) -> None:
    p = result["pooled"]
    row = " ".join(f"{p['per_class'][f'T{i}']['f1']:.3f}".rjust(7) for i in range(4))
    print(f"{result['source']:8s} {result['input_mode']:14s} macro_f1={p['macro_f1']:.4f} acc={p['accuracy']:.4f}  "
          f"[Fix,Cos,SlS,SlE]={row}  grouped_mean={result['grouped_mean_macro_f1']:.4f}+/-{result['grouped_std_macro_f1']:.4f}")


def main() -> None:
    records = build()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for src in ("oracle", "crepe"):
        for mode in ("shape_only", "shape_velocity"):
            key = f"{src}_{mode}"
            print(f"=== training {key} ===")
            results[key] = run_condition(records, src, mode)
            _print(results[key])

    out = {k: {kk: vv for kk, vv in v.items() if kk != "folds"} for k, v in results.items()}
    (OUT_DIR / "cnn_results.json").write_text(json.dumps(out, indent=2) + "\n")
    # folds (with per-primitive test predictions, needed for duration/span/T2-T3 slicing) saved separately
    full = {k: v for k, v in results.items()}
    with open(OUT_DIR / "cnn_results_full.json", "w") as fh:
        json.dump(full, fh, indent=2)
    print(f"\nsaved to {OUT_DIR / 'cnn_results.json'} and cnn_results_full.json")


if __name__ == "__main__":
    main()
