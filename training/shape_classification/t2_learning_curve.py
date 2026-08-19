"""Does T2 (Sloped-start) look data-starved, or hard regardless of volume?

Controlled ablation: identical to Step 23 B1 / Step 25-26's A0 (frozen CREPE
q(x)+dq/dx, ContourCNN, class-balanced sampler, unweighted CE, same
optimizer/epoch budget/patience/seed/grouped folds) -- the ONLY variable is
how many DISTINCT T2 training primitives are available per fold, subsampled
to a fixed fraction {0.25, 0.5, 0.75, 1.0} of what's actually there.

Every other class's training data is left untouched at 100%, and val/test
are NEVER touched -- only the T2 slice of TRAIN is subsampled, deterministically
(fixed seed per fold x fraction, no cherry-picking).

This deliberately does NOT bypass class-balanced sampling: the sampler's
inverse-frequency weighting is recomputed on the post-truncation class
counts, exactly reproducing what "if we had only collected fraction*100% of
T2 examples in the first place" would actually look like, exposure-wise --
distinguishing "the model never sees T2 enough" (which balancing already
addresses at any fraction) from "the model doesn't have enough DISTINCT T2
examples to generalize from" (which shrinks by construction as fraction
drops).

If T2 F1 rises substantially and monotonically with fraction: data-starved,
more T2 collection would plausibly help.
If T2 F1 is roughly flat across fractions: not primarily a volume problem --
T2 is hard given any amount of this kind of evidence, echoing Step 23's own
finding that balanced exposure alone didn't fix it.
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
from training.shape_classification.cnn_model import (  # noqa: E402
    BATCH_SIZE, MAX_EPOCHS, N_FOLDS, PATIENCE, SEED, ContourCNN, channel_stats, count_params, get_device, standardize,
)
from training.shape_classification.dataset import OUT_DIR, build  # noqa: E402
from training.shape_classification.metrics_utils import eval_metrics  # noqa: E402
from training.shape_classification.step23_train import class_weights_inverse_freq, select_and_remap  # noqa: E402

FOUR_CLASS_NAMES = ("Fixed", "Cosine", "Sloped-start", "Sloped-end")
T2_CLASS_ID = 2  # index into FOUR_CLASS_NAMES / class_ids=(0,1,2,3), unaffected by remap
FRACTIONS = (0.25, 0.5, 0.75, 1.0)
OUT_PATH = OUT_DIR / "t2_learning_curve.json"


def subsample_class(
    X: np.ndarray, y: np.ndarray, meta: list[dict], class_id: int, fraction: float, seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict], int, int]:
    """Keep ALL rows not of `class_id`; keep a deterministic `fraction` of
    `class_id`'s rows. Returns (X, y, meta, n_before, n_after)."""
    is_target = y == class_id
    target_idx = np.where(is_target)[0]
    other_idx = np.where(~is_target)[0]
    n_before = len(target_idx)
    if fraction >= 1.0:
        keep_idx = np.concatenate([other_idx, target_idx])
    else:
        rng = np.random.default_rng(seed)
        n_keep = max(1, int(round(n_before * fraction)))
        keep_target = rng.choice(target_idx, size=n_keep, replace=False)
        keep_idx = np.concatenate([other_idx, keep_target])
    keep_idx = np.sort(keep_idx)
    n_after = int(is_target[keep_idx].sum())
    return X[keep_idx], y[keep_idx], [meta[i] for i in keep_idx], n_before, n_after


def run_fold(
    records: list[dict], fold_index: int, fraction: float, *, manifest_name: str = "grouped_kfold_k5_seed42.json",
) -> dict[str, Any]:
    torch.manual_seed(SEED + fold_index)
    np.random.seed(SEED + fold_index)
    device = get_device()
    class_ids = (0, 1, 2, 3)
    n_classes = 4

    manifest = load_kfold_manifest(REPO_ROOT, manifest_name)
    split = build_fold_split(manifest, fold_index, seed=SEED)
    X_train, y_train, meta_train = select_and_remap(records, "crepe", "shape_velocity", set(split.train_recording_ids), class_ids)
    X_val, y_val, _ = select_and_remap(records, "crepe", "shape_velocity", set(split.val_recording_ids), class_ids)
    X_test, y_test, meta_test = select_and_remap(records, "crepe", "shape_velocity", set(split.test_recording_ids), class_ids)

    X_train, y_train, meta_train, n_t2_before, n_t2_after = subsample_class(
        X_train, y_train, meta_train, T2_CLASS_ID, fraction, seed=SEED + fold_index + int(fraction * 1000),
    )

    mu, sigma = channel_stats(X_train)
    X_train = standardize(X_train, mu, sigma)
    X_val = standardize(X_val, mu, sigma)
    X_test = standardize(X_test, mu, sigma)

    model = ContourCNN(in_channels=2, n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    Xtr = torch.from_numpy(X_train).to(device); ytr = torch.from_numpy(y_train).to(device)
    Xva = torch.from_numpy(X_val).to(device)
    Xte = torch.from_numpy(X_test).to(device)

    n = len(Xtr)
    per_class_w = class_weights_inverse_freq(y_train, n_classes)
    sample_weights = per_class_w[y_train]
    torch_gen = torch.Generator().manual_seed(SEED + fold_index)

    best_f1, best_epoch, stale, best_state = -1.0, -1, 0, None
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        probs = sample_weights / sample_weights.sum()
        perm = torch.multinomial(torch.from_numpy(probs), num_samples=n, replacement=True, generator=torch_gen).numpy()
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
        val_f1 = eval_metrics(val_pred, y_val, FOUR_CLASS_NAMES)["macro_f1"]
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
    test_metrics = eval_metrics(test_pred, y_test, FOUR_CLASS_NAMES)

    return {
        "fold": fold_index, "fraction": fraction, "best_epoch": best_epoch, "val_macro_f1": best_f1,
        "test_metrics": test_metrics, "n_train": n, "n_t2_before_subsample": n_t2_before, "n_t2_after_subsample": n_t2_after,
        "n_params": count_params(model),
        "test_pred": test_pred.tolist(), "test_true": y_test.tolist(),
        "test_recording_ids": [r["recording_id"] for r in meta_test],
    }


def run_fraction(records: list[dict], fraction: float) -> dict[str, Any]:
    folds = [run_fold(records, f, fraction) for f in range(N_FOLDS)]
    pooled_pred = np.concatenate([np.array(f["test_pred"]) for f in folds])
    pooled_true = np.concatenate([np.array(f["test_true"]) for f in folds])
    pooled = eval_metrics(pooled_pred, pooled_true, FOUR_CLASS_NAMES)
    per_fold_f1 = {f["fold"]: f["test_metrics"]["macro_f1"] for f in folds}
    per_fold_t2_f1 = {f["fold"]: f["test_metrics"]["per_class"]["Sloped-start"]["f1"] for f in folds}
    return {
        "fraction": fraction, "pooled": pooled, "per_fold_macro_f1": per_fold_f1, "per_fold_t2_f1": per_fold_t2_f1,
        "grouped_mean_macro_f1": float(np.mean(list(per_fold_f1.values()))),
        "n_t2_train_by_fold": {f["fold"]: {"before": f["n_t2_before_subsample"], "after": f["n_t2_after_subsample"]} for f in folds},
        "folds": folds,
    }


def main() -> None:
    records = build()
    results = {}
    for frac in FRACTIONS:
        print(f"=== T2 fraction={frac} ===")
        r = run_fraction(records, frac)
        results[str(frac)] = r
        p = r["pooled"]
        t2 = p["per_class"]["Sloped-start"]
        print(f"  macro_f1={p['macro_f1']:.4f}  T2: P={t2['precision']:.3f} R={t2['recall']:.3f} F1={t2['f1']:.3f}  "
              f"n_t2_train (by fold)={r['n_t2_train_by_fold']}")

    print("\n=== T2 learning curve summary ===")
    print(f"{'fraction':>10s} {'macro_f1':>10s} {'T2_F1':>8s} {'T2_P':>8s} {'T2_R':>8s} {'mean_n_t2_train':>16s}")
    for frac in FRACTIONS:
        r = results[str(frac)]
        p = r["pooled"]["per_class"]["Sloped-start"]
        mean_n = np.mean([v["after"] for v in r["n_t2_train_by_fold"].values()])
        print(f"{frac:>10.2f} {r['pooled']['macro_f1']:>10.4f} {p['f1']:>8.3f} {p['precision']:>8.3f} {p['recall']:>8.3f} {mean_n:>16.1f}")

    out = {frac: {k: v for k, v in r.items() if k != "folds"} for frac, r in results.items()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nsaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
