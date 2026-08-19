"""Step 22 sections 4/17: CP condition -- how sensitive is CREPE normalized-
shape recognition to boundary error? Applies the ALREADY-TRAINED, frozen
CREPE shape_only CNN (one checkpoint per fold, from cnn_model.py) to test
primitives whose extraction window is perturbed from the GT boundary by a
small predetermined set of offsets. No retraining; the label never changes,
only which slice of the CREPE contour is read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.folds import build_fold_split, load_kfold_manifest  # noqa: E402
from training.metrics import frame_metrics  # noqa: E402
from training.shape_classification.cnn_model import ContourCNN, N_FOLDS, RUN_DIR, SEED, get_device  # noqa: E402
from training.shape_classification.contours import crepe_contour  # noqa: E402
from training.shape_classification.dataset import OUT_DIR, build, derive_contour, load_recording_lookup  # noqa: E402

# Symmetric window shift (start and end both move by the same delta,
# duration preserved) -- the primary sweep.
SYMMETRIC_DELTAS_MS = (-100, -50, -20, 0, 20, 50, 100)
# One-sided perturbations at a single moderate magnitude, to separate
# start- vs. end-boundary sensitivity (section 17's "if straightforward").
ONE_SIDED_DELTAS_MS = (-50, 50)


def _extract(prim, lookup, start_delta_s: float, end_delta_s: float):
    rec = lookup[prim["recording_id"]]
    contour = crepe_contour(
        rec["frame_time_s"], rec["crepe_log2"], prim["start_s"], prim["end_s"],
        start_delta_s=start_delta_s, end_delta_s=end_delta_s, lane_duration_s=rec["duration_s"],
    )
    return derive_contour(contour)


def _load_model(fold: int) -> tuple[ContourCNN, np.ndarray, np.ndarray]:
    ckpt = torch.load(RUN_DIR / "crepe" / "shape_only" / f"fold_{fold}" / "best.pt",
                       map_location="cpu", weights_only=False)
    model = ContourCNN(in_channels=1)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["mu"], ckpt["sigma"]


def evaluate_condition(records, lookup, start_delta_s: float, end_delta_s: float) -> dict:
    device = get_device()
    manifest = load_kfold_manifest(REPO_ROOT)
    pooled_pred, pooled_true = [], []
    per_fold_f1 = {}

    for fold in range(N_FOLDS):
        split = build_fold_split(manifest, fold, seed=SEED)
        test_ids = set(split.test_recording_ids)
        model, mu, sigma = _load_model(fold)
        model.to(device)

        X, y = [], []
        for r in records:
            if r["recording_id"] not in test_ids:
                continue
            d = _extract(r, lookup, start_delta_s, end_delta_s)
            if d is None:
                continue
            X.append(d["q"][None, :])
            y.append(r["canonical_type"])
        if not X:
            continue
        X = np.stack(X).astype(np.float32)
        y = np.array(y, dtype=np.int64)
        X = (X - mu[None, :, None]) / sigma[None, :, None]
        with torch.no_grad():
            pred = model(torch.from_numpy(X).to(device)).argmax(dim=-1).cpu().numpy()
        per_fold_f1[fold] = frame_metrics(pred, y)["macro_f1"]
        pooled_pred.append(pred); pooled_true.append(y)

    pooled_pred = np.concatenate(pooled_pred); pooled_true = np.concatenate(pooled_true)
    pooled = frame_metrics(pooled_pred, pooled_true)
    return {
        "start_delta_ms": start_delta_s * 1000, "end_delta_ms": end_delta_s * 1000,
        "pooled": pooled, "per_fold_macro_f1": per_fold_f1,
        "grouped_mean_macro_f1": float(np.mean(list(per_fold_f1.values()))),
    }


def main() -> None:
    records = build()
    lookup = load_recording_lookup()

    results = {"symmetric": {}, "start_only": {}, "end_only": {}}
    print("=== Step 22 §17 boundary perturbation (frozen CREPE shape_only CNN) ===")
    print("\n-- symmetric window shift --")
    for d_ms in SYMMETRIC_DELTAS_MS:
        d_s = d_ms / 1000.0
        res = evaluate_condition(records, lookup, d_s, d_s)
        results["symmetric"][d_ms] = res
        p = res["pooled"]
        row = " ".join(f"{p['per_class'][f'T{i}']['f1']:.3f}".rjust(6) for i in range(4))
        print(f"delta={d_ms:+4d}ms  macro_f1={p['macro_f1']:.4f}  [Fix,Cos,SlS,SlE]={row}  "
              f"grouped_mean={res['grouped_mean_macro_f1']:.4f}")

    print("\n-- start-boundary only (end fixed at GT) --")
    for d_ms in ONE_SIDED_DELTAS_MS:
        res = evaluate_condition(records, lookup, d_ms / 1000.0, 0.0)
        results["start_only"][d_ms] = res
        print(f"start_delta={d_ms:+4d}ms  macro_f1={res['pooled']['macro_f1']:.4f}  "
              f"grouped_mean={res['grouped_mean_macro_f1']:.4f}")

    print("\n-- end-boundary only (start fixed at GT) --")
    for d_ms in ONE_SIDED_DELTAS_MS:
        res = evaluate_condition(records, lookup, 0.0, d_ms / 1000.0)
        results["end_only"][d_ms] = res
        print(f"end_delta={d_ms:+4d}ms  macro_f1={res['pooled']['macro_f1']:.4f}  "
              f"grouped_mean={res['grouped_mean_macro_f1']:.4f}")

    (OUT_DIR / "boundary_perturbation.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nsaved to {OUT_DIR / 'boundary_perturbation.json'}")


if __name__ == "__main__":
    main()
