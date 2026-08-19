"""Step 22 section 8: a tiny interpretable analytic classifier (logistic
regression) over the section 6 geometric feature set only -- are the four
classes already nearly separable from a handful of interpretable summary
statistics, before any learned sequence model? Same grouped 5-fold
recording-level splits as the rest of the project (section 12); no
hyperparameter search.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.folds import build_fold_split, load_kfold_manifest  # noqa: E402
from training.metrics import frame_metrics  # noqa: E402
from training.shape_classification.dataset import CLASS_NAMES, OUT_DIR, build  # noqa: E402
from training.shape_classification.normalize import analytic_feature_vector  # noqa: E402

N_FOLDS = 5
SEED = 42


def build_xy(records: list[dict], source_key: str, recording_ids: set[str]) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for r in records:
        if r["recording_id"] not in recording_ids or r[source_key] is None:
            continue
        X.append(analytic_feature_vector(r[source_key]["features"]))
        y.append(r["canonical_type"])
    return np.stack(X) if X else np.zeros((0, 7)), np.array(y, dtype=np.int64)


def run(records: list[dict], source_key: str) -> dict:
    manifest = load_kfold_manifest(REPO_ROOT)
    pooled_pred, pooled_true = [], []
    per_fold = {}
    for fold in range(N_FOLDS):
        split = build_fold_split(manifest, fold, seed=SEED)
        X_train, y_train = build_xy(records, source_key, set(split.train_recording_ids))
        X_test, y_test = build_xy(records, source_key, set(split.test_recording_ids))
        if len(X_train) == 0 or len(X_test) == 0:
            continue

        scaler = StandardScaler().fit(X_train)
        clf = LogisticRegression(multi_class="multinomial", max_iter=2000, random_state=SEED)
        clf.fit(scaler.transform(X_train), y_train)
        pred = clf.predict(scaler.transform(X_test))

        m = frame_metrics(pred, y_test)
        per_fold[fold] = m["macro_f1"]
        pooled_pred.append(pred); pooled_true.append(y_test)

    pooled_pred = np.concatenate(pooled_pred); pooled_true = np.concatenate(pooled_true)
    pooled = frame_metrics(pooled_pred, pooled_true)
    return {
        "source": source_key, "pooled": pooled, "per_fold_macro_f1": per_fold,
        "grouped_mean_macro_f1": float(np.mean(list(per_fold.values()))),
        "grouped_std_macro_f1": float(np.std(list(per_fold.values()))),
    }


def _print(result: dict) -> None:
    p = result["pooled"]
    row = " ".join(f"{p['per_class'][f'T{i}']['f1']:.3f}".rjust(7) for i in range(4))
    print(f"{result['source']:8s} macro_f1={p['macro_f1']:.4f} acc={p['accuracy']:.4f}  "
          f"[{CLASS_NAMES[0][:4]},{CLASS_NAMES[1][:4]},{CLASS_NAMES[2][:4]},{CLASS_NAMES[3][:4]}]={row}  "
          f"grouped_mean={result['grouped_mean_macro_f1']:.4f}+/-{result['grouped_std_macro_f1']:.4f}")


def main() -> None:
    records = build()
    results = {src: run(records, src) for src in ("oracle", "crepe")}
    (OUT_DIR / "analytic_baseline.json").write_text(json.dumps(results, indent=2) + "\n")
    print("=== Step 22 §8 analytic baseline (logistic regression) ===")
    for src in ("oracle", "crepe"):
        _print(results[src])
    print(f"\nsaved to {OUT_DIR / 'analytic_baseline.json'}")


if __name__ == "__main__":
    main()
