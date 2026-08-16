"""Step 13 section 4: lightweight, non-architectural probes of whether
octave-invariant relative-pitch-contour features carry T0-T3 discriminative
information. Multinomial logistic regression (sklearn, no tuning grid), for
each of: oracle (GT), HPS argmax, HPS+D3, learned argmax, learned+D3,
fused+D3.

Two feature constructions are reported:

1. `sliding_window` -- fixed 21-frame (~210ms) window centered on every
   eligible frame (windows.py::eligible_centers/relative_contour), the
   literal "fixed-size relative-contour feature" the spec describes.
2. `primitive_aligned` -- the fairer test. A pilot run of (1) showed near-
   identical, near-zero macro F1 for EVERY source including oracle GT pitch
   (T0/T2/T3 F1 ~0, majority-class T1 collapse) -- suspicious, since GT
   pitch trivially determines trajectory shape by construction. Step 10's
   duration table shows ~40% of primitive-frames belong to primitives
   *shorter than* the 210ms window, so a fixed sliding window frequently
   spans multiple primitives of different types, contaminating even the
   trivially-separable "flat" (T0) class. `primitive_aligned` resamples
   each already-labeled primitive segment (derived from the existing
   per-frame trajectory_type array, contiguous-run boundaries -- no
   canonicalization change, see windows.py::primitive_segments) to a fixed
   20-point relative contour, one example per primitive instance. Still a
   fixed-size feature, still plain logistic regression -- not a new
   architecture, just a fairer window alignment for the same question.

Both reuse the existing grouped fold split so recordings never leak between
train and test; validation recordings are folded into train (no
hyperparameter search happens here). Not a production classifier -- no
class weighting, no oversampling, no sequence model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.folds import build_fold_split, load_kfold_manifest  # noqa: E402
from training.metrics import TYPE_NAMES  # noqa: E402
from training.pitch_diagnostics.common import write_json  # noqa: E402
from training.pitch_diagnostics.relative_pitch.path_cache import REL_DIR, build  # noqa: E402
from training.pitch_diagnostics.relative_pitch.windows import (  # noqa: E402
    eligible_centers, primitive_segments, relative_contour, resample_contour,
)

FEATURE_SOURCES = ("oracle", "hps_argmax", "hps_d3", "learned_argmax", "learned_d3", "fused_d3")
TYPE_ORDER = (0, 1, 2, 3)


def _path_for(record: dict, source: str) -> np.ndarray:
    return record["true_cents"] if source == "oracle" else record[f"{source}_cents"]


def _build_sliding(records: list[dict], recording_ids: set) -> dict:
    per_rec_centers, labels_by_rec = {}, {}
    for r in records:
        if r["recording_id"] not in recording_ids:
            continue
        centers = eligible_centers(r)
        per_rec_centers[r["recording_id"]] = centers
        labels_by_rec[r["recording_id"]] = r["trajectory_type"][centers]

    out = {}
    for source in FEATURE_SOURCES:
        X_parts, y_parts = [], []
        for r in records:
            rid = r["recording_id"]
            if rid not in recording_ids:
                continue
            centers = per_rec_centers[rid]
            if len(centers) == 0:
                continue
            path = _path_for(r, source)
            feats = np.stack([relative_contour(path, c) for c in centers], axis=0)
            X_parts.append(feats); y_parts.append(labels_by_rec[rid])
        out[source] = (np.concatenate(X_parts), np.concatenate(y_parts)) if X_parts else (np.zeros((0, 20)), np.zeros((0,), dtype=int))
    return out


def _build_primitive(records: list[dict], recording_ids: set) -> dict:
    per_rec_segs = {}
    for r in records:
        if r["recording_id"] not in recording_ids:
            continue
        per_rec_segs[r["recording_id"]] = primitive_segments(r)

    out = {}
    for source in FEATURE_SOURCES:
        X_parts, y_parts = [], []
        for r in records:
            rid = r["recording_id"]
            if rid not in recording_ids:
                continue
            segs = per_rec_segs[rid]
            if not segs:
                continue
            path = _path_for(r, source)
            feats = np.stack([resample_contour(path, s, e) for s, e, _t in segs], axis=0)
            labels = np.array([t for _s, _e, t in segs])
            X_parts.append(feats); y_parts.append(labels)
        out[source] = (np.concatenate(X_parts), np.concatenate(y_parts)) if X_parts else (np.zeros((0, 20)), np.zeros((0,), dtype=int))
    return out


def _run_probe(records: list[dict], builder, feature_dim: int) -> dict:
    manifest = load_kfold_manifest(REPO_ROOT)
    pooled_true = {s: [] for s in FEATURE_SOURCES}
    pooled_pred = {s: [] for s in FEATURE_SOURCES}
    per_fold = []

    for fold in range(5):
        split = build_fold_split(manifest, fold, seed=42)
        train_ids = set(split.train_recording_ids) | set(split.val_recording_ids)
        test_ids = set(split.test_recording_ids)
        train_data = builder(records, train_ids)
        test_data = builder(records, test_ids)

        fold_entry = {"fold": fold, "n_train": {}, "n_test": {}, "macro_f1": {}}
        for source in FEATURE_SOURCES:
            X_train, y_train = train_data[source]
            X_test, y_test = test_data[source]
            fold_entry["n_train"][source] = int(len(y_train))
            fold_entry["n_test"][source] = int(len(y_test))
            if len(y_train) == 0 or len(y_test) == 0 or len(np.unique(y_train)) < 2:
                fold_entry["macro_f1"][source] = None
                continue
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            clf = LogisticRegression(max_iter=2000)
            clf.fit(X_train_s, y_train)
            y_pred = clf.predict(X_test_s)
            fold_entry["macro_f1"][source] = float(f1_score(y_test, y_pred, average="macro", labels=TYPE_ORDER))
            pooled_true[source].append(y_test)
            pooled_pred[source].append(y_pred)
        per_fold.append(fold_entry)
        print(f"  fold {fold}:", {s: (round(v, 3) if v is not None else None) for s, v in fold_entry["macro_f1"].items()})

    summary = {}
    for source in FEATURE_SOURCES:
        if not pooled_true[source]:
            continue
        y_true = np.concatenate(pooled_true[source])
        y_pred = np.concatenate(pooled_pred[source])
        cm = confusion_matrix(y_true, y_pred, labels=TYPE_ORDER)
        per_class_f1 = f1_score(y_true, y_pred, average=None, labels=TYPE_ORDER)
        summary[source] = {
            "n": int(len(y_true)),
            "accuracy": float(np.mean(y_true == y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=TYPE_ORDER)),
            "per_class_f1": {TYPE_NAMES[t]: float(per_class_f1[i]) for i, t in enumerate(TYPE_ORDER)},
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_labels": [TYPE_NAMES[t] for t in TYPE_ORDER],
            "class_support": {TYPE_NAMES[t]: int(np.sum(y_true == t)) for t in TYPE_ORDER},
        }
    return {"feature_dim": feature_dim, "per_fold": per_fold, "pooled_summary": summary}


def main() -> None:
    records = build()

    print("=== sliding-window probe (21-frame, unaligned to primitive boundaries) ===")
    sliding = _run_probe(records, _build_sliding, feature_dim=20)

    print("\n=== primitive-aligned probe (one example per labeled primitive, 20-point resample) ===")
    primitive = _run_probe(records, _build_primitive, feature_dim=20)

    write_json(REL_DIR / "probe_result.json", {"sliding_window": sliding, "primitive_aligned": primitive})

    for name, res in (("sliding_window", sliding), ("primitive_aligned", primitive)):
        print(f"\n=== Step 13 probe pooled summary: {name} ===")
        for source, v in res["pooled_summary"].items():
            print(f"{source:16s} n={v['n']:7d} acc={v['accuracy']*100:5.1f}% macro_f1={v['macro_f1']:.3f}  "
                  f"per_class_f1={ {k: round(vv,3) for k,vv in v['per_class_f1'].items()} }")


if __name__ == "__main__":
    main()
