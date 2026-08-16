"""Evaluation metrics for framewise trajectory prediction."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

NUM_TYPES = 4
TYPE_NAMES = {0: "T0", 1: "T1", 2: "T2", 3: "T3"}


def _eval_mask(padding_mask: np.ndarray, valid_target: np.ndarray) -> np.ndarray:
    return (~padding_mask) & valid_target


def collect_frame_predictions(
    type_logits: torch.Tensor,
    trajectory_type: torch.Tensor,
    valid_target: torch.Tensor,
    padding_mask: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    mask = _eval_mask(padding_mask.numpy(), valid_target.numpy())
    preds = type_logits.argmax(dim=-1).numpy()
    labels = trajectory_type.numpy()
    return preds[mask], labels[mask]


def frame_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, Any]:
    if len(y_true) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}, "support": {}}
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", labels=list(range(NUM_TYPES)), zero_division=0)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_TYPES)),
        target_names=[TYPE_NAMES[i] for i in range(NUM_TYPES)],
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        TYPE_NAMES[i]: {
            "precision": report[TYPE_NAMES[i]]["precision"],
            "recall": report[TYPE_NAMES[i]]["recall"],
            "f1": report[TYPE_NAMES[i]]["f1-score"],
        }
        for i in range(NUM_TYPES)
    }
    support = {TYPE_NAMES[i]: int(report[TYPE_NAMES[i]]["support"]) for i in range(NUM_TYPES)}
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro),
        "per_class": per_class,
        "support": support,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(NUM_TYPES))
        ).tolist(),
    }


def majority_baseline(train_histogram: dict[int, int]) -> int:
    if not train_histogram:
        return 0
    return max(train_histogram, key=lambda k: train_histogram[k])


def baseline_frame_metrics(
    y_true: np.ndarray,
    majority_class: int,
) -> dict[str, float]:
    if len(y_true) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0}
    y_pred = np.full_like(y_true, majority_class)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=list(range(NUM_TYPES)), zero_division=0)
        ),
    }


def t1_provenance_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    provenance: np.ndarray,
) -> dict[str, Any]:
    """Stratify T1 frames by provenance bucket."""
    t1_mask = y_true == 1
    out: dict[str, Any] = {}
    for bucket in ("raw_t1", "t4_decomposition", "t5_decomposition", "t6_decomposition", "other"):
        mask = t1_mask & (provenance == bucket)
        if not np.any(mask):
            out[bucket] = {"count": 0}
            continue
        preds = y_pred[mask]
        labels = y_true[mask]
        confused = Counter(int(p) for p in preds if p != 1)
        out[bucket] = {
            "count": int(mask.sum()),
            "recall": float((preds == labels).mean()),
            "top_confused": dict(confused.most_common(3)),
        }
    return out


def aggregate_trajectory_predictions(
    frame_times: np.ndarray,
    type_logits: np.ndarray,
    padding_mask: np.ndarray,
    primitives: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Mean softmax logits over evaluable frames per primitive."""
    probs = torch.softmax(torch.from_numpy(type_logits), dim=-1).numpy()
    results: list[dict[str, Any]] = []
    frame_count_hist: Counter[int] = Counter()

    for prim in primitives:
        start, end = prim["start_s"], prim["end_s"]
        in_prim = (frame_times >= start) & (frame_times < end) & (~padding_mask)
        n_frames = int(in_prim.sum())
        if n_frames == 0:
            frame_count_hist[0] += 1
            continue
        frame_count_hist[min(n_frames, 3) if n_frames <= 3 else 3] += 1
        mean_prob = probs[in_prim].mean(axis=0)
        pred = int(mean_prob.argmax())
        results.append(
            {
                "predicted_type": pred,
                "canonical_type": int(prim["canonical_type"]),
                "n_frames": n_frames,
                "mean_prob": mean_prob.tolist(),
                "start_s": start,
                "end_s": end,
            }
        )

    hist = {
        "0_frames": frame_count_hist.get(0, 0),
        "1_frame": frame_count_hist.get(1, 0),
        "2_frames": frame_count_hist.get(2, 0),
        "3plus_frames": sum(v for k, v in frame_count_hist.items() if k >= 3),
    }
    return results, hist


def trajectory_accuracy(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0
    correct = sum(1 for r in results if r["predicted_type"] == r["canonical_type"])
    return correct / len(results)


def trajectory_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}, "support": {}, "n": 0}
    y_pred = np.array([r["predicted_type"] for r in results], dtype=np.int64)
    y_true = np.array([r["canonical_type"] for r in results], dtype=np.int64)
    out = frame_metrics(y_pred, y_true)
    out["n"] = len(results)
    return out


CONFUSION_PAIRS = (
    (1, 2),
    (1, 3),
    (2, 0),
    (2, 1),
    (2, 3),
    (3, 0),
    (3, 1),
    (3, 2),
)


def confusion_pair_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for src, dst in CONFUSION_PAIRS:
        out[f"true_T{src}_pred_T{dst}"] = int(((y_true == src) & (y_pred == dst)).sum())
    return out


def pitch_error_metrics(pred_cents: np.ndarray, true_cents: np.ndarray) -> dict[str, float]:
    if len(true_cents) == 0:
        return {
            "mae_cents": 0.0,
            "median_ae_cents": 0.0,
            "pct_within_10": 0.0,
            "pct_within_25": 0.0,
            "pct_within_50": 0.0,
            "support": 0,
        }
    err = np.abs(pred_cents - true_cents)
    return {
        "mae_cents": float(err.mean()),
        "median_ae_cents": float(np.median(err)),
        "pct_within_10": float((err <= 10).mean()),
        "pct_within_25": float((err <= 25).mean()),
        "pct_within_50": float((err <= 50).mean()),
        "support": int(len(err)),
    }


def pitch_metrics_by_type(
    pred_cents: np.ndarray,
    true_cents: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for t in range(NUM_TYPES):
        mask = y_true == t
        m = pitch_error_metrics(pred_cents[mask], true_cents[mask])
        out[TYPE_NAMES[t]] = m
    return out


def pitch_metrics_by_provenance(
    pred_cents: np.ndarray,
    true_cents: np.ndarray,
    y_true: np.ndarray,
    provenance: np.ndarray,
) -> dict[str, dict[str, float]]:
    t1 = y_true == 1
    out: dict[str, dict[str, float]] = {}
    for bucket in ("raw_t1", "t4_decomposition", "t5_decomposition", "t6_decomposition", "other"):
        mask = t1 & (provenance == bucket)
        out[bucket] = pitch_error_metrics(pred_cents[mask], true_cents[mask])
    return out


PITCH_ERROR_BUCKETS = (
    ("le_10", 0.0, 10.0),
    ("10_25", 10.0, 25.0),
    ("25_50", 25.0, 50.0),
    ("50_100", 50.0, 100.0),
    ("gt_100", 100.0, None),
)


def type_metrics_by_pitch_error(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    pred_cents: np.ndarray,
    true_cents: np.ndarray,
) -> dict[str, Any]:
    err = np.abs(pred_cents - true_cents)
    out: dict[str, Any] = {}
    for name, lo, hi in PITCH_ERROR_BUCKETS:
        if hi is None:
            mask = err > lo
        else:
            mask = (err > lo) & (err <= hi) if lo > 0 else err <= hi
        m = frame_metrics(y_pred[mask], y_true[mask])
        dist = {TYPE_NAMES[i]: int((y_true[mask] == i).sum()) for i in range(NUM_TYPES)}
        out[name] = {
            "support": int(mask.sum()),
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "class_distribution": dist,
        }
    return out


def pitch_error_by_type_correctness(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    pred_cents: np.ndarray,
    true_cents: np.ndarray,
) -> dict[str, dict[str, float]]:
    err = np.abs(pred_cents - true_cents)
    correct = y_pred == y_true
    def stats(mask: np.ndarray) -> dict[str, float]:
        if not np.any(mask):
            return {"mean_ae_cents": 0.0, "median_ae_cents": 0.0, "support": 0}
        return {
            "mean_ae_cents": float(err[mask].mean()),
            "median_ae_cents": float(np.median(err[mask])),
            "support": int(mask.sum()),
        }
    return {"type_correct": stats(correct), "type_incorrect": stats(~correct)}


def class_distribution(y: np.ndarray) -> dict[str, dict[str, float | int]]:
    n = int(len(y))
    out: dict[str, dict[str, float | int]] = {}
    for i in range(NUM_TYPES):
        count = int((y == i).sum()) if n else 0
        out[TYPE_NAMES[i]] = {
            "count": count,
            "pct": float(count / n) if n else 0.0,
        }
    return out


def mean_softmax_entropy(logits: np.ndarray) -> float:
    if len(logits) == 0:
        return 0.0
    x = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(x)
    probs = exp / np.clip(exp.sum(axis=-1, keepdims=True), 1e-12, None)
    ent = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=-1)
    return float(ent.mean())


BOUNDARY_MS_BUCKETS = (
    ("0_20", 0.0, 20.0),
    ("20_50", 20.0, 50.0),
    ("50_100", 50.0, 100.0),
    ("100_250", 100.0, 250.0),
    ("gt_250", 250.0, None),
)

DURATION_S_BUCKETS = (
    ("lt_100ms", 0.0, 0.10),
    ("100_250ms", 0.10, 0.25),
    ("250_500ms", 0.25, 0.50),
    ("500ms_1s", 0.50, 1.00),
    ("gt_1s", 1.00, None),
)

FINE_PITCH_ERROR_BUCKETS = (
    ("le_10", 0.0, 10.0),
    ("10_25", 10.0, 25.0),
    ("25_50", 25.0, 50.0),
    ("50_100", 50.0, 100.0),
    ("100_300", 100.0, 300.0),
    ("300_600", 300.0, 600.0),
    ("gt_600", 600.0, None),
)


def _bucket_mask(values: np.ndarray, lo: float, hi: float | None) -> np.ndarray:
    finite = np.isfinite(values)
    if hi is None:
        return finite & (values > lo)
    if lo == 0:
        return finite & (values >= lo) & (values <= hi)
    return finite & (values > lo) & (values <= hi)


def nearest_boundary_distance_ms(
    frame_times: np.ndarray,
    primitives: list[dict[str, Any]],
) -> np.ndarray:
    if not primitives:
        return np.full(len(frame_times), np.nan, dtype=np.float64)
    bounds = np.array(
        [p["start_s"] for p in primitives] + [p["end_s"] for p in primitives],
        dtype=np.float64,
    )
    return np.min(np.abs(frame_times[:, None] - bounds[None, :]), axis=1) * 1000.0


def duration_s_per_frame(
    frame_times: np.ndarray,
    primitives: list[dict[str, Any]],
) -> np.ndarray:
    dur = np.full(len(frame_times), np.nan, dtype=np.float64)
    for prim in primitives:
        start, end = prim["start_s"], prim["end_s"]
        mask = (frame_times >= start) & (frame_times < end)
        dur[mask] = end - start
    return dur


def metrics_by_buckets(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    values: np.ndarray,
    buckets: tuple[tuple[str, float, float | None], ...],
    *,
    pred_cents: np.ndarray | None = None,
    true_cents: np.ndarray | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, lo, hi in buckets:
        mask = _bucket_mask(values, lo, hi)
        block: dict[str, Any] = {
            "support": int(mask.sum()),
            **frame_metrics(y_pred[mask], y_true[mask]),
        }
        if pred_cents is not None and true_cents is not None:
            block["pitch"] = pitch_error_metrics(pred_cents[mask], true_cents[mask])
        out[name] = block
    return out


def trajectory_metrics_by_duration(
    traj_results: list[dict[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, lo, hi in DURATION_S_BUCKETS:
        selected = [
            r
            for r in traj_results
            if _bucket_mask(np.array([float(r["end_s"] - r["start_s"])]), lo, hi)[0]
        ]
        out[name] = trajectory_metrics(selected)
    return out


def type_metrics_by_fine_pitch_error(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    pred_cents: np.ndarray,
    true_cents: np.ndarray,
) -> dict[str, Any]:
    err = np.abs(pred_cents - true_cents)
    out: dict[str, Any] = {}
    for name, lo, hi in FINE_PITCH_ERROR_BUCKETS:
        mask = _bucket_mask(err, lo, hi)
        m = frame_metrics(y_pred[mask], y_true[mask])
        dist = {TYPE_NAMES[i]: int((y_true[mask] == i).sum()) for i in range(NUM_TYPES)}
        out[name] = {
            "support": int(mask.sum()),
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "class_distribution": dist,
        }
    return out
