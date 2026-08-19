"""Step 23: a k-class generalization of `training.metrics.frame_metrics`
(which hardcodes NUM_TYPES=4/T0-T3) -- same output shape (accuracy,
macro_f1, per_class precision/recall/F1, support, confusion_matrix), usable
for the 4-class main experiment as well as the 2-class (T2 vs T3) and
3-class (Cosine/Sloped-start/Sloped-end) diagnostics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def eval_metrics(y_pred: np.ndarray, y_true: np.ndarray, names: tuple[str, ...]) -> dict[str, Any]:
    n = len(names)
    labels = list(range(n))
    if len(y_true) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}, "support": {}, "confusion_matrix": []}
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, target_names=list(names),
                                    output_dict=True, zero_division=0)
    per_class = {names[i]: {"precision": report[names[i]]["precision"],
                             "recall": report[names[i]]["recall"],
                             "f1": report[names[i]]["f1-score"]} for i in range(n)}
    support = {names[i]: int(report[names[i]]["support"]) for i in range(n)}
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {"accuracy": float(acc), "macro_f1": float(macro), "per_class": per_class,
            "support": support, "confusion_matrix": cm}


def prediction_frequency(y_pred: np.ndarray, names: tuple[str, ...]) -> dict[str, float]:
    n = len(names)
    total = max(len(y_pred), 1)
    counts = np.bincount(y_pred, minlength=n)
    return {names[i]: float(counts[i]) / total for i in range(n)}
