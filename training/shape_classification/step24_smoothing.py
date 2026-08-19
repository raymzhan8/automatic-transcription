"""Step 24 section 17: the ONE gated smoothing control. Only invoked by
step24_experiments.py if the raw-vs-robust comparison shows frame-level
jitter/outliers materially hurt CREPE template fitting. A single short
median filter, window fixed BEFORE looking at any Step 24 classification
result: 5 native 10ms frames (50ms) -- short enough to preserve genuine
T1-T3 bend transitions (Step 19/20 found real motion on a 10-40ms scale),
long enough to be a real smoothing operation. No sweep, no alternative
filter family.
"""

from __future__ import annotations

import numpy as np

from training.shape_classification.contours import crepe_contour
from training.shape_classification.dataset import derive_contour, load_recording_lookup
from training.shape_classification.metrics_utils import eval_metrics
from training.shape_classification.templates import TEMPLATE_NAMES, template_errors

SMOOTH_WINDOW_FRAMES = 5  # 50ms centered median filter, fixed before any Step 24 result was examined


def median_filter_1d(x: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    padded = np.pad(x, (half, half), mode="edge")
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.median(padded[i:i + window])
    return out


def run_smoothing_control(records: list[dict], fold_map: dict[str, int]) -> dict:
    lookup = load_recording_lookup()
    smoothed_log2 = {rid: median_filter_1d(v["crepe_log2"], SMOOTH_WINDOW_FRAMES) for rid, v in lookup.items()}

    scored = []
    for r in records:
        rid = r["recording_id"]
        rec = lookup[rid]
        contour = crepe_contour(rec["frame_time_s"], smoothed_log2[rid], r["start_s"], r["end_s"],
                                 lane_duration_s=rec["duration_s"])
        d = derive_contour(contour)
        if d is None:
            continue
        errs = template_errors(d["r"], d["span_cents"], robust=False)
        scored.append({
            "primitive_id": r["primitive_id"], "recording_id": rid, "fold": fold_map.get(rid),
            "true": r["canonical_type"], "pred": int(np.argmin(errs)), "errors": errs,
            "duration_s": r["duration_s"], "span_cents": abs(d["span_cents"]),
        })

    pred = np.array([s["pred"] for s in scored]); true = np.array([s["true"] for s in scored])
    pooled = eval_metrics(pred, true, TEMPLATE_NAMES)
    return {"pooled": pooled, "window_frames": SMOOTH_WINDOW_FRAMES, "scored": scored}
