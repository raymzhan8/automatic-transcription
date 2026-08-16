"""Step 12: shared per-recording processing for the register-resolution
diagnostics. Reuses Step 11's frozen HPS (baselines_b.harmonic_product_cents /
hps_salience.hps_salience_probs, k=2,3, untouched) and the trained
harmonic_salience_abs checkpoints (salience_models.HarmonicSalienceModel,
re-evaluated over the corrected full-CQT candidate range per plan decision #1
-- the scorer is built from position-agnostic 1x1 convs, so no retraining is
needed to widen the candidate axis).

One pass per recording computes everything the Phase-A diagnostics need for
both HPS and learned salience, so the model forward pass / HPS product only
happen once per recording.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.features import N_BINS  # noqa: E402
from training.framewise_dataset import RecordingLaneIndex  # noqa: E402
from training.normalization import log2_hz_to_cents, normalize_cqt  # noqa: E402
from training.metrics import pitch_error_metrics  # noqa: E402
from training.pitch_diagnostics.common import (  # noqa: E402
    OUT_DIR,
    PRIMARY_LANE,
    bin_from_hz,
    linear_mag,
    octave_adjusted_error,
    write_json,
)
from training.pitch_diagnostics.hps_salience import hps_salience_probs  # noqa: E402
from training.pitch_diagnostics.salience_common import (  # noqa: E402
    SALIENCE_SIGMA_CENTS,
    candidate_hz,
    distribution_entropy,
    target_rank_in_distribution,
)
from training.pitch_diagnostics.salience_models import HarmonicSalienceModel  # noqa: E402
from training.pitch_diagnostics.train_salience import VARIANT_CONFIG  # noqa: E402

REG_DIR = OUT_DIR / "register_resolution"
FIG_DIR = REG_DIR / "figures"
FULL_LO_BIN, FULL_HI_BIN = 0, N_BINS  # the §2 fix: full physically-meaningful CQT range
LARGE_FAILURE_RECORDINGS = ["6417585554a0bfbd8de2d3ff", "6824de49abc4705438ce918b"]


def extended_pitch_metrics(pred_cents: np.ndarray, true_cents: np.ndarray) -> dict[str, Any]:
    base = pitch_error_metrics(pred_cents, true_cents)
    if len(true_cents) == 0:
        base.update({
            "pct_within_100": 0.0, "octave_adjusted_mae": 0.0, "octave_adjusted_median_ae": 0.0,
            "correct_octave": 0.0, "octave_plus1": 0.0, "octave_minus1": 0.0, "within_1_octave": 0.0,
        })
        return base
    err = np.abs(pred_cents - true_cents)
    oct_err, oct_k = octave_adjusted_error(pred_cents, true_cents)
    base["pct_within_100"] = float((err <= 100).mean())
    base["octave_adjusted_mae"] = float(oct_err.mean())
    base["octave_adjusted_median_ae"] = float(np.median(oct_err))
    base["correct_octave"] = float((oct_k == 0).mean())
    base["octave_plus1"] = float((oct_k == 1).mean())
    base["octave_minus1"] = float((oct_k == -1).mean())
    base["within_1_octave"] = float(np.isin(oct_k, [-1, 0, 1]).mean())
    return base


def full_range_cand_cents() -> np.ndarray:
    hz = candidate_hz(FULL_LO_BIN, FULL_HI_BIN)
    return 1200.0 * np.log2(hz)


def load_learned_model_full_range(variant: str, fold: int) -> tuple[HarmonicSalienceModel, dict]:
    return load_learned_model(variant, fold, FULL_LO_BIN, FULL_HI_BIN)


def load_learned_model(variant: str, fold: int, lo_bin: int, hi_bin: int) -> tuple[HarmonicSalienceModel, dict]:
    cfg = VARIANT_CONFIG[variant]
    model = HarmonicSalienceModel(
        candidate_lo_bin=lo_bin, candidate_hi_bin=hi_bin,
        harmonic_ks=cfg["harmonic_ks"], hidden=cfg["hidden"],
    )
    run_name = {"local": "local_salience_abs", "harmonic": "harmonic_salience_abs"}[variant]
    ckpt = torch.load(OUT_DIR / "runs" / run_name / f"fold_{fold}" / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def native_range() -> tuple[int, int]:
    """Step 11's originally-trained candidate range (34, 244). Used for the
    learned model in all Step 12 diagnostics AFTER the §2 reconciliation
    step found that widening to the full CQT range causes severe
    out-of-distribution degradation (grouped MAE 284.2c -> 370.5c, driven by
    a new +1/+2 octave bias into the newly-added high-frequency bins the
    scorer never saw gradient signal for). Retraining is out of scope for
    Step 12, so the model's valid operating domain is its native range --
    see step11_reconciliation.json / docs/step_12 for the full writeup."""
    import json
    path = OUT_DIR / "harmonic_salience_candidate_range.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    return int(d["candidate_lo_bin"]), int(d["candidate_hi_bin"])


def _boundary_distance_s(times: np.ndarray, starts: np.ndarray, ends: np.ndarray, hop_s: float) -> np.ndarray:
    """Distance in seconds from each frame time to the nearest GT primitive
    boundary (start or end index). Evaluation-metadata only (spec §28)."""
    if len(starts) == 0 and len(ends) == 0:
        return np.full(len(times), np.inf)
    boundary_times = np.concatenate([
        np.asarray(starts, dtype=np.float64) * hop_s,
        np.asarray(ends, dtype=np.float64) * hop_s,
    ])
    boundary_times = np.sort(boundary_times)
    idx = np.searchsorted(boundary_times, times)
    idx_lo = np.clip(idx - 1, 0, len(boundary_times) - 1)
    idx_hi = np.clip(idx, 0, len(boundary_times) - 1)
    d_lo = np.abs(times - boundary_times[idx_lo])
    d_hi = np.abs(times - boundary_times[idx_hi])
    return np.minimum(d_lo, d_hi)


def _topk_from_probs(probs: np.ndarray, cand_cents: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """probs: [F,T]. Returns (top_k_cents [k,T], top_k_scores [k,T]), sorted descending."""
    order = np.argsort(-probs, axis=0)[:k]  # [k,T]
    top_scores = np.take_along_axis(probs, order, axis=0)
    top_cents = cand_cents[order]
    return top_cents, top_scores


def _nearest_bin_score(probs: np.ndarray, hz: np.ndarray, lo_bin: int = 0, hi_bin: int | None = None) -> np.ndarray:
    """probs: [F,T] over CQT bins [lo_bin,hi_bin). hz: [T] target frequency
    (may be GT, GT/2, or 2*GT). Returns score at nearest bin per frame,
    clipped into range (out-of-range GT/2 or 2*GT reads the boundary bin's
    score, which is representative for "is there any evidence near the edge
    of what this model/map can see")."""
    hi_bin = hi_bin if hi_bin is not None else lo_bin + probs.shape[0]
    abs_bins = np.clip(np.rint(np.asarray(bin_from_hz(hz))), lo_bin, hi_bin - 1).astype(np.int64)
    rel_bins = abs_bins - lo_bin
    t_idx = np.arange(probs.shape[1])
    return probs[rel_bins, t_idx]


def per_recording_frame_data(
    rec_id: str,
    fold: int,
    index: RecordingLaneIndex,
    learned_model: HarmonicSalienceModel,
    mu: np.ndarray,
    sigma: np.ndarray,
    cand_cents_full: np.ndarray,
    cand_cents_learned: np.ndarray | None = None,
) -> dict[str, Any] | None:
    """HPS is always evaluated over the full 0-360 range (cand_cents_full;
    it's always valid, see load_or_compute_candidate_range's note). The
    learned model is evaluated over whatever range `learned_model` was
    constructed with -- pass the matching `cand_cents_learned` (defaults to
    cand_cents_full for the §2 full-range reconciliation pass only)."""
    if cand_cents_learned is None:
        cand_cents_learned = cand_cents_full
    lane = next(x for x in index.lanes if x.recording_id == rec_id)
    frames = index._frames[(rec_id, PRIMARY_LANE)]
    cqt_log = index._features[rec_id]["cqt_log"]
    n = min(cqt_log.shape[1], lane.n_frames)
    cqt_log = cqt_log[:, :n]
    times = frames["frame_time_s"][:n]
    valid = frames["valid_target"][:n] & (times < lane.duration_s)
    if not np.any(valid):
        return None

    mag = linear_mag(cqt_log)
    tonic_hz = lane.fundamental_hz
    tonic_term = 1200.0 * np.log2(tonic_hz)

    true_log2 = frames["pitch_log2_hz"][:n].astype(np.float64)
    true_cents_abs_full = 1200.0 * true_log2
    true_hz_full = np.exp2(true_log2)

    # ---- HPS full-range salience map (frozen k=2,3 product, generalized) ----
    hps_probs_full = hps_salience_probs(mag, FULL_LO_BIN, FULL_HI_BIN)  # [360,T]
    hps_idx = hps_probs_full.argmax(axis=0)
    hps_argmax_cents_abs = cand_cents_full[hps_idx]

    # ---- learned model salience map, over ITS OWN candidate range ----
    lrn_lo, lrn_hi = learned_model.candidate_lo_bin, learned_model.candidate_hi_bin
    spec = normalize_cqt(cqt_log, mu, sigma).astype(np.float32)
    spec_t = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = learned_model(spec_t)[0].numpy()  # [lrn_hi-lrn_lo, T]
    logits = logits - logits.max(axis=0, keepdims=True)
    learned_probs_full = np.exp(logits)
    learned_probs_full /= np.maximum(learned_probs_full.sum(axis=0, keepdims=True), 1e-12)
    learned_idx = learned_probs_full.argmax(axis=0)
    learned_argmax_cents_abs = cand_cents_learned[learned_idx]

    def _valid_rel(x_abs: np.ndarray) -> np.ndarray:
        return (x_abs - tonic_term)[valid]

    true_rel = _valid_rel(true_cents_abs_full)
    hps_argmax_rel = _valid_rel(hps_argmax_cents_abs)
    learned_argmax_rel = _valid_rel(learned_argmax_cents_abs)

    hps_oct_err, hps_k = octave_adjusted_error(hps_argmax_rel, true_rel)
    learned_oct_err, learned_k = octave_adjusted_error(learned_argmax_rel, true_rel)

    hps_probs_v = hps_probs_full[:, valid]
    learned_probs_v = learned_probs_full[:, valid]
    true_cents_abs_v = true_cents_abs_full[valid]
    true_hz_v = true_hz_full[valid]

    hps_rank = target_rank_in_distribution(hps_probs_v, true_cents_abs_v, cand_cents_full, tolerance_cents=SALIENCE_SIGMA_CENTS)
    learned_rank = target_rank_in_distribution(learned_probs_v, true_cents_abs_v, cand_cents_learned, tolerance_cents=SALIENCE_SIGMA_CENTS)
    hps_entropy = distribution_entropy(hps_probs_v, axis=0)
    learned_entropy = distribution_entropy(learned_probs_v, axis=0)

    # top5_cents stored TONIC-RELATIVE (subtract tonic_term) so they're directly
    # comparable to true_cents downstream (oracle-topk etc.) -- cand_cents_full/
    # cand_cents_learned are absolute cents, this was previously a units bug.
    hps_top5_cents_abs, hps_top5_scores = _topk_from_probs(hps_probs_v, cand_cents_full, 5)
    learned_top5_cents_abs, learned_top5_scores = _topk_from_probs(learned_probs_v, cand_cents_learned, 5)
    hps_top5_cents = hps_top5_cents_abs - tonic_term
    learned_top5_cents = learned_top5_cents_abs - tonic_term
    hps_margin12 = hps_top5_scores[0] - hps_top5_scores[1]
    learned_margin12 = learned_top5_scores[0] - learned_top5_scores[1]

    hps_gt2 = _nearest_bin_score(hps_probs_v, true_hz_v / 2.0, 0, N_BINS)
    hps_gt = _nearest_bin_score(hps_probs_v, true_hz_v, 0, N_BINS)
    hps_2gt = _nearest_bin_score(hps_probs_v, true_hz_v * 2.0, 0, N_BINS)
    learned_gt2 = _nearest_bin_score(learned_probs_v, true_hz_v / 2.0, lrn_lo, lrn_hi)
    learned_gt = _nearest_bin_score(learned_probs_v, true_hz_v, lrn_lo, lrn_hi)
    learned_2gt = _nearest_bin_score(learned_probs_v, true_hz_v * 2.0, lrn_lo, lrn_hi)

    hop_s = float(frames["hop_s"]) if "hop_s" in frames else 0.01
    boundary_dist = _boundary_distance_s(
        times[valid], frames["boundary_start_frame_indices"], frames["boundary_end_frame_indices"], hop_s,
    )

    return {
        "recording_id": rec_id, "fold": fold, "n_valid": int(valid.sum()), "times": times[valid],
        "true_cents": true_rel,
        "trajectory_type": frames["trajectory_type"][:n][valid],
        "dp_dt": frames["dp_dt_log2_hz_per_s"][:n][valid],
        "boundary_dist_s": boundary_dist,
        "hps": {
            "argmax_cents": hps_argmax_rel, "octave_k": hps_k, "octave_err": hps_oct_err,
            "rank": hps_rank, "entropy": hps_entropy, "margin12": hps_margin12,
            "top5_cents": hps_top5_cents, "top5_scores": hps_top5_scores,
            "gt2_score": hps_gt2, "gt_score": hps_gt, "2gt_score": hps_2gt,
        },
        "learned": {
            "argmax_cents": learned_argmax_rel, "octave_k": learned_k, "octave_err": learned_oct_err,
            "rank": learned_rank, "entropy": learned_entropy, "margin12": learned_margin12,
            "top5_cents": learned_top5_cents, "top5_scores": learned_top5_scores,
            "gt2_score": learned_gt2, "gt_score": learned_gt, "2gt_score": learned_2gt,
        },
    }
