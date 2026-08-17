"""Step 16 sections 14-16: justified diagnostic pitch-path corrections
(Q1 register, Q2 lag, Q4 motion-magnitude -- Q3/jitter-correction skipped,
undersupported: section 10 found jitter is a secondary, T0-localized
phenomenon while the dominant failure across T1-T3 is UNDER-motion, not
excess noise), and their downstream effect using Step 15's FROZEN P0
classifier (`training/train_pitch_motion_ablation.py`'s condition P0 =
Fused+D3 -> fixed phi -> shared TCN) -- no retraining, per spec section 15's
preference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.folds import build_fold_split, load_kfold_manifest  # noqa: E402
from training.framewise_models import FramewiseConditionalTCNModel  # noqa: E402
from training.metrics import TYPE_NAMES, frame_metrics  # noqa: E402
from training.pitch_diagnostics.common import octave_adjusted_error, write_json  # noqa: E402
from training.pitch_diagnostics.pitch_audit.common import AUDIT_DIR, build_bundles  # noqa: E402
from training.relative_pitch_features import PITCH_FEATURE_DIM, compute_phi, standardize_phi  # noqa: E402

RUN_DIR = REPO_ROOT / "output" / "pitch_motion_ablation"
LAG_SEARCH_FRAMES = range(-10, 11)
MOTION_CORRECTION_SCALE_OFFSET = 5  # calibrate the Q4 amplification factor from the 50ms scale (clearest R_k signal)


def _global_best_lag(gt_cents: np.ndarray, est_cents: np.ndarray, valid: np.ndarray) -> int:
    from training.pitch_diagnostics.pitch_audit.common import delta_at_offset, both_valid_mask
    d_gt = delta_at_offset(gt_cents, 2)
    d_est = delta_at_offset(est_cents, 2)
    mvalid = both_valid_mask(valid, 2)
    best_lag, best_mae = 0, np.inf
    for lag in LAG_SEARCH_FRAMES:
        if lag >= 0:
            est_s, gt_s, m_s = d_est[lag:], d_gt[: len(d_gt) - lag] if lag else d_gt, mvalid[lag:] & (mvalid[: len(mvalid) - lag] if lag else mvalid)
        else:
            L = -lag
            est_s, gt_s, m_s = d_est[: len(d_est) - L], d_gt[L:], mvalid[: len(mvalid) - L] & mvalid[L:]
        m = m_s & np.isfinite(est_s) & np.isfinite(gt_s)
        if m.sum() < 50:
            continue
        mae = float(np.mean(np.abs(est_s[m] - gt_s[m])))
        if mae < best_mae:
            best_mae, best_lag = mae, lag
    return best_lag


def build_q1_register_corrected(b: dict) -> np.ndarray:
    """GT-only octave correction, within-octave estimated error preserved."""
    _err, k = octave_adjusted_error(b["est_cents"], b["gt_cents"])
    k = np.nan_to_num(k, nan=0.0)
    return b["est_cents"] - 1200.0 * k


def build_q2_lag_corrected(b: dict) -> tuple[np.ndarray, int]:
    lag = _global_best_lag(b["gt_cents"], b["est_cents"], b["valid"])
    n = len(b["est_cents"])
    q2 = np.full(n, np.nan)
    if lag >= 0:
        q2[: n - lag] = b["est_cents"][lag:]
    else:
        L = -lag
        q2[L:] = b["est_cents"][: n - L]
    # fill edges by holding nearest value (diagnostic only, keeps array dense)
    valid_idx = np.flatnonzero(np.isfinite(q2))
    if len(valid_idx):
        q2[: valid_idx[0]] = q2[valid_idx[0]]
        q2[valid_idx[-1] + 1:] = q2[valid_idx[-1]]
    return q2, lag


def build_q4_motion_amplified(b: dict, factor: float) -> np.ndarray:
    """Amplify frame-to-frame octave-unwrapped deltas by a single global
    factor (derived from section 4's R_50ms attenuation ratio), then
    cumulatively reconstruct a path. Diagnostic only -- the factor is a
    single scalar calibrated from this audit's own GT-vs-EST comparison,
    not a per-frame GT lookup."""
    from training.relative_pitch_features import octave_unwrap
    n = len(b["est_cents"])
    raw_delta = np.zeros(n)
    raw_delta[1:] = octave_unwrap(b["est_cents"][1:] - b["est_cents"][:-1])
    raw_delta = np.nan_to_num(raw_delta, nan=0.0)
    amplified = raw_delta * factor
    q4 = np.cumsum(amplified)
    q4 += b["est_cents"][0] if np.isfinite(b["est_cents"][0]) else 0.0
    return q4


def _load_p0(fold: int, device):
    ckpt = torch.load(RUN_DIR / "condition_P0" / f"fold_{fold}" / "best.pt", map_location="cpu", weights_only=False)
    model = FramewiseConditionalTCNModel(use_audio=False, use_pitch=True, pitch_dim=PITCH_FEATURE_DIM).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["phi_mu"], ckpt["phi_sigma"]


@torch.no_grad()
def _eval_variant(bundles_by_rid: dict, cents_by_rid: dict, device) -> dict:
    manifest = load_kfold_manifest(REPO_ROOT)
    pooled_pred, pooled_true = [], []
    for fold in range(5):
        split = build_fold_split(manifest, fold, seed=42)
        model, phi_mu, phi_sigma = _load_p0(fold, device)  # loaded once per fold, not per recording
        for rid in split.test_recording_ids:
            b = bundles_by_rid[rid]
            cents = cents_by_rid[rid]
            phi = compute_phi(cents.astype(np.float64), b["valid"])
            std = standardize_phi(phi, phi_mu, phi_sigma)
            x = torch.from_numpy(std[np.newaxis]).to(device)  # [1, T, 4]
            logits = model(None, x)[0].numpy()
            mask = b["valid"]
            pred = logits.argmax(axis=-1)[mask]
            true = b["trajectory_type"][mask]
            pooled_pred.append(pred); pooled_true.append(true)
    pred = np.concatenate(pooled_pred); true = np.concatenate(pooled_true)
    return frame_metrics(pred, true)


def main() -> None:
    device = torch.device("cpu")
    bundles = build_bundles()
    bundles_by_rid = {b["recording_id"]: b for b in bundles}

    # calibrate Q4's global amplification factor from R_50ms (GT-moving-only), reusing motion.py's own computation
    from training.pitch_diagnostics.pitch_audit.common import delta_at_offset, both_valid_mask
    abs_est, abs_gt = [], []
    for b in bundles:
        d_est = delta_at_offset(b["est_cents"], MOTION_CORRECTION_SCALE_OFFSET)
        d_gt = delta_at_offset(b["gt_cents"], MOTION_CORRECTION_SCALE_OFFSET)
        m = both_valid_mask(b["valid"], MOTION_CORRECTION_SCALE_OFFSET) & np.isfinite(d_est) & np.isfinite(d_gt) & (np.abs(d_gt) > 1e-6)
        abs_est.append(np.abs(d_est[m])); abs_gt.append(np.abs(d_gt[m]))
    R50 = float(np.median(np.concatenate(abs_est)) / np.median(np.concatenate(abs_gt)))
    q4_factor = 1.0 / R50
    print(f"Q4 amplification factor = 1/R_50ms = 1/{R50:.3f} = {q4_factor:.3f}")

    variants = {"Q0_baseline": {}, "Q1_register_corrected": {}, "Q2_lag_corrected": {}, "Q4_motion_amplified": {}}
    q2_lags = {}
    for b in bundles:
        rid = b["recording_id"]
        variants["Q0_baseline"][rid] = b["est_cents"]
        variants["Q1_register_corrected"][rid] = build_q1_register_corrected(b)
        q2, lag = build_q2_lag_corrected(b)
        variants["Q2_lag_corrected"][rid] = q2
        q2_lags[rid] = lag
        variants["Q4_motion_amplified"][rid] = build_q4_motion_amplified(b, q4_factor)

    results = {}
    for name, cents_by_rid in variants.items():
        m = _eval_variant(bundles_by_rid, cents_by_rid, device)
        results[name] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "per_class": m["per_class"]}
        print(name, "macro_f1=", round(m["macro_f1"], 4), "acc=", round(m["accuracy"], 4))

    out = {
        "q4_amplification_factor": q4_factor, "q4_R_50ms": R50,
        "q2_lags_frames": q2_lags,
        "results": results,
        "reference_P0": None,  # filled by report from Step 15's evaluation_result.json
    }
    write_json(AUDIT_DIR / "counterfactual_results.json", out)


if __name__ == "__main__":
    main()
