"""Step 20 Phase B: retrain the pitch-estimation pipeline (harmonic-salience
CNN, HPS+learned fusion, Fused+D3 Viterbi decode) on the `A1a_cqt_fs0.5`
frontend Phase A selected, and re-measure both pitch-estimation quality and
downstream P0 trajectory macro F1 against the frozen fs=1 numbers.

Scope, stated explicitly (mirrors this project's practice of narrowing scope
transparently rather than silently): this reproduces Phase B items (a)-(d)
and (f) from docs/step_20_acoustic_frontend_bakeoff.md's own spec --
fs=0.5 feature cache + fold CQT/pitch stats (already done, see
`output/phase_b_fs0.5_shadow/`), salience CNN retrain, D3-only register/
fusion recalibration (D4's octave penalty is skipped: `dense_pitch_path.py`
only ever consumes D3 hyperparameters, so D4 regridding would be extra work
the downstream comparison doesn't need), and the Fused+D3 dense pitch path
rebuild. Item (e), Step 19's full pre-decoder audit (rank/motion-contrast/
continuity/zero-delta decomposition), is NOT re-run in full; pitch-
estimation quality is instead reported via the same MAE/octave-accuracy
metrics Steps 12/12.5/13 already use, directly comparable to the frozen
fs=1 numbers in those docs. Item (f), retraining the P0 trajectory
classifier, is done via `train_pitch_motion_ablation.py`'s existing
`--pitch-variant` mechanism (a `fs0.5` variant added there).

The frozen fs=1 production artifacts (checkpoints, `decoder_ablation.json`,
`fusion_viterbi_result.json`, `dense_fused_d3_log2hz.pkl`) are never
overwritten: every Phase B artifact uses a `_fs0.5` run-name/filename
suffix and reads CQT data from a separate shadow repo root
(`output/phase_b_fs0.5_shadow/`, symlinked to production for everything
except `features/`/`normalization/`, which hold real fs=0.5 content).

Usage:
    python -m training.pitch_diagnostics.pitch_audit.phase_b_fs0_5 salience
    python -m training.pitch_diagnostics.pitch_audit.phase_b_fs0_5 register
    python -m training.pitch_diagnostics.pitch_audit.phase_b_fs0_5 densepath
    python -m training.pitch_diagnostics.pitch_audit.phase_b_fs0_5 compare
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.folds import build_fold_split, load_kfold_manifest  # noqa: E402
from training.framewise_dataset import RecordingLaneIndex  # noqa: E402
from training.normalization import load_fold_cqt_stats, normalize_cqt  # noqa: E402
from training.pitch_diagnostics.common import OUT_DIR, PRIMARY_LANE, linear_mag, write_json  # noqa: E402
from training.pitch_diagnostics.hps_salience import hps_salience_probs  # noqa: E402
from training.pitch_diagnostics.register_resolution.common import (  # noqa: E402
    FULL_HI_BIN, FULL_LO_BIN, REG_DIR, candidate_hz, extended_pitch_metrics, native_range,
)
from training.pitch_diagnostics.register_resolution.decoders import (  # noqa: E402
    LAMBDA_T_GRID, _states_for_frame, viterbi_decode,
)
from training.pitch_diagnostics.register_resolution.fusion import RATIOS, fused_argmax  # noqa: E402
from training.pitch_diagnostics.salience_common import load_or_compute_candidate_range  # noqa: E402
from training.pitch_diagnostics.salience_models import HarmonicSalienceModel  # noqa: E402
from training.pitch_diagnostics.train_salience import VARIANT_CONFIG, train_one  # noqa: E402
from training.pitch_diagnostics.relative_pitch.path_cache import REL_DIR  # noqa: E402

SHADOW = REPO_ROOT / "output" / "phase_b_fs0.5_shadow"
SALIENCE_RUN_NAME = "harmonic_salience_abs_fs0.5"
REGISTER_RESULT_PATH = REG_DIR / "phase_b_fs0.5_hyperparams_and_metrics.json"
DENSE_CACHE_PATH = REL_DIR / "dense_fused_d3_log2hz_fs0.5.pkl"
NATIVE_HOP_S = 0.01


# ---------------------------------------------------------------- (b) salience retrain


def run_salience() -> None:
    index = RecordingLaneIndex.build(SHADOW)
    candidate_range = load_or_compute_candidate_range(SHADOW, index)  # loads frozen Step 11 file (frontend-independent)
    results = []
    for fold in range(5):
        r = train_one(
            variant="harmonic", fold=fold, run_name=SALIENCE_RUN_NAME,
            repo_root=SHADOW, index=index, candidate_range=candidate_range,
        )
        results.append(r)
        print(f"fold {fold}: best_val_mae={r['best_val_mae_cents']:.1f}c @ epoch {r['best_epoch']}")
    summary = {"run_name": SALIENCE_RUN_NAME, "folds": results,
               "mean_val_mae": float(np.mean([r["best_val_mae_cents"] for r in results]))}
    write_json(OUT_DIR / "runs" / SALIENCE_RUN_NAME / "cv_summary.json", summary)
    print(f"mean_val_mae={summary['mean_val_mae']:.1f}c (fs=1 harmonic reference: see docs/step_11_harmonic_salience.md)")


def _load_salience_model_fs0_5(fold: int, lo_bin: int, hi_bin: int) -> HarmonicSalienceModel:
    cfg = VARIANT_CONFIG["harmonic"]
    model = HarmonicSalienceModel(candidate_lo_bin=lo_bin, candidate_hi_bin=hi_bin,
                                   harmonic_ks=cfg["harmonic_ks"], hidden=cfg["hidden"])
    ckpt = torch.load(OUT_DIR / "runs" / SALIENCE_RUN_NAME / f"fold_{fold}" / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ---------------------------------------------------------------- (c) register/fusion recalibration (D3 only)


def _bundle(rec_id: str, index: RecordingLaneIndex, model, mu, sigma, lo: int, hi: int) -> dict:
    lane = next(x for x in index.lanes if x.recording_id == rec_id)
    frames = index._frames[(rec_id, PRIMARY_LANE)]
    cqt_log = index._features[rec_id]["cqt_log"]
    n = min(cqt_log.shape[1], lane.n_frames)
    cqt_log = cqt_log[:, :n]
    times = frames["frame_time_s"][:n]
    valid = frames["valid_target"][:n] & (times < lane.duration_s)
    tonic_term = 1200.0 * np.log2(lane.fundamental_hz)
    true_rel = (1200.0 * frames["pitch_log2_hz"][:n].astype(np.float64) - tonic_term)[valid]

    mag = linear_mag(cqt_log)
    hps_probs_full = hps_salience_probs(mag, FULL_LO_BIN, FULL_HI_BIN)[:, valid]
    hps_probs_shared = hps_salience_probs(mag, lo, hi)[:, valid]
    spec = normalize_cqt(cqt_log, mu, sigma).astype(np.float32)
    spec_t = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(spec_t)[0].numpy()
    logits = logits - logits.max(axis=0, keepdims=True)
    learned_probs = np.exp(logits)
    learned_probs /= np.maximum(learned_probs.sum(axis=0, keepdims=True), 1e-12)
    learned_probs = learned_probs[:, valid]

    t_valid = times[valid]
    dt_steps = np.empty(len(t_valid))
    dt_steps[0] = 1.0
    dt_steps[1:] = np.maximum((t_valid[1:] - t_valid[:-1]) / NATIVE_HOP_S, 1.0)

    return {
        "recording_id": rec_id, "hps_probs_full": hps_probs_full, "hps_probs_shared": hps_probs_shared,
        "learned_probs": learned_probs, "true_cents": true_rel, "tonic_term": tonic_term, "dt_steps": dt_steps,
        "trajectory_type": frames["trajectory_type"][:n][valid],
    }


def _fused_probs(hps_probs: np.ndarray, learned_probs: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    score = alpha * np.log(np.maximum(learned_probs, 1e-12)) + beta * np.log(np.maximum(hps_probs, 1e-12))
    score = score - score.max(axis=0, keepdims=True)
    probs = np.exp(score)
    probs /= np.maximum(probs.sum(axis=0, keepdims=True), 1e-12)
    return probs


def _decode(probs: np.ndarray, cand_cents: np.ndarray, lo: int, dt_steps: np.ndarray, tonic_term: float, lambda_t: float) -> np.ndarray:
    T = probs.shape[1]
    states = [_states_for_frame(probs[:, t], cand_cents, lo) for t in range(T)]
    path_abs = viterbi_decode(states, dt_steps, lambda_t, 0.0)
    return path_abs - tonic_term


def _select_lambda_t(bundles: list[dict], probs_fn, cand_cents: np.ndarray, lo: int) -> tuple[float, dict]:
    val_mae = {}
    for lt in LAMBDA_T_GRID:
        preds, trues = [], []
        for b in bundles:
            probs = probs_fn(b)
            pred = _decode(probs, cand_cents, lo, b["dt_steps"], b["tonic_term"], lt)
            preds.append(pred); trues.append(b["true_cents"])
        val_mae[lt] = float(np.abs(np.concatenate(preds) - np.concatenate(trues)).mean())
    best_lt = min(val_mae, key=val_mae.get)
    return best_lt, val_mae


def run_register() -> None:
    """Recalibrates: HPS D3 lambda_t, learned D3 lambda_t, fusion (alpha,beta),
    fused D3 lambda_t -- exactly the four fields `dense_pitch_path.py`'s
    `_load_fixed_hyperparams` consumes, D3 only (no D4 octave-penalty grid,
    see module docstring)."""
    index = RecordingLaneIndex.build(SHADOW)
    manifest = load_kfold_manifest(SHADOW)
    lrn_lo, lrn_hi = native_range()  # frozen Step 11 range, frontend-independent
    cand_cents_hps_full = 1200.0 * np.log2(candidate_hz(FULL_LO_BIN, FULL_HI_BIN))
    cand_cents_learned = 1200.0 * np.log2(candidate_hz(lrn_lo, lrn_hi))

    by_fold: dict[int, dict] = {}
    pooled = {m: {"pred": [], "true": []} for m in ("hps_d3", "learned_d3", "fused_d3")}

    for fold in range(5):
        split = build_fold_split(manifest, fold, seed=42)
        model = _load_salience_model_fs0_5(fold, lrn_lo, lrn_hi)
        mu, sigma = load_fold_cqt_stats(fold, SHADOW)

        val_bundles = [_bundle(rid, index, model, mu, sigma, lrn_lo, lrn_hi) for rid in split.val_recording_ids]

        hps_lt, _ = _select_lambda_t(val_bundles, lambda b: b["hps_probs_full"], cand_cents_hps_full, FULL_LO_BIN)
        learned_lt, _ = _select_lambda_t(val_bundles, lambda b: b["learned_probs"], cand_cents_learned, lrn_lo)

        val_scores = {r: {"pred": [], "true": []} for r in RATIOS}
        for b in val_bundles:
            for (a, bw) in RATIOS:
                pred = fused_argmax(b["hps_probs_shared"], b["learned_probs"], cand_cents_learned, a, bw, b["tonic_term"])
                val_scores[(a, bw)]["pred"].append(pred); val_scores[(a, bw)]["true"].append(b["true_cents"])
        best_ratio, best_mae = (1.0, 0.0), float("inf")
        for r in RATIOS:
            p = np.concatenate(val_scores[r]["pred"]); t = np.concatenate(val_scores[r]["true"])
            mae = float(np.abs(p - t).mean())
            if mae < best_mae:
                best_mae, best_ratio = mae, r
        alpha, beta = best_ratio
        for b in val_bundles:
            b["fused_probs"] = _fused_probs(b["hps_probs_shared"], b["learned_probs"], alpha, beta)
        fused_lt, _ = _select_lambda_t(val_bundles, lambda b: b["fused_probs"], cand_cents_learned, lrn_lo)

        by_fold[fold] = {"hps_lambda_t": hps_lt, "learned_lambda_t": learned_lt,
                          "fusion_ratio": [alpha, beta], "fused_lambda_t": fused_lt}
        print(f"fold {fold}: hps_lt={hps_lt} learned_lt={learned_lt} ratio=({alpha},{beta}) fused_lt={fused_lt}")

        for rid in split.test_recording_ids:
            b = _bundle(rid, index, model, mu, sigma, lrn_lo, lrn_hi)
            fused_probs = _fused_probs(b["hps_probs_shared"], b["learned_probs"], alpha, beta)
            preds = {
                "hps_d3": _decode(b["hps_probs_full"], cand_cents_hps_full, FULL_LO_BIN, b["dt_steps"], b["tonic_term"], hps_lt),
                "learned_d3": _decode(b["learned_probs"], cand_cents_learned, lrn_lo, b["dt_steps"], b["tonic_term"], learned_lt),
                "fused_d3": _decode(fused_probs, cand_cents_learned, lrn_lo, b["dt_steps"], b["tonic_term"], fused_lt),
            }
            for m, pred in preds.items():
                pooled[m]["pred"].append(pred); pooled[m]["true"].append(b["true_cents"])

    summary = {m: extended_pitch_metrics(np.concatenate(v["pred"]), np.concatenate(v["true"])) for m, v in pooled.items()}
    out = {"by_fold": by_fold, "test_summary_pooled": summary}
    write_json(REGISTER_RESULT_PATH, out)
    print("\n=== fs=0.5 D3 pooled test summary ===")
    for m, v in summary.items():
        print(f"{m:12s} MAE {v['mae_cents']:7.1f}  oct_adj {v['octave_adjusted_mae']:6.1f}  correct_oct {v['correct_octave']*100:5.1f}%")
    print(f"saved to {REGISTER_RESULT_PATH}")


# ---------------------------------------------------------------- (d) dense Fused+D3 path rebuild


def run_densepath(force: bool = True) -> dict[str, np.ndarray]:
    if DENSE_CACHE_PATH.exists() and not force:
        with open(DENSE_CACHE_PATH, "rb") as fh:
            return pickle.load(fh)

    hp = {int(k): v for k, v in json.loads(REGISTER_RESULT_PATH.read_text())["by_fold"].items()}
    index = RecordingLaneIndex.build(SHADOW)
    manifest = load_kfold_manifest(SHADOW)
    lrn_lo, lrn_hi = native_range()
    cand_cents_learned = 1200.0 * np.log2(candidate_hz(lrn_lo, lrn_hi))

    out: dict[str, np.ndarray] = {}
    for fold in range(5):
        split = build_fold_split(manifest, fold, seed=42)
        model = _load_salience_model_fs0_5(fold, lrn_lo, lrn_hi)
        mu, sigma = load_fold_cqt_stats(fold, SHADOW)
        h = hp[fold]

        for rec_id in split.test_recording_ids:
            lane = next(x for x in index.lanes if x.recording_id == rec_id)
            frames = index._frames[(rec_id, PRIMARY_LANE)]
            cqt_log = index._features[rec_id]["cqt_log"]
            n = min(cqt_log.shape[1], lane.n_frames)
            cqt_log = cqt_log[:, :n]

            mag = linear_mag(cqt_log)
            hps_probs_shared = hps_salience_probs(mag, lrn_lo, lrn_hi)
            spec = normalize_cqt(cqt_log, mu, sigma).astype(np.float32)
            spec_t = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = model(spec_t)[0].numpy()
            logits = logits - logits.max(axis=0, keepdims=True)
            learned_probs = np.exp(logits)
            learned_probs /= np.maximum(learned_probs.sum(axis=0, keepdims=True), 1e-12)

            fused_probs = _fused_probs(hps_probs_shared, learned_probs, *h["fusion_ratio"])
            dt_steps = np.ones(n, dtype=np.float64)
            states = [_states_for_frame(fused_probs[:, t], cand_cents_learned, lrn_lo) for t in range(n)]
            path_abs_cents = viterbi_decode(states, dt_steps, h["fused_lambda_t"], 0.0)
            path_log2_hz = (path_abs_cents / 1200.0).astype(np.float32)

            if n < frames["frame_time_s"].shape[0]:
                pad = frames["frame_time_s"].shape[0] - n
                path_log2_hz = np.pad(path_log2_hz, (0, pad), mode="edge")
            out[rec_id] = path_log2_hz
            print(f"  fold {fold} {rec_id}: n_frames={len(path_log2_hz)}")

    REL_DIR.mkdir(parents=True, exist_ok=True)
    with open(DENSE_CACHE_PATH, "wb") as fh:
        pickle.dump(out, fh)
    print(f"Saved dense pitch paths for {len(out)} recordings to {DENSE_CACHE_PATH}")
    return out


def load_dense_estimated_pitch_fs0_5(path: Path = DENSE_CACHE_PATH) -> dict[str, np.ndarray]:
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------- driver


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("salience", "all"):
        print("=== (b) salience retrain, fs=0.5 ===")
        run_salience()
    if stage in ("register", "all"):
        print("\n=== (c) register/fusion recalibration, fs=0.5 ===")
        run_register()
    if stage in ("densepath", "all"):
        print("\n=== (d) dense Fused+D3 path rebuild, fs=0.5 ===")
        run_densepath(force=True)


if __name__ == "__main__":
    main()
