"""Step 11 §11-18: train the harmonic-salience / local-salience-control models.

Soft target (§11): Y(f,t) is a Gaussian in cents around the true pitch, built
directly in ABSOLUTE log2(Hz) space for both candidate bins and target
(distance-in-cents is tonic-invariant, so no per-example tonic bookkeeping is
needed at target-construction time — see salience_common.gaussian_soft_target_log2hz_torch).
sigma=30 cents (~1.8 CQT bins; documented in salience_common.py).

Loss (§12): KL(target || predicted) on log-softmax logits, masked by
valid_target & ~padding_mask.

Protocol matches Step 10's train_pitch.py exactly: AdamW lr=1e-3 wd=1e-4,
batch=8, seed=42 (+fold offset), early stopping on validation MAE (argmax
decode; expected-value decode is also computed for the argmax-vs-expected
comparison required by §13, but argmax is used for model selection/early
stopping to keep the protocol identical across folds/variants).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.folds import prepare_fold  # noqa: E402
from training.framewise_dataset import FramewiseExcerptDataset, RecordingLaneIndex, collate_excerpts  # noqa: E402
from training.metrics import pitch_error_metrics  # noqa: E402
from training.normalization import load_fold_cqt_stats, log2_hz_to_cents, normalize_cqt  # noqa: E402
from training.pitch_diagnostics.common import OUT_DIR, PRIMARY_LANE, write_json  # noqa: E402
from training.pitch_diagnostics.salience_common import (  # noqa: E402
    SALIENCE_SIGMA_CENTS,
    candidate_hz,
    gaussian_soft_target_log2hz_torch,
    load_or_compute_candidate_range,
)
from training.pitch_diagnostics.salience_models import HarmonicSalienceModel, count_params  # noqa: E402

VARIANT_CONFIG = {
    # Both variants use the same hidden width so the only architectural
    # difference is which harmonics feed the shared scorer (the B-vs-C
    # ablation itself); resulting param counts (local=4609, harmonic=5185)
    # are within ~11% of each other. Widened from the tiny-overfit debug
    # settings (hidden=24/16) per the capacity test showing local's plateau
    # is not parameter-count-limited.
    "local": {"harmonic_ks": (1,), "hidden": 64},
    "harmonic": {"harmonic_ks": (1, 2, 3, 4), "hidden": 64},
}
PARAM_BUDGET = 100_000


def build_model(variant: str, candidate_lo_bin: int, candidate_hi_bin: int, *, temporal_smoothing: bool = False) -> HarmonicSalienceModel:
    cfg = VARIANT_CONFIG[variant]
    return HarmonicSalienceModel(
        candidate_lo_bin=candidate_lo_bin,
        candidate_hi_bin=candidate_hi_bin,
        harmonic_ks=cfg["harmonic_ks"],
        hidden=cfg["hidden"],
        temporal_smoothing=temporal_smoothing,
    )


def decode(logits: torch.Tensor, candidate_log2_hz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """logits: [B,F_cand,T]. Returns (argmax_log2_hz, expected_log2_hz), both [B,T]."""
    idx = logits.argmax(dim=1)  # [B,T]
    argmax_log2_hz = candidate_log2_hz[idx]
    probs = F.softmax(logits, dim=1)
    expected_log2_hz = (probs * candidate_log2_hz.view(1, -1, 1)).sum(dim=1)
    return argmax_log2_hz, expected_log2_hz


def kl_loss(logits: torch.Tensor, target_dist: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)  # [B,F,T]
    per_elem = target_dist * (torch.log(target_dist.clamp_min(1e-12)) - log_probs)  # [B,F,T]
    per_frame = per_elem.sum(dim=1)  # [B,T]
    if mask.any():
        return per_frame[mask].mean()
    return per_frame.sum() * 0.0


def run_epoch(
    model: HarmonicSalienceModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    candidate_log2_hz: torch.Tensor,
    sigma_cents: float,
) -> dict[str, Any]:
    train = optimizer is not None
    model.train(train)
    total_loss, n_batches = 0.0, 0
    argmax_pred, expected_pred, true_c, fund = [], [], [], []

    for batch in loader:
        spec = batch["spec"].to(device)
        valid = batch["valid_target"].to(device)
        pad = batch["padding_mask"].to(device)
        mask = (~pad) & valid
        pitch_log2 = batch["pitch_log2_hz"].to(device)
        # pitch_log2_hz is NaN at invalid/padded frames by schema convention.
        # Those frames are excluded from the loss via `mask`, but 0*NaN=NaN
        # during backward would otherwise corrupt the shared 1x1-conv weight
        # gradients (accumulated over every time position) — sanitize first.
        pitch_log2_safe = torch.nan_to_num(pitch_log2, nan=0.0)

        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(spec)
        target_dist = gaussian_soft_target_log2hz_torch(pitch_log2_safe, candidate_log2_hz, sigma_cents=sigma_cents)
        loss = kl_loss(logits, target_dist, mask)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1

        with torch.no_grad():
            argmax_log2, expected_log2 = decode(logits.detach(), candidate_log2_hz)
        mask_np = mask.cpu().numpy()
        fund_np = batch["fundamental_hz"].numpy()  # [B]
        fund_b = np.broadcast_to(fund_np[:, None], mask_np.shape)
        argmax_pred.append(np.asarray(log2_hz_to_cents(argmax_log2.cpu().numpy(), 1.0))[mask_np])
        expected_pred.append(np.asarray(log2_hz_to_cents(expected_log2.cpu().numpy(), 1.0))[mask_np])
        true_c.append(np.asarray(log2_hz_to_cents(pitch_log2.cpu().numpy(), 1.0))[mask_np])
        fund.append(fund_b[mask_np])

    argmax_c = np.concatenate(argmax_pred) if argmax_pred else np.array([])
    expected_c = np.concatenate(expected_pred) if expected_pred else np.array([])
    true_abs_c = np.concatenate(true_c) if true_c else np.array([])
    fund_c = np.concatenate(fund) if fund else np.array([])
    # log2_hz_to_cents(x, 1.0) = 1200*log2(x) i.e. absolute-referenced cents;
    # subtract the per-example tonic term (also 1200*log2(fundamental_hz)) to
    # get the tonic-relative cents used everywhere else in this repo.
    tonic_term = 1200.0 * np.log2(np.maximum(fund_c, 1e-12))
    argmax_rel = argmax_c - tonic_term
    expected_rel = expected_c - tonic_term
    true_rel = true_abs_c - tonic_term

    argmax_metrics = pitch_error_metrics(argmax_rel, true_rel)
    expected_metrics = pitch_error_metrics(expected_rel, true_rel)
    return {
        "loss": total_loss / max(n_batches, 1),
        "argmax": argmax_metrics,
        "expected": expected_metrics,
        "mae_cents": argmax_metrics["mae_cents"],
    }


def train_one(
    *,
    variant: str,
    fold: int,
    run_name: str,
    tiny_overfit: int = 0,
    max_epochs: int | None = None,
    temporal_smoothing: bool = False,
    repo_root: Path = REPO_ROOT,
    index: RecordingLaneIndex | None = None,
    candidate_range: dict[str, Any] | None = None,
) -> dict[str, Any]:
    seed = 42
    torch.manual_seed(seed + fold)
    np.random.seed(seed + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split, _summary = prepare_fold(repo_root, fold, seed=seed)
    index = index or RecordingLaneIndex.build(repo_root)
    candidate_range = candidate_range or load_or_compute_candidate_range(repo_root, index)
    lo_bin, hi_bin = candidate_range["candidate_lo_bin"], candidate_range["candidate_hi_bin"]
    cand_hz = candidate_hz(lo_bin, hi_bin)
    candidate_log2_hz = torch.from_numpy(np.log2(cand_hz).astype(np.float32)).to(device)

    mu_cqt, sigma_cqt = load_fold_cqt_stats(fold, repo_root)

    model = build_model(variant, lo_bin, hi_bin, temporal_smoothing=temporal_smoothing).to(device)
    n_params = count_params(model)
    if n_params > PARAM_BUDGET:
        raise RuntimeError(f"salience model too large: {n_params} > {PARAM_BUDGET}")

    train_ids = split.train_recording_ids
    val_ids = split.val_recording_ids

    if tiny_overfit:
        cache_n = tiny_overfit
        train_ds = FramewiseExcerptDataset(
            index, train_ids, mu_cqt, sigma_cqt, seed=seed + fold,
            excerpts_per_epoch=cache_n, cache_excerpts=cache_n,
        )
        val_ds = train_ds  # tiny-overfit sanity check: can the model memorize this exact set?
    else:
        train_ds = FramewiseExcerptDataset(
            index, train_ids, mu_cqt, sigma_cqt, seed=seed + fold, excerpts_per_epoch=512,
        )
        val_ds = FramewiseExcerptDataset(
            index, val_ids, mu_cqt, sigma_cqt, seed=seed + fold + 1000,
            excerpts_per_epoch=max(len(val_ids) * 20, 64),
        )

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False, collate_fn=collate_excerpts)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_excerpts)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = max_epochs or (150 if tiny_overfit else 20)
    patience = 150 if tiny_overfit else 5
    best_mae, best_epoch, best_state, stale = 1e9, -1, None, 0
    history = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, opt, device, candidate_log2_hz=candidate_log2_hz, sigma_cents=SALIENCE_SIGMA_CENTS)
        va = run_epoch(model, val_loader, None, device, candidate_log2_hz=candidate_log2_hz, sigma_cents=SALIENCE_SIGMA_CENTS)
        history.append({"epoch": epoch, "train": tr, "val": va})
        print(
            f"{run_name} fold {fold} ep {epoch}: train loss={tr['loss']:.4f} MAE={tr['mae_cents']:.1f} "
            f"val loss={va['loss']:.4f} MAE={va['mae_cents']:.1f}"
        )
        if va["mae_cents"] < best_mae:
            best_mae = va["mae_cents"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience and not tiny_overfit:
                print("early stop", epoch)
                break
    assert best_state is not None
    model.load_state_dict(best_state)

    run_dir = OUT_DIR / "runs" / run_name / f"fold_{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state": best_state,
        "n_params": n_params,
        "variant": variant,
        "harmonic_ks": VARIANT_CONFIG[variant]["harmonic_ks"],
        "candidate_lo_bin": lo_bin,
        "candidate_hi_bin": hi_bin,
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_mae": best_mae,
    }
    torch.save(ckpt, run_dir / "best.pt")

    result = {
        "run_name": run_name,
        "variant": variant,
        "fold": fold,
        "n_params": n_params,
        "candidate_lo_bin": lo_bin,
        "candidate_hi_bin": hi_bin,
        "sigma_cents": SALIENCE_SIGMA_CENTS,
        "temporal_smoothing": temporal_smoothing,
        "best_epoch": best_epoch,
        "best_val_mae_cents": best_mae,
        "tiny_overfit": tiny_overfit,
        "final_train": history[-1]["train"],
        "final_val": history[-1]["val"],
        "initial_train": history[0]["train"],
        "elapsed_s": time.time() - t0,
        "history_tail": history[-3:],
    }
    write_json(run_dir / "result.json", result)
    torch.save({"history": history}, run_dir / "history.pt")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=("local", "harmonic"), default="harmonic")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--all-folds", action="store_true")
    p.add_argument("--tiny-overfit", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--temporal-smoothing", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"{args.variant}_salience_abs"
    folds = list(range(5)) if args.all_folds else [args.fold]
    shared_index = RecordingLaneIndex.build(REPO_ROOT)
    candidate_range = load_or_compute_candidate_range(REPO_ROOT, shared_index)
    results = []
    for fold in folds:
        results.append(
            train_one(
                variant=args.variant,
                fold=fold,
                run_name=run_name,
                tiny_overfit=args.tiny_overfit,
                max_epochs=args.max_epochs,
                temporal_smoothing=args.temporal_smoothing,
                index=shared_index,
                candidate_range=candidate_range,
            )
        )
    if not args.tiny_overfit:
        summary = {
            "run_name": run_name,
            "variant": args.variant,
            "folds": results,
            "mean_val_mae": float(np.mean([r["best_val_mae_cents"] for r in results])),
        }
        write_json(OUT_DIR / "runs" / run_name / "cv_summary.json", summary)
        print(json.dumps({k: v for k, v in summary.items() if k != "folds"}, indent=2))


if __name__ == "__main__":
    main()
