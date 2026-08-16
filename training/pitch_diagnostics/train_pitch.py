"""Train/eval the Step 10 frequency-preserving pitch-only model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.folds import build_fold_split, load_kfold_manifest, prepare_fold  # noqa: E402
from training.framewise_dataset import RecordingLaneIndex  # noqa: E402
from training.metrics import pitch_error_metrics  # noqa: E402
from training.normalization import (  # noqa: E402
    denormalize_pitch,
    load_fold_cqt_stats,
    load_fold_pitch_stats,
    log2_hz_to_cents,
    normalize_cqt,
)
from training.pitch_diagnostics.common import (  # noqa: E402
    OUT_DIR,
    PRIMARY_LANE,
    tonic_align_cqt,
    write_json,
)
from training.pitch_diagnostics.dataset import PitchExcerptDataset  # noqa: E402
from training.pitch_diagnostics.models import (  # noqa: E402
    FREQ_AFTER_POOL,
    FreqPreservingPitchModel,
    bins_to_cents,
    count_params,
)
from training.features import N_BINS as CQT_BINS  # noqa: E402


def collate_pitch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "spec": torch.stack([b["spec"] for b in batch]),
        "valid_target": torch.stack([b["valid_target"] for b in batch]),
        "padding_mask": torch.stack([b["padding_mask"] for b in batch]),
        "pitch_cents": torch.stack([b["pitch_cents"] for b in batch]),
        "pitch_standardized": torch.stack([b["pitch_standardized"] for b in batch]),
        "recording_id": [b["recording_id"] for b in batch],
    }


def aligned_cqt_stats(index: RecordingLaneIndex, rec_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros(CQT_BINS, dtype=np.float64)
    sumsq = np.zeros(CQT_BINS, dtype=np.float64)
    count = 0
    for rec_id in rec_ids:
        lane = next(x for x in index.lanes if x.recording_id == rec_id)
        cqt = tonic_align_cqt(index._features[rec_id]["cqt_log"], lane.fundamental_hz)
        sums += cqt.sum(axis=1)
        sumsq += (cqt.astype(np.float64) ** 2).sum(axis=1)
        count += cqt.shape[1]
    mu = sums / max(count, 1)
    var = sumsq / max(count, 1) - mu ** 2
    sigma = np.maximum(np.sqrt(np.maximum(var, 0.0)), 1e-3)
    return mu.astype(np.float32), sigma.astype(np.float32)


def train_cents_range(index: RecordingLaneIndex, rec_ids: list[str]) -> tuple[float, float]:
    vals = []
    for rec_id in rec_ids:
        lane = next(x for x in index.lanes if x.recording_id == rec_id)
        frames = index._frames[(rec_id, PRIMARY_LANE)]
        mask = frames["valid_target"]
        cents = log2_hz_to_cents(frames["pitch_log2_hz"][mask], lane.fundamental_hz)
        vals.append(np.asarray(cents, dtype=np.float64))
    arr = np.concatenate(vals)
    lo, hi = np.quantile(arr, [0.005, 0.995])
    return float(lo), float(hi)


def choose_bin_width(lo: float, hi: float) -> dict[str, Any]:
    span = hi - lo
    rows = []
    for width in (10.0, 20.0, 25.0):
        n = int(np.ceil(span / width))
        rows.append({"width": width, "n_classes": n, "mean_support_if_uniform": None, "span": span})
    # Prefer 25 cents unless that yields < 40 classes (too coarse for ~5 octaves)
    chosen = 25.0
    n = int(np.ceil(span / chosen))
    return {"lo": lo, "hi": hi, "width": chosen, "n_classes": n, "candidates": rows, "span_cents": span}


def make_bin_centers(lo: float, hi: float, width: float, n: int) -> np.ndarray:
    return lo + (np.arange(n) + 0.5) * width


def cents_to_bin_index(cents: torch.Tensor, lo: float, width: float, n: int) -> torch.Tensor:
    idx = torch.floor((cents - lo) / width).long()
    return idx.clamp(0, n - 1)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    head: str,
    mu_pitch: float,
    sigma_pitch: float,
    bin_meta: dict[str, Any] | None,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    n_batches = 0
    pred_c, true_c = [], []
    smooth = torch.nn.SmoothL1Loss()
    ce = torch.nn.CrossEntropyLoss()
    centers = None
    if bin_meta is not None:
        centers = torch.tensor(bin_meta["centers"], dtype=torch.float32, device=device)

    for batch in loader:
        spec = batch["spec"].to(device)
        valid = batch["valid_target"].to(device)
        pad = batch["padding_mask"].to(device)
        mask = (~pad) & valid
        if train:
            optimizer.zero_grad(set_to_none=True)
        out = model(spec)
        if head == "scalar":
            target = batch["pitch_standardized"].to(device)
            if mask.any():
                loss = smooth(out[mask], target[mask])
            else:
                loss = out.sum() * 0
            pred_std = out.detach()
            pred_cents = denormalize_pitch(pred_std.cpu().numpy(), mu_pitch, sigma_pitch)
        else:
            assert bin_meta is not None and centers is not None
            cents = batch["pitch_cents"].to(device)
            idx = cents_to_bin_index(cents, bin_meta["lo"], bin_meta["width"], bin_meta["n_classes"])
            b, t, n = out.shape
            if mask.any():
                loss = ce(out[mask], idx[mask])
            else:
                loss = out.sum() * 0
            argmax_c, _ = bins_to_cents(out.detach(), centers)
            pred_cents = argmax_c.cpu().numpy()
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
        true_np = batch["pitch_cents"].numpy()
        mask_np = mask.cpu().numpy()
        pred_c.append(pred_cents[mask_np])
        true_c.append(true_np[mask_np])

    p = np.concatenate(pred_c) if pred_c else np.array([])
    t = np.concatenate(true_c) if true_c else np.array([])
    metrics = pitch_error_metrics(p, t)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


@torch.no_grad()
def eval_recordings(
    model: torch.nn.Module,
    index: RecordingLaneIndex,
    rec_ids: list[str],
    device: torch.device,
    *,
    mu_cqt: np.ndarray,
    sigma_cqt: np.ndarray,
    mu_pitch: float,
    sigma_pitch: float,
    head: str,
    tonic_align: bool,
    bin_meta: dict[str, Any] | None,
    time_lo_frac: float = 0.0,
    time_hi_frac: float = 1.0,
    chunk_frames: int = 800,
) -> dict[str, Any]:
    model.eval()
    centers = None
    if bin_meta is not None:
        centers = torch.tensor(bin_meta["centers"], dtype=torch.float32, device=device)
    preds, trues = [], []
    per_rec = {}
    for rec_id in rec_ids:
        lane = next(x for x in index.lanes if x.recording_id == rec_id)
        frames = index._frames[(rec_id, PRIMARY_LANE)]
        cqt = index._features[rec_id]["cqt_log"]
        if tonic_align:
            cqt = tonic_align_cqt(cqt, lane.fundamental_hz)
        spec = normalize_cqt(cqt, mu_cqt, sigma_cqt)
        n = min(cqt.shape[1], lane.n_frames)
        times = frames["frame_time_s"][:n]
        valid = frames["valid_target"][:n] & (times < lane.duration_s)
        if time_lo_frac <= 0.0 and time_hi_frac >= 1.0:
            mask = valid
        else:
            # Match PitchExcerptDataset: window over valid-frame order.
            idx = np.flatnonzero(valid)
            mask = np.zeros(n, dtype=bool)
            if idx.size:
                lo = int(time_lo_frac * idx.size)
                hi = max(lo + 1, int(time_hi_frac * idx.size))
                mask[idx[lo:hi]] = True
        true = np.asarray(log2_hz_to_cents(frames["pitch_log2_hz"][:n], lane.fundamental_hz))
        pred = np.full(n, np.nan, dtype=np.float64)
        # Chunk along time so long concerts do not OOM in one forward.
        for start in range(0, n, chunk_frames):
            end = min(start + chunk_frames, n)
            if not np.any(mask[start:end]):
                continue
            x = torch.from_numpy(spec[np.newaxis, np.newaxis, :, start:end].astype(np.float32)).to(device)
            out = model(x)
            if head == "scalar":
                pred[start:end] = denormalize_pitch(out[0].cpu().numpy(), mu_pitch, sigma_pitch)
            else:
                assert centers is not None
                argmax_c, _ = bins_to_cents(out, centers)
                pred[start:end] = argmax_c[0].cpu().numpy()
        per_rec[rec_id] = pitch_error_metrics(pred[mask], true[mask])
        preds.append(pred[mask])
        trues.append(true[mask])
    p = np.concatenate(preds) if preds else np.array([])
    t = np.concatenate(trues) if trues else np.array([])
    return {
        "overall": pitch_error_metrics(p, t),
        "mean_pitch_baseline": pitch_error_metrics(np.full_like(t, mu_pitch), t) if len(t) else pitch_error_metrics(p, t),
        "per_recording": per_rec,
    }


def train_one(
    *,
    fold: int,
    head: str,
    tonic_align: bool,
    run_name: str,
    tiny_overfit: int = 0,
    max_epochs: int | None = None,
    within_recording: bool = False,
    repo_root: Path = REPO_ROOT,
    index: RecordingLaneIndex | None = None,
) -> dict[str, Any]:
    seed = 42
    torch.manual_seed(seed + fold)
    np.random.seed(seed + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split, _summary = prepare_fold(repo_root, fold, seed=seed)
    index = index or RecordingLaneIndex.build(repo_root)
    mu_pitch, sigma_pitch = load_fold_pitch_stats(fold, repo_root)

    if tonic_align:
        mu_cqt, sigma_cqt = aligned_cqt_stats(index, split.train_recording_ids)
    else:
        mu_cqt, sigma_cqt = load_fold_cqt_stats(fold, repo_root)

    bin_meta = None
    n_bins = None
    if head == "bins":
        lo, hi = train_cents_range(index, split.train_recording_ids)
        choice = choose_bin_width(lo, hi)
        n_bins = choice["n_classes"]
        centers = make_bin_centers(lo, hi, choice["width"], n_bins)
        # support per bin
        vals = []
        for rec_id in split.train_recording_ids:
            lane = next(x for x in index.lanes if x.recording_id == rec_id)
            frames = index._frames[(rec_id, PRIMARY_LANE)]
            cents = np.asarray(log2_hz_to_cents(frames["pitch_log2_hz"][frames["valid_target"]], lane.fundamental_hz))
            vals.append(cents)
        arr = np.concatenate(vals)
        idx = np.floor((np.clip(arr, lo, hi - 1e-6) - lo) / choice["width"]).astype(int)
        idx = np.clip(idx, 0, n_bins - 1)
        counts = np.bincount(idx, minlength=n_bins)
        choice["mean_support_per_bin"] = float(counts.mean())
        choice["median_support_per_bin"] = float(np.median(counts))
        choice["empty_bins"] = int((counts == 0).sum())
        choice["centers"] = centers.tolist()
        bin_meta = choice

    model = FreqPreservingPitchModel(n_pitch_bins=n_bins).to(device)
    n_params = count_params(model)
    if n_params > 80_000:
        raise RuntimeError(f"pitch model too large: {n_params}")

    train_lo, train_hi = (0.0, 0.7) if within_recording else (0.0, 1.0)
    val_ids = split.train_recording_ids if within_recording else split.val_recording_ids
    train_ids = split.train_recording_ids[:1] if tiny_overfit else (
        index.lanes and [l.recording_id for l in index.lanes] if within_recording else split.train_recording_ids
    )
    if within_recording and not tiny_overfit:
        train_ids = [l.recording_id for l in index.lanes]

    ds_kw = dict(
        mu_pitch=mu_pitch,
        sigma_pitch=sigma_pitch,
        tonic_align=tonic_align,
        time_lo_frac=train_lo,
        time_hi_frac=train_hi,
    )
    train_ds = PitchExcerptDataset(
        index, train_ids, mu_cqt, sigma_cqt,
        seed=seed + fold,
        excerpts_per_epoch=tiny_overfit or 512,
        cache_excerpts=tiny_overfit or None,
        **ds_kw,
    )
    val_ds = PitchExcerptDataset(
        index, val_ids, mu_cqt, sigma_cqt,
        seed=seed + fold + 1000,
        excerpts_per_epoch=64 if tiny_overfit else max(len(val_ids) * 20, 64),
        time_lo_frac=0.7 if within_recording else 0.0,
        time_hi_frac=1.0,
        tonic_align=tonic_align,
        mu_pitch=mu_pitch,
        sigma_pitch=sigma_pitch,
    )
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False, collate_fn=collate_pitch)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_pitch)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = max_epochs or (100 if tiny_overfit else 20)
    patience = 100 if tiny_overfit else 5
    best_mae = 1e9
    best_state = None
    best_epoch = -1
    stale = 0
    history = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr = run_epoch(
            model, train_loader, opt, device, head=head,
            mu_pitch=mu_pitch, sigma_pitch=sigma_pitch, bin_meta=bin_meta,
        )
        va = run_epoch(
            model, val_loader, None, device, head=head,
            mu_pitch=mu_pitch, sigma_pitch=sigma_pitch, bin_meta=bin_meta,
        )
        history.append({"epoch": epoch, "train": tr, "val": va})
        print(
            f"{run_name} fold {fold} ep {epoch}: "
            f"train MAE={tr['mae_cents']:.1f} val MAE={va['mae_cents']:.1f} loss={tr['loss']:.3f}"
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
        "head": head,
        "tonic_align": tonic_align,
        "fold": fold,
        "mu_pitch": mu_pitch,
        "sigma_pitch": sigma_pitch,
        "bin_meta": bin_meta,
        "best_epoch": best_epoch,
        "best_val_mae": best_mae,
    }
    torch.save(ckpt, run_dir / "best.pt")
    if tonic_align:
        np.savez(run_dir / "aligned_cqt_stats.npz", mu=mu_cqt, sigma=sigma_cqt)

    test_ids = split.test_recording_ids
    eval_lo, eval_hi = (0.7, 1.0) if within_recording else (0.0, 1.0)
    if within_recording:
        test_ids = [l.recording_id for l in index.lanes]
    test = eval_recordings(
        model, index, test_ids, device,
        mu_cqt=mu_cqt, sigma_cqt=sigma_cqt, mu_pitch=mu_pitch, sigma_pitch=sigma_pitch,
        head=head, tonic_align=tonic_align, bin_meta=bin_meta,
        time_lo_frac=eval_lo, time_hi_frac=eval_hi,
    )
    result = {
        "run_name": run_name,
        "fold": fold,
        "n_params": n_params,
        "head": head,
        "tonic_align": tonic_align,
        "within_recording": within_recording,
        "best_epoch": best_epoch,
        "best_val_mae_cents": best_mae,
        "tiny_overfit": tiny_overfit,
        "bin_meta": {k: v for k, v in (bin_meta or {}).items() if k != "centers"} if bin_meta else None,
        "test": test,
        "mean_pitch_baseline_mae_cents": (test.get("mean_pitch_baseline") or {}).get("mae_cents"),
        "elapsed_s": time.time() - t0,
        "history_tail": history[-3:],
    }
    # mean baseline from test overall: need true cents. Re-eval cheaply from per_recording? skip and compute in eval_recordings
    write_json(run_dir / "result.json", result)
    torch.save({"history": history}, run_dir / "history.pt")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--head", choices=("scalar", "bins"), default="scalar")
    p.add_argument("--tonic-align", action="store_true")
    p.add_argument("--within-recording", action="store_true")
    p.add_argument("--run-name", type=str, default="scalar_abs")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--all-folds", action="store_true")
    p.add_argument("--tiny-overfit", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    folds = list(range(5)) if args.all_folds else [args.fold]
    results = []
    # Reuse one RecordingLaneIndex across folds to avoid multi-GB memory growth.
    shared_index = RecordingLaneIndex.build(REPO_ROOT)
    for fold in folds:
        results.append(
            train_one(
                fold=fold,
                head=args.head,
                tonic_align=args.tonic_align,
                run_name=args.run_name,
                tiny_overfit=args.tiny_overfit,
                max_epochs=args.max_epochs,
                within_recording=args.within_recording,
                index=shared_index,
            )
        )
    summary = {
        "run_name": args.run_name,
        "folds": results,
        "mean_test_mae": float(np.mean([r["test"]["overall"]["mae_cents"] for r in results])),
        "n_beats_mean_baseline": int(
            sum(
                1
                for r in results
                if r["test"]["overall"]["mae_cents"]
                < r["test"]["mean_pitch_baseline"]["mae_cents"]
            )
        ),
    }
    write_json(OUT_DIR / "runs" / args.run_name / "cv_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "folds"}, indent=2))


if __name__ == "__main__":
    main()
