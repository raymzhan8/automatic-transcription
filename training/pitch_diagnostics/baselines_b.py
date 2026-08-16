"""Diagnostic B: CQT peak baselines + librosa.pyin on valid frames."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import librosa
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.canonical.visualize_targets import source_audio_path  # noqa: E402
from dataset.canonical.schema import recordings_dir  # noqa: E402
from training.features import (  # noqa: E402
    BINS_PER_OCTAVE,
    CQT_HOP,
    FMIN,
    N_BINS,
    SR,
    interpolate_cqt_to_target_grid,
)
from training.folds import build_fold_split, load_kfold_manifest  # noqa: E402
from training.framewise_dataset import RecordingLaneIndex  # noqa: E402
from training.metrics import pitch_error_metrics  # noqa: E402
from training.normalization import log2_hz_to_cents  # noqa: E402
from training.pitch_diagnostics.common import (  # noqa: E402
    OUT_DIR,
    PRIMARY_LANE,
    cents_from_hz,
    hz_from_bin,
    linear_mag,
    write_json,
)
import json


def _valid(index: RecordingLaneIndex, rec_id: str, duration_s: float) -> np.ndarray:
    frames = index._frames[(rec_id, PRIMARY_LANE)]
    return frames["valid_target"] & (frames["frame_time_s"] < duration_s)


def cqt_argmax_cents(mag: np.ndarray, tonic_hz: float, bin_lo: int = 0, bin_hi: int | None = None) -> np.ndarray:
    sl = mag[bin_lo:bin_hi] if bin_hi is not None else mag[bin_lo:]
    peaks = sl.argmax(axis=0) + bin_lo
    hz = hz_from_bin(peaks.astype(np.float64))
    return np.asarray(cents_from_hz(hz, tonic_hz), dtype=np.float64)


def harmonic_product_cents(mag: np.ndarray, tonic_hz: float) -> np.ndarray:
    """Log-frequency harmonic product spectrum (first 3 harmonics), then argmax."""
    f_bins, n_t = mag.shape
    hps = mag.copy()
    for k in (2, 3):
        # downsample frequency by k (approx harmonic on log grid: k * f is +72*log2(k) bins)
        shift = int(round(BINS_PER_OCTAVE * np.log2(k)))
        padded = np.zeros_like(mag)
        if shift < f_bins:
            padded[: f_bins - shift] = mag[shift:]
        hps = hps * np.maximum(padded, 1e-12)
    return cqt_argmax_cents(hps, tonic_hz)


def eval_fold(
    index: RecordingLaneIndex,
    rec_ids: list[str],
    pred_fn,
) -> dict[str, Any]:
    preds, trues = [], []
    per_rec = {}
    for rec_id in rec_ids:
        lane = next(x for x in index.lanes if x.recording_id == rec_id)
        frames = index._frames[(rec_id, PRIMARY_LANE)]
        mag = linear_mag(index._features[rec_id]["cqt_log"])
        n = min(mag.shape[1], lane.n_frames)
        mag = mag[:, :n]
        mask = _valid(index, rec_id, lane.duration_s)[:n]
        true = np.asarray(log2_hz_to_cents(frames["pitch_log2_hz"][:n], lane.fundamental_hz))
        pred = pred_fn(mag, lane.fundamental_hz)
        pv, tv = pred[mask], true[mask]
        per_rec[rec_id] = pitch_error_metrics(pv, tv)
        preds.append(pv)
        trues.append(tv)
    p = np.concatenate(preds) if preds else np.array([])
    t = np.concatenate(trues) if trues else np.array([])
    return {"overall": pitch_error_metrics(p, t), "per_recording": per_rec}


def _valid_time_segments(
    times: np.ndarray,
    mask: np.ndarray,
    *,
    pad_s: float = 1.0,
    merge_gap_s: float = 30.0,
) -> list[tuple[float, float]]:
    """Merge valid-frame times into padded [t0, t1] segments for local pyin.

    Large merge gaps avoid hundreds of tiny pyin calls on dense annotations.
    """
    vt = np.asarray(times[mask], dtype=np.float64)
    if vt.size == 0:
        return []
    segs: list[tuple[float, float]] = []
    a = float(vt[0])
    b = float(vt[0])
    for t in vt[1:]:
        t = float(t)
        if t - b > merge_gap_s:
            segs.append((max(0.0, a - pad_s), b + pad_s))
            a = t
        b = t
    segs.append((max(0.0, a - pad_s), b + pad_s))
    return segs


def pyin_recording(
    audio_path: Path,
    times: np.ndarray,
    tonic_hz: float,
    *,
    valid_mask: np.ndarray | None = None,
    pad_s: float = 1.0,
    chunk_s: float = 60.0,
) -> np.ndarray:
    """Run librosa.pyin only around valid-target segments (full concerts are huge)."""
    y, sr = librosa.load(str(audio_path), sr=SR, mono=True)
    fmax = FMIN * (2 ** (N_BINS / BINS_PER_OCTAVE))
    hop = CQT_HOP
    n = len(y)
    out_hz = np.full(len(times), tonic_hz, dtype=np.float64)
    mask = np.ones(len(times), dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    segments = _valid_time_segments(times, mask, pad_s=pad_s)
    chunk = max(int(chunk_s * sr), hop * 8)

    def _run_chunk(i0: int, i1: int) -> None:
        i0 = (i0 // hop) * hop
        if i1 - i0 < hop * 4:
            return
        y_chunk = y[i0:i1]
        f0, voiced, _ = librosa.pyin(
            y_chunk,
            fmin=FMIN,
            fmax=fmax,
            sr=sr,
            hop_length=hop,
        )
        native = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop) + (i0 / sr)
        f0 = np.asarray(f0, dtype=np.float64)
        voiced = np.asarray(voiced, dtype=bool)
        f0 = np.where(voiced & np.isfinite(f0), f0, np.nan)
        in_seg = (times >= native[0]) & (times <= native[-1] + hop / sr)
        if not np.any(in_seg):
            return
        interp = np.interp(times[in_seg], native, f0, left=np.nan, right=np.nan)
        good = np.isfinite(interp)
        out_hz[np.where(in_seg)[0][good]] = interp[good]

    for t0, t1 in segments:
        i0 = int(max(0, np.floor(t0 * sr)))
        i1 = int(min(n, np.ceil(t1 * sr)))
        start = i0
        while start < i1:
            end = min(start + chunk, i1)
            _run_chunk(start, end)
            if end >= i1:
                break
            start = end - hop * 4  # tiny overlap
    return np.asarray(cents_from_hz(out_hz, tonic_hz), dtype=np.float64)


def main() -> None:
    index = RecordingLaneIndex.build(REPO_ROOT)
    manifest = load_kfold_manifest(REPO_ROOT)
    methods = {}

    def argmax(mag, tonic):
        return cqt_argmax_cents(mag, tonic)

    def neighborhood(mag, tonic):
        lo = int(np.clip(np.rint(BINS_PER_OCTAVE * np.log2(max(tonic, 1e-6) / FMIN)), 0, N_BINS - 2))
        hi = int(np.clip(lo + 2 * BINS_PER_OCTAVE, lo + 1, N_BINS))
        return cqt_argmax_cents(mag, tonic, bin_lo=lo, bin_hi=hi)

    def hps(mag, tonic):
        return harmonic_product_cents(mag, tonic)

    for name, fn in (("cqt_argmax", argmax), ("tonic_neighborhood", neighborhood), ("harmonic_product", hps)):
        folds = []
        for i in range(5):
            split = build_fold_split(manifest, i, seed=42)
            folds.append({"fold": i, **eval_fold(index, split.test_recording_ids, fn)})
        methods[name] = {"folds": folds}

    try:
        pyin_folds = []
        rec_docs = {
            p.stem: json.loads(p.read_text())
            for p in recordings_dir(REPO_ROOT).glob("*.json")
        }
        for i in range(5):
            split = build_fold_split(manifest, i, seed=42)
            preds, trues = [], []
            per_rec = {}
            skipped = []
            for rec_id in split.test_recording_ids:
                lane = next(x for x in index.lanes if x.recording_id == rec_id)
                doc = rec_docs[rec_id]
                audio = source_audio_path(doc, REPO_ROOT)
                if audio is None:
                    skipped.append(rec_id)
                    continue
                frames = index._frames[(rec_id, PRIMARY_LANE)]
                times = frames["frame_time_s"]
                n = min(len(times), lane.n_frames)
                mask = _valid(index, rec_id, lane.duration_s)[:n]
                true = np.asarray(
                    log2_hz_to_cents(frames["pitch_log2_hz"][:n], lane.fundamental_hz)
                )
                pred = pyin_recording(audio, times[:n], lane.fundamental_hz)
                pv, tv = pred[mask], true[mask]
                per_rec[rec_id] = pitch_error_metrics(pv, tv)
                preds.append(pv)
                trues.append(tv)
            p = np.concatenate(preds) if preds else np.array([])
            t = np.concatenate(trues) if trues else np.array([])
            pyin_folds.append(
                {
                    "fold": i,
                    "overall": pitch_error_metrics(p, t),
                    "per_recording": per_rec,
                    "skipped": skipped,
                }
            )
        methods["pyin"] = {
            "folds": pyin_folds,
            "note": (
                "librosa.pyin hop=220 fmin=75 fmax=2400; interpolated to 10 ms "
                "grid; unvoiced filled with tonic"
            ),
        }
    except Exception as exc:
        methods["pyin"] = {
            "skipped": True,
            "reason": f"librosa.pyin unavailable in this environment: {type(exc).__name__}: {exc}",
        }
    write_json(OUT_DIR / "baselines.json", methods)
    print("Diagnostic B written", OUT_DIR / "baselines.json")
    for name, block in methods.items():
        if block.get("skipped"):
            print(name, "skipped:", block["reason"])
            continue
        maes = [f["overall"]["mae_cents"] for f in block["folds"]]
        print(name, "MAE folds", [round(m, 1) for m in maes], "mean", round(float(np.mean(maes)), 1))


if __name__ == "__main__":
    main()
