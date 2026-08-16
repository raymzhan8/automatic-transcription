"""Run librosa.pyin on grouped test folds only; write incremental JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.canonical.schema import recordings_dir  # noqa: E402
from dataset.canonical.visualize_targets import source_audio_path  # noqa: E402
from training.features import BINS_PER_OCTAVE, CQT_HOP, FMIN, N_BINS, SR  # noqa: E402
from training.folds import build_fold_split, load_kfold_manifest  # noqa: E402
from training.framewise_dataset import RecordingLaneIndex  # noqa: E402
from training.metrics import pitch_error_metrics  # noqa: E402
from training.normalization import log2_hz_to_cents  # noqa: E402
from training.pitch_diagnostics.baselines_b import pyin_recording  # noqa: E402
from training.pitch_diagnostics.common import OUT_DIR, PRIMARY_LANE, write_json  # noqa: E402


def main() -> None:
    index = RecordingLaneIndex.build(REPO_ROOT)
    manifest = load_kfold_manifest(REPO_ROOT)
    rec_docs = {
        p.stem: json.loads(p.read_text())
        for p in recordings_dir(REPO_ROOT).glob("*.json")
    }
    folds = []
    out_path = OUT_DIR / "pyin.json"
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
                print(f"fold {i} skip {rec_id}: no audio", flush=True)
                continue
            frames = index._frames[(rec_id, PRIMARY_LANE)]
            times = frames["frame_time_s"]
            n = min(len(times), lane.n_frames)
            mask = frames["valid_target"][:n] & (times[:n] < lane.duration_s)
            true = np.asarray(log2_hz_to_cents(frames["pitch_log2_hz"][:n], lane.fundamental_hz))
            print(
                f"fold {i} pyin {rec_id} valid={float(mask.sum())*0.01:.1f}s ...",
                flush=True,
            )
            t0 = __import__("time").time()
            pred = pyin_recording(
                audio, times[:n], lane.fundamental_hz, valid_mask=mask
            )
            print(f"  done in {__import__('time').time()-t0:.1f}s", flush=True)
            pv, tv = pred[mask], true[mask]
            per_rec[rec_id] = pitch_error_metrics(pv, tv)
            print(f"  MAE={per_rec[rec_id]['mae_cents']:.1f}", flush=True)
            preds.append(pv)
            trues.append(tv)
        p = np.concatenate(preds) if preds else np.array([])
        t = np.concatenate(trues) if trues else np.array([])
        folds.append(
            {
                "fold": i,
                "overall": pitch_error_metrics(p, t),
                "per_recording": per_rec,
                "skipped": skipped,
            }
        )
        write_json(
            out_path,
            {
                "folds": folds,
                "note": (
                    f"librosa.pyin hop={CQT_HOP} fmin={FMIN} "
                    f"fmax={FMIN * (2 ** (N_BINS / BINS_PER_OCTAVE))}; "
                    "interpolated to 10 ms grid; unvoiced filled with tonic"
                ),
            },
        )
        print(f"fold {i} overall MAE={folds[-1]['overall']['mae_cents']:.1f}", flush=True)
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
