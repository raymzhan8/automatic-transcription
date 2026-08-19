"""Step 21: dense per-recording pitch path from a pretrained, generic
monophonic pitch tracker (CREPE, via torchcrepe) run on the already-
separated "vocals" stem (falling back to source audio), frozen as of Step
21 as the project's default estimated-pitch source, replacing the
CQT/salience/Viterbi pipeline (`dense_pitch_path.py`, D1). No training --
CREPE's public pretrained weights only. Produces the same {recording_id:
log2_hz per native 10ms frame} shape as dense_framewise_argmax_path.py /
dense_pitch_path.py so it can be evaluated with the identical Step 16-17
diagnostics via pitch_audit/common.py's `_load_estimated_pitch_variant`,
and used as a `--pitch-variant CREPE` override in
train_pitch_motion_ablation.py.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torchcrepe

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.canonical.visualize_targets import source_audio_path  # noqa: E402
from training.framewise_dataset import PRIMARY_LANE, RecordingLaneIndex  # noqa: E402
from training.pitch_diagnostics.relative_pitch.path_cache import REL_DIR  # noqa: E402

DENSE_CACHE_PATH = REL_DIR / "dense_crepe_log2hz.pkl"
CREPE_SR = 16000
HOP_SAMPLES = 160  # 10ms at 16kHz, matches the canonical grid
FMIN_HZ, FMAX_HZ = 65.0, 900.0  # a bit wider than the 104-778Hz candidate band, no hard edge clipping
MODEL = "full"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def _vocals_or_source_path(recording_doc: dict, repo_root: Path) -> Path:
    for item in recording_doc.get("audio", {}).get("files") or []:
        if item.get("role") == "vocals" and item.get("relpath"):
            p = repo_root / item["relpath"]
            if p.exists():
                return p
    return source_audio_path(recording_doc, repo_root)


def build(force: bool = False) -> dict[str, np.ndarray]:
    if DENSE_CACHE_PATH.exists() and not force:
        print(f"Loading cached CREPE paths from {DENSE_CACHE_PATH}")
        with open(DENSE_CACHE_PATH, "rb") as fh:
            return pickle.load(fh)

    index = RecordingLaneIndex.build(REPO_ROOT)
    out: dict[str, np.ndarray] = {}
    t_start = time.time()
    for i, lane in enumerate(index.lanes):
        rid = lane.recording_id
        frames = index._frames[(rid, PRIMARY_LANE)]
        target_times = frames["frame_time_s"]

        rec_path = REPO_ROOT / "output" / "canonical" / "v1" / "recordings" / f"{rid}.json"
        recording_doc = json.loads(rec_path.read_text(encoding="utf-8"))
        audio_path = _vocals_or_source_path(recording_doc, REPO_ROOT)

        audio, _sr = librosa.load(audio_path, sr=CREPE_SR, mono=True)
        audio_t = torch.from_numpy(audio).unsqueeze(0)
        freq = torchcrepe.predict(
            audio_t, CREPE_SR, hop_length=HOP_SAMPLES, fmin=FMIN_HZ, fmax=FMAX_HZ,
            model=MODEL, batch_size=1024, device=DEVICE,
        )
        freq = freq.squeeze(0).cpu().numpy().astype(np.float64)
        native_times = np.arange(len(freq)) * (HOP_SAMPLES / CREPE_SR)

        log2_native = np.log2(np.maximum(freq, 1e-6))
        log2_aligned = np.interp(target_times, native_times, log2_native)
        out[rid] = log2_aligned.astype(np.float32)
        print(f"[{i+1}/{len(index.lanes)}] {rid} done, n={len(log2_aligned)} ({time.time()-t_start:.1f}s elapsed)", flush=True)

    REL_DIR.mkdir(parents=True, exist_ok=True)
    with open(DENSE_CACHE_PATH, "wb") as fh:
        pickle.dump(out, fh)
    print(f"Saved CREPE paths for {len(out)} recordings to {DENSE_CACHE_PATH}")
    return out


if __name__ == "__main__":
    paths = build(force=True)
    total = sum(len(v) for v in paths.values())
    print(f"Total recordings: {len(paths)}, total frames: {total}")
