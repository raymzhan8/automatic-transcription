"""Diagnostic F: CQT-argmax + visibility on raw / denoised / vocals stems (small subset)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.canonical.audio_refs import find_source_audio  # noqa: E402
from dataset.canonical.schema import DERIVED_STEM_SUFFIXES, recordings_dir  # noqa: E402
from training.features import SR, extract_features_at_target_centers  # noqa: E402
from training.framewise_dataset import RecordingLaneIndex  # noqa: E402
from training.metrics import pitch_error_metrics  # noqa: E402
from training.normalization import log2_hz_to_cents  # noqa: E402
from training.pitch_diagnostics.baselines_b import cqt_argmax_cents  # noqa: E402
from training.pitch_diagnostics.common import (  # noqa: E402
    OUT_DIR,
    PRIMARY_LANE,
    bin_from_hz,
    clip_bin,
    hz_from_cents,
    linear_mag,
    summarize_array,
    write_json,
)


def stem_path(recording_id: str, role: str) -> Path | None:
    source = find_source_audio(REPO_ROOT, recording_id)
    if source is None:
        return None
    if role == "source":
        return source if source.exists() else None
    suffix = DERIVED_STEM_SUFFIXES.get(role)
    if not suffix:
        return None
    path = REPO_ROOT / "output" / "denoised" / recording_id / f"{source.stem}{suffix}"
    return path if path.exists() else None


def visibility_ratio(cqt_log: np.ndarray, target_hz: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mag = linear_mag(cqt_log)
    bins = clip_bin(bin_from_hz(target_hz))
    ratios = []
    for t in np.where(mask)[0]:
        col = mag[:, t]
        mx = col.max()
        ratios.append(col[int(bins[t])] / max(mx, 1e-12))
    return np.asarray(ratios)


def main() -> None:
    index = RecordingLaneIndex.build(REPO_ROOT)
    coverage = []
    both = []
    for lane in index.lanes:
        roles = {role: stem_path(lane.recording_id, role) for role in ("source", "denoised", "vocals")}
        coverage.append(
            {
                "recording_id": lane.recording_id,
                "has_source": roles["source"] is not None,
                "has_denoised": roles["denoised"] is not None,
                "has_vocals": roles["vocals"] is not None,
            }
        )
        if roles["denoised"] and roles["vocals"]:
            both.append(lane.recording_id)

    write_json(OUT_DIR / "stem_coverage.json", {"recordings": coverage, "n_with_denoised_and_vocals": len(both)})
    subset = both[:4]
    if not subset:
        write_json(OUT_DIR / "stems.json", {"skipped": True, "reason": "no recordings with both denoised and vocals stems"})
        print("Diagnostic F skipped: no dual stems")
        return

    try:
        _ = librosa.cqt
    except Exception as exc:
        write_json(
            OUT_DIR / "stems.json",
            {
                "subset": subset,
                "skipped": True,
                "reason": (
                    "cached stems are available, but librosa CQT is unavailable in "
                    f"this environment: {type(exc).__name__}: {exc}"
                ),
            },
        )
        print("Diagnostic F skipped: librosa CQT unavailable")
        return

    rec_docs = {p.stem: json.loads(p.read_text()) for p in recordings_dir(REPO_ROOT).glob("*.json")}
    results = {}
    for rec_id in subset:
        lane = next(x for x in index.lanes if x.recording_id == rec_id)
        frames = index._frames[(rec_id, PRIMARY_LANE)]
        n = lane.n_frames
        mask = frames["valid_target"][:n] & (frames["frame_time_s"][:n] < lane.duration_s)
        true = np.asarray(log2_hz_to_cents(frames["pitch_log2_hz"][:n], lane.fundamental_hz))
        target_hz = np.asarray(hz_from_cents(true, lane.fundamental_hz))
        rec_block = {}
        for role in ("source", "denoised", "vocals"):
            path = stem_path(rec_id, role)
            if path is None:
                rec_block[role] = {"missing": True}
                continue
            y, _ = librosa.load(str(path), sr=SR, mono=True)
            duration = len(y) / SR
            _, cqt = extract_features_at_target_centers(y, duration)
            tlen = min(cqt.shape[1], n)
            mag = linear_mag(cqt[:, :tlen])
            pred = cqt_argmax_cents(mag, lane.fundamental_hz)
            m = mask[:tlen]
            rec_block[role] = {
                "argmax": pitch_error_metrics(pred[m], true[:tlen][m]),
                "target_over_max": summarize_array(visibility_ratio(cqt[:, :tlen], target_hz[:tlen], m)),
            }
        results[rec_id] = rec_block
        print(rec_id, {k: (v.get("argmax") or {}).get("mae_cents") for k, v in rec_block.items()})

    write_json(OUT_DIR / "stems.json", {"subset": subset, "results": results})
    print("Diagnostic F written", OUT_DIR / "stems.json")


if __name__ == "__main__":
    main()
