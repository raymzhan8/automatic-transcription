"""CLI: precompute CQT log-magnitude features on the canonical 10 ms grid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.canonical.build import exported_recording_ids  # noqa: E402
from dataset.canonical.frames import _audio_duration  # noqa: E402
from dataset.canonical.schema import features_dir, recordings_dir  # noqa: E402
from dataset.canonical.visualize_targets import source_audio_path  # noqa: E402
from training.features import (  # noqa: E402
    SR,
    extract_features_at_target_centers,
    feature_metadata,
)


def build_recording_features(
    recording_doc: dict,
    *,
    repo_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    audio_path = source_audio_path(recording_doc, repo_root)
    if audio_path is None:
        raise FileNotFoundError(
            f"source audio missing for {recording_doc['recording_id']}"
        )
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    duration_s = _audio_duration(recording_doc)
    frame_time_s, cqt_log = extract_features_at_target_centers(y, duration_s)
    meta = feature_metadata()
    return cqt_log, frame_time_s, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-ids", nargs="*")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()

    ids = args.recording_ids or exported_recording_ids(args.repo_root)
    out_dir = features_dir(args.repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    built = 0
    for recording_id in ids:
        rec_path = recordings_dir(args.repo_root) / f"{recording_id}.json"
        if not rec_path.exists():
            print(f"skip missing {recording_id}", file=sys.stderr)
            continue
        recording_doc = json.loads(rec_path.read_text(encoding="utf-8"))
        try:
            cqt_log, frame_time_s, meta = build_recording_features(
                recording_doc, repo_root=args.repo_root
            )
        except FileNotFoundError as exc:
            print(f"skip {recording_id}: {exc}", file=sys.stderr)
            continue

        out_path = out_dir / f"{recording_id}.npz"
        np.savez_compressed(
            out_path,
            cqt_log=cqt_log,
            frame_time_s=frame_time_s,
            recording_id=recording_id,
            **meta,
        )
        size = out_path.stat().st_size
        total_bytes += size
        built += 1
        print(
            f"{recording_id}: cqt {cqt_log.shape} "
            f"({size / 1e6:.1f} MB) -> {out_path.name}"
        )

    print(f"Built {built} feature files, total {total_bytes / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
