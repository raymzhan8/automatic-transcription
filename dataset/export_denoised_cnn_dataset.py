"""Batch-denoise recordings and export CNN dataset variants (denoised / vocals)."""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from idtap import Piece, SwaraClient
from idtap.classes.trajectory import Trajectory
from PIL import Image
from vocalprep import VocalPipeline
from vocalprep.pipeline import (
    _ACCOMP_STEM_LABELS,
    _CLEAN_STEM_LABELS,
    _NOISE_STEM_LABELS,
    _VOCAL_STEM_LABELS,
    DEFAULT_DENOISE_MODEL,
    DEFAULT_VOCAL_MODEL,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

INST = 0
VOCAL_INSTRUMENTS = frozenset({"Vocal_M", "Vocal_F"})
STRING_IDX = 0
SR = 22050
AUDIO_FORMAT = "wav"
SILENT_TRAJECTORY_ID = 12
SKIP_IDTAP_IDS = {7, 8, 9, 10, 11, 13}

MIN_FREQUENCY = 75
MAX_FREQUENCY = 2400
BINS_PER_OCTAVE = 72
HOP_LENGTH = 512
N_BINS = int(np.ceil(BINS_PER_OCTAVE * np.log2(MAX_FREQUENCY / MIN_FREQUENCY)))
CLIP_DURATION = 1.0
CQT_FRAMES_1S = int(np.ceil(CLIP_DURATION * SR / HOP_LENGTH))
EXPORT_IMAGE_WIDTH = CQT_FRAMES_1S * 4

COMPOSITE_LABELS: dict[int, list[int]] = {
    4: [2, 1],
    5: [1, 3],
}

METADATA_COLUMNS = [
    "piece_id",
    "piece_title",
    "traj_index",
    "segment_index",
    "unique_id",
    "idtap_id",
    "idtap_name",
    "label",
    "abs_start",
    "abs_end",
    "duration",
    "image_path",
    "clip_path",
]

SKIPPED_COLUMNS = [
    "piece_id",
    "traj_index",
    "unique_id",
    "idtap_id",
    "idtap_name",
    "reason",
]


def replace_zeros(data: np.ndarray) -> np.ndarray:
    nonzero = data[np.nonzero(data)]
    if nonzero.size == 0:
        return data
    out = data.copy()
    out[data == 0] = np.min(nonzero)
    return out


def compute_spec_display(y: np.ndarray, sample_rate: int) -> np.ndarray:
    constantq = np.abs(
        librosa.cqt(
            y,
            sr=sample_rate,
            hop_length=HOP_LENGTH,
            fmin=MIN_FREQUENCY,
            n_bins=N_BINS,
            bins_per_octave=BINS_PER_OCTAVE,
        )
    )
    return np.flipud(np.log10(replace_zeros(constantq)))


def load_clip(audio_path: str | Path, offset: float, sample_rate: int = SR) -> tuple[np.ndarray, int]:
    y, loaded_sr = librosa.load(
        audio_path,
        sr=sample_rate,
        mono=True,
        offset=offset,
        duration=CLIP_DURATION,
    )
    target_samples = int(round(CLIP_DURATION * loaded_sr))
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
    elif len(y) > target_samples:
        y = y[:target_samples]
    return y, loaded_sr


def crop_spec_to_clip(spec_log: np.ndarray) -> np.ndarray:
    _, n_frames = spec_log.shape
    if n_frames > CQT_FRAMES_1S:
        return spec_log[:, :CQT_FRAMES_1S]
    if n_frames < CQT_FRAMES_1S:
        pad_value = float(spec_log.min())
        pad = np.full(
            (spec_log.shape[0], CQT_FRAMES_1S - n_frames),
            pad_value,
            dtype=spec_log.dtype,
        )
        return np.hstack([spec_log, pad])
    return spec_log


def spec_to_magma_rgb(spec_log: np.ndarray) -> np.ndarray:
    spec = crop_spec_to_clip(spec_log)
    spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
    spec_min = float(spec.min())
    spec_max = float(spec.max())
    if spec_max - spec_min < 1e-10:
        norm = np.zeros_like(spec, dtype=np.float64)
    else:
        norm = np.clip((spec - spec_min) / (spec_max - spec_min), 0.0, 1.0)
    rgba = plt.cm.magma(norm)
    return (rgba[..., :3] * 255).astype(np.uint8)


def save_spec_png(spec_log: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = spec_to_magma_rgb(spec_log)
    img = Image.fromarray(rgb)
    img = img.resize((EXPORT_IMAGE_WIDTH, rgb.shape[0]), Image.Resampling.BILINEAR)
    img.save(path)


def build_traj_selections(piece_obj: Piece) -> list[dict[str, object]]:
    trajectories = piece_obj.all_trajectories(inst=INST, string_idx=STRING_IDX)
    start_times = piece_obj.traj_start_times(inst=INST, string_idx=STRING_IDX)
    selections: list[dict[str, object]] = []
    for idx, traj in enumerate(trajectories):
        if idx >= len(start_times):
            break
        traj_start = float(start_times[idx])
        selections.append(
            {
                "traj": traj,
                "index": idx,
                "start": traj_start,
                "end": traj_start + float(traj.dur_tot),
            }
        )
    return selections


def iter_labeled_segments(traj: Trajectory) -> list[tuple[int | str, float, float]]:
    if traj.id in SKIP_IDTAP_IDS:
        return []
    if traj.id == SILENT_TRAJECTORY_ID:
        return [("silent", 0.0, 1.0)]
    if traj.id in {0, 1, 2, 3}:
        return [(traj.id, 0.0, 1.0)]
    if traj.id in COMPOSITE_LABELS:
        labels = COMPOSITE_LABELS[traj.id]
        if traj.dur_array and len(traj.dur_array) >= len(labels):
            fracs = [float(x) for x in traj.dur_array[: len(labels)]]
        elif len(labels) == 2:
            fracs = [1 / 3, 2 / 3]
        else:
            fracs = [1.0 / len(labels)] * len(labels)
        starts = [0.0]
        for frac in fracs:
            starts.append(starts[-1] + frac)
        starts[-1] = 1.0
        return [
            (label, starts[i], starts[i + 1]) for i, label in enumerate(labels)
        ]
    if traj.id == 6:
        n_segments = len(traj.dur_array) if traj.dur_array else max(len(traj.pitches) - 1, 1)
        if traj.dur_array and len(traj.dur_array) >= n_segments:
            fracs = [float(x) for x in traj.dur_array[:n_segments]]
        else:
            fracs = [1.0 / n_segments] * n_segments
        starts = [0.0]
        for frac in fracs:
            starts.append(starts[-1] + frac)
        starts[-1] = 1.0
        return [(1, starts[i], starts[i + 1]) for i in range(n_segments)]
    return []


def trajectory_export_label(traj: Trajectory) -> int | str | None:
    segments = iter_labeled_segments(traj)
    if not segments:
        return None
    return segments[0][0]


def track_instrument_name(piece_obj: Piece, track_index: int = INST) -> str:
    instrumentation = piece_obj.instrumentation or []
    if track_index >= len(instrumentation):
        return ""
    instrument = instrumentation[track_index]
    return getattr(instrument, "name", str(instrument))


def piece_uses_vocal_separation(piece_obj: Piece, track_index: int = INST) -> bool:
    """True when the transcribed track is a vocal instrument (Vocal_M / Vocal_F)."""
    return track_instrument_name(piece_obj, track_index) in VOCAL_INSTRUMENTS


def find_raw_audio(output_dir: Path, piece_id: str, piece_title: str) -> Path | None:
    candidates = list(output_dir.glob(f"*_{piece_id}.wav"))
    if candidates:
        return candidates[0]
    title_path = output_dir / f"{piece_title}_{piece_id}.wav"
    if title_path.exists():
        return title_path
    return None


def run_denoise_only(
    raw_audio: Path,
    denoise_dir: Path,
    *,
    denoise_model: str = DEFAULT_DENOISE_MODEL,
) -> Path:
    """Denoise without karaoke separation (for Sitar, Sarangi, etc.)."""
    denoise_dir.mkdir(parents=True, exist_ok=True)
    base = raw_audio.stem
    denoised_path = denoise_dir / f"{base}.denoised.wav"
    if denoised_path.exists():
        print(f"  using cached denoised stem in {denoise_dir}")
        return denoised_path

    print(f"  running denoise only (skipping vocal separation) on {raw_audio.name} ...")
    pipe = VocalPipeline(output_dir=denoise_dir, denoise_model=denoise_model)
    names: dict[str, str] = {}
    for label in _CLEAN_STEM_LABELS:
        names[label] = f"{base}.denoised"
    for label in _NOISE_STEM_LABELS:
        names[label] = f"{base}.noise"
    outputs = pipe._get_denoiser().separate(str(raw_audio), custom_output_names=names)
    denoised = pipe._find_output(outputs, f"{base}.denoised")
    if not denoised.exists():
        raise FileNotFoundError(f"Denoise stage did not produce output for {raw_audio.name}")
    return denoised


def run_vocal_separation(
    separation_input: Path,
    denoise_dir: Path,
    *,
    base: str,
    vocal_model: str = DEFAULT_VOCAL_MODEL,
) -> Path:
    """Karaoke vocal separation on denoised (or raw) audio."""
    denoise_dir.mkdir(parents=True, exist_ok=True)
    print(f"  running vocal separation ({vocal_model}) on {separation_input.name} ...")
    pipe = VocalPipeline(
        output_dir=denoise_dir,
        denoise=False,
        vocal_model=vocal_model,
    )
    names: dict[str, str] = {}
    for label in _VOCAL_STEM_LABELS:
        names[label] = f"{base}.vocals"
    for label in _ACCOMP_STEM_LABELS:
        names[label] = f"{base}.accompaniment"
    outputs = pipe._get_separator().separate(
        str(separation_input),
        custom_output_names=names,
    )
    vocals = pipe._find_output(outputs, f"{base}.vocals")
    if not vocals.exists():
        raise FileNotFoundError(
            f"Separation stage did not produce vocals for {separation_input.name}"
        )
    return vocals


def run_denoise_and_separate(
    raw_audio: Path,
    denoise_dir: Path,
    *,
    denoise_model: str = DEFAULT_DENOISE_MODEL,
    vocal_model: str = DEFAULT_VOCAL_MODEL,
    force_vocal_separation: bool = False,
) -> tuple[Path, Path]:
    """Denoise then karaoke vocal separation (denoise → separate on denoised mix)."""
    denoise_dir.mkdir(parents=True, exist_ok=True)
    base = raw_audio.stem
    denoised_path = denoise_dir / f"{base}.denoised.wav"
    vocals_path = denoise_dir / f"{base}.vocals.wav"

    if denoised_path.exists():
        print(f"  using cached denoised stem in {denoise_dir}")
    else:
        denoised_path = run_denoise_only(
            raw_audio,
            denoise_dir,
            denoise_model=denoise_model,
        )

    if vocals_path.exists() and not force_vocal_separation:
        print(f"  using cached vocals stem in {denoise_dir}")
        return denoised_path, vocals_path

    if force_vocal_separation and vocals_path.exists():
        print(f"  re-running vocal separation (--force-vocal-separation)")
        vocals_path.unlink()

    vocals_path = run_vocal_separation(
        denoised_path,
        denoise_dir,
        base=base,
        vocal_model=vocal_model,
    )
    return denoised_path, vocals_path


def process_piece_audio(
    raw_audio: Path,
    denoise_dir: Path,
    *,
    separate_vocals: bool,
    denoise_model: str = DEFAULT_DENOISE_MODEL,
    vocal_model: str = DEFAULT_VOCAL_MODEL,
    force_vocal_separation: bool = False,
) -> tuple[Path, Path | None]:
    """Return denoised path and optional vocals stem path."""
    if separate_vocals:
        denoised_path, vocals_path = run_denoise_and_separate(
            raw_audio,
            denoise_dir,
            denoise_model=denoise_model,
            vocal_model=vocal_model,
            force_vocal_separation=force_vocal_separation,
        )
        return denoised_path, vocals_path
    return run_denoise_only(raw_audio, denoise_dir, denoise_model=denoise_model), None


def export_piece_variant(
    piece_obj: Piece,
    piece_id: str,
    audio_path: Path,
    cnn_output_dir: Path,
    *,
    clear_dirs: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    piece_title = str(piece_obj.title or piece_id)
    dataset_dir = cnn_output_dir / piece_id
    images_dir = dataset_dir / "images"
    clips_dir = dataset_dir / "clips"

    if clear_dirs:
        for export_dir in (images_dir, clips_dir):
            if export_dir.exists():
                shutil.rmtree(export_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    traj_selections = build_traj_selections(piece_obj)
    metadata_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []

    for sel in traj_selections:
        traj = sel["traj"]
        label = trajectory_export_label(traj)
        if label is None:
            if traj.id in SKIP_IDTAP_IDS:
                skipped_rows.append(
                    {
                        "piece_id": piece_id,
                        "traj_index": sel["index"],
                        "unique_id": traj.unique_id,
                        "idtap_id": traj.id,
                        "idtap_name": traj.name_,
                        "reason": "skipped_idtap_type",
                    }
                )
            continue

        y, loaded_sr = load_clip(audio_path, offset=sel["start"])
        spec_log = compute_spec_display(y, loaded_sr)
        label_token = str(label)
        stem = f"{sel['index']:04d}_{label_token}"
        image_path = images_dir / f"{stem}.png"
        clip_path = clips_dir / f"{stem}.{AUDIO_FORMAT}"

        save_spec_png(spec_log, image_path)
        sf.write(clip_path, y, loaded_sr)

        metadata_rows.append(
            {
                "piece_id": piece_id,
                "piece_title": piece_title,
                "traj_index": sel["index"],
                "segment_index": 0,
                "unique_id": traj.unique_id,
                "idtap_id": traj.id,
                "idtap_name": traj.name_,
                "label": label,
                "abs_start": sel["start"],
                "abs_end": sel["start"] + CLIP_DURATION,
                "duration": CLIP_DURATION,
                "image_path": str(image_path),
                "clip_path": str(clip_path),
            }
        )

    with (dataset_dir / "metadata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(metadata_rows)

    with (dataset_dir / "skipped.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SKIPPED_COLUMNS)
        writer.writeheader()
        writer.writerows(skipped_rows)

    return metadata_rows, skipped_rows


def combine_metadata(cnn_dir: Path) -> None:
    all_dir = cnn_dir / "all"
    all_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    all_skipped: list[dict[str, object]] = []

    for piece_dir in sorted(cnn_dir.iterdir()):
        if not piece_dir.is_dir() or piece_dir.name == "all":
            continue
        metadata_path = piece_dir / "metadata.csv"
        if metadata_path.exists():
            with metadata_path.open(newline="") as f:
                all_rows.extend(list(csv.DictReader(f)))
        skipped_path = piece_dir / "skipped.csv"
        if skipped_path.exists():
            with skipped_path.open(newline="") as f:
                all_skipped.extend(list(csv.DictReader(f)))

    with (all_dir / "metadata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    with (all_dir / "skipped.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SKIPPED_COLUMNS)
        writer.writeheader()
        writer.writerows(all_skipped)

    print(f"  combined {len(all_rows)} rows -> {all_dir / 'metadata.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output",
    )
    parser.add_argument(
        "--piece-ids",
        nargs="*",
        default=None,
        help="Limit to specific piece IDs (default: all in cnn_dataset)",
    )
    parser.add_argument(
        "--skip-denoise",
        action="store_true",
        help="Skip vocalprep; assume stems already exist under output/denoised/",
    )
    parser.add_argument(
        "--denoise-model",
        default=DEFAULT_DENOISE_MODEL,
        help="audio-separator denoise model filename",
    )
    parser.add_argument(
        "--vocal-model",
        default=DEFAULT_VOCAL_MODEL,
        help="audio-separator karaoke vocal model filename",
    )
    parser.add_argument(
        "--force-vocal-separation",
        action="store_true",
        help="Re-run karaoke separation even when cached .vocals.wav exists",
    )
    parser.add_argument(
        "--vocals-only",
        action="store_true",
        help="Only export cnn_dataset_vocals (skip denoised variant)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    raw_cnn_dir = output_dir / "cnn_dataset"
    denoised_cnn_dir = output_dir / "cnn_dataset_denoised"
    vocals_cnn_dir = output_dir / "cnn_dataset_vocals"
    denoise_root = output_dir / "denoised"

    if args.piece_ids:
        piece_ids = args.piece_ids
    else:
        piece_ids = sorted(
            d.name
            for d in raw_cnn_dir.iterdir()
            if d.is_dir() and d.name != "all"
        )

    client = SwaraClient()

    for i, piece_id in enumerate(piece_ids, start=1):
        print(f"\n[{i}/{len(piece_ids)}] piece {piece_id}")
        try:
            piece_data = client.get_piece(piece_id)
            piece_obj = Piece.from_json(piece_data)
            piece_title = str(piece_obj.title or piece_id)

            raw_audio = find_raw_audio(output_dir, piece_id, piece_title)
            if raw_audio is None:
                print(f"  SKIP: no raw audio found for {piece_title}")
                continue

            instrument = track_instrument_name(piece_obj)
            separate_vocals = piece_uses_vocal_separation(piece_obj)
            print(
                f"  instrument={instrument!r}, "
                f"vocal_separation={'yes' if separate_vocals else 'no (instrumental)'}"
            )
            if separate_vocals:
                print(f"  vocal_model={args.vocal_model}")

            piece_denoise_dir = denoise_root / piece_id
            if args.skip_denoise:
                base = raw_audio.stem
                denoised_path = piece_denoise_dir / f"{base}.denoised.wav"
                if not denoised_path.exists():
                    print(f"  SKIP: cached denoised stem missing in {piece_denoise_dir}")
                    continue
                vocals_path = None
                if separate_vocals:
                    vocals_path = piece_denoise_dir / f"{base}.vocals.wav"
                    if not vocals_path.exists():
                        print(f"  SKIP: cached vocals stem missing in {piece_denoise_dir}")
                        continue
            else:
                denoised_path, vocals_path = process_piece_audio(
                    raw_audio,
                    piece_denoise_dir,
                    separate_vocals=separate_vocals,
                    denoise_model=args.denoise_model,
                    vocal_model=args.vocal_model,
                    force_vocal_separation=args.force_vocal_separation,
                )

            if not args.vocals_only:
                metadata_denoised, _ = export_piece_variant(
                    piece_obj, piece_id, denoised_path, denoised_cnn_dir
                )
            else:
                metadata_denoised = []
            vocals_piece_dir = vocals_cnn_dir / piece_id
            if separate_vocals and vocals_path is not None:
                metadata_vocals, _ = export_piece_variant(
                    piece_obj, piece_id, vocals_path, vocals_cnn_dir
                )
                if args.vocals_only:
                    print(f"  exported vocals={len(metadata_vocals)}")
                else:
                    print(
                        f"  exported denoised={len(metadata_denoised)}, "
                        f"vocals={len(metadata_vocals)}"
                    )
            else:
                if vocals_piece_dir.exists():
                    shutil.rmtree(vocals_piece_dir)
                if not args.vocals_only:
                    print(
                        f"  exported denoised={len(metadata_denoised)}, "
                        f"vocals=skipped (non-vocal instrument)"
                    )
                else:
                    print("  vocals=skipped (non-vocal instrument)")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue

    print("\nCombining metadata ...")
    if not args.vocals_only:
        combine_metadata(denoised_cnn_dir)
    combine_metadata(vocals_cnn_dir)
    print("Done.")


if __name__ == "__main__":
    main()
