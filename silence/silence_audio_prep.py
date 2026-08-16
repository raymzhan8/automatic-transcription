"""Prepare denoised / separated audio stems for silence detection."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import librosa  # noqa: E402
import numpy as np  # noqa: E402
from idtap.classes.piece import Piece  # noqa: E402
from vocalprep.pipeline import DEFAULT_DENOISE_MODEL, DEFAULT_VOCAL_MODEL  # noqa: E402

from dataset.export_denoised_cnn_dataset import (  # noqa: E402
    find_raw_audio,
    piece_uses_vocal_separation,
    process_piece_audio,
)

AudioVariant = Literal["raw", "denoised", "vocals"]


def should_separate_vocals(
    piece_obj: Piece | None,
    *,
    force_vocal_separation: bool = False,
) -> bool:
    if force_vocal_separation:
        return True
    if piece_obj is None:
        return False
    return piece_uses_vocal_separation(piece_obj)


def resolve_piece_stem_path(
    raw_audio: Path,
    denoise_dir: Path,
    variant: AudioVariant,
    *,
    separate_vocals: bool,
    skip_denoise: bool = False,
    denoise_model: str = DEFAULT_DENOISE_MODEL,
    vocal_model: str = DEFAULT_VOCAL_MODEL,
    force_vocal_separation: bool = False,
) -> Path:
    """Return the WAV path to use for silence detection on this piece."""
    if variant == "raw":
        return raw_audio

    base = raw_audio.stem
    denoised_path = denoise_dir / f"{base}.denoised.wav"
    vocals_path = denoise_dir / f"{base}.vocals.wav"

    if skip_denoise:
        if variant == "denoised":
            if not denoised_path.exists():
                raise FileNotFoundError(f"Missing cached denoised stem: {denoised_path}")
            return denoised_path
        if not vocals_path.exists():
            raise FileNotFoundError(f"Missing cached vocals stem: {vocals_path}")
        return vocals_path

    denoised_path, vocals_path = process_piece_audio(
        raw_audio,
        denoise_dir,
        separate_vocals=separate_vocals,
        denoise_model=denoise_model,
        vocal_model=vocal_model,
        force_vocal_separation=force_vocal_separation,
    )
    if variant == "denoised":
        return denoised_path
    if vocals_path is None:
        raise FileNotFoundError(
            f"Vocals stem unavailable for {raw_audio.name}. "
            "This piece may be instrumental, or pass --force-vocal-separation."
        )
    return vocals_path


def load_variant_audio(
    *,
    piece_id: str,
    piece_title: str,
    audio_dir: Path,
    denoise_root: Path,
    variant: AudioVariant,
    sr: int,
    piece_obj: Piece | None = None,
    skip_denoise: bool = False,
    force_vocal_separation: bool = False,
    denoise_model: str = DEFAULT_DENOISE_MODEL,
    vocal_model: str = DEFAULT_VOCAL_MODEL,
) -> tuple[np.ndarray, Path, dict[str, object]]:
    raw_audio = find_raw_audio(audio_dir, piece_id, piece_title)
    if raw_audio is None:
        raise FileNotFoundError(f"No raw audio found for piece {piece_id}")

    separate_vocals = should_separate_vocals(
        piece_obj,
        force_vocal_separation=force_vocal_separation,
    )
    denoise_dir = denoise_root / piece_id
    stem_path = resolve_piece_stem_path(
        raw_audio,
        denoise_dir,
        variant,
        separate_vocals=separate_vocals,
        skip_denoise=skip_denoise,
        denoise_model=denoise_model,
        vocal_model=vocal_model,
        force_vocal_separation=force_vocal_separation,
    )
    y, loaded_sr = librosa.load(stem_path, sr=sr, mono=True)
    if loaded_sr != sr:
        raise ValueError(f"Expected sample rate {sr}, got {loaded_sr} from {stem_path}")

    meta = {
        "piece_id": piece_id,
        "variant": variant,
        "raw_audio": str(raw_audio),
        "stem_path": str(stem_path),
        "separate_vocals": separate_vocals,
        "skip_denoise": skip_denoise,
    }
    return y, stem_path, meta
