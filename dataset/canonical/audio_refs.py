"""The ``audio.files`` block: repo-relative paths plus native file properties.

Never absolute paths — ``relpath`` is relative to the repository root, matching
the convention already used by the CNN ``metadata.csv``. The sha256 makes it
detectable when a stem was re-rendered with a different model, which would
otherwise silently invalidate cached spectrograms.
"""

from __future__ import annotations

import hashlib
import wave
from pathlib import Path
from typing import Any

from .schema import DERIVED_STEM_SUFFIXES

_HASH_CHUNK_BYTES = 1 << 20


def find_source_audio(repo_root: Path, recording_id: str) -> Path | None:
    """Locate ``output/<title>_<recording_id>.wav`` by its id suffix."""
    candidates = sorted((repo_root / "output").glob(f"*_{recording_id}.wav"))
    return candidates[0] if candidates else None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_properties(path: Path) -> dict[str, Any]:
    """Native sample rate / channels / frame count, without decoding samples."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return {
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "n_frames": int(info.frames),
            "duration_s": float(info.frames) / float(info.samplerate),
        }
    except Exception:
        with wave.open(str(path), "rb") as handle:
            sample_rate = handle.getframerate()
            n_frames = handle.getnframes()
            return {
                "sample_rate": int(sample_rate),
                "channels": int(handle.getnchannels()),
                "n_frames": int(n_frames),
                "duration_s": float(n_frames) / float(sample_rate),
            }


def describe_file(
    path: Path,
    role: str,
    repo_root: Path,
    *,
    compute_sha256: bool = True,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "role": role,
        "relpath": path.resolve().relative_to(repo_root.resolve()).as_posix(),
    }
    entry.update(audio_properties(path))
    entry["sha256"] = sha256_of(path) if compute_sha256 else None
    return entry


def build_audio_block(
    recording_id: str,
    audio_id: str | None,
    repo_root: Path,
    *,
    compute_sha256: bool = True,
) -> dict[str, Any]:
    """Source WAV plus whatever stems the ``output/denoised/`` cache already holds."""
    files: list[dict[str, Any]] = []
    source = find_source_audio(repo_root, recording_id)
    if source is not None:
        files.append(describe_file(source, "source", repo_root, compute_sha256=compute_sha256))

        stem_dir = repo_root / "output" / "denoised" / recording_id
        base = source.stem
        for role, suffix in DERIVED_STEM_SUFFIXES.items():
            stem_path = stem_dir / f"{base}{suffix}"
            if stem_path.exists():
                files.append(
                    describe_file(stem_path, role, repo_root, compute_sha256=compute_sha256)
                )

    return {"audio_id": audio_id, "files": files}


def source_entry(audio_block: dict[str, Any]) -> dict[str, Any] | None:
    for entry in audio_block.get("files", []):
        if entry["role"] == "source":
            return entry
    return None
