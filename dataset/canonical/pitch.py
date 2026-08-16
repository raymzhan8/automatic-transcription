"""Raw and derived pitch blocks.

``raw`` keeps only the four wire components (swara / oct / raised / log_offset).
Everything frequency-flavoured lives in the derived block, because it depends on
``raga.fundamental`` and the raga's stratified ratios rather than on the pitch
itself.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from idtap.classes.pitch import Pitch

# swara indices whose ``raised`` flag carries no meaning; idtap forces them True.
_RAISED_INVARIANT_SWARAS = frozenset({0, 4})

_REDUNDANT_WIRE_PITCH_KEYS = ("ratios", "fundamental")


def raw_pitch(wire_pitch: dict[str, Any]) -> dict[str, Any]:
    """The four verbatim wire components, snake_cased.

    ``ratios`` and ``fundamental`` are dropped: they duplicate ``raga`` and are
    embedded on only some recordings. Their presence is recorded separately in
    ``raw.pitch_wire_keys_present``.
    """
    return {
        "swara": wire_pitch.get("swara"),
        "oct": wire_pitch.get("oct"),
        "raised": wire_pitch.get("raised"),
        "log_offset": wire_pitch.get("logOffset", wire_pitch.get("log_offset")),
    }


def pitch_wire_keys(wire_pitches: Iterable[dict[str, Any]]) -> list[str]:
    """Sorted union of keys seen across a trajectory's wire pitches."""
    seen: set[str] = set()
    for wire_pitch in wire_pitches or []:
        seen.update(wire_pitch.keys())
    return sorted(seen)


def has_redundant_ratio_keys(wire_pitches: Iterable[dict[str, Any]]) -> bool:
    keys = set(pitch_wire_keys(wire_pitches))
    return any(k in keys for k in _REDUNDANT_WIRE_PITCH_KEYS)


def pitch_key(
    swara: int | None,
    octave: int | None,
    raised: bool | None,
    log_offset: float | None,
) -> str | None:
    """Canonical identity token used for the same-pitch test.

    Formatted to fixed precision from all four raw components. Sa and Pa are
    normalised to raised, mirroring idtap, so that two wire encodings of the
    same pitch never compare unequal.
    """
    if swara is None or octave is None:
        return None
    effective_raised = True if int(swara) in _RAISED_INVARIANT_SWARAS else bool(raised)
    offset = 0.0 if log_offset is None else float(log_offset)
    return f"{int(swara)}|{int(octave)}|{'R' if effective_raised else 'r'}|{offset:+.6f}"


def derive_pitch(
    raw: dict[str, Any],
    *,
    ratios: Sequence[Any],
    fundamental: float,
) -> dict[str, Any] | None:
    """Build the derived Pitch block from a raw pitch plus raga context."""
    if raw is None or raw.get("swara") is None:
        return None

    swara = int(raw["swara"])
    octave = int(raw.get("oct") or 0)
    raised = bool(raw.get("raised", True))
    log_offset = float(raw.get("log_offset") or 0.0)

    pitch = Pitch(
        {
            "swara": swara,
            "oct": octave,
            "raised": raised,
            "log_offset": log_offset,
            "ratios": list(ratios),
            "fundamental": float(fundamental),
        }
    )
    frequency = float(pitch.frequency)

    return {
        "swara": swara,
        "oct": octave,
        "raised": raised,
        "log_offset": log_offset,
        "pitch_key": pitch_key(swara, octave, raised, log_offset),
        "sargam": pitch.sargam_letter,
        "octaved_sargam": pitch.octaved_sargam_letter,
        "frequency_hz": frequency,
        "log2_hz": math.log2(frequency),
        "cents_above_tonic": 1200.0 * math.log2(frequency / float(fundamental)),
        "numbered_pitch": int(pitch.numbered_pitch),
    }


def cents_between(from_pitch: dict[str, Any] | None, to_pitch: dict[str, Any] | None) -> float | None:
    """Signed interval in cents, or ``None`` when either endpoint is missing."""
    if not from_pitch or not to_pitch:
        return None
    return 1200.0 * (to_pitch["log2_hz"] - from_pitch["log2_hz"])


def log2_between(from_pitch: dict[str, Any] | None, to_pitch: dict[str, Any] | None) -> float | None:
    if not from_pitch or not to_pitch:
        return None
    return to_pitch["log2_hz"] - from_pitch["log2_hz"]


def pitch_range_cents(pitches: Sequence[dict[str, Any] | None]) -> float | None:
    present = [p for p in pitches if p]
    if not present:
        return None
    log2s = [p["log2_hz"] for p in present]
    return 1200.0 * (max(log2s) - min(log2s))
