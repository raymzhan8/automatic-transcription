"""Excerpt dataset options for tonic-aligned CQT and within-recording time windows."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from training.framewise_dataset import FramewiseExcerptDataset, LaneRecord
from training.normalization import log2_hz_to_cents, normalize_cqt, standardize_pitch
from training.pitch_diagnostics.common import tonic_align_cqt
from training.sampling import (
    EXCERPT_FRAMES,
    choose_excerpt_start,
    sample_anchor_index,
    slice_excerpt,
)


class PitchExcerptDataset(FramewiseExcerptDataset):
    def __init__(
        self,
        *args: Any,
        tonic_align: bool = False,
        time_lo_frac: float = 0.0,
        time_hi_frac: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.tonic_align = tonic_align
        self.time_lo_frac = time_lo_frac
        self.time_hi_frac = time_hi_frac
        super().__init__(*args, **kwargs)

    def _anchor_valid(self, lane: LaneRecord, valid: np.ndarray) -> np.ndarray:
        """Restrict anchors to a time window.

        When ``time_lo_frac``/``time_hi_frac`` cut the timeline, the window is
        applied over **valid-frame indices** (annotation order), not absolute
        recording time. Absolute-time cuts leave most late windows empty because
        annotations are front-loaded in many pieces.
        """
        n = len(valid)
        if self.time_lo_frac <= 0.0 and self.time_hi_frac >= 1.0:
            return valid
        idx = np.flatnonzero(valid)
        if idx.size == 0:
            return valid
        lo = int(self.time_lo_frac * idx.size)
        hi = max(lo + 1, int(self.time_hi_frac * idx.size))
        keep = idx[lo:hi]
        out = np.zeros(n, dtype=bool)
        out[keep] = True
        return out

    def _sample_one(self) -> dict[str, Any]:
        lane = self.pool[int(self.rng.integers(0, len(self.pool)))]
        frames = self.index._frames[(lane.recording_id, lane.lane_id)]
        valid = self._anchor_valid(lane, frames["valid_target"])
        if not np.any(valid):
            valid = frames["valid_target"]
        anchor = sample_anchor_index(valid, self.rng)
        start = choose_excerpt_start(
            anchor, lane.n_frames, lane.duration_s, rng=self.rng
        )
        excerpt = slice_excerpt(
            lane.n_frames, lane.duration_s, frames["valid_target"], start_idx=start
        )
        end = min(excerpt.end_idx, lane.n_frames)
        actual_len = end - start
        pad_len = EXCERPT_FRAMES - actual_len

        cqt_full = self.index._features[lane.recording_id]["cqt_log"]
        if self.tonic_align:
            cqt_full = tonic_align_cqt(cqt_full, lane.fundamental_hz)
        cqt_slice = cqt_full[:, start:end]
        if pad_len > 0:
            cqt_slice = np.pad(cqt_slice, ((0, 0), (0, pad_len)), mode="constant")
        cqt_norm = normalize_cqt(cqt_slice, self.mu_cqt, self.sigma_cqt)

        def take(name: str) -> np.ndarray:
            arr = frames[name][start:end]
            if pad_len > 0:
                fill: Any = False if arr.dtype == bool else ("" if arr.dtype.kind in ("U", "O") else 0)
                arr = np.pad(arr, (0, pad_len), constant_values=fill)
            return arr

        padding_mask = excerpt.padding_mask.copy()
        if actual_len < EXCERPT_FRAMES:
            padding_mask[actual_len:] = True

        pitch_log2 = take("pitch_log2_hz").astype(np.float32)
        pitch_cents = log2_hz_to_cents(pitch_log2, lane.fundamental_hz).astype(np.float32)
        if self.mu_pitch is not None and self.sigma_pitch is not None:
            pitch_std = standardize_pitch(pitch_cents, self.mu_pitch, self.sigma_pitch)
        else:
            pitch_std = np.zeros_like(pitch_cents)

        return {
            "spec": torch.from_numpy(cqt_norm[np.newaxis].astype(np.float32)),
            "valid_target": torch.from_numpy(take("valid_target")),
            "padding_mask": torch.from_numpy(padding_mask),
            "pitch_cents": torch.from_numpy(pitch_cents),
            "pitch_standardized": torch.from_numpy(pitch_std),
            "recording_id": lane.recording_id,
            "fundamental_hz": lane.fundamental_hz,
        }
