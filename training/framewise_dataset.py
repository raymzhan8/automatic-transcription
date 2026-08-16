"""Framewise dataset: recording index, excerpt sampling, batch collation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.canonical.analyze_step_5_5 import t1_origin
from dataset.canonical.frames import _audio_duration
from dataset.canonical.schema import (
    features_dir,
    frames_dir,
    frames_npz_name,
    primitives_dir,
    recordings_dir,
)
from training.normalization import log2_hz_to_cents, normalize_cqt, standardize_pitch
from training.relative_pitch_features import compute_phi
from training.sampling import (
    EXCERPT_FRAMES,
    EXCERPT_S,
    choose_excerpt_start,
    sample_anchor_index,
    slice_excerpt,
)

PRIMARY_LANE = "0:0"


@dataclass(frozen=True)
class LaneRecord:
    recording_id: str
    lane_id: str
    performance_group_id: str
    fundamental_hz: float
    duration_s: float
    n_frames: int


@dataclass
class RecordingLaneIndex:
    repo_root: Path
    lanes: list[LaneRecord]
    _features: dict[str, dict[str, np.ndarray]]
    _frames: dict[tuple[str, str], dict[str, np.ndarray]]
    _primitives: dict[str, dict[str, dict[str, Any]]]

    @classmethod
    def build(cls, repo_root: Path, *, lane: str = PRIMARY_LANE) -> RecordingLaneIndex:
        features: dict[str, dict[str, np.ndarray]] = {}
        frames: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        primitives_lookup: dict[str, dict[str, dict[str, Any]]] = {}
        lanes: list[LaneRecord] = []

        for rec_path in sorted(recordings_dir(repo_root).glob("*.json")):
            recording_doc = json.loads(rec_path.read_text(encoding="utf-8"))
            recording_id = recording_doc["recording_id"]
            feat_path = features_dir(repo_root) / f"{recording_id}.npz"
            frame_path = frames_dir(repo_root) / frames_npz_name(recording_id, lane)
            if not feat_path.exists() or not frame_path.exists():
                continue

            feat = np.load(feat_path)
            frame = np.load(frame_path, allow_pickle=True)
            features[recording_id] = {
                "cqt_log": feat["cqt_log"],
                "frame_time_s": feat["frame_time_s"],
            }
            frames[(recording_id, lane)] = {k: frame[k] for k in frame.files}

            prim_path = primitives_dir(repo_root) / f"{recording_id}.json"
            prim_by_id: dict[str, dict[str, Any]] = {}
            if prim_path.exists():
                prim_doc = json.loads(prim_path.read_text(encoding="utf-8"))
                for prim in prim_doc.get("primitives", []):
                    if prim["lane_id"] != lane:
                        continue
                    prim_by_id[prim["primitive_id"]] = {
                        "start_s": prim["start_s"],
                        "end_s": prim["end_s"],
                        "canonical_type": prim["canonical_type"],
                        "t1_provenance": t1_origin(prim),
                    }
            primitives_lookup[recording_id] = prim_by_id

            pg_id = recording_doc.get("performance", {}).get("performance_group_id", "")
            lanes.append(
                LaneRecord(
                    recording_id=recording_id,
                    lane_id=lane,
                    performance_group_id=pg_id,
                    fundamental_hz=float(recording_doc["raga"]["fundamental_hz"]),
                    duration_s=_audio_duration(recording_doc),
                    n_frames=int(frame["n_frames"]),
                )
            )

        return cls(
            repo_root=repo_root,
            lanes=lanes,
            _features=features,
            _frames=frames,
            _primitives=primitives_lookup,
        )

    def filter_recordings(self, recording_ids: set[str]) -> list[LaneRecord]:
        return [lane for lane in self.lanes if lane.recording_id in recording_ids]

    def get_primitive(self, recording_id: str, primitive_id: str) -> dict[str, Any] | None:
        return self._primitives.get(recording_id, {}).get(str(primitive_id))

    def primitives_for_recording(self, recording_id: str) -> list[dict[str, Any]]:
        return list(self._primitives.get(recording_id, {}).values())


class FramewiseExcerptDataset(Dataset):
    """Random valid-anchor 4 s excerpts from a pool of recordings."""

    def __init__(
        self,
        index: RecordingLaneIndex,
        recording_ids: list[str],
        mu_cqt: np.ndarray,
        sigma_cqt: np.ndarray,
        *,
        mu_pitch: float | None = None,
        sigma_pitch: float | None = None,
        seed: int = 42,
        excerpts_per_epoch: int | None = None,
        cache_excerpts: int | None = None,
        estimated_pitch: dict[str, np.ndarray] | None = None,
        compute_pitch_features: bool = False,
    ) -> None:
        self.index = index
        self.pool = index.filter_recordings(set(recording_ids))
        if not self.pool:
            raise ValueError("empty recording pool")
        self.mu_cqt = mu_cqt
        self.sigma_cqt = sigma_cqt
        self.mu_pitch = mu_pitch
        self.sigma_pitch = sigma_pitch
        self.estimated_pitch = estimated_pitch
        self.compute_pitch_features = compute_pitch_features
        self.rng = np.random.default_rng(seed)
        self.excerpts_per_epoch = excerpts_per_epoch or max(len(self.pool) * 50, 256)
        self._cache: list[dict[str, Any]] | None = None
        if cache_excerpts:
            self._cache = [self._sample_one() for _ in range(cache_excerpts)]

    def __len__(self) -> int:
        if self._cache is not None:
            return len(self._cache)
        return self.excerpts_per_epoch

    def _sample_one(self) -> dict[str, Any]:
        lane = self.pool[int(self.rng.integers(0, len(self.pool)))]
        frames = self.index._frames[(lane.recording_id, lane.lane_id)]
        valid = frames["valid_target"]
        anchor = sample_anchor_index(valid, self.rng)
        start = choose_excerpt_start(
            anchor, lane.n_frames, lane.duration_s, rng=self.rng
        )
        excerpt = slice_excerpt(
            lane.n_frames, lane.duration_s, valid, start_idx=start
        )

        end = min(excerpt.end_idx, lane.n_frames)
        actual_len = end - start
        pad_len = EXCERPT_FRAMES - actual_len

        cqt_full = self.index._features[lane.recording_id]["cqt_log"]
        cqt_slice = cqt_full[:, start:end]
        if pad_len > 0:
            cqt_slice = np.pad(cqt_slice, ((0, 0), (0, pad_len)), mode="constant")
        cqt_norm = normalize_cqt(cqt_slice, self.mu_cqt, self.sigma_cqt)

        def take(name: str) -> np.ndarray:
            arr = frames[name][start:end]
            if pad_len > 0:
                if arr.dtype == bool:
                    fill: Any = False
                elif arr.dtype.kind in ("U", "O"):
                    fill = ""
                else:
                    fill = 0
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

        sample = {
            "spec": torch.from_numpy(cqt_norm[np.newaxis].astype(np.float32)),
            "trajectory_type": torch.from_numpy(take("trajectory_type").astype(np.int64)),
            "valid_target": torch.from_numpy(take("valid_target")),
            "padding_mask": torch.from_numpy(padding_mask),
            "pitch_log2_hz": torch.from_numpy(pitch_log2),
            "pitch_cents": torch.from_numpy(pitch_cents),
            "pitch_standardized": torch.from_numpy(pitch_std),
            "phase": torch.from_numpy(take("phase").astype(np.float32)),
            "primitive_id": take("primitive_id"),
            "recording_id": lane.recording_id,
            "lane_id": lane.lane_id,
            "fundamental_hz": lane.fundamental_hz,
            "excerpt_start_idx": start,
            "anchor_idx": anchor,
        }

        if self.compute_pitch_features:
            local_valid = take("valid_target")
            phi_oracle = compute_phi(pitch_cents.astype(np.float64), local_valid)
            sample["phi_oracle"] = torch.from_numpy(phi_oracle)
            if self.estimated_pitch is not None:
                rec_est = self.estimated_pitch[lane.recording_id]
                est_slice = rec_est[start:end]
                if pad_len > 0:
                    est_slice = np.pad(est_slice, (0, pad_len), constant_values=0.0)
                est_cents = log2_hz_to_cents(est_slice.astype(np.float64), lane.fundamental_hz)
                phi_estimated = compute_phi(est_cents, local_valid)
                sample["phi_estimated"] = torch.from_numpy(phi_estimated)
            else:
                sample["phi_estimated"] = torch.zeros_like(sample["phi_oracle"])

        return sample

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._cache is not None:
            return self._cache[idx]
        return self._sample_one()


def collate_excerpts(batch: list[dict[str, Any]]) -> dict[str, Any]:
    padding_mask = torch.stack([b["padding_mask"] for b in batch])
    out: dict[str, Any] = {
        "spec": torch.stack([b["spec"] for b in batch]),
        "trajectory_type": torch.stack([b["trajectory_type"] for b in batch]),
        "valid_target": torch.stack([b["valid_target"] for b in batch]),
        "padding_mask": padding_mask,
        "pitch_log2_hz": torch.stack([b["pitch_log2_hz"] for b in batch]),
        "pitch_cents": torch.stack([b["pitch_cents"] for b in batch]),
        "pitch_standardized": torch.stack([b["pitch_standardized"] for b in batch]),
        "phase": torch.stack([b["phase"] for b in batch]),
        "recording_id": [b["recording_id"] for b in batch],
        "lane_id": [b["lane_id"] for b in batch],
        "fundamental_hz": torch.tensor([b["fundamental_hz"] for b in batch], dtype=torch.float32),
        "excerpt_start_idx": torch.tensor([b["excerpt_start_idx"] for b in batch]),
        "anchor_idx": torch.tensor([b["anchor_idx"] for b in batch]),
        "primitive_id": [b["primitive_id"] for b in batch],
        "lengths": (~padding_mask).sum(dim=1).long(),
    }
    if "phi_oracle" in batch[0]:
        out["phi_oracle"] = torch.stack([b["phi_oracle"] for b in batch])
        out["phi_estimated"] = torch.stack([b["phi_estimated"] for b in batch])
    return out


class FullRecordingDataset(Dataset):
    """Full recording for evaluation (variable length, normalized)."""

    def __init__(
        self,
        index: RecordingLaneIndex,
        recording_ids: list[str],
        mu_cqt: np.ndarray,
        sigma_cqt: np.ndarray,
        *,
        mu_pitch: float | None = None,
        sigma_pitch: float | None = None,
        estimated_pitch: dict[str, np.ndarray] | None = None,
        compute_pitch_features: bool = False,
    ) -> None:
        self.index = index
        self.pool = index.filter_recordings(set(recording_ids))
        self.mu_cqt = mu_cqt
        self.sigma_cqt = sigma_cqt
        self.mu_pitch = mu_pitch
        self.sigma_pitch = sigma_pitch
        self.estimated_pitch = estimated_pitch
        self.compute_pitch_features = compute_pitch_features

    def __len__(self) -> int:
        return len(self.pool)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        lane = self.pool[idx]
        cqt = self.index._features[lane.recording_id]["cqt_log"]
        cqt_norm = normalize_cqt(cqt, self.mu_cqt, self.sigma_cqt)
        frames = self.index._frames[(lane.recording_id, lane.lane_id)]
        times = frames["frame_time_s"]
        padding_mask = times >= lane.duration_s
        pitch_log2 = frames["pitch_log2_hz"].astype(np.float32)
        pitch_cents = log2_hz_to_cents(pitch_log2, lane.fundamental_hz).astype(np.float32)
        if self.mu_pitch is not None and self.sigma_pitch is not None:
            pitch_std = standardize_pitch(pitch_cents, self.mu_pitch, self.sigma_pitch)
        else:
            pitch_std = np.zeros_like(pitch_cents)
        valid_target = frames["valid_target"]
        sample = {
            "spec": torch.from_numpy(cqt_norm[np.newaxis].astype(np.float32)),
            "trajectory_type": torch.from_numpy(frames["trajectory_type"].astype(np.int64)),
            "valid_target": torch.from_numpy(valid_target),
            "padding_mask": torch.from_numpy(padding_mask),
            "pitch_log2_hz": torch.from_numpy(pitch_log2),
            "pitch_cents": torch.from_numpy(pitch_cents),
            "pitch_standardized": torch.from_numpy(pitch_std),
            "frame_time_s": frames["frame_time_s"],
            "primitive_id": frames["primitive_id"],
            "recording_id": lane.recording_id,
            "lane_id": lane.lane_id,
            "duration_s": lane.duration_s,
            "fundamental_hz": lane.fundamental_hz,
        }
        if self.compute_pitch_features:
            phi_oracle = compute_phi(pitch_cents.astype(np.float64), valid_target)
            sample["phi_oracle"] = torch.from_numpy(phi_oracle)
            if self.estimated_pitch is not None:
                est = self.estimated_pitch[lane.recording_id][: len(pitch_cents)]
                est_cents = log2_hz_to_cents(est.astype(np.float64), lane.fundamental_hz)
                phi_estimated = compute_phi(est_cents, valid_target)
                sample["phi_estimated"] = torch.from_numpy(phi_estimated)
            else:
                sample["phi_estimated"] = torch.zeros_like(sample["phi_oracle"])
        return sample


def collate_variable_length(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_t = max(b["spec"].shape[-1] for b in batch)
    specs, types, valid, pad_mask, pitch, times, prim_ids = [], [], [], [], [], [], []
    for b in batch:
        t = b["spec"].shape[-1]
        pad = max_t - t
        spec = b["spec"]
        if pad:
            spec = torch.nn.functional.pad(spec, (0, pad))
        specs.append(spec)
        for key, container, fill in (
            ("trajectory_type", types, 0),
            ("valid_target", valid, False),
            ("padding_mask", pad_mask, True),
            ("pitch_log2_hz", pitch, 0.0),
        ):
            tensor = b[key]
            if pad:
                tensor = torch.nn.functional.pad(tensor, (0, pad), value=fill)
            container.append(tensor)
        ft = b["frame_time_s"]
        if pad:
            ft = np.pad(ft, (0, pad), constant_values=ft[-1] if len(ft) else 0.0)
        times.append(ft)
        prim_ids.append(b["primitive_id"])

    result = {
        "spec": torch.stack(specs),
        "trajectory_type": torch.stack(types),
        "valid_target": torch.stack(valid),
        "padding_mask": torch.stack(pad_mask),
        "pitch_log2_hz": torch.stack(pitch),
        "pitch_cents": torch.stack([_pad_1d(b["pitch_cents"], max_t - b["pitch_cents"].shape[-1], 0.0) for b in batch]),
        "pitch_standardized": torch.stack(
            [_pad_1d(b["pitch_standardized"], max_t - b["pitch_standardized"].shape[-1], 0.0) for b in batch]
        ),
        "frame_time_s": times,
        "primitive_id": prim_ids,
        "recording_id": [b["recording_id"] for b in batch],
        "lane_id": [b["lane_id"] for b in batch],
        "fundamental_hz": torch.tensor([b["fundamental_hz"] for b in batch], dtype=torch.float32),
        "lengths": (~torch.stack(pad_mask)).sum(dim=1).long(),
    }
    if "phi_oracle" in batch[0]:
        result["phi_oracle"] = torch.stack([
            _pad_2d(b["phi_oracle"], max_t - b["phi_oracle"].shape[0]) for b in batch
        ])
        result["phi_estimated"] = torch.stack([
            _pad_2d(b["phi_estimated"], max_t - b["phi_estimated"].shape[0]) for b in batch
        ])
    return result


def _pad_1d(tensor: torch.Tensor, pad: int, fill: float) -> torch.Tensor:
    if pad:
        return torch.nn.functional.pad(tensor, (0, pad), value=fill)
    return tensor


def _pad_2d(tensor: torch.Tensor, pad: int) -> torch.Tensor:
    if pad:
        return torch.nn.functional.pad(tensor, (0, 0, 0, pad), value=0.0)
    return tensor
