"""PyTorch dataset for CQT spectrogram trajectory classification."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

FULL_LABEL_TO_IDX: dict[str, int] = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "silent": 4,
}

# Default 5-class config (backward compatible).
LABEL_TO_IDX = FULL_LABEL_TO_IDX
IDX_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_TO_IDX.items()}
NUM_CLASSES = len(LABEL_TO_IDX)
CLASS_NAMES = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]


def build_label_config(
    exclude_labels: list[str] | None = None,
) -> tuple[dict[str, int], dict[int, str], list[str], int]:
    excluded = {normalize_label(label) for label in (exclude_labels or [])}
    remaining = [
        label for label in FULL_LABEL_TO_IDX if label not in excluded
    ]
    if not remaining:
        raise ValueError("At least one label must remain after exclusions.")
    label_to_idx = {label: idx for idx, label in enumerate(remaining)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    class_names = [idx_to_label[i] for i in range(len(remaining))]
    return label_to_idx, idx_to_label, class_names, len(class_names)


def normalize_label(label: object) -> str:
    if isinstance(label, str):
        return label
    return str(int(label))


def label_to_idx(label: object, label_map: dict[str, int] | None = None) -> int:
    token = normalize_label(label)
    mapping = label_map or LABEL_TO_IDX
    if token not in mapping:
        raise ValueError(f"Unknown label: {label!r}")
    return mapping[token]


def resolve_image_path(image_path: str | Path, project_root: Path) -> Path:
    path = Path(image_path)
    if path.is_absolute() and path.exists():
        return path
    candidate = project_root / path
    if candidate.exists():
        return candidate
    if path.exists():
        return path
    raise FileNotFoundError(f"Image not found: {image_path}")


def build_transforms(*, augment: bool = False) -> transforms.Compose:
    steps: list[Callable] = [transforms.ToTensor()]
    if augment:
        steps.insert(
            0,
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        )
    return transforms.Compose(steps)


class SpecImageDataset(Dataset):
    """Dataset over CNN spectrogram PNGs referenced by metadata.csv."""

    def __init__(
        self,
        metadata_path: str | Path,
        *,
        project_root: str | Path | None = None,
        indices: list[int] | None = None,
        augment: bool = False,
        label_to_idx_map: dict[str, int] | None = None,
    ) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[2])
        self.metadata_path = Path(metadata_path)
        self.label_to_idx_map = label_to_idx_map or LABEL_TO_IDX
        frame = pd.read_csv(self.metadata_path)
        if indices is not None:
            frame = frame.loc[indices].reset_index(drop=True)
        self.frame = frame
        self.transform = build_transforms(augment=augment)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[idx]
        image_path = resolve_image_path(row["image_path"], self.project_root)
        label = label_to_idx(row["label"], self.label_to_idx_map)
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, label

    def row(self, idx: int) -> pd.Series:
        return self.frame.iloc[idx]
