"""Loss functions for framewise training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FramewiseTypeLoss(nn.Module):
    """Unweighted cross-entropy on (~padding_mask) & valid_target frames."""

    def __init__(self, ignore_index: int = -1) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        type_logits: torch.Tensor,
        trajectory_type: torch.Tensor,
        valid_target: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = (~padding_mask) & valid_target
        labels = trajectory_type.clone()
        labels[~mask] = self.ce.ignore_index
        b, t, c = type_logits.shape
        return self.ce(type_logits.reshape(b * t, c), labels.reshape(b * t))


class FramewiseMultitaskLoss(nn.Module):
    """Type CE + pitch Smooth L1 on fold-standardized cents (B1+)."""

    def __init__(
        self,
        *,
        lambda_type: float = 1.0,
        lambda_pitch: float = 1.0,
        ignore_index: int = -1,
    ) -> None:
        super().__init__()
        self.type_loss = FramewiseTypeLoss(ignore_index=ignore_index)
        self.lambda_type = lambda_type
        self.lambda_pitch = lambda_pitch

    def forward(
        self,
        type_logits: torch.Tensor,
        pitch_pred: torch.Tensor,
        trajectory_type: torch.Tensor,
        pitch_target: torch.Tensor,
        valid_target: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        type_l = self.type_loss(type_logits, trajectory_type, valid_target, padding_mask)
        mask = (~padding_mask) & valid_target
        if mask.any():
            pitch_l = F.smooth_l1_loss(pitch_pred[mask], pitch_target[mask])
        else:
            pitch_l = pitch_pred.sum() * 0.0
        total = self.lambda_type * type_l + self.lambda_pitch * pitch_l
        return total, {"type_loss": type_l, "pitch_loss": pitch_l}
